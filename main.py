# main.py
# FastAPI backend for TapeDeck Time Machine
import os
import re
import time
import json
import sqlite3
import logging
from typing import List, Optional, Dict
from datetime import datetime

import requests
import jwt  # PyJWT

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

# --- Import simulator bits ---
from playlist_creator import (
    resolve_candidate_slugs,
    nearest_billboard_chart_date,
    fetch_chart_entries_strict,
    simulate_rotations,
)

# -------------------- Config / Logging --------------------
APP_NAME = "timedeck-api"
log = logging.getLogger(APP_NAME)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

ALLOWED = os.getenv("ALLOWED_ORIGINS", "https://anthonyklemm.github.io,https://tapedecktimemachine.com,https://www.tapedecktimemachine.com").split(",")
CACHE_DIR = os.getenv("CACHE_DIR", "/data")  # Correct path for Render Persistent Disks
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_DB = os.path.join(CACHE_DIR, "yt_cache.sqlite")

DISCOGS_TOKEN = os.getenv("DISCOGS_TOKEN", "").strip()

# --- MODIFICATIONS START HERE ---

# Strip whitespace from env vars
APPLE_TEAM_ID = os.getenv("APPLE_TEAM_ID", "").strip()
APPLE_KEY_ID = os.getenv("APPLE_KEY_ID", "").strip()
APPLE_STOREFRONT = os.getenv("APPLE_STOREFRONT", "us").strip()
APPLE_PRIVATE_KEY_RAW = os.getenv("APPLE_PRIVATE_KEY", "").strip()
APPLE_PRIVATE_KEY = None

# Check if the key is mangled (a single line with \n)
if APPLE_PRIVATE_KEY_RAW and "\\n" in APPLE_PRIVATE_KEY_RAW:
    log.info("Found escaped newlines in APPLE_PRIVATE_KEY, replacing them.")
    # Rebuild the key with actual newlines
    APPLE_PRIVATE_KEY = APPLE_PRIVATE_KEY_RAW.replace("\\n", "\n")
elif APPLE_PRIVATE_KEY_RAW:
    # Assume it's a correct, multi-line key
    APPLE_PRIVATE_KEY = APPLE_PRIVATE_KEY_RAW

# --- MODIFICATIONS END HERE ---

# -------------------- Pydantic Models (defined at the top) --------------------
class SimReq(BaseModel):
    date: str
    genre: Optional[str] = None
    station: Optional[str] = None
    hours: float = 3.0
    repeat_gap_min: int = Field(90, ge=0)
    seed: str = "97"
    limit: int = Field(40, ge=1, le=1000)

class Track(BaseModel):
    artist: str
    title: str

class ResolveReq(BaseModel):
    tracks: List[Track]
    limit: Optional[int] = 50

class AppleCreateRequest(BaseModel):
    userToken: str
    name: str
    tracks: List[Track]

# -------------------- FastAPI App + CORS --------------------
app = FastAPI(title="TapeDeck API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=600,
)

# -------------------- Utils --------------------
def _norm_key(artist: str, title: str) -> str:
    def clean(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    return f"{clean(artist)} :: {clean(title)}"

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, video_id TEXT, ts INTEGER)")
    return conn

def _cache_get(conn: sqlite3.Connection, k: str) -> Optional[str]:
    cur = conn.execute("SELECT video_id FROM cache WHERE k=?", (k,))
    row = cur.fetchone()
    return row[0] if row else None

def _cache_put(conn: sqlite3.Connection, k: str, vid: str) -> None:
    conn.execute("INSERT OR REPLACE INTO cache (k, video_id, ts) VALUES (?,?,?)", (k, vid, int(time.time())))
    conn.commit()

def _extract_video_id(u: str) -> Optional[str]:
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", u) or re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", u)
    return m.group(1) if m else None

def _discogs_find_video_id(artist: str, title: str) -> Optional[str]:
    if not DISCOGS_TOKEN: return None
    q = f"{artist} - {title}"
    try:
        r = requests.get(
            "https://api.discogs.com/database/search",
            params={"q": q, "per_page": 1, "page": 1, "token": DISCOGS_TOKEN},
            timeout=12,
        )
        if r.status_code != 200: return None
        results = r.json().get("results", [])
        if not results: return None

        res_url = results[0].get("resource_url")
        if not res_url: return None

        time.sleep(1) # Be nice to Discogs API
        d = requests.get(res_url, timeout=12)
        if d.status_code != 200: return None
        
        for v in d.json().get("videos", []) or []:
            uri = v.get("uri") or ""
            if "youtube.com" in uri:
                vid = _extract_video_id(uri)
                if vid: return vid
    except Exception as e:
        log.warning("Discogs lookup failed for %s — %s", q, e)
    return None

# -------------------- Endpoints --------------------
@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat() + "Z"}

@app.post("/v1/simulate")
def simulate(req: SimReq):
    try:
        slugs = resolve_candidate_slugs(req.station, req.genre)
        d = datetime.strptime(req.date, "%Y-%m-%d")
        chart_date = nearest_billboard_chart_date(d)
        entries = fetch_chart_entries_strict(slugs, chart_date, limit=req.limit, tolerance_days=7)
        start_dt = d.replace(hour=6, minute=0, second=0, microsecond=0)
        spins = simulate_rotations(
            entries=entries, start_time=start_dt, hours=req.hours,
            average_song_minutes=3.5, min_gap_minutes=req.repeat_gap_min,
            seed=int(req.seed or "97"),
        )
        return {
            "tracks": [
                {"timestamp": s.timestamp.isoformat(timespec="minutes"), "artist": s.artist, "title": s.title, "source_rank": s.source_rank}
                for s in spins
            ],
        }
    except Exception as e:
        log.exception("simulate failed")
        return JSONResponse(status_code=500, content={"detail": f"simulate failed: {e}"})

@app.post("/v1/yt/resolve")
def yt_resolve(req: ResolveReq):
    try:
        conn = _db()
        ids, dropped, seen = [], [], set()
        for t in req.tracks:
            if len(ids) >= (req.limit or 50): break
            k = _norm_key(t.artist, t.title)
            vid = _cache_get(conn, k)
            if not vid:
                time.sleep(0.5) # Pace requests to avoid rate limits
                vid = _discogs_find_video_id(t.artist, t.title)
                if vid: _cache_put(conn, k, vid)
            if vid and vid not in seen:
                seen.add(vid)
                ids.append(vid)
        return {"ids": ids}
    except Exception as e:
        log.exception("yt/resolve failed")
        return JSONResponse(status_code=500, content={"detail": f"yt/resolve failed: {e}"})

@app.get("/v1/apple/dev-token")
def apple_dev_token():
    try:
        # --- MODIFIED: Stricter check ---
        if not (APPLE_TEAM_ID and APPLE_KEY_ID and APPLE_PRIVATE_KEY):
            log.error("Apple keys not configured. TEAM_ID, KEY_ID, or PRIVATE_KEY is missing.")
            return JSONResponse(status_code=500, content={"detail": "Apple keys not configured"})
        # --- END MODIFIED ---
            
        now = int(time.time())
        payload = {"iss": APPLE_TEAM_ID, "iat": now, "exp": now + (60 * 55)}
        
        # This will now use the un-mangled key
        token = jwt.encode(payload, APPLE_PRIVATE_KEY, algorithm="ES256", headers={"kid": APPLE_KEY_ID, "alg": "ES256"})
        
        return {"token": token, "storefront": APPLE_STOREFRONT}
    except Exception as e:
        # This block will now catch any PyJWT errors from a bad (but non-empty) key
        log.exception("dev-token failed")
        return JSONResponse(status_code=500, content={"detail": f"dev-token failed: {e}"})

@app.post("/v1/apple/create-playlist")
def apple_create_playlist(req: AppleCreateRequest):
    if not (APPLE_TEAM_ID and APPLE_KEY_ID and APPLE_PRIVATE_KEY):
        return JSONResponse(status_code=500, content={"detail": "Apple keys not configured"})
    try:
        now = int(time.time())
        dev_token = jwt.encode({"iss": APPLE_TEAM_ID, "iat": now, "exp": now + 1500}, APPLE_PRIVATE_KEY, algorithm="ES256", headers={"kid": APPLE_KEY_ID, "alg": "ES256"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Dev token generation failed: {e}"})
    
    headers = {"Authorization": f"Bearer {dev_token}", "Music-User-Token": req.userToken}
    
    try:
        playlist_payload = {"attributes": {"name": req.name, "description": "Generated by TapeDeck Time Machine"}}
        r_create = requests.post("https://api.music.apple.com/v1/me/library/playlists", headers=headers, json=playlist_payload)
        r_create.raise_for_status()
        playlist_id = r_create.json()["data"][0]["id"]
    except Exception as e:
        log.exception("Failed to create AM playlist")
        return JSONResponse(status_code=500, content={"detail": f"Playlist creation failed: {e}"})

    song_ids = []
    storefront = APPLE_STOREFRONT or "us"
    for track in req.tracks[:100]:
        try:
            params = {"term": f"{track.artist} {track.title}", "limit": 1, "types": "songs"}
            r_search = requests.get(f"https://api.music.apple.com/v1/catalog/{storefront}/search", headers=headers, params=params)
            r_search.raise_for_status()
            results = r_search.json().get("results", {}).get("songs", {}).get("data", [])
            if results: song_ids.append({"id": results[0]["id"], "type": "songs"})
        except Exception:
            log.warning(f"Could not find track '{track.artist} - {track.title}'")
        time.sleep(0.2)
    
    if not song_ids:
        return {"detail": "Playlist created, but no tracks matched.", "added_count": 0, "total_tracks": len(req.tracks)}

    try:
        r_add = requests.post(f"https://api.music.apple.com/v1/me/library/playlists/{playlist_id}/tracks", headers=headers, json={"data": song_ids})
        r_add.raise_for_status()
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Adding tracks failed: {e}"})
        
    return {"added_count": len(song_ids), "total_tracks": len(req.tracks), "playlist_id": playlist_id}

@app.get("/")
def root():
    return PlainTextResponse(f"{APP_NAME} OK")