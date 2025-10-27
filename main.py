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

from fastapi import FastAPI, Query
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

ALLOWED = os.getenv(
    "ALLOWED_ORIGINS",
    "https://anthonyklemm.github.io,https://tapedecktimemachine.com,https://www.tapedecktimemachine.com",
).split(",")
CACHE_DIR = os.getenv("CACHE_DIR", "/data")  # Render persistent disk path
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_DB = os.path.join(CACHE_DIR, "yt_cache.sqlite")

DISCOGS_TOKEN = os.getenv("DISCOGS_TOKEN", "").strip()

APPLE_TEAM_ID = os.getenv("APPLE_TEAM_ID", "").strip()
APPLE_KEY_ID = os.getenv("APPLE_KEY_ID", "").strip()
APPLE_STOREFRONT = os.getenv("APPLE_STOREFRONT", "us").strip()
APPLE_PRIVATE_KEY_RAW = os.getenv("APPLE_PRIVATE_KEY", "").strip()
APPLE_PRIVATE_KEY = None

log.info("--- Apple Key Diagnostics ---")
log.info(f"APPLE_TEAM_ID loaded: {bool(APPLE_TEAM_ID)}, Length: {len(APPLE_TEAM_ID)}")
log.info(f"APPLE_KEY_ID loaded: {bool(APPLE_KEY_ID)}, Length: {len(APPLE_KEY_ID)}")
log.info(f"APPLE_PRIVATE_KEY_RAW loaded: {bool(APPLE_PRIVATE_KEY_RAW)}, Length: {len(APPLE_PRIVATE_KEY_RAW)}")

if APPLE_PRIVATE_KEY_RAW and "\\n" in APPLE_PRIVATE_KEY_RAW:
    log.info("Found escaped newlines in APPLE_PRIVATE_KEY, replacing them.")
    APPLE_PRIVATE_KEY = APPLE_PRIVATE_KEY_RAW.replace("\\n", "\n")
elif APPLE_PRIVATE_KEY_RAW:
    log.info("No escaped newlines found in private key. Using as-is.")
    APPLE_PRIVATE_KEY = APPLE_PRIVATE_KEY_RAW
else:
    log.warning("APPLE_PRIVATE_KEY is empty after loading and stripping.")
log.info("-----------------------------")

# -------------------- Pydantic Models --------------------
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
app = FastAPI(title="TapeDeck API", version="1.2")

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
    # keep simple & robust
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", u) or re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", u)
    return m.group(1) if m else None

def _discogs_find_video_id(artist: str, title: str) -> Optional[str]:
    if not DISCOGS_TOKEN:
        return None
    q = f"{artist} - {title}"
    try:
        r = requests.get(
            "https://api.discogs.com/database/search",
            params={"q": q, "per_page": 1, "page": 1, "token": DISCOGS_TOKEN},
            timeout=12,
        )
        if r.status_code != 200:
            return None
        results = r.json().get("results", [])
        if not results:
            return None

        res_url = results[0].get("resource_url")
        if not res_url:
            return None

        time.sleep(1)  # be nice
        d = requests.get(res_url, timeout=12)
        if d.status_code != 200:
            return None

        for v in d.json().get("videos", []) or []:
            uri = v.get("uri") or ""
            if "youtube.com" in uri or "youtu.be" in uri:
                vid = _extract_video_id(uri)
                if vid:
                    return vid
    except Exception as e:
        log.warning("Discogs lookup failed for %s — %s", q, e)
    return None

def _require_apple_keys():
    if not (APPLE_TEAM_ID and APPLE_KEY_ID and APPLE_PRIVATE_KEY):
        raise RuntimeError("Apple keys not configured")

def _mint_dev_token(ttl_seconds: int = 55 * 60) -> str:
    """Create a short-lived Apple developer token."""
    _require_apple_keys()
    now = int(time.time())
    payload = {"iss": APPLE_TEAM_ID, "iat": now, "exp": now + ttl_seconds}
    token = jwt.encode(
        payload,
        APPLE_PRIVATE_KEY,
        algorithm="ES256",
        headers={"kid": APPLE_KEY_ID},
    )
    return token

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
            entries=entries,
            start_time=start_dt,
            hours=req.hours,
            average_song_minutes=3.5,
            min_gap_minutes=req.repeat_gap_min,
            seed=int(req.seed or "97"),
        )
        return {
            "tracks": [
                {
                    "timestamp": s.timestamp.isoformat(timespec="minutes"),
                    "artist": s.artist,
                    "title": s.title,
                    "source_rank": s.source_rank,
                }
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
        ids, seen = [], set()
        for t in req.tracks:
            if len(ids) >= (req.limit or 50):
                break
            k = _norm_key(t.artist, t.title)
            vid = _cache_get(conn, k)
            if not vid:
                time.sleep(0.5)  # Pace requests
                vid = _discogs_find_video_id(t.artist, t.title)
                if vid:
                    _cache_put(conn, k, vid)
            if vid and vid not in seen:
                seen.add(vid)
                ids.append(vid)
        return {"ids": ids}
    except Exception as e:
        log.exception("yt/resolve failed")
        return JSONResponse(status_code=500, content={"detail": f"yt/resolve failed: {e}"})

# ---------- Apple: developer token ----------
@app.get("/v1/apple/dev-token")
def apple_dev_token():
    try:
        token = _mint_dev_token()
        log.info(f"Successfully generated a dev token. Length: {len(token)}")
        return {"token": token, "storefront": APPLE_STOREFRONT}
    except Exception as e:
        log.exception("dev-token failed")
        return JSONResponse(status_code=500, content={"detail": f"dev-token failed: {e}"})

# ---------- Apple: create library playlist, then add found songs ----------
@app.post("/v1/apple/create-playlist")
def apple_create_playlist(req: AppleCreateRequest):
    try:
        dev_token = _mint_dev_token(ttl_seconds=25 * 60)
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Dev token generation failed: {e}"})

    headers = {
        "Authorization": f"Bearer {dev_token}",
        "Music-User-Token": req.userToken,
    }

    # 1) Create empty library playlist
    try:
        payload = {"attributes": {"name": req.name, "description": "Generated by TapeDeck Time Machine"}}
        r_create = requests.post(
            "https://api.music.apple.com/v1/me/library/playlists",
            headers=headers,
            json=payload,
            timeout=20,
        )
        r_create.raise_for_status()
        playlist_id = r_create.json()["data"][0]["id"]  # library IDs look like "p.xxxxx"
    except Exception as e:
        log.exception("Failed to create AM playlist")
        return JSONResponse(status_code=500, content={"detail": f"Playlist creation failed: {e}"})

    # 2) Resolve songs (simple 1-by-1 search)
    song_ids = []
    storefront = APPLE_STOREFRONT or "us"
    for track in req.tracks[:100]:
        try:
            params = {"term": f"{track.artist} {track.title}", "limit": 1, "types": "songs"}
            r_search = requests.get(
                f"https://api.music.apple.com/v1/catalog/{storefront}/search",
                headers=headers,
                params=params,
                timeout=12,
            )
            if r_search.status_code == 200:
                results = r_search.json().get("results", {}).get("songs", {}).get("data", [])
                if results:
                    song_ids.append({"id": results[0]["id"], "type": "songs"})
        except Exception:
            log.warning("Could not find track '%s - %s'", track.artist, track.title)
        time.sleep(0.18)

    # 3) Add to playlist (ok if empty)
    added = 0
    if song_ids:
        try:
            r_add = requests.post(
                f"https://api.music.apple.com/v1/me/library/playlists/{playlist_id}/tracks",
                headers=headers,
                json={"data": song_ids},
                timeout=20,
            )
            r_add.raise_for_status()
            added = len(song_ids)
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"Adding tracks failed: {e}"})

    # Library playlists usually do NOT have public URLs
    return {
        "added_count": added,
        "total_tracks": len(req.tracks),
        "playlist_id": playlist_id,
        "id": playlist_id,
        "storefront": storefront,
        "url": None,  # no public URL for library playlists; frontend will fall back correctly
    }

# ---------- Apple: song meta for a given song id (used by the mini-player title) ----------
@app.get("/v1/apple/song-meta")
def apple_song_meta(id: str = Query(..., description="Apple song id (catalog)"), storefront: str = Query("us")):
    """
    Returns minimal metadata for a catalog song id so the web app can show
    title/artist even when MusicKit's nowPlayingItem is sparse.
    Response: { id, name, artistName, url }
    """
    try:
        dev_token = _mint_dev_token(ttl_seconds=10 * 60)
        url = f"https://api.music.apple.com/v1/catalog/{storefront}/songs/{id}"
        r = requests.get(url, headers={"Authorization": f"Bearer {dev_token}"}, timeout=12)
        if r.status_code != 200:
            return JSONResponse(status_code=r.status_code, content={"detail": f"Apple catalog error: {r.text}"})
        data = r.json().get("data", [])
        if not data:
            return JSONResponse(status_code=404, content={"detail": "Song not found"})
        attrs = data[0].get("attributes", {}) or {}
        return {
            "id": id,
            "name": attrs.get("name"),
            "artistName": attrs.get("artistName"),
            "url": attrs.get("url"),
        }
    except Exception as e:
        log.exception("song-meta failed")
        return JSONResponse(status_code=500, content={"detail": f"song-meta failed: {e}"})

@app.get("/")
def root():
    return PlainTextResponse(f"{APP_NAME} OK")
