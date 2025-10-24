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
    seed: Optional[str] = None
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

class PlaylistAttributes(BaseModel):
    url: Optional[str] = None

class PlaylistData(BaseModel):
    id: str
    attributes: Optional[PlaylistAttributes] = None

class PlaylistResponse(BaseModel):
    data: List[PlaylistData]

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
    def clean(s: str) -> str: s = s.lower(); s = re.sub(r"[^\w\s]", " ", s); s = re.sub(r"\s+", " ", s).strip(); return s
    return f"{clean(artist)} :: {clean(title)}"
def _db() -> sqlite3.Connection: conn = sqlite3.connect(CACHE_DB); conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, video_id TEXT, ts INTEGER)"); return conn
def _cache_get(conn: sqlite3.Connection, k: str) -> Optional[str]: cur = conn.execute("SELECT video_id FROM cache WHERE k=?", (k,)); row = cur.fetchone(); return row[0] if row else None
def _cache_put(conn: sqlite3.Connection, k: str, vid: str) -> None: conn.execute("INSERT OR REPLACE INTO cache (k, video_id, ts) VALUES (?,?,?)", (k, vid, int(time.time()))); conn.commit()
def _extract_video_id(u: str) -> Optional[str]: m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", u) or re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", u); return m.group(1) if m else None

def _discogs_find_video_id(artist: str, title: str) -> Optional[str]:
    if not DISCOGS_TOKEN: return None
    q = f"{artist} - {title}"
    try:
        r = requests.get("https://api.discogs.com/database/search", params={"q": q, "per_page": 1, "page": 1, "token": DISCOGS_TOKEN}, timeout=12)
        if r.status_code != 200: return None
        results = r.json().get("results", [])
        if not results: return None
        res_url = results[0].get("resource_url")
        if not res_url: return None
        time.sleep(1);
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
        seed_val = 97
        if req.seed is not None:
            try: seed_val = int(req.seed)
            except ValueError: log.warning(f"Invalid seed value '{req.seed}', using default 97.")
        spins = simulate_rotations(
            entries=entries, start_time=start_dt, hours=req.hours,
            average_song_minutes=3.5, min_gap_minutes=req.repeat_gap_min,
            seed=seed_val,
        )
        return {"tracks": [{"timestamp": s.timestamp.isoformat(timespec="minutes"), "artist": s.artist, "title": s.title, "source_rank": s.source_rank} for s in spins],}
    except Exception as e:
        log.exception("simulate failed")
        return JSONResponse(status_code=500, content={"detail": f"simulate failed: {e}"})

@app.post("/v1/yt/resolve")
def yt_resolve(req: ResolveReq):
    conn = None
    try:
        conn = _db(); ids, dropped, seen = [], [], set()
        for t in req.tracks:
            if len(ids) >= (req.limit or 50): break
            k = _norm_key(t.artist, t.title);
            vid = _cache_get(conn, k)
            # --- CORRECTED Syntax: Moved logic to separate lines ---
            if not vid:
                time.sleep(0.5)
                vid = _discogs_find_video_id(t.artist, t.title)
                if vid:
                    _cache_put(conn, k, vid)
            # --- END CORRECTION ---
            if vid and vid not in seen:
                seen.add(vid);
                ids.append(vid)
        return {"ids": ids}
    except Exception as e:
        log.exception("yt/resolve failed");
        return JSONResponse(status_code=500, content={"detail": f"yt/resolve failed: {e}"})
    finally:
        if conn: conn.close()

@app.get("/v1/apple/dev-token")
def apple_dev_token():
    try:
        if not (APPLE_TEAM_ID and APPLE_KEY_ID and APPLE_PRIVATE_KEY):
            log.error("CRITICAL: Apple keys not configured. TEAM_ID, KEY_ID, or PRIVATE_KEY is missing or empty after processing.")
            return JSONResponse(status_code=500, content={"detail": "Apple keys not configured"})
        now = int(time.time())
        payload = {"iss": APPLE_TEAM_ID, "iat": now, "exp": now + (60 * 55)}
        token = jwt.encode(payload, APPLE_PRIVATE_KEY, algorithm="ES256", headers={"kid": APPLE_KEY_ID})
        log.info(f"Successfully generated a dev token. Length: {len(token)}")
        return {"token": token, "storefront": APPLE_STOREFRONT}
    except Exception as e:
        log.exception(f"CRITICAL: dev-token failed during jwt.encode(). Error: {e}")
        return JSONResponse(status_code=500, content={"detail": f"dev-token failed: {e}"})

@app.post("/v1/apple/create-playlist")
def apple_create_playlist(req: AppleCreateRequest) -> Dict[str, any]:
    if not (APPLE_TEAM_ID and APPLE_KEY_ID and APPLE_PRIVATE_KEY):
        log.error("Apple keys not configured during playlist creation.")
        return JSONResponse(status_code=500, content={"detail": "Apple keys not configured"})

    # --- Generate Dev Token ---
    try:
        now = int(time.time())
        dev_token = jwt.encode({"iss": APPLE_TEAM_ID, "iat": now, "exp": now + 1500},
                               APPLE_PRIVATE_KEY, algorithm="ES256",
                               headers={"kid": APPLE_KEY_ID})
    except Exception as e:
        log.exception("Dev token generation failed in create-playlist")
        return JSONResponse(status_code=500, content={"detail": f"Dev token generation failed: {e}"})

    headers = {"Authorization": f"Bearer {dev_token}", "Music-User-Token": req.userToken}
    storefront = APPLE_STOREFRONT or "us"

    # --- Create Playlist ---
    playlist_id: Optional[str] = None
    playlist_url: Optional[str] = None
    try:
        playlist_payload = {
            "attributes": { "name": req.name, "description": "Generated by TapeDeck Time Machine" }
        }
        log.info(f"Creating playlist '{req.name}'...")
        r_create = requests.post("https://api.music.apple.com/v1/me/library/playlists",
                                 headers=headers, json=playlist_payload, timeout=20)
        r_create.raise_for_status()
        created_data = r_create.json()
        playlist_response = PlaylistResponse(**created_data)
        if playlist_response.data and playlist_response.data[0]:
            playlist_id = playlist_response.data[0].id
            if playlist_response.data[0].attributes: playlist_url = playlist_response.data[0].attributes.url
            log.info(f"Playlist created successfully. ID: {playlist_id}, URL: {playlist_url}")
        else: log.error("Playlist created, but response parsing failed."); raise ValueError("Playlist created, bad response.")
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 500
        error_detail = f"Playlist creation request failed: {e}"; try:
            if e.response is not None: error_detail = e.response.json().get('errors', [{}])[0].get('detail', str(e))
        except Exception: pass
        log.exception("Failed to create AM playlist (RequestException)"); return JSONResponse(status_code=status_code, content={"detail": error_detail})
    except Exception as e:
        log.exception("Failed to create AM playlist (Other Exception)"); return JSONResponse(status_code=500, content={"detail": f"Playlist creation failed: {e}"})

    if not playlist_id: log.error("Playlist creation ok but no ID extracted."); return JSONResponse(status_code=500, content={"detail": "Playlist created, failed ID retrieval."})

    # --- Add Tracks ---
    song_ids_to_add = []
    log.info(f"Searching for {len(req.tracks)} tracks to add...")
    search_headers = {"Authorization": f"Bearer {dev_token}"}
    for track in req.tracks[:100]:
        try:
            params = {"term": f"{track.artist} {track.title}", "limit": 1, "types": "songs"}
            r_search = requests.get(f"https://api.music.apple.com/v1/catalog/{storefront}/search", headers=search_headers, params=params, timeout=10)
            r_search.raise_for_status()
            results = r_search.json().get("results", {}).get("songs", {}).get("data", [])
            if results: song_ids_to_add.append({"id": results[0]["id"], "type": "songs"})
            else: log.warning(f"Could not find track: '{track.artist} - {track.title}'")
        except Exception as e: log.warning(f"Search failed for track '{track.artist} - {track.title}': {e}")
        time.sleep(0.15)

    added_count = 0
    if not song_ids_to_add:
        log.warning("No tracks found to add.")
        return { "detail": "Playlist created, but no tracks matched.", "added_count": 0, "total_tracks_searched": len(req.tracks), "playlist_id": playlist_id, "playlist_url": playlist_url, "storefront": storefront }

    try:
        log.info(f"Adding {len(song_ids_to_add)} tracks to playlist {playlist_id}...")
        add_payload = {"data": song_ids_to_add}
        r_add = requests.post(f"https://api.music.apple.com/v1/me/library/playlists/{playlist_id}/tracks", headers=headers, json=add_payload, timeout=30)
        r_add.raise_for_status()
        added_count = len(song_ids_to_add)
        log.info("Tracks added successfully.")
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 500
        error_detail = f"Adding tracks failed: {e}"; try:
            if e.response is not None: error_detail = e.response.json().get('errors', [{}])[0].get('detail', str(e))
        except Exception: pass
        log.exception("Failed to add tracks (RequestException)"); return JSONResponse(status_code=status_code, content={ "detail": f"Playlist created, adding tracks failed: {error_detail}", "added_count": 0, "total_tracks_searched": len(req.tracks), "playlist_id": playlist_id, "playlist_url": playlist_url, "storefront": storefront })
    except Exception as e:
        log.exception("Failed to add tracks (Other Exception)"); return JSONResponse(status_code=500, content={ "detail": f"Playlist created, adding tracks failed: {e}", "added_count": 0, "total_tracks_searched": len(req.tracks), "playlist_id": playlist_id, "playlist_url": playlist_url, "storefront": storefront })

    # --- Return Success ---
    return { "added_count": added_count, "total_tracks_searched": len(req.tracks), "playlist_id": playlist_id, "playlist_url": playlist_url, "storefront": storefront }

@app.get("/")
def root():
    return PlainTextResponse(f"{APP_NAME} OK")

