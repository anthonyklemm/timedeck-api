# main.py
# FastAPI backend for TapeDeck Time Machine
import os
import re
import time
import json
import sqlite3
import logging
import base64  # Added for Spotify auth
import secrets  # NEW: for state/session ids
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from urllib.parse import urlencode  # Added for Spotify auth

import requests
import jwt  # PyJWT

from fastapi import FastAPI, Query, HTTPException, Cookie  # NEW: Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
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

# --- General Config ---
ALLOWED = os.getenv(
    "ALLOWED_ORIGINS",
    "https://anthonyklemm.github.io,https://tapedecktimemachine.com,https://www.tapedecktimemachine.com,http://127.0.0.1:8888,http://localhost:8000",
).split(",")
CACHE_DIR = os.getenv("CACHE_DIR", "/data")  # Render persistent disk path
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_DB = os.path.join(CACHE_DIR, "yt_cache.sqlite")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://tapedecktimemachine.com/app.html")

# --- Session / Cookies (NEW) ---
SESSION_COOKIE_NAME = "td_session"
SESS_DB = os.path.join(CACHE_DIR, "sessions.sqlite")

# --- YouTube/Discogs Config ---
DISCOGS_TOKEN = os.getenv("DISCOGS_TOKEN", "").strip()

# --- Apple Music Config ---
APPLE_TEAM_ID = os.getenv("APPLE_TEAM_ID", "").strip()
APPLE_KEY_ID = os.getenv("APPLE_KEY_ID", "").strip()
APPLE_STOREFRONT = os.getenv("APPLE_STOREFRONT", "us").strip()
APPLE_PRIVATE_KEY_RAW = os.getenv("APPLE_PRIVATE_KEY", "").strip()
APPLE_PRIVATE_KEY = None  # Processed below

# --- Spotify Config ---
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", f"https://{APP_NAME}.onrender.com/v1/spotify/callback").strip()
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE_URL = "https://api.spotify.com/v1/"
# Added user-read-email to make /v1/me consistent across accounts
SPOTIFY_SCOPES = "playlist-modify-public playlist-modify-private user-read-email"

# --- Process Apple Key ---
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
log.info("--- End Apple Key Diagnostics ---")

# --- Log Spotify Config ---
log.info("--- Spotify Config ---")
log.info(f"SPOTIFY_CLIENT_ID loaded: {bool(SPOTIFY_CLIENT_ID)}")
log.info(f"SPOTIFY_CLIENT_SECRET loaded: {bool(SPOTIFY_CLIENT_SECRET)}")
log.info(f"SPOTIFY_REDIRECT_URI: {SPOTIFY_REDIRECT_URI}")
log.info(f"FRONTEND_URL: {FRONTEND_URL}")
log.info("--- End Spotify Config ---")


# -------------------- Pydantic Models --------------------
class SimReq(BaseModel):
    date: str
    genre: Optional[str] = None
    station: Optional[str] = None
    hours: float = 3.0
    repeat_gap_min: int = Field(90, ge=0)
    seed: Optional[str] = "97"
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

class SpotifyCreateRequest(BaseModel):
    # NOW OPTIONAL (legacy support). Session cookie is preferred.
    accessToken: Optional[str] = None
    name: str
    tracks: List[Track]


# -------------------- FastAPI App + CORS --------------------
app = FastAPI(title="TapeDeck API", version="1.4")

log.info(f"Allowed CORS origins: {ALLOWED}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED if o.strip()],
    allow_credentials=True,   # important for cookie sessions
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=600,
)


# -------------------- General Utils --------------------
def _norm_key(artist: str, title: str) -> str:
    def clean(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    return f"{clean(artist)} :: {clean(title)}"

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, video_id TEXT, ts INTEGER)")
    return conn

def _cache_get(conn: sqlite3.Connection, k: str) -> Optional[str]:
    try:
        cur = conn.execute("SELECT video_id FROM cache WHERE k=?", (k,))
        row = cur.fetchone()
        return row[0] if row else None
    except sqlite3.Error as e:
        log.error(f"SQLite cache_get error for key '{k}': {e}")
        return None

def _cache_put(conn: sqlite3.Connection, k: str, vid: str) -> None:
    try:
        conn.execute("INSERT OR REPLACE INTO cache (k, video_id, ts) VALUES (?,?,?)", (k, vid, int(time.time())))
        conn.commit()
    except sqlite3.Error as e:
        log.error(f"SQLite cache_put error for key '{k}': {e}")

def _extract_video_id(u: str) -> Optional[str]:
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
        if r.status_code == 429:
            log.warning("Discogs rate limit hit.")
            time.sleep(5)
            return None
        if r.status_code != 200:
            log.warning(f"Discogs search for '{q}' failed: Status {r.status_code}")
            return None
        results = r.json().get("results", [])
        if not results:
            return None

        res_url = results[0].get("resource_url")
        if not res_url:
            return None

        time.sleep(1.1)
        d_headers = {"User-Agent": "TapeDeckTimeMachine/1.0 (+https://tapedecktimemachine.com)"}
        d = requests.get(res_url, headers=d_headers, timeout=12)
        if d.status_code != 200:
            log.warning(f"Discogs fetch for resource '{res_url}' failed: Status {d.status_code}")
            return None

        for v in d.json().get("videos", []) or []:
            uri = v.get("uri") or ""
            if "youtube.com" in uri or "youtu.be" in uri:
                vid = _extract_video_id(uri)
                if vid:
                    return vid
    except requests.RequestException as e:
        log.warning(f"Discogs request failed for '{q}': {e}")
    except Exception as e:
        log.exception(f"Unexpected error in Discogs lookup for '{q}': {e}")
    return None


# ---------- Session storage (Spotify) ----------
def _sess_db() -> sqlite3.Connection:
    conn = sqlite3.connect(SESS_DB)
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.execute("""
      CREATE TABLE IF NOT EXISTS spotify_sessions (
        session_id   TEXT PRIMARY KEY,
        refresh_token TEXT NOT NULL,
        access_token  TEXT,
        expires_at    INTEGER,
        user_id       TEXT,
        scope         TEXT,
        created_at    INTEGER
      )
    """)
    conn.execute("""
      CREATE TABLE IF NOT EXISTS oauth_state (
        state TEXT PRIMARY KEY,
        created_at INTEGER
      )
    """)
    return conn

def _save_oauth_state(state: str):
    conn = _sess_db()
    conn.execute("INSERT OR REPLACE INTO oauth_state(state, created_at) VALUES(?, ?)", (state, int(time.time())))
    conn.commit()
    conn.close()

def _pop_oauth_state(state: str) -> bool:
    conn = _sess_db()
    cur = conn.execute("SELECT state FROM oauth_state WHERE state=?", (state,))
    ok = cur.fetchone() is not None
    conn.execute("DELETE FROM oauth_state WHERE state=?", (state,))
    conn.commit(); conn.close()
    return ok

def _create_session(refresh_token: str, access_token: str, expires_in: int, user_id: str, scope: str) -> str:
    session_id = secrets.token_urlsafe(32)
    expires_at = int(time.time()) + int(expires_in or 3600)
    conn = _sess_db()
    conn.execute(
        "INSERT OR REPLACE INTO spotify_sessions(session_id, refresh_token, access_token, expires_at, user_id, scope, created_at) VALUES(?,?,?,?,?,?,?)",
        (session_id, refresh_token, access_token, expires_at, user_id, scope, int(time.time()))
    )
    conn.commit(); conn.close()
    return session_id

def _get_session(session_id: str):
    conn = _sess_db()
    cur = conn.execute("SELECT refresh_token, access_token, expires_at, user_id, scope FROM spotify_sessions WHERE session_id=?", (session_id,))
    row = cur.fetchone()
    conn.close()
    return row  # (refresh, access, exp, uid, scope) or None

def _update_session_tokens(session_id: str, access_token: str, expires_in: int):
    conn = _sess_db()
    conn.execute("UPDATE spotify_sessions SET access_token=?, expires_at=? WHERE session_id=?",
                 (access_token, int(time.time()) + int(expires_in or 3600), session_id))
    conn.commit(); conn.close()


# --- Apple Music Utils ---
def _require_apple_keys():
    if not (APPLE_TEAM_ID and APPLE_KEY_ID and APPLE_PRIVATE_KEY):
        raise RuntimeError("Apple keys not configured")

def _mint_dev_token(ttl_seconds: int = 55 * 60) -> str:
    _require_apple_keys()
    now = int(time.time())
    payload = {"iss": APPLE_TEAM_ID, "iat": now, "exp": now + ttl_seconds}
    token = jwt.encode(payload, APPLE_PRIVATE_KEY, algorithm="ES256", headers={"kid": APPLE_KEY_ID})
    return token


# --- Spotify Utils ---
def _spotify_get_user_id(token: str) -> Optional[str]:
    headers = {"Authorization": f"Bearer {token}"}
    response = None
    try:
        response = requests.get(SPOTIFY_API_BASE_URL + "me", headers=headers, timeout=10)
        response.raise_for_status()
        user_info = response.json()
        log.info(f"Fetched Spotify user info for ID: {user_info.get('id')}")
        return user_info.get("id")
    except requests.RequestException as e:
        status_code = response.status_code if response is not None else 'N/A'
        log.error(f"Spotify API error getting user ID: {e} (Status: {status_code})")
        raise HTTPException(status_code=status_code if isinstance(status_code, int) else 500, detail=f"Failed to get Spotify user ID: {e}")

def _spotify_search_track(token: str, artist: str, title: str) -> Optional[str]:
    headers = {"Authorization": f"Bearer {token}"}
    query = f"artist:{artist} track:{title}".replace('"', '')
    params = {"q": query, "type": "track", "limit": 1}
    response = None
    try:
        response = requests.get(SPOTIFY_API_BASE_URL + "search", headers=headers, params=params, timeout=10)
        response.raise_for_status()
        results = response.json().get("tracks", {}).get("items", [])
        if results:
            uri = results[0].get("uri")
            log.debug(f"Spotify search success: '{artist} - {title}' -> {uri}")
            return uri
        else:
            log.warning(f"Spotify search: Track not found: {artist} - {title}")
            return None
    except requests.RequestException as e:
        status_code = response.status_code if response is not None else 'N/A'
        log.error(f"Spotify API error searching track '{artist} - {title}': {e} (Status: {status_code})")
        return None

def _spotify_create_playlist(token: str, user_id: str, name: str, description: str = "") -> Optional[Dict]:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = json.dumps({"name": name, "description": description, "public": False})
    url = f"{SPOTIFY_API_BASE_URL}users/{user_id}/playlists"
    response = None
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=15)
        response.raise_for_status()
        playlist_data = response.json()
        playlist_id = playlist_data.get("id")
        playlist_url = playlist_data.get("external_urls", {}).get("spotify")
        log.info(f"Spotify playlist '{name}' created successfully (ID: {playlist_id}).")
        return {"id": playlist_id, "url": playlist_url}
    except requests.RequestException as e:
        status_code = response.status_code if response is not None else 'N/A'
        response_text = response.text if response is not None else str(e)
        log.error(f"Spotify API error creating playlist: {e} (Status: {status_code}) Response: {response_text}")
        raise HTTPException(status_code=status_code if isinstance(status_code, int) else 500, detail=f"Failed to create Spotify playlist: {e}")

def _spotify_add_tracks_to_playlist(token: str, playlist_id: str, track_uris: List[str]) -> bool:
    if not track_uris:
        log.info("No Spotify track URIs to add to playlist.")
        return True
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    success = True
    response = None
    for i in range(0, len(track_uris), 100):
        batch_uris = track_uris[i:i+100]
        payload = json.dumps({"uris": batch_uris})
        url = f"{SPOTIFY_API_BASE_URL}playlists/{playlist_id}/tracks"
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=20)
            response.raise_for_status()
            log.info(f"Added batch of {len(batch_uris)} tracks to Spotify playlist {playlist_id}.")
        except requests.RequestException as e:
            status_code = response.status_code if response is not None else 'N/A'
            response_text = response.text if response is not None else str(e)
            log.error(f"Spotify API error adding tracks batch: {e} (Status: {status_code}) Response: {response_text}")
            success = False
            break
        time.sleep(0.1)
    return success

def _spotify_refresh(refresh_token: str) -> Dict:
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="Spotify integration not configured on server.")
    auth_string = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(auth_string.encode()).decode()}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"grant_type": "refresh_token", "refresh_token": refresh_token}
    r = requests.post(SPOTIFY_TOKEN_URL, headers=headers, data=data, timeout=15)
    try:
        r.raise_for_status()
    except requests.RequestException as e:
        log.error(f"Spotify refresh failed: {r.text if r is not None else e}")
        raise HTTPException(status_code=401, detail="spotify_refresh_failed")
    return r.json()

def _access_from_session(session_id: str) -> Tuple[str, Optional[str]]:
    row = _get_session(session_id)
    if not row:
        raise HTTPException(status_code=401, detail="no_session")
    refresh_token, access_token, expires_at, user_id, _scope = row
    if not access_token or int(time.time()) >= int(expires_at or 0) - 60:
        t = _spotify_refresh(refresh_token)
        access_token = t.get("access_token")
        expires_in = int(t.get("expires_in", 3600))
        if not access_token:
            raise HTTPException(status_code=401, detail="spotify_refresh_no_access_token")
        _update_session_tokens(session_id, access_token, expires_in)
    return access_token, user_id


# -------------------- API Endpoints --------------------
@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat() + "Z"}


@app.post("/v1/simulate")
def simulate_playlist(req: SimReq):
    try:
        slugs = resolve_candidate_slugs(req.station, req.genre)
        d = datetime.strptime(req.date, "%Y-%m-%d")
        chart_date = nearest_billboard_chart_date(d)
        log.info(f"Simulating playlist for date={req.date}, genre={req.genre}, station={req.station}, hours={req.hours}. Using chart date: {chart_date}")
        entries = fetch_chart_entries_strict(slugs, chart_date, limit=req.limit, tolerance_days=7)
        start_dt = d.replace(hour=6, minute=0, second=0, microsecond=0)
        spins = simulate_rotations(
            entries=entries, start_time=start_dt, hours=req.hours,
            average_song_minutes=3.5, min_gap_minutes=req.repeat_gap_min,
            seed=int(req.seed or "97"),
        )
        log.info(f"Simulation complete. Generated {len(spins)} spins.")
        return {"tracks": [{"timestamp": s.timestamp.isoformat(timespec="minutes"), "artist": s.artist, "title": s.title, "source_rank": s.source_rank} for s in spins]}
    except Exception as e:
        log.exception("simulate failed")
        raise HTTPException(status_code=500, detail=f"simulate failed: {e}")


@app.post("/v1/yt/resolve")
def yt_resolve(req: ResolveReq):
    conn = None
    try:
        conn = _db()
        ids, seen = [], set()
        log.info(f"Resolving YouTube IDs for {len(req.tracks)} tracks (limit {req.limit or 50}).")
        for i, t in enumerate(req.tracks):
            if len(ids) >= (req.limit or 50): break
            k = _norm_key(t.artist, t.title)
            vid = _cache_get(conn, k)
            if not vid:
                time.sleep(0.6)
                vid = _discogs_find_video_id(t.artist, t.title)
                if vid: _cache_put(conn, k, vid)
                else: log.warning(f"YT Resolve: Failed via Discogs for '{t.artist} - {t.title}'")
            if vid and vid not in seen:
                seen.add(vid); ids.append(vid)
            elif vid in seen:
                log.debug(f"YT Resolve ({i+1}): Duplicate skipped for '{t.artist} - {t.title}'")
        log.info(f"YouTube resolution finished. Found {len(ids)} unique IDs.")
        return {"ids": ids}
    except Exception as e:
        log.exception("yt/resolve failed")
        raise HTTPException(status_code=500, detail=f"yt/resolve failed: {e}")
    finally:
        if conn: conn.close()


# ---------- Apple Music Endpoints ----------
@app.get("/v1/apple/dev-token")
def apple_dev_token():
    try:
        token = _mint_dev_token()
        log.info("Generated Apple dev token.")
        return {"token": token, "storefront": APPLE_STOREFRONT}
    except RuntimeError as e:
        log.error(f"Apple keys not configured: {e}")
        raise HTTPException(status_code=500, detail="Apple Music integration not configured on server.")
    except Exception as e:
        log.exception("apple/dev-token failed")
        raise HTTPException(status_code=500, detail=f"apple/dev-token failed: {e}")

@app.post("/v1/apple/create-playlist")
def apple_create_playlist(req: AppleCreateRequest):
    playlist_name = req.name
    log.info(f"Attempting to create Apple Music playlist named '{playlist_name}'")
    try:
        dev_token = _mint_dev_token(ttl_seconds=25 * 60)
    except Exception as e:
        log.exception("Apple dev token generation failed during playlist creation.")
        raise HTTPException(status_code=500, detail=f"Dev token generation failed: {e}")

    headers = {"Authorization": f"Bearer {dev_token}", "Music-User-Token": req.userToken}
    playlist_id = None
    playlist_url = None

    # 1) Create empty library playlist
    try:
        payload = {"attributes": {"name": playlist_name, "description": "Generated by TapeDeckTimeMachine"}}
        r_create = requests.post("https://api.music.apple.com/v1/me/library/playlists", headers=headers, json=payload, timeout=20)
        r_create.raise_for_status()
        playlist_data = r_create.json()["data"][0]
        playlist_id = playlist_data["id"]
        log.info(f"Apple Music playlist created with ID: {playlist_id}")
    except requests.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 500
        response_text = e.response.text if e.response is not None else str(e)
        log.error(f"Failed to create Apple Music playlist: {e} (Status: {status_code}) Response: {response_text}")
        raise HTTPException(status_code=status_code, detail=f"Apple Music playlist creation failed: {response_text}")
    except Exception as e:
        log.exception("Unexpected error creating Apple Music playlist")
        raise HTTPException(status_code=500, detail=f"Playlist creation failed: {e}")

    # 2) Resolve songs
    song_ids_to_add = []
    storefront = APPLE_STOREFRONT or "us"
    log.info(f"Resolving {len(req.tracks)} tracks for Apple Music playlist {playlist_id}...")
    for track in req.tracks[:150]:
        try:
            params = {"term": f"{track.artist} {track.title}", "limit": 1, "types": "songs"}
            search_url = f"https://api.music.apple.com/v1/catalog/{storefront}/search"
            r_search = requests.get(search_url, headers=headers, params=params, timeout=12)

            if r_search.status_code == 429:
                log.warning("Apple Music API rate limit hit during search. Skipping remaining tracks.")
                break
            r_search.raise_for_status()

            results = r_search.json().get("results", {}).get("songs", {}).get("data", [])
            if results:
                song_id = results[0]["id"]
                song_ids_to_add.append({"id": song_id, "type": "songs"})
            else:
                log.warning(f"Apple Resolve: Could not find '{track.artist} - {track.title}'")
        except requests.RequestException as e:
            status_code = e.response.status_code if e.response is not None else 500
            log.warning(f"Apple Music search request failed for '{track.artist} - {track.title}': {e} (Status: {status_code})")
            time.sleep(0.5)
        except Exception as e:
            log.exception(f"Unexpected error resolving Apple Music track '{track.artist} - {track.title}'")
        time.sleep(0.2)

    # 3) Add resolved tracks
    added_count = 0
    if song_ids_to_add:
        log.info(f"Adding {len(song_ids_to_add)} resolved tracks to Apple Music playlist {playlist_id}...")
        try:
            add_url = f"https://api.music.apple.com/v1/me/library/playlists/{playlist_id}/tracks"
            r_add = requests.post(add_url, headers=headers, json={"data": song_ids_to_add}, timeout=30)
            r_add.raise_for_status()
            if r_add.status_code == 204:
                added_count = len(song_ids_to_add)
                log.info(f"Successfully added {added_count} tracks to Apple Music playlist {playlist_id}.")
            else:
                log.warning(f"Adding tracks to Apple Music playlist {playlist_id} returned status {r_add.status_code}, expected 204.")
                added_count = len(song_ids_to_add)
        except requests.RequestException as e:
            status_code = e.response.status_code if e.response is not None else 500
            response_text = e.response.text if e.response is not None else str(e)
            log.error(f"Failed to add tracks to Apple Music playlist {playlist_id}: {e} (Status: {status_code}) Response: {response_text}")
            added_count = 0
        except Exception as e:
            log.exception(f"Unexpected error adding tracks to Apple Music playlist {playlist_id}")
            added_count = 0
    else:
        log.warning(f"No tracks resolved for Apple Music playlist {playlist_id}, nothing to add.")

    return {
        "added_count": added_count,
        "total_tracks": len(req.tracks),
        "playlist_id": playlist_id,
        "id": playlist_id,
        "storefront": storefront,
        "url": playlist_url,
    }

@app.get("/v1/apple/song-meta")
def apple_song_meta(id: str = Query(..., description="Apple song id (catalog)"), storefront: str = Query("us")):
    try:
        dev_token = _mint_dev_token(ttl_seconds=10 * 60)
        url = f"https://api.music.apple.com/v1/catalog/{storefront}/songs/{id}"
        r = requests.get(url, headers={"Authorization": f"Bearer {dev_token}"}, timeout=12)
        if r.status_code != 200:
            log.warning(f"Apple song-meta request failed for ID {id}: Status {r.status_code} - {r.text}")
            raise HTTPException(status_code=r.status_code, detail=f"Apple catalog error: {r.text}")
        data = r.json().get("data", [])
        if not data:
            log.warning(f"Apple song-meta: Song ID {id} not found in storefront {storefront}")
            raise HTTPException(status_code=404, detail="Song not found")
        attrs = data[0].get("attributes", {}) or {}
        return {"id": id, "name": attrs.get("name"), "artistName": attrs.get("artistName"), "url": attrs.get("url")}
    except Exception as e:
        log.exception(f"apple/song-meta failed for ID {id}")
        raise HTTPException(status_code=500, detail=f"song-meta failed: {e}")


# ---------- Spotify Endpoints (NEW session flow) ----------
@app.get("/v1/spotify/login")
async def spotify_login():
    if not SPOTIFY_CLIENT_ID:
        log.error("Spotify login attempt failed: SPOTIFY_CLIENT_ID not configured.")
        raise HTTPException(status_code=500, detail="Spotify integration not configured on server.")
    state = secrets.token_urlsafe(16)
    _save_oauth_state(state)
    auth_params = {
        "client_id": SPOTIFY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "scope": SPOTIFY_SCOPES,
        "state": state,
    }
    auth_url = f"{SPOTIFY_AUTH_URL}?{urlencode(auth_params)}"
    log.info(f"Redirecting user to Spotify for authorization: {auth_url}")
    return RedirectResponse(auth_url)

@app.get("/v1/spotify/callback")
async def spotify_callback(code: Optional[str] = None, error: Optional[str] = None, state: Optional[str] = None):
    if error:
        log.error(f"Spotify authorization failed with error: {error}")
        return RedirectResponse(f"{FRONTEND_URL}#spotify_error={error}")
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code from Spotify.")
    if not _pop_oauth_state(state or ""):
        log.warning("Spotify callback with invalid/missing state.")
        return RedirectResponse(f"{FRONTEND_URL}#spotify_error=bad_state")
    if not (SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET):
        raise HTTPException(status_code=500, detail="Spotify integration not configured on server.")

    try:
        auth_string = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
        headers = {"Authorization": f"Basic {base64.b64encode(auth_string.encode()).decode()}",
                   "Content-Type": "application/x-www-form-urlencoded"}
        payload = {"grant_type": "authorization_code", "code": code, "redirect_uri": SPOTIFY_REDIRECT_URI}
        log.info(f"Exchanging Spotify code for tokens (code: {code[:10]}...)")
        response = requests.post(SPOTIFY_TOKEN_URL, headers=headers, data=payload, timeout=15)
        response.raise_for_status()
        token_info = response.json()
        access_token = token_info.get("access_token")
        refresh_token = token_info.get("refresh_token")
        expires_in = int(token_info.get("expires_in", 3600))
        if not access_token or not refresh_token:
            log.error(f"Token exchange missing fields: {token_info}")
            return RedirectResponse(f"{FRONTEND_URL}#spotify_error=token_exchange_failed")

        # Fetch user id once and create session
        uid = _spotify_get_user_id(access_token) or ""
        session_id = _create_session(refresh_token, access_token, expires_in, uid, SPOTIFY_SCOPES)

        # Set secure cookie and redirect back WITHOUT tokens
        resp = RedirectResponse(f"{FRONTEND_URL}#spotify=ok")
        resp.set_cookie(
            SESSION_COOKIE_NAME,
            session_id,
            httponly=True,
            secure=True,
            samesite="none",  # works cross-site; ensure HTTPS
            max_age=30 * 24 * 3600,
            path="/",
        )
        log.info("Spotify token exchange successful and session cookie set.")
        return resp

    except requests.RequestException as e:
        log.error(f"Error exchanging Spotify code for tokens: {e}")
        return RedirectResponse(f"{FRONTEND_URL}#spotify_error=token_exchange_failed")
    except Exception as e:
        log.exception("Unexpected error during Spotify callback handling.")
        return RedirectResponse(f"{FRONTEND_URL}#spotify_error=internal_server_error")

@app.get("/v1/spotify/status")
def spotify_status(td_session: Optional[str] = Cookie(None)):
    if not td_session:
        return {"signed_in": False}
    try:
        _access_from_session(td_session)  # will refresh if needed
        return {"signed_in": True}
    except HTTPException:
        return {"signed_in": False}

@app.post("/v1/spotify/logout")
def spotify_logout(td_session: Optional[str] = Cookie(None)):
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE_NAME, path="/")
    if td_session:
        conn = _sess_db()
        conn.execute("DELETE FROM spotify_sessions WHERE session_id=?", (td_session,))
        conn.commit(); conn.close()
    return resp

@app.post("/v1/spotify/create-playlist")
async def spotify_create_playlist(req: SpotifyCreateRequest, td_session: Optional[str] = Cookie(None)):
    """
    Preferred: authenticate via cookie session (auto-refresh).
    Legacy: if body contains accessToken, use that path.
    """
    if td_session:
        access_token, user_id = _access_from_session(td_session)
        if not user_id:
            # if session was created before we stored uid, fetch now
            user_id = _spotify_get_user_id(access_token)
    elif req.accessToken:
        access_token = req.accessToken
        user_id = _spotify_get_user_id(access_token)
    else:
        raise HTTPException(status_code=401, detail="spotify_auth_required")

    playlist_name = req.name
    tracks_to_resolve = req.tracks
    log.info(f"Received request to create Spotify playlist '{playlist_name}' with {len(tracks_to_resolve)} tracks.")

    # 2. Create the empty playlist
    playlist_info = _spotify_create_playlist(access_token, user_id, playlist_name, description="Generated by TapeDeckTimeMachine")
    if not playlist_info or not playlist_info.get("id"):
        raise HTTPException(status_code=500, detail="Failed to create Spotify playlist.")
    playlist_id = playlist_info["id"]
    playlist_url = playlist_info.get("url")

    # 3. Search for track URIs
    log.info(f"Searching Spotify for {len(tracks_to_resolve)} tracks...")
    track_uris_to_add: List[str] = []
    failed_searches = 0
    for i, track in enumerate(tracks_to_resolve[:150]):
        uri = _spotify_search_track(access_token, track.artist, track.title)
        if uri:
            track_uris_to_add.append(uri)
        else:
            failed_searches += 1
        if (i + 1) % 10 == 0:
            time.sleep(0.1)
        elif (i + 1) % 50 == 0:
            time.sleep(0.5)

    log.info(f"Spotify search complete. Found {len(track_uris_to_add)} URIs. Failed to find {failed_searches} tracks.")

    # 4. Add found tracks
    add_success = _spotify_add_tracks_to_playlist(access_token, playlist_id, track_uris_to_add)
    if not add_success:
        log.warning(f"Failed to add some or all tracks to Spotify playlist {playlist_id}.")

    return {
        "added_count": len(track_uris_to_add),
        "total_tracks": len(tracks_to_resolve),
        "playlist_id": playlist_id,
        "url": playlist_url,
        "add_success": add_success
    }


# ---------- Root Endpoint ----------
@app.get("/")
def root():
    return PlainTextResponse(f"{APP_NAME} OK")
