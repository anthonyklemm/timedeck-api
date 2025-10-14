# main.py
# FastAPI backend for TapeDeck Time Machine
# Endpoints:
#   GET  /health
#   POST /v1/simulate        -> uses playlist_creator.py to build spins
#   POST /v1/yt/resolve      -> resolves {artist,title} -> YouTube videoIds (cache + Discogs)
#   GET  /v1/apple/dev-token -> emits a MusicKit developer token (JWT)

import os
import re
import time
import json
import sqlite3
import logging
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta

import requests
import jwt  # PyJWT

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

# --- import simulator bits from your existing script ---
from playlist_creator import (
    resolve_candidate_slugs,
    nearest_billboard_chart_date,
    fetch_chart_entries_strict,
    simulate_rotations,
)

# -------------------- config / logging --------------------

APP_NAME = "timedeck-api"
log = logging.getLogger(APP_NAME)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

ALLOWED = os.getenv("ALLOWED_ORIGINS", "https://anthonyklemm.github.io").split(",")
CACHE_DIR = os.getenv("CACHE_DIR", "/data")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_DB = os.path.join(CACHE_DIR, "yt_cache.sqlite")

DISCOGS_TOKEN = os.getenv("DISCOGS_TOKEN", "").strip()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()

APPLE_TEAM_ID = os.getenv("APPLE_TEAM_ID")
APPLE_KEY_ID = os.getenv("APPLE_KEY_ID")
APPLE_PRIVATE_KEY = os.getenv("APPLE_PRIVATE_KEY")
APPLE_STOREFRONT = os.getenv("APPLE_STOREFRONT", "us")

# -------------------- FastAPI app + CORS --------------------

app = FastAPI(title="TapeDeck API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=600,
)

# -------------------- utils --------------------

def _norm_key(artist: str, title: str) -> str:
    def clean(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    return f"{clean(artist)} :: {clean(title)}"

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(CACHE_DB)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, video_id TEXT, ts INTEGER)"
    )
    return conn

def _cache_get(conn: sqlite3.Connection, k: str) -> Optional[str]:
    cur = conn.execute("SELECT video_id FROM cache WHERE k=?", (k,))
    row = cur.fetchone()
    return row[0] if row else None

def _cache_put(conn: sqlite3.Connection, k: str, vid: str) -> None:
    conn.execute("INSERT OR REPLACE INTO cache (k, video_id, ts) VALUES (?,?,?)",
                 (k, vid, int(time.time())))
    conn.commit()

def _is_youtube_url(u: str) -> bool:
    return "youtube.com/watch" in u or "youtu.be/" in u

def _extract_video_id(u: str) -> Optional[str]:
    # Handles typical forms like https://www.youtube.com/watch?v=ABC123 and https://youtu.be/ABC123
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{8,})", u)
    if m:
        return m.group(1)
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{8,})", u)
    if m:
        return m.group(1)
    return None

def _discogs_find_video_id(artist: str, title: str) -> Optional[str]:
    if not DISCOGS_TOKEN:
        return None
    q = f"{artist} - {title}"
    try:
        r = requests.get(
            "https://api.discogs.com/database/search",
            params={"q": q, "per_page": 5, "page": 1, "token": DISCOGS_TOKEN},
            timeout=12,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        results = data.get("results", [])[:5]
        for item in results:
            res_url = item.get("resource_url")
            if not res_url:
                continue
            d = requests.get(res_url, timeout=12)
            if d.status_code != 200:
                continue
            obj = d.json()
            for v in obj.get("videos", []) or []:
                uri = v.get("uri") or v.get("url") or ""
                if _is_youtube_url(uri):
                    vid = _extract_video_id(uri)
                    if vid:
                        return vid
    except Exception as e:
        log.warning("Discogs lookup failed for %s — %s", q, e)
    return None

# -------------------- models --------------------

class SimReq(BaseModel):
    date: str                   # "YYYY-MM-DD"
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
    region: Optional[str] = "US"
    limit: Optional[int] = 50

# -------------------- endpoints --------------------

@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat() + "Z"}

@app.post("/v1/simulate")
def simulate(req: SimReq):
    try:
        # Resolve slugs for the user's genre/station
        slugs = resolve_candidate_slugs(req.station, req.genre)

        # Billboard chart week (Saturday on/after given date)
        # nearest_billboard_chart_date expects a datetime
        d = datetime.strptime(req.date, "%Y-%m-%d")
        chart_date = nearest_billboard_chart_date(d)

        # Strict fetch for that week
        entries = fetch_chart_entries_strict(slugs, chart_date, limit=req.limit, tolerance_days=7)

        # Start 06:00; avg length fixed at 3.5 as per your UX
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
            "date": req.date,
            "genre": req.genre or "",
            "hours": req.hours,
            "repeat_gap_min": req.repeat_gap_min,
            "seed": req.seed,
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
    """
    Cache-first YouTube resolver.
    - looks in sqlite cache
    - tries Discogs videos as a fallback
    (You can extend with MusicBrainz or YouTube Data API later.)
    """
    try:
        conn = _db()
        ids: List[str] = []
        dropped: List[Dict] = []
        seen: set = set()
        stats = {"cache": 0, "discogs": 0}

        for t in req.tracks:
            if len(ids) >= (req.limit or 50):
                break
            k = _norm_key(t.artist, t.title)
            vid = _cache_get(conn, k)
            if vid:
                stats["cache"] += 1
            else:
                vid = _discogs_find_video_id(t.artist, t.title)
                if vid:
                    stats["discogs"] += 1
                    _cache_put(conn, k, vid)

            # dedupe and collect
            if vid and vid not in seen:
                seen.add(vid)
                ids.append(vid)
            else:
                dropped.append({"artist": t.artist, "title": t.title, "reason": "not_found"})

        return {"ids": ids, "dropped": dropped, "sources": stats}
    except Exception as e:
        log.exception("yt/resolve failed")
        return JSONResponse(status_code=500, content={"detail": f"yt/resolve failed: {e}"})

@app.get("/v1/apple/dev-token")
def apple_dev_token():
    """
    Issue a MusicKit developer token from env:
      APPLE_TEAM_ID, APPLE_KEY_ID, APPLE_PRIVATE_KEY
    """
    try:
        if not (APPLE_TEAM_ID and APPLE_KEY_ID and APPLE_PRIVATE_KEY):
            return JSONResponse(status_code=400, content={"detail": "Apple keys not configured"})
        now = int(time.time())
        payload = {
            "iss": APPLE_TEAM_ID,
            "iat": now,
            "exp": now + 60 * 55,   # ~55 minutes
        }
        token = jwt.encode(
            payload,
            APPLE_PRIVATE_KEY,
            algorithm="ES256",
            headers={"kid": APPLE_KEY_ID, "alg": "ES256"},
        )
        return {"token": token, "storefront": APPLE_STOREFRONT}
    except Exception as e:
        log.exception("dev-token failed")
        return JSONResponse(status_code=500, content={"detail": f"dev-token failed: {e}"})

@app.get("/")
def root():
    return PlainTextResponse(f"{APP_NAME} OK")

# -------------------- uvicorn entry (Render uses: uvicorn main:app ...) --------------------
# no __main__ needed for Render; keep file simple
