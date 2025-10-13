# main.py
import os
import time
import json
import base64
import sqlite3
import re
from typing import Optional, List

import jwt  # PyJWT
import requests
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import aiohttp
    import asyncio
except Exception:
    aiohttp = None
    asyncio = None

DB_PATH = os.environ.get("YT_CACHE_DB", "yt_cache.sqlite")

app = FastAPI(title="TimeDeck API", version="1.0")

# CORS for your GitHub Pages origin(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later (e.g., ["https://anthonyklemm.github.io"])
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- SQLite cache ----------
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS yt_cache(
      artist TEXT, title TEXT, year TEXT, region TEXT,
      video_id TEXT, source TEXT, verified INTEGER,
      updated_at INTEGER,
      PRIMARY KEY (artist, title, COALESCE(year,''), COALESCE(region,''))
    );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vid ON yt_cache(video_id);")
    return conn

def cache_get(conn, artist, title, year, region):
    row = conn.execute("""
      SELECT video_id, verified FROM yt_cache
      WHERE artist=? AND title=? AND COALESCE(year,'')=COALESCE(?, '')
        AND COALESCE(region,'')=COALESCE(?, '')
    """, (artist, title, year, region)).fetchone()
    return (row[0], int(row[1])) if row else (None, 0)

def cache_put(conn, artist, title, year, region, video_id, source, verified=1):
    conn.execute("""
      INSERT OR REPLACE INTO yt_cache
      (artist,title,year,region,video_id,source,verified,updated_at)
      VALUES (?,?,?,?,?,?,?,?)
    """, (artist, title, year, region, video_id, source, int(verified), int(time.time())))

# ---------- Models ----------
class Track(BaseModel):
    artist: str
    title: str
    year: Optional[str] = None

# ---------- Utility: extract YouTube id ----------
YT_ID_RE = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|v/))([A-Za-z0-9_-]{11})",
    re.IGNORECASE,
)

def extract_yt_id(url: str) -> Optional[str]:
    if not url:
        return None
    m = YT_ID_RE.search(url)
    if m:
        return m.group(1)
    # try query param v= (case sensitive id)
    if "v=" in url:
        v = url.split("v=", 1)[1].split("&", 1)[0]
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", v):
            return v
    return None

# ---------- Discogs lookup (release->videos) ----------
DISCOGS_API = "https://api.discogs.com"
def discogs_lookup(session, artist: str, title: str, timeout: float = 3.0) -> Optional[str]:
    token = os.environ.get("DISCOGS_TOKEN")
    if not token:
        return None
    q = f"{artist} - {title}"
    try:
        r = session.get(
            f"{DISCOGS_API}/database/search",
            params={"q": q, "type": "release", "per_page": 1, "token": token},
            timeout=timeout,
            headers={"User-Agent": "TimeDeck/1.0 (+github.com/anthonyklemm)"}
        )
        if r.status_code != 200:
            return None
        res = r.json()
        if not res.get("results"):
            return None
        item = res["results"][0]
        # Follow to the release endpoint to get 'videos'
        rid = item.get("id")
        if not rid:
            return None
        r2 = session.get(
            f"{DISCOGS_API}/releases/{rid}",
            timeout=timeout,
            headers={"User-Agent": "TimeDeck/1.0 (+github.com/anthonyklemm)"}
        )
        if r2.status_code != 200:
            return None
        rel = r2.json()
        vids = rel.get("videos") or []
        for v in vids:
            vid = extract_yt_id(v.get("uri") or "")
            if vid:
                return vid
        return None
    except Exception:
        return None

# ---------- Health ----------
@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}

# ---------- Apple Music dev token ----------
@app.get("/dev-token")
def dev_token():
    team_id = os.environ.get("APPLE_MUSIC_TEAM_ID")
    key_id = os.environ.get("APPLE_MUSIC_KEY_ID")
    p8 = os.environ.get("APPLE_MUSIC_PRIVATE_KEY")  # raw .p8 contents
    if not (team_id and key_id and p8):
        raise HTTPException(500, "Apple Music credentials not configured")
    # normalize possible \n escapes
    p8_norm = p8.replace("\\n", "\n").strip()
    now = int(time.time())
    payload = {
        "iss": team_id,
        "iat": now,
        "exp": now + 60 * 30,   # 30 minutes is plenty for browser
    }
    token = jwt.encode(payload, p8_norm, algorithm="ES256", headers={"alg": "ES256", "kid": key_id})
    return {"token": token}

# ---------- YouTube resolve: FAST (cache-only) ----------
@app.post("/v1/resolve_batch_fast")
def resolve_batch_fast(tracks: List[Track], region: str = Query("US")):
    conn = _conn()
    out = []
    for t in tracks:
        vid, verified = cache_get(conn, t.artist, t.title, t.year, region)
        out.append({
            "artist": t.artist, "title": t.title, "year": t.year,
            "videoId": vid, "verified": bool(verified), "source": "cache" if vid else None
        })
    conn.close()
    return {"items": out}

# ---------- YouTube resolve: bounded, async ----------
SEM_LIMIT = int(os.environ.get("RESOLVE_CONCURRENCY", "8"))
PER_ITEM_TIMEOUT = float(os.environ.get("RESOLVE_TIMEOUT", "3.0"))

async def resolve_one_async(session, conn, t: Track, region: str):
    # check cache (again) in case other task filled it
    vid, verified = cache_get(conn, t.artist, t.title, t.year, region)
    if vid:
        return {"artist": t.artist, "title": t.title, "year": t.year,
                "videoId": vid, "source": "cache", "verified": bool(verified)}

    # Discogs (time-boxed)
    try:
        async with asyncio.timeout(PER_ITEM_TIMEOUT):
            # run Discogs call in threadpool (requests is blocking)
            def _work():
                with requests.Session() as s:
                    return discogs_lookup(s, t.artist, t.title, timeout=PER_ITEM_TIMEOUT)
            vid = await asyncio.to_thread(_work)
    except Exception:
        vid = None

    if vid:
        cache_put(conn, t.artist, t.title, t.year, region, vid, "discogs", verified=1)
        return {"artist": t.artist, "title": t.title, "year": t.year,
                "videoId": vid, "source": "discogs", "verified": True}

    return {"artist": t.artist, "title": t.title, "year": t.year,
            "videoId": None, "source": "miss", "verified": False}

@app.post("/v1/resolve_batch")
async def resolve_batch(payload: dict, region: str = Query("US")):
    if aiohttp is None or asyncio is None:
        raise HTTPException(500, "aiohttp/asyncio not available on server")

    tracks = [Track(**t) if isinstance(t, dict) else t for t in payload.get("tracks", [])]
    if not tracks:
        return {"items": []}

    conn = _conn()
    sem = asyncio.Semaphore(SEM_LIMIT)

    async def guarded(t: Track):
        async with sem:
            async with aiohttp.ClientSession(headers={"User-Agent": "TimeDeck/1.0"}) as session:
                return await resolve_one_async(session, conn, t, region)

    # 7s wall-time overall
    done, pending = await asyncio.wait([guarded(t) for t in tracks], timeout=7.0)
    for p in pending: p.cancel()

    out = []
    for d in done:
        try:
            out.append(d.result())
        except Exception:
            pass

    conn.commit()
    conn.close()
    return {"items": out}

# ---------- (Optional) Simulation passthrough ----------
# If your existing /v1/simulate is already working in your repo, you can keep it.
# Below is a tiny wrapper that calls your existing endpoint code if present,
# otherwise it returns a friendly error so the FE can show a message.

@app.post("/v1/simulate")
def simulate_passthrough(body: dict):
    # Expecting your repo to already implement a real simulator.
    # This stub just guards so the endpoint exists.
    raise HTTPException(500, "simulate not wired here; keep your existing /v1/simulate implementation in timedeck-api.")
