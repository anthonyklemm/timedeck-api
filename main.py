import csv
import json
import sqlite3, os, pathlib
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import time, jwt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# Where you’ve been saving the cache (match your seeding script/output)
CACHE_DIR = os.getenv("CACHE_DIR", "/opt/render/project/src/cache")
DB_PATH = os.getenv("YT_CACHE_DB", os.path.join(CACHE_DIR, "yt_cache.sqlite"))

# Create cache dir if missing (harmless if it already exists)
pathlib.Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
# ---------- Pydantic models ----------

class SimRequest(BaseModel):
    date: str = Field(..., description="YYYY-MM-DD")
    genre: str
    hours: float = 3.0
    repeat_gap_min: int = 90
    seed: Optional[str] = None
    limit: Optional[int] = None  # optional 'first N' rows to return


class Track(BaseModel):
    timestamp: str
    artist: str
    title: str
    source_rank: Optional[int] = None


class SimResponse(BaseModel):
    date: str
    genre: str
    hours: float
    repeat_gap_min: int
    seed: Optional[str]
    tracks: List[Track]

class TrackIn(BaseModel):
    artist: str = Field(..., min_length=1)
    title:  str = Field(..., min_length=1)

class ResolveReq(BaseModel):
    tracks: List[dict]   # [{artist, title, year?, timestamp?}]
    region: Optional[str] = "US"
    limit: Optional[int] = 50

class ResolveResp(BaseModel):
    video_ids: list[str]
    count: int
    misses: list[dict]
    
# ---------- FastAPI app ----------

app = FastAPI(title="TimeDeck API", version="1.0.0")

# IMPORTANT: the Origin for GitHub Pages is just scheme+host (no path)
allowed = os.getenv("ALLOWED_ORIGINS", "")
allow_origins = [o.strip() for o in allowed.split(",") if o.strip()]
if not allow_origins:
    # safe default for local tests; keep tight in prod
    allow_origins = ["http://localhost:8000", "http://127.0.0.1:8000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,          # OK even if you don’t use cookies
    allow_methods=["*"],             # lets POST/OPTIONS through
    allow_headers=["*"],             # allow custom headers
    expose_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat() + "Z"}


@app.get("/v1/apple/dev-token")
def apple_dev_token():
    # allow preset token OR mint from keys
    preset = os.getenv("APPLE_MUSIC_DEV_TOKEN")
    if preset:
        return {"token": preset}

    # accept either APPLE_MUSIC_* or APPLE_* names
    team_id = os.getenv("APPLE_MUSIC_TEAM_ID") or os.getenv("APPLE_TEAM_ID")
    key_id  = os.getenv("APPLE_MUSIC_KEY_ID")  or os.getenv("APPLE_KEY_ID")
    p8      = os.getenv("APPLE_MUSIC_PRIVATE_KEY") or os.getenv("APPLE_PRIVATE_KEY")

    missing = [k for k,v in {
        "TEAM_ID": team_id, "KEY_ID": key_id, "PRIVATE_KEY": p8
    }.items() if not v]
    if missing:
        raise HTTPException(status_code=500, detail=f"Apple keys not configured (missing: {', '.join(missing)})")

    p8_norm = p8.replace("\\n", "\n").strip()
    now = int(time.time())
    ttl = int(os.getenv("APPLE_MUSIC_TOKEN_TTL", "1800"))  # 30 min default
    token = jwt.encode(
        {"iss": team_id, "iat": now, "exp": now + ttl},
        p8_norm,
        algorithm="ES256",
        headers={"alg": "ES256", "kid": key_id},
    )
    return {"token": token}

# keep the alias so FE can hit /dev-token
@app.get("/dev-token")
def dev_token_compat():
    return apple_dev_token()

@app.post("/v1/yt/resolve")
def yt_resolve(req: ResolveReq):
    # call your cache-first resolver and return:
    # { "ids": ["zvCBSSwgtg4", ...], "dropped": [...], "sources": {...} }
    return your_resolve_impl(req)

# ---------- Helpers ----------

def run_playlist_creator(req: SimRequest) -> Path:
    """
    Call playlist_creator.py as a subprocess, write outputs to a tmp dir,
    and return the path to the produced CSV file.
    """
    repo_root = Path(__file__).resolve().parent
    script = repo_root / "playlist_creator.py"
    if not script.exists():
        raise HTTPException(status_code=500, detail="playlist_creator.py not found in API container")

    tmpdir = Path(tempfile.mkdtemp(prefix="sim_"))
    out_base = tmpdir / "sim_output"

    # Build the command — we keep avg-mins fixed at 3.5 as agreed
    cmd = [
        "python", str(script),
        "--date", req.date,
        "--genre", req.genre,
        "--hours", str(req.hours),
        "--avg-mins", "3.5",
        "--min-gap", str(req.repeat_gap_min),
        "--out", str(out_base)
    ]
    if req.seed:
        cmd += ["--seed", req.seed]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"failed to start playlist_creator: {e}")

    if proc.returncode != 0:
        log_tail = proc.stdout[-2000:] if proc.stdout else ""
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"playlist_creator failed ({proc.returncode}):\n{log_tail}")

    csv_path = out_base.with_suffix(".csv")
    if not csv_path.exists():
        log_tail = proc.stdout[-2000:] if proc.stdout else ""
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"CSV not found at {csv_path}.\n{log_tail}")

    # Leave tmpdir in place so we can read the CSV; caller cleans up.
    return csv_path


def read_tracks(csv_path: Path, limit: Optional[int]) -> List[Track]:
    rows: List[Track] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            rows.append(
                Track(
                    timestamp=r.get("timestamp", ""),
                    artist=r.get("artist", ""),
                    title=r.get("title", ""),
                    source_rank=int(r["source_rank"]) if r.get("source_rank") else None,
                )
            )
            if limit and len(rows) >= limit:
                break
    return rows

def _lookup_from_cache(conn: sqlite3.Connection, artist: str, title: str) -> str | None:
    """
    Works with the safe-cache script schema:
      CREATE TABLE IF NOT EXISTS yt_cache (
          key TEXT PRIMARY KEY,        -- lowercased "artist — title"
          video_id TEXT NOT NULL,      -- canonical case (YouTube IDs are case-sensitive)
          src TEXT,                    -- discogs|musicbrainz|search|cache
          conf REAL                    -- 0..1 confidence
      );
    """
    k = f"{artist.strip().lower()} — {title.strip().lower()}"
    row = conn.execute("SELECT video_id FROM yt_cache WHERE key = ? LIMIT 1", (k,)).fetchone()
    return row[0] if row else None

# ---------- Routes ----------

@app.post("/v1/simulate", response_model=SimResponse)
def simulate(req: SimRequest):
    # light validation
    try:
        datetime.strptime(req.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")

    csv_path = None
    try:
        csv_path = run_playlist_creator(req)
        tracks = read_tracks(csv_path, req.limit)
        return SimResponse(
            date=req.date,
            genre=req.genre,
            hours=req.hours,
            repeat_gap_min=req.repeat_gap_min,
            seed=req.seed,
            tracks=tracks,
        )
    finally:
        # clean up temp directory
        if csv_path:
            shutil.rmtree(csv_path.parent, ignore_errors=True)
            
@app.post("/v1/yt/resolve", response_model=ResolveResp)
def yt_resolve(body: ResolveReq):
    video_ids, misses = [], []

    # If the DB isn't there yet, return empty (don’t 404)
    if not os.path.exists(DB_PATH):
        return ResolveResp(video_ids=[], count=0, misses=body.tracks[: body.limit])

    # Query cache
    try:
        with sqlite3.connect(DB_PATH) as conn:
            for t in body.tracks[: body.limit]:
                vid = _lookup_from_cache(conn, t.artist, t.title)
                if vid:
                    video_ids.append(vid)
                else:
                    misses.append({"artist": t.artist, "title": t.title})
    except Exception as e:
        # Don’t leak internals; return empty but log server-side
        print(f"[yt/resolve] cache error: {e}")
        return ResolveResp(video_ids=[], count=0, misses=body.tracks[: body.limit])

    return ResolveResp(video_ids=video_ids, count=len(video_ids), misses=misses)
