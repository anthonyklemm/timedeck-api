import csv, os, subprocess, uuid, json, shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------- Config ----------
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
DISCOGS_TOKEN   = os.getenv("DISCOGS_TOKEN", "")
DISCOGS_KEY     = os.getenv("DISCOGS_KEY", "")
DISCOGS_SECRET  = os.getenv("DISCOGS_SECRET", "")

# Allow your GitHub Pages front-end to call this API
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(title="TapeDeck API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class SimulateRequest(BaseModel):
    date: str = Field(..., description="YYYY-MM-DD")
    genre: str = Field(..., description="e.g., alternative, hot-100, rock")
    hours: float = Field(4, description="Total hours of simulated spins")
    min_gap: int = Field(90, description="Minutes before a song can repeat")
    seed: Optional[str] = Field(None, description="Seed for deterministic runs")

class PlaylistItem(BaseModel):
    timestamp: str
    artist: str
    title: str
    year: Optional[str] = None
    source_rank: Optional[int] = None
    videoId: Optional[str] = None
    resolve_source: Optional[str] = None
    confidence: Optional[float] = None

class SimulateResponse(BaseModel):
    playlist_id: str
    title: str
    count: int
    items: List[PlaylistItem]

class ResolveRequest(BaseModel):
    playlist_id: str
    region: str = "US"
    keep_duplicates: bool = False
    prefer_search: bool = False
    use_discogs: bool = True
    use_musicbrainz: bool = False
    fill_with_search: bool = True

class ResolveResponse(BaseModel):
    playlist_id: str
    unique_videos: int
    video_ids: List[str]
    open_url: str
    dropped: List[Dict[str, str]] = []

# ---------- Helpers ----------
def playlist_dir(pid: str) -> Path:
    d = DATA_DIR / pid
    d.mkdir(parents=True, exist_ok=True)
    return d

def read_csv_items(csv_path: Path) -> List[Dict[str, Any]]:
    out = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append({
                "timestamp": row.get("timestamp") or "",
                "artist": row.get("artist") or "",
                "title": row.get("title") or "",
                "year": row.get("year") or None,
                "source_rank": int(row.get("source_rank")) if row.get("source_rank") else None,
            })
    return out

# ---------- Endpoints ----------
@app.post("/v1/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    # 1) Create a playlist id & folder
    pid = "pl_" + uuid.uuid4().hex[:12]
    pdir = playlist_dir(pid)

    # 2) Call your working simulator script
    #    It should write pdir / "playlist.csv" and pdir / "playlist.m3u"
    out_prefix = pdir / "playlist"
    cmd = [
        "python", "playlist_creator.py",
        "--date", req.date,
        "--genre", req.genre,
        "--hours", str(req.hours),
        "--min-gap", str(req.min_gap),
        "--out", str(out_prefix),
    ]
    if req.seed:
        cmd += ["--seed", req.seed]

    try:
        cp = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"simulate failed: {e.stdout}\n{e.stderr}")

    csv_path = out_prefix.with_suffix(".csv")
    if not csv_path.exists():
        raise HTTPException(status_code=500, detail="simulate succeeded but CSV not found")

    items = read_csv_items(csv_path)
    title = f"TimeDeck — {req.date} — {req.genre}"
    return {
        "playlist_id": pid,
        "title": title,
        "count": len(items),
        "items": items,
    }

@app.post("/v1/resolve/youtube", response_model=ResolveResponse)
def resolve_youtube(req: ResolveRequest):
    pdir = playlist_dir(req.playlist_id)
    csv_path = pdir / "playlist.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="playlist CSV not found; call /v1/simulate first")

    outdir = pdir / "youtube"
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build command to run your resolver with options
    cmd = [
        "python", "yt_ids_from_sources_safe_cache.py",
        "--csv", str(csv_path),
        "--outdir", str(outdir),
        "--region", req.region,
        "--limit", "1000",
    ]
    if req.fill_with_search: cmd += ["--fill-with-search"]
    if req.prefer_search:    cmd += ["--prefer-search"]
    if not req.use_discogs:  cmd += ["--no-discogs"]
    if not req.use_musicbrainz: cmd += ["--no-mb"]

    try:
        cp = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"resolve failed: {e.stdout}\n{e.stderr}")

    vid_file = outdir / "video_ids.txt"
    if not vid_file.exists():
        raise HTTPException(status_code=500, detail="resolver wrote no video_ids.txt")

    video_ids = [ln.strip() for ln in vid_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    open_url = "https://www.youtube.com/watch_videos?video_ids=" + ",".join(video_ids)

    dropped = []
    dropped_csv = outdir / "dropped.csv"
    if dropped_csv.exists():
        with dropped_csv.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                dropped.append({"videoId": row.get("videoId",""), "reason": row.get("reason","")})

    if not req.keep_duplicates:
        # Already deduped by your resolver, but keep this guard.
        seen, uniq = set(), []
        for vid in video_ids:
            if vid not in seen:
                seen.add(vid); uniq.append(vid)
        video_ids = uniq

    return {
        "playlist_id": req.playlist_id,
        "unique_videos": len(video_ids),
        "video_ids": video_ids,
        "open_url": open_url,
        "dropped": dropped,
    }

@app.get("/healthz")
def healthz():
    return {"ok": True, "time": datetime.utcnow().isoformat()+"Z"}
