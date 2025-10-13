import csv
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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


# ---------- FastAPI app ----------

app = FastAPI(title="TimeDeck API", version="1.0.0")


@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat() + "Z"}


# Apple Music dev token endpoint you already use from the front-end
@app.get("/v1/apple/dev-token")
def apple_dev_token():
    token = os.getenv("APPLE_MUSIC_DEV_TOKEN", "")
    if not token:
        raise HTTPException(status_code=500, detail="APPLE_MUSIC_DEV_TOKEN not set")
    return {"token": token}


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
