import os, time, jwt, csv, io, textwrap, requests, random
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import billboard  # pip install billboard.py
from fastapi.middleware.cors import CORSMiddleware

# ------------------ Apple credentials (ENV-based) ------------------
TEAM_ID = os.getenv("APPLE_TEAM_ID", "")
KEY_ID  = os.getenv("APPLE_KEY_ID", "")
STOREFRONT = os.getenv("APPLE_STOREFRONT", "us")

def _read_private_key() -> str:
    inline = os.getenv("APPLE_PRIVATE_KEY")
    if inline:
        return textwrap.dedent(inline).strip()
    p = os.getenv("APPLE_PRIVATE_KEY_PATH")
    if p and os.path.exists(p):
        with open(p, "r") as f:
            return f.read()
    raise RuntimeError("Set APPLE_PRIVATE_KEY or APPLE_PRIVATE_KEY_PATH (and APPLE_TEAM_ID / APPLE_KEY_ID).")

PRIVATE_KEY = _read_private_key()
APPLE_API = "https://api.music.apple.com/v1"

def make_developer_token() -> str:
    now = int(time.time())
    payload = {"iss": TEAM_ID, "iat": now, "exp": now + 60*60*12}
    headers = {"alg": "ES256", "kid": KEY_ID}
    return jwt.encode(payload, PRIVATE_KEY, algorithm="ES256", headers=headers)

def am_headers(dev_token: str, user_token: str):
    return {"Authorization": f"Bearer {dev_token}",
            "Music-User-Token": user_token,
            "Accept": "application/json"}

# ------------------ Simulator ------------------

GENRE_TO_CHARTS = {
    "alternative": ["modern-rock-tracks", "alternative-airplay", "alternative-songs"],
    "alt":         ["modern-rock-tracks", "alternative-airplay", "alternative-songs"],
    "pop":   ["pop-songs", "radio-songs", "hot-100"],
    "top40": ["pop-songs", "radio-songs", "hot-100"],
    "chr":   ["pop-songs", "radio-songs", "hot-100"],
    "rock":            ["mainstream-rock", "rock-songs", "hot-100"],
    "mainstream-rock": ["mainstream-rock", "rock-songs", "hot-100"],
    "aaa":               ["adult-alternative-songs", "triple-a", "radio-songs"],
    "adult-alternative": ["adult-alternative-songs", "triple-a", "radio-songs"],
    "country": ["country-songs", "hot-100"],
    "hiphop":  ["r-b-hip-hop-songs", "hot-100"],
    "rnb":     ["r-b-hip-hop-songs", "hot-100"],
    "latin":   ["latin-songs", "hot-100"],
    "hot100":  ["hot-100"],
}
STATION_TO_GENRE = {"kroq":"alternative","kiis":"pop","z100":"pop","wxrt":"aaa","q101":"alternative","97x":"alternative"}

class ChartEntry(BaseModel):
    rank: int
    artist: str
    title: str

class Spin(BaseModel):
    timestamp: str  # ISO string
    artist: str
    title: str
    source_rank: int

def resolve_candidate_slugs(station: Optional[str], genre: Optional[str]) -> List[str]:
    g = None
    if station:
        key = "".join(c for c in station.lower() if c.isalnum())
        for k, gg in STATION_TO_GENRE.items():
            if k in key: g = gg; break
    if not g and genre: g = genre.lower().strip()
    return GENRE_TO_CHARTS.get(g, ["hot-100"])

def nearest_billboard_chart_date(user_date: datetime) -> str:
    dow = user_date.weekday()  # Mon=0..Sat=5
    days_until_sat = (5 - dow) % 7
    return (user_date + timedelta(days=days_until_sat)).strftime("%Y-%m-%d")

def fetch_chart_entries_strict(slugs: List[str], chart_date: str, limit: int = 40, tolerance_days: int = 7) -> List[ChartEntry]:
    if not hasattr(billboard, "ChartData"):
        raise RuntimeError("Wrong 'billboard' module; install billboard.py")
    target_dt = datetime.strptime(chart_date, "%Y-%m-%d")
    last_err = None
    for slug in slugs:
        try:
            chart = billboard.ChartData(slug, date=chart_date)
            if not chart.date: continue
            returned_dt = datetime.strptime(chart.date, "%Y-%m-%d")
            if abs((returned_dt - target_dt).days) > tolerance_days:
                continue
            out = [ChartEntry(rank=int(e.rank), artist=str(e.artist), title=str(e.title)) for e in chart[:limit]]
            out.sort(key=lambda x: x.rank)
            if out: return out
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"No chart within ±{tolerance_days} days for {chart_date}. Last error: {last_err}")

def default_weight_for_rank(rank: int) -> float:
    base = 1.0 if rank<=5 else 0.65 if rank<=10 else 0.40 if rank<=20 else 0.22 if rank<=30 else 0.12
    decay = max(1.05 - 0.01*rank, 0.60)
    return base*decay

def simulate(entries: List[ChartEntry], start_time: datetime, hours: float, min_gap: int, seed: int) -> List[Spin]:
    AVG_MIN = 3.5  # fixed
    random.seed(seed)
    weights = [(i, default_weight_for_rank(e.rank)) for i,e in enumerate(entries)]
    total = sum(w for _,w in weights) or 1.0
    pool = [(i, w/total) for i,w in weights]

    def choose(now, last_played):
        adjusted=[]
        for i,p in pool:
            last = last_played.get(i)
            mins = (now-last).total_seconds()/60 if last else 9999
            adjusted.append((i, p*(0.01 if mins<min_gap else 1.0)))
        tot = sum(p for _,p in adjusted) or 1.0
        r = random.random(); acc=0.0
        for i,p in adjusted:
            acc += p/tot
            if r<=acc: return i
        return adjusted[-1][0]

    n = max(1, int(hours*60/AVG_MIN))
    last_played={}; now=start_time; spins=[]
    for _ in range(n):
        idx = choose(now, last_played)
        e = entries[idx]
        spins.append(Spin(timestamp=now.isoformat(), artist=e.artist, title=e.title, source_rank=e.rank))
        last_played[idx]=now
        now += timedelta(minutes=AVG_MIN)
    return spins

def to_csv_text(spins: List[Spin]) -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["timestamp","artist","title","source_rank"])
    for s in spins: w.writerow([s.timestamp, s.artist, s.title, s.source_rank])
    return out.getvalue()

def to_m3u_text(spins: List[Spin], title_comment: Optional[str]) -> str:
    lines=["#EXTM3U"]
    if title_comment: lines.append(f"#PLAYLIST:{title_comment}")
    for s in spins:
        t = datetime.fromisoformat(s.timestamp).strftime("%H:%M")
        lines.append(f"#EXTINF:-1,{t} — {s.artist} - {s.title} (rank {s.source_rank})")
        lines.append(f"{s.artist} - {s.title}")
    return "\n".join(lines)

# ------------------ FastAPI setup ------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your GitHub Pages origin later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    return open("static/index.html", "r", encoding="utf-8").read()

@app.get("/dev-token")
def dev_token():
    return {"token": make_developer_token(), "storefront": STOREFRONT}

class SimulateParams(BaseModel):
    date: str                 # YYYY-MM-DD
    hours: float = 3.0
    genre: Optional[str] = None
    station: Optional[str] = None
    min_gap: int = 90
    seed: int = 97
    limit: int = 40

@app.post("/simulate")
def simulate_endpoint(p: SimulateParams):
    try:
        d = datetime.strptime(p.date, "%Y-%m-%d")
        start_dt = d.replace(hour=6, minute=0, second=0)  # fixed 06:00
    except Exception:
        raise HTTPException(400, "Bad date.")
    slugs = resolve_candidate_slugs(p.station, p.genre)
    chart_date = nearest_billboard_chart_date(d)
    entries = fetch_chart_entries_strict(slugs, chart_date, limit=p.limit, tolerance_days=7)
    spins = simulate(entries, start_dt, p.hours, p.min_gap, p.seed)
    title = " | ".join(filter(None, [p.station or p.genre or slugs[0], p.date, f"{p.hours}h"]))
    return {
        "meta": {"slugs": slugs, "chart_week": chart_date, "used": len(entries), "title": title, "input_year": d.year},
        "spins": [s.dict() for s in spins],
        "csv": to_csv_text(spins),
        "m3u": to_m3u_text(spins, title_comment=title),
    }

class SpinModel(BaseModel):
    timestamp: str
    artist: str
    title: str
    source_rank: int

class CreatePayload(BaseModel):
    userToken: str
    name: str
    spins: List[SpinModel]
    storefront: Optional[str] = None

def search_song(dev_token, user_token, storefront, artist, title, year_hint=None):
    q = f"{artist} {title}" + (f" {year_hint}" if year_hint else "")
    params = {"term": q, "limit": 5, "types": "songs"}
    r = requests.get(f"{APPLE_API}/catalog/{storefront}/search",
                     headers=am_headers(dev_token, user_token), params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    try:
        return data["results"]["songs"]["data"][0]["id"]
    except Exception:
        return None

@app.post("/create-from-sim")
def create_from_sim(payload: CreatePayload):
    dev = make_developer_token()
    sf = payload.storefront or STOREFRONT

    # Derive year hint from the first spin's timestamp
    year_hint = None
    try:
        if payload.spins:
            year_hint = datetime.fromisoformat(payload.spins[0].timestamp).year
    except Exception:
        year_hint = None

    # 1) create playlist
    body = {"attributes": {"name": payload.name,
                       "description": "Generated with TapeDeck Time Machine"}}
    r = requests.post(f"{APPLE_API}/me/library/playlists",
                      headers=am_headers(dev, payload.userToken), json=body, timeout=20)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, f"Create playlist failed: {r.text}")
    playlist_id = r.json()["data"][0]["id"]

    # 2) resolve tracks
    catalog_ids=[]; misses=[]
    for s in payload.spins:
        cid = search_song(dev, payload.userToken, sf, s.artist, s.title, year_hint)
        if cid: catalog_ids.append(cid)
        else: misses.append({"artist": s.artist, "title": s.title})

    # 3) add in chunks
    CHUNK=80
    for i in range(0, len(catalog_ids), CHUNK):
        batch = {"data": [{"id": cid, "type": "songs"} for cid in catalog_ids[i:i+CHUNK]]}
        r = requests.post(f"{APPLE_API}/me/library/playlists/{playlist_id}/tracks",
                          headers=am_headers(dev, payload.userToken), json=batch, timeout=30)
        if r.status_code >= 400:
            raise HTTPException(r.status_code, f"Add tracks failed at batch {i}: {r.text}")

    return {"playlist_id": playlist_id, "added": len(catalog_ids), "misses": misses}
