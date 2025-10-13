#!/usr/bin/env python3
# playlist_creator.py  (strict historical retrieval)
#
# Examples:
#   python playlist_creator.py --date 2002-07-07 --station "97X Tampa" --hours 3 --out ./97x_2002-07-07_3h
#   python playlist_creator.py --date 2002-07-07 --genre alternative --hours 24 --avg-mins 3.3 --min-gap 95 --seed 20020707 --out ./alt_full_day

import argparse
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import billboard  # pip install billboard.py


# -------- Mapping: Genre/Station → candidate Billboard slugs (ordered by history) --------

GENRE_TO_CHARTS = {
    # Alternative / Modern Rock
    "alternative": ["modern-rock-tracks", "alternative-airplay", "alternative-songs"],
    "alt":         ["modern-rock-tracks", "alternative-airplay", "alternative-songs"],

    # Pop / Top 40
    "pop":   ["pop-songs", "radio-songs", "hot-100"],
    "top40": ["pop-songs", "radio-songs", "hot-100"],
    "chr":   ["pop-songs", "radio-songs", "hot-100"],

    # Rock
    "rock":            ["mainstream-rock", "rock-songs", "hot-100"],
    "mainstream-rock": ["mainstream-rock", "rock-songs", "hot-100"],

    # AAA
    "aaa":               ["adult-alternative-songs", "triple-a", "radio-songs"],
    "adult-alternative": ["adult-alternative-songs", "triple-a", "radio-songs"],

    # Country / Hip-hop / Latin
    "country": ["country-songs", "hot-100"],
    "hiphop":  ["r-b-hip-hop-songs", "hot-100"],
    "rnb":     ["r-b-hip-hop-songs", "hot-100"],
    "latin":   ["latin-songs", "hot-100"],

    # Fallback bucket
    "hot100":  ["hot-100"],
}

STATION_TO_GENRE = {
    # LA
    "kroq": "alternative",
    "kiis": "pop",
    # NYC
    "z100": "pop",
    # Chicago
    "wxrt": "aaa",
    "q101": "alternative",
    # Tampa example
    "97x": "alternative",
}

# -------- Data types --------

@dataclass
class ChartEntry:
    rank: int
    artist: str
    title: str

@dataclass
class Spin:
    timestamp: datetime
    artist: str
    title: str
    source_rank: int


# -------- Helpers: resolve chart set, date logic --------

def resolve_candidate_slugs(station: Optional[str], genre: Optional[str]) -> List[str]:
    g = None
    if station:
        key = "".join(c for c in station.lower() if c.isalnum())
        for k, gg in STATION_TO_GENRE.items():
            if k in key:
                g = gg
                break
    if not g and genre:
        g = genre.lower().strip()

    if g and g in GENRE_TO_CHARTS:
        return GENRE_TO_CHARTS[g]
    # Ultimate fallback
    return ["hot-100"]


def nearest_billboard_chart_date(user_date: datetime) -> str:
    """Billboard chart dates are Saturdays. Choose next Saturday on/after user date."""
    dow = user_date.weekday()  # Mon=0..Sat=5
    days_until_sat = (5 - dow) % 7
    chart_dt = user_date + timedelta(days=days_until_sat)
    return chart_dt.strftime("%Y-%m-%d")


def parse_chart_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


# -------- Fetch with strict date validation --------

def fetch_chart_entries_strict(chart_slugs: List[str], chart_date: str, limit: int = 40, tolerance_days: int = 7) -> List[ChartEntry]:
    """
    Try slugs in order; ensure the chart returned is within ±tolerance_days of requested chart_date.
    If not, try next slug. If none match, raise.
    """
    if not hasattr(billboard, "ChartData"):
        raise RuntimeError("Wrong module imported as 'billboard'. Install with: pip install billboard.py")

    target_dt = parse_chart_date(chart_date)
    last_err = None
    tried = []

    for slug in chart_slugs:
        if slug in tried:
            continue
        tried.append(slug)
        try:
            chart = billboard.ChartData(slug, date=chart_date)
            # billboard.py returns chart.date (string like '2002-07-13')
            returned_dt = parse_chart_date(chart.date) if chart.date else None
            if not returned_dt or abs((returned_dt - target_dt).days) > tolerance_days:
                print(f"[warn] '{slug}' @ {chart_date} returned {chart.date} (out of tolerance) — trying next slug.")
                continue

            entries = [
                ChartEntry(rank=int(e.rank), artist=str(e.artist), title=str(e.title))
                for e in chart[:limit]
            ]
            entries.sort(key=lambda x: x.rank)
            if entries:
                print(f"[ok] Billboard '{slug}' @ {chart.date} • {len(entries)} entries")
                return entries
        except Exception as e:
            last_err = e
            print(f"[warn] Fetch failed for '{slug}' @ {chart_date}: {e}")

    raise RuntimeError(f"No historical chart matched within ±{tolerance_days} days of {chart_date}. Tried: {tried}. Last error: {last_err}")


# -------- Rotation simulator --------

def default_weight_for_rank(rank: int) -> float:
    if rank <= 5:   base = 1.0
    elif rank <= 10: base = 0.65
    elif rank <= 20: base = 0.40
    elif rank <= 30: base = 0.22
    else:            base = 0.12
    decay = max(1.05 - 0.01 * rank, 0.60)
    return base * decay


def build_weighted_pool(entries: List[ChartEntry]) -> List[Tuple[int, float]]:
    weights = []
    for idx, e in enumerate(entries):
        w = default_weight_for_rank(e.rank)
        weights.append((idx, w))
    total = sum(w for _, w in weights) or 1.0
    return [(i, w / total) for i, w in weights]


def choose_next(pool: List[Tuple[int, float]], last_played_at: dict, now: datetime, min_gap_min: int) -> int:
    adjusted = []
    for idx, p in pool:
        last = last_played_at.get(idx)
        minutes_since = (now - last).total_seconds() / 60.0 if last else 9999.0
        adjusted.append((idx, p * (0.01 if minutes_since < min_gap_min else 1.0)))

    total = sum(p for _, p in adjusted) or 1.0
    adjusted = [(i, p / total) for i, p in adjusted]

    r = random.random()
    acc = 0.0
    for i, p in adjusted:
        acc += p
        if r <= acc:
            return i
    return adjusted[-1][0]


def simulate_rotations(
        entries: List[ChartEntry],
        start_time: datetime,
        hours: float,
        average_song_minutes: float,
        min_gap_minutes: int,
        seed: int
) -> List[Spin]:
    random.seed(seed)
    pool = build_weighted_pool(entries)
    n_spins = max(1, int(hours * 60 / average_song_minutes))
    spins: List[Spin] = []
    now = start_time
    last_played_at = {}
    for _ in range(n_spins):
        idx = choose_next(pool, last_played_at, now, min_gap_minutes)
        e = entries[idx]
        spins.append(Spin(timestamp=now, artist=e.artist, title=e.title, source_rank=e.rank))
        last_played_at[idx] = now
        now += timedelta(minutes=average_song_minutes)
    return spins


# -------- Exporters --------

def export_csv(spins: List[Spin], path: Path) -> Path:
    df = pd.DataFrame([{
        "timestamp": s.timestamp.isoformat(),
        "artist": s.artist,
        "title": s.title,
        "source_rank": s.source_rank
    } for s in spins])
    df.to_csv(path, index=False)
    return path


def export_m3u(spins: List[Spin], path: Path, title_comment: Optional[str] = None) -> Path:
    lines = ["#EXTM3U"]
    if title_comment:
        lines.append(f"#PLAYLIST:{title_comment}")
    for s in spins:
        extinf = f"#EXTINF:-1,{s.timestamp.strftime('%H:%M')} — {s.artist} - {s.title} (rank {s.source_rank})"
        lines.append(extinf)
        lines.append(f"{s.artist} - {s.title}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# -------- CLI --------

def main():
    ap = argparse.ArgumentParser(description="Build a simulated radio playlist from historical Billboard weekly charts (strict).")
    ap.add_argument("--date", required=True, help="Target date (YYYY-MM-DD).")
    ap.add_argument("--genre", help="Genre (e.g., alternative, pop, rock, country, hiphop, latin).")
    ap.add_argument("--station", help="Station name (e.g., '97X Tampa', 'KROQ', 'Z100').")
    ap.add_argument("--hours", type=float, default=3.0, help="How many hours to simulate (default 3.0).")
    ap.add_argument("--start", default="06:00", help="Start time (HH:MM 24h, default 06:00).")
    ap.add_argument("--avg-mins", type=float, default=3.5, help="Average song length in minutes (default 3.5).")
    ap.add_argument("--min-gap", type=int, default=90, help="Min minutes before a track can repeat (default 90).")
    ap.add_argument("--seed", type=int, default=97, help="RNG seed for reproducibility (default 97).")
    ap.add_argument("--limit", type=int, default=40, help="Top-N chart records to use (default 40).")
    ap.add_argument("--out", type=Path, required=True, help="Output file prefix (no extension).")
    args = ap.parse_args()

    if not args.genre and not args.station:
        raise SystemExit("Provide either --genre or --station (or both).")

    # Parse date/time
    try:
        d = datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        raise SystemExit("Invalid --date. Use YYYY-MM-DD (e.g., 2002-07-07).")
    try:
        hh, mm = args.start.split(":")
        start_time = d.replace(hour=int(hh), minute=int(mm), second=0)
    except Exception:
        raise SystemExit("Invalid --start. Use HH:MM 24h (e.g., 06:00).")

    # Resolve candidate slugs & target chart week
    slugs = resolve_candidate_slugs(args.station, args.genre)
    chart_date = nearest_billboard_chart_date(d)

    # Fetch entries strictly for that week
    entries = fetch_chart_entries_strict(slugs, chart_date, limit=args.limit, tolerance_days=7)
    if not entries:
        raise SystemExit("No chart entries returned.")

    # Simulate
    spins = simulate_rotations(
        entries=entries,
        start_time=start_time,
        hours=args.hours,
        average_song_minutes=args.avg_mins,
        min_gap_minutes=args.min_gap,
        seed=args.seed
    )

    # Title/comment
    label_bits = [args.station or args.genre or slugs[0], args.date, f"{args.hours}h"]
    playlist_title = " | ".join([str(b) for b in label_bits if b])

    # Export
    csv_path = export_csv(spins, args.out.with_suffix(".csv"))
    m3u_path = export_m3u(spins, args.out.with_suffix(".m3u"), title_comment=playlist_title)

    print(f"[ok] Slugs tried: {slugs}")
    print(f"[ok] Chart week: {chart_date} • Entries used: {len(entries)}")
    print(f"[ok] Simulated spins: {len(spins)} from {start_time.strftime('%Y-%m-%d %H:%M')} over {args.hours} hours")
    print(f"[ok] Wrote CSV: {csv_path}")
    print(f"[ok] Wrote M3U: {m3u_path}")
    if args.station:
        print("[note] Station→genre mapping is heuristic; tweak STATION_TO_GENRE / GENRE_TO_CHARTS if needed.")


if __name__ == "__main__":
    main()
