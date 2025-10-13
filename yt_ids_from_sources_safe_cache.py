#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yt_ids_from_sources_safe_cache.py

Safe, fast resolver for YouTube videoIds from (artist, title[, year]):
- Uses Discogs + MusicBrainz + YouTube Search (v3) with **short-circuit** order.
- Caches **only verified** (embeddable, region-OK, processed) IDs.
- Re-verifies cached entries by default; can purge stale ones.
- Writes: video_ids.txt, matches.csv, misses.csv, dropped.csv, embed.html

Quick start (search-first, no MB; fastest):
  PYTHONUNBUFFERED=1 python -u yt_ids_from_sources_safe_cache.py \
    --csv "$HOME/Downloads/attachments (2)/alt_full_day1.csv" \
    --outdir ./yt_seed_fast \
    --region US \
    --limit 50 \
    --fill-with-search \
    --force-refresh \
    --prefer-search \
    --no-mb

Env:
  export YOUTUBE_API_KEY='YOUR_YT_KEY'
  export DISCOGS_TOKEN='YOUR_DISCOGS_PERSONAL_TOKEN'     # or:
  # export DISCOGS_KEY='YOUR_KEY'
  # export DISCOGS_SECRET='YOUR_SECRET'

pip install requests
"""

import os, re, csv, time, html, sqlite3, argparse, threading, requests
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# ---------- Config ----------
DISCOGS_TOKEN  = "gDHcyymczFqdmrmNFRiCAzAaEotfNfYzOQuxlldu"
DISCOGS_KEY    = os.getenv("DISCOGS_KEY")
DISCOGS_SECRET = os.getenv("DISCOGS_SECRET")
YOUTUBE_API_KEY = "REDACTED_YT_KEY"
DEFAULT_REGION = os.getenv("YT_REGION", "US")
HEADERS = {"User-Agent": "TimeDeck/1.0 (contact: you@example.com)"}

DISCOGS_SEARCH = "https://api.discogs.com/database/search"
MUSICBRAINZ_SEARCH = "https://musicbrainz.org/ws/2/recording"
MUSICBRAINZ_RECORDING = "https://musicbrainz.org/ws/2/recording/"
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

CACHE_DB_DEFAULT = "yt_cache.sqlite"

BAD_TITLE_PAT = re.compile(
    r"(karaoke|cover|tribute|nightcore|sped\s*up|slowed|8d|fan\s*made|remix|edit|lyrics|audio\s+only|live|rehearsal|demo|reaction|performance)",
    re.I,
)

def norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s*\(.*?\)\s*", "", s)  # drop parentheticals
    s = re.sub(r"\s+", " ", s)
    return s

def cache_key(artist: str, title: str, year: Optional[str], country: str) -> str:
    return f"{norm(artist).lower()}|{norm(title).lower()}|{(year or '').lower()}|{country.upper()}"

# ---------- SQLite cache ----------
def db_connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("""
    CREATE TABLE IF NOT EXISTS cache (
      key TEXT PRIMARY KEY,
      artist TEXT, title TEXT, year TEXT, country TEXT,
      video_id TEXT, source TEXT, confidence REAL,
      checked_at INTEGER
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_artist_title ON cache(artist, title)")
    return conn

def cache_get(conn, key: str) -> Optional[Dict]:
    row = conn.execute(
        "SELECT artist,title,year,country,video_id,source,confidence FROM cache WHERE key=?",
        (key,)
    ).fetchone()
    if not row: return None
    a,t,y,c,vid,src,conf = row
    return {"artist":a,"title":t,"year":y,"country":c,"videoId":vid,"source":src,"confidence":conf}

def cache_put_verified(conn, key: str, a: str, t: str, y: Optional[str], c: str, vid: str, src: str, conf: float):
    # Only after passing verify
    conn.execute("""INSERT OR REPLACE INTO cache
    (key,artist,title,year,country,video_id,source,confidence,checked_at)
    VALUES (?,?,?,?,?,?,?,?,strftime('%s','now'))""", (key,a,t,y,c,vid,src,conf))
    conn.commit()

def cache_delete(conn, key: str):
    conn.execute("DELETE FROM cache WHERE key=?", (key,))
    conn.commit()

# ---------- Discogs (with throttle) ----------
_discogs_lock = threading.Lock()
_last_discogs_ts = 0.0
DISCOGS_MIN_INTERVAL = 1.1  # ~60/min

def _discogs_throttle():
    global _last_discogs_ts
    with _discogs_lock:
        now = time.time()
        delta = now - _last_discogs_ts
        wait = max(0.0, DISCOGS_MIN_INTERVAL - delta)
        if wait > 0: time.sleep(wait)
        _last_discogs_ts = time.time()

def _discogs_auth() -> Dict[str, str]:
    if DISCOGS_TOKEN: return {"token": DISCOGS_TOKEN}
    if DISCOGS_KEY and DISCOGS_SECRET: return {"key": DISCOGS_KEY, "secret": DISCOGS_SECRET}
    return {}

def extract_video_id(url: str) -> Optional[str]:
    # Case-sensitive capture. Do NOT lowercase.
    m = re.search(r"(?:v=|youtu\.be/|embed/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None

def discogs_find_youtube(artist: str, title: str) -> Optional[str]:
    def get(url: str, params: dict, tries=3) -> Optional[requests.Response]:
        for attempt in range(1, tries+1):
            _discogs_throttle()
            try:
                r = requests.get(url, params=params, headers=HEADERS, timeout=20)
                if r.status_code == 429:
                    ra = r.headers.get("Retry-After")
                    time.sleep(float(ra) if ra else 2.0*attempt)
                    continue
                r.raise_for_status()
                return r
            except requests.RequestException:
                if attempt == tries: return None
                time.sleep(2.0*attempt)
        return None

    q = f"{artist} - {title}"
    base = {"q": q, "per_page": 5}
    base.update(_discogs_auth())

    # Primary search: releases; then masters if needed
    r = get(DISCOGS_SEARCH, {**base, "type": "release"})
    results = (r.json().get("results", []) if r else [])[:5]
    if len(results) < 3:
        r2 = get(DISCOGS_SEARCH, {**base, "type": "master"})
        if r2: results += r2.json().get("results", [])[:5]

    def pick_from_videos(videos: List[Dict]) -> Optional[str]:
        best_id, best_score = None, -1.0
        a_low, t_low = norm(artist).lower(), norm(title).lower()
        for v in videos or []:
            uri_orig = (v.get("uri") or "")
            if "youtube.com" not in uri_orig.lower():
                continue
            tt = v.get("title") or ""
            if BAD_TITLE_PAT.search(tt):
                continue
            sc = 0.0
            tl = tt.lower()
            if "official" in tl: sc += 0.4
            if "vevo" in tl or "topic" in tl: sc += 0.25
            if a_low in tl: sc += 0.25
            if t_low in tl: sc += 0.5
            vid = extract_video_id(uri_orig)  # preserve case
            if vid and sc > best_score:
                best_id, best_score = vid, sc
        return best_id

    for item in results:
        url = item.get("resource_url")
        if not url: continue
        d = get(url, {})
        if d:
            obj = d.json()
            pick = pick_from_videos(obj.get("videos"))
            if pick: return pick
            murl = obj.get("master_url")
            if murl:
                md = get(murl, {})
                if md:
                    pick = pick_from_videos(md.json().get("videos"))
                    if pick: return pick
    return None

# ---------- MusicBrainz ----------
def musicbrainz_find_youtube(artist: str, title: str) -> Optional[str]:
    params = {"query": f'recording:"{norm(title)}" AND artist:"{artist}"', "fmt":"json", "limit":5}
    try:
        r = requests.get(MUSICBRAINZ_SEARCH, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        recs = (r.json().get("recordings", []) or [])[:5]
        for rec in recs:
            rid = rec.get("id")
            if not rid: continue
            time.sleep(1.1)  # be nice
            rr = requests.get(
                MUSICBRAINZ_RECORDING + rid,
                params={"inc":"url-rels","fmt":"json"},
                headers=HEADERS, timeout=15
            )
            if not rr.ok: continue
            rels = rr.json().get("relations", []) or []
            for rel in rels:
                url = (rel.get("url") or {}).get("resource", "")
                if "youtube.com" in url.lower():
                    vid = extract_video_id(url)  # preserve case
                    if vid: return vid
    except requests.RequestException:
        return None
    return None

# ---------- YouTube helpers ----------
def yt_search(api_key: str, query: str, region: str, max_results: int = 10) -> List[Dict]:
    params = {
        "key": api_key, "q": query, "part": "snippet", "type": "video",
        "maxResults": max_results, "regionCode": region, "videoCategoryId": "10",
        "safeSearch": "none"
    }
    r = requests.get(YOUTUBE_SEARCH_URL, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json().get("items", [])

def yt_videos(api_key: str, ids: List[str]) -> Dict[str, Dict]:
    out = {}
    for i in range(0, len(ids), 50):
        batch = ids[i:i+50]
        params = {"key": api_key, "id": ",".join(batch), "part": "status,contentDetails,snippet"}
        r = requests.get(YOUTUBE_VIDEOS_URL, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        for item in r.json().get("items", []):
            out[item["id"]] = item
    return out

def pick_best_candidate(artist: str, title: str, candidates: List[Dict]) -> Optional[Tuple[str,float,str]]:
    if not candidates: return None
    na, nt = norm(artist).lower(), norm(title).lower()
    def score(it: Dict) -> float:
        sn=it["snippet"]; t=sn.get("title",""); ch=sn.get("channelTitle",""); s=0.0
        tnorm=norm(t).lower()
        if nt in tnorm: s+=0.6
        if na in ch.lower(): s+=0.4
        if "vevo" in ch.lower() or "topic" in ch.lower(): s+=0.25
        if "official" in t.lower(): s+=0.15
        if BAD_TITLE_PAT.search(t): s-=0.7
        return s
    best=max(candidates, key=score); vid=best["id"]["videoId"]
    return vid, max(0.05,min(1.0, score(best))), best["snippet"].get("channelTitle","")

def verify_ids(ids: List[str], api_key: Optional[str], region: str) -> Tuple[List[str], Dict[str,str]]:
    if not ids: return [], {}
    if not api_key: return ids[:], {}
    meta = yt_videos(api_key, ids)
    keep, drops = [], {}
    for vid in ids:
        m = meta.get(vid)
        if not m:
            drops[vid] = "not_found_in_api"; continue
        st = (m.get("status") or {})
        if st.get("uploadStatus") != "processed":
            drops[vid] = f"status:{st.get('uploadStatus')}"; continue
        if st.get("embeddable") is False:
            drops[vid] = "non_embeddable"; continue
        rr = (m.get("contentDetails") or {}).get("regionRestriction", {}) or {}
        blocked = set(rr.get("blocked", []) or [])
        allowed = set(rr.get("allowed", []) or [])
        if allowed and region not in allowed:
            drops[vid] = "not_in_allowed_region"; continue
        if region in blocked:
            drops[vid] = "region_blocked"; continue
        keep.append(vid)
    return keep, drops

# ---------- Resolve pipeline (short-circuit; verify before caching) ----------
def resolve_candidates(artist: str, title: str, year: Optional[str], region: str,
                       allow_search: bool, no_discogs: bool = False, no_mb: bool = False,
                       prefer_search: bool = False) -> List[Tuple[str,float,str]]:
    """
    Return a list of (videoId, confidence, source) in PRIORITY order,
    but short-circuit: we stop as soon as we get a candidate from a higher tier.
    """
    out: List[Tuple[str,float,str]] = []

    def try_search() -> Optional[Tuple[str,float,str]]:
        if allow_search and YOUTUBE_API_KEY:
            q = f"{artist} - {title} {(year or '')}".strip()
            try:
                items = yt_search(YOUTUBE_API_KEY, q, region=region, max_results=10)
                pick = pick_best_candidate(artist, title, items)
                if pick: return (pick[0], pick[1], "search")
            except requests.RequestException:
                return None
        return None

    tiers = ["discogs", "mb", "search"]
    if prefer_search:
        tiers = ["search", "discogs", "mb"]

    for tier in tiers:
        if tier == "discogs" and not no_discogs:
            vid = discogs_find_youtube(artist, title)
            if vid:
                out.append((vid, 0.90, "discogs"))
                return out
        elif tier == "mb" and not no_mb:
            vid = musicbrainz_find_youtube(artist, title)
            if vid:
                out.append((vid, 0.85, "musicbrainz"))
                return out
        elif tier == "search":
            pick = try_search()
            if pick:
                out.append(pick)
                return out

    return out  # may be empty

# ---------- HTML writer ----------
def write_embed_html(ids: List[str], title: str, out_path: Path):
    if not ids:
        out_path.write_text(
            "<!doctype html><meta charset='utf-8'><title>Empty</title><p>No playable videos.</p>",
            encoding="utf-8"
        ); return
    first, rest = ids[0], ids[1:]
    playlist_param = ",".join(rest)
    open_url = "https://www.youtube.com/watch_videos?video_ids=" + ",".join(ids)
    doc=f"""<!doctype html>
<html lang="en"><meta charset="utf-8">
<title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body{{margin:0;background:#0B0F1A;color:#ECECF1;font-family:-apple-system,system-ui,Segoe UI,Roboto,Arial,sans-serif}}
  .wrap{{max-width:900px;margin:20px auto;padding:12px}}
  .meta{{opacity:.8;margin:8px 0 14px}}
  #frameWrap{{background:#0f1424;border-radius:12px;box-shadow:0 8px 30px rgba(2,5,12,.45);height:360px;display:flex;align-items:center;justify-content:center}}
  iframe{{width:100%;height:360px;border:0;border-radius:12px}}
  a.btn{{display:inline-block;margin:10px 8px 0 0;padding:10px 14px;border-radius:10px;border:1px solid #1d2340;background:#0b1120;color:#ECECF1;text-decoration:none;font-weight:700}}
</style>
<div class="wrap">
  <h1 style="margin:8px 0">{html.escape(title)}</h1>
  <div class="meta">YouTube queue • {len(ids)} videos</div>
  <div id="frameWrap">
    <iframe id="ytp" allow="autoplay; encrypted-media" allowfullscreen title="YouTube player"
      src="https://www.youtube.com/embed/{first}?playlist={playlist_param}&autoplay=0&playsinline=1&rel=0&modestbranding=1&origin=http://localhost">
    </iframe>
  </div>
  <div><a class="btn" href="{html.escape(open_url)}" target="_blank" rel="noopener">Open in YouTube</a></div>
</div>
<script>(function(){{try{{const f=document.getElementById('ytp');const u=new URL(f.src);u.searchParams.set('origin',window.location.origin);f.src=u.toString();}}catch(e){{}}}})();</script>
"""
    out_path.write_text(doc, encoding="utf-8")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Discogs/MusicBrainz→YouTube IDs (safe cache, verified only).")
    ap.add_argument("--csv", required=True, help="Input CSV: timestamp,artist,title,source_rank[,year]")
    ap.add_argument("--outdir", default="yt_seed_out", help="Output directory")
    ap.add_argument("--region", default=DEFAULT_REGION, help="Region code (e.g., US)")
    ap.add_argument("--limit", type=int, default=50, help="Max tracks to resolve")
    ap.add_argument("--skip", type=int, default=0, help="Skip first N rows")
    ap.add_argument("--cache", default=CACHE_DB_DEFAULT, help="SQLite cache path")

    # Behavior toggles
    ap.add_argument("--no-search", action="store_true", help="Disable YouTube search fallback")
    ap.add_argument("--fill-with-search", action="store_true", help="Try search to replace dropped IDs")
    ap.add_argument("--no-verify", action="store_true", help="Skip verification (NOT recommended)")
    ap.add_argument("--force-refresh", action="store_true", help="Ignore cached IDs (do fresh lookups)")
    ap.add_argument("--purge-cache", action="store_true", help="Delete any failing cached IDs encountered")

    # New speed/ordering flags
    ap.add_argument("--no-discogs", action="store_true", help="Disable Discogs lookup")
    ap.add_argument("--no-mb", action="store_true", help="Disable MusicBrainz lookup")
    ap.add_argument("--prefer-search", action="store_true", help="Try YouTube search first")

    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    conn = db_connect(Path(args.cache))

    # Load rows
    rows=[]
    with open(args.csv, newline="", encoding="utf-8") as f:
        rdr=csv.DictReader(f)
        for i,r in enumerate(rdr):
            if i < args.skip: continue
            rows.append({"artist": r["artist"].strip(), "title": r["title"].strip(), "year": (r.get("year") or "").strip() or None})
            if len(rows) >= args.limit: break
    print(f"[info] Loaded {len(rows)} rows; region={args.region}; limit={args.limit}; cache={args.cache}", flush=True)

    results, misses = [], []
    src_counts = {"cache":0,"discogs":0,"musicbrainz":0,"search":0,"search_fill":0}

    # Step 1: Gather candidates (respect cache unless force-refresh)
    candidates_per_row: List[List[Tuple[str,float,str]]] = []
    for idx, r in enumerate(rows, 1):
        a,t,y = r["artist"], r["title"], r["year"]
        key = cache_key(a,t,y,args.region)
        row_candidates: List[Tuple[str,float,str]] = []

        if not args.force_refresh:
            cached = cache_get(conn, key)
            if cached and cached.get("videoId"):
                row_candidates.append((cached["videoId"], float(cached.get("confidence",0.9)), "cache"))

        if not row_candidates:
            # short-circuit: stop at first tier that yields a candidate
            row_candidates.extend(
                resolve_candidates(
                    a, t, y, args.region,
                    allow_search=(not args.no_search),
                    no_discogs=args.no_discogs,
                    no_mb=args.no_mb,
                    prefer_search=args.prefer_search,
                )
            )

        candidates_per_row.append(row_candidates)
        time.sleep(0.15)

    # Step 2: Verify per row; keep first that passes; cache it
    kept_ids_in_order: List[str] = []
    dropped_reasons: Dict[str,str] = {}

    for idx, (r, cands) in enumerate(zip(rows, candidates_per_row), 1):
        a,t,y = r["artist"], r["title"], r["year"]
        key = cache_key(a,t,y,args.region)
        if not cands:
            print(f"[{idx:03}/{len(rows)}] ❌ {a} — {t} (reason=no_candidates)", flush=True)
            misses.append({**r, "reason":"no_candidates"})
            continue

        passed_vid, passed_src, passed_conf = None, None, 0.0
        for vid, conf, src in cands:
            if args.no_verify:
                passed_vid, passed_src, passed_conf = vid, src, conf
                break
            keep, drops = verify_ids([vid], YOUTUBE_API_KEY, args.region)
            if keep:
                passed_vid, passed_src, passed_conf = vid, src, conf
                break
            else:
                reason = list(drops.values())[0] if drops else "verification_failed"
                dropped_reasons[vid] = reason
                if src == "cache" and args.purge_cache:
                    cache_delete(conn, key)

        if passed_vid:
            cache_put_verified(conn, key, a, t, y, args.region, passed_vid, passed_src, passed_conf)
            src_counts[passed_src] = src_counts.get(passed_src,0)+1
            kept_ids_in_order.append(passed_vid)
            results.append({**r, "videoId": passed_vid, "source": passed_src, "confidence": passed_conf})
            print(f"[{idx:03}/{len(rows)}] ✅ {a} — {t} → {passed_vid}  ({passed_src}, conf={passed_conf:.2f})", flush=True)
        else:
            if args.fill_with_search and not args.no_search and YOUTUBE_API_KEY:
                q=f"{a} - {t} {(y or '')}".strip()
                try:
                    items=yt_search(YOUTUBE_API_KEY, q, region=args.region, max_results=12)
                    pick=pick_best_candidate(a, t, items)
                    if pick:
                        vid2, conf2, _ = pick
                        keep2, drops2 = verify_ids([vid2], YOUTUBE_API_KEY, args.region)
                        if keep2:
                            cache_put_verified(conn, key, a, t, y, args.region, keep2[0], "search_fill", conf2)
                            src_counts["search_fill"] += 1
                            kept_ids_in_order.append(keep2[0])
                            results.append({**r, "videoId": keep2[0], "source":"search_fill", "confidence": conf2})
                            print(f"[{idx:03}/{len(rows)}] ✅ {a} — {t} → {keep2[0]}  (search_fill, conf={conf2:.2f})", flush=True)
                            continue
                        else:
                            if drops2: dropped_reasons.update(drops2)
                except requests.RequestException:
                    pass
            print(f"[{idx:03}/{len(rows)}] ❌ {a} — {t} (reason=no_verified_candidate)", flush=True)
            misses.append({**r, "reason":"no_verified_candidate"})

        time.sleep(0.1)

    # Step 3: De-dupe preserving order (for embed queue)
    seen=set(); final_ids=[]
    for vid in kept_ids_in_order:
        if vid not in seen:
            seen.add(vid); final_ids.append(vid)

    # Write outputs
    (outdir/Path("video_ids.txt")).write_text("\n".join(final_ids), encoding="utf-8")

    with open(outdir/Path("matches.csv"),"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["artist","title","year","videoId","source","confidence"])
        for r in results:
            w.writerow([r["artist"], r["title"], r.get("year",""), r["videoId"], r["source"], f"{r['confidence']:.2f}"])

    if misses:
        with open(outdir/Path("misses.csv"),"w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["artist","title","year","reason"])
            for m in misses: w.writerow([m["artist"], m["title"], m.get("year",""), m["reason"]])

    with open(outdir/Path("dropped.csv"),"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["videoId","reason"])
        for vid,reason in dropped_reasons.items(): w.writerow([vid, reason])

    title=f"TimeDeck Mini Player — {len(final_ids)} videos"
    write_embed_html(final_ids, title, outdir/Path("embed.html"))

    print(f"\n[done] Resolved {len(final_ids)} / {len(rows)} playable IDs", flush=True)
    print(f"[out] {outdir/'video_ids.txt'}", flush=True)
    print(f"[out] {outdir/'matches.csv'}   (cache: {args.cache})", flush=True)
    if misses: print(f"[out] {outdir/'misses.csv'}", flush=True)
    print(f"[report] Dropped {len(dropped_reasons)} videos → {outdir/'dropped.csv'}", flush=True)
    print(f"[open] {outdir/'embed.html'}\n", flush=True)
    print(f"[sources] cache:{src_counts.get('cache',0)} discogs:{src_counts.get('discogs',0)} musicbrainz:{src_counts.get('musicbrainz',0)} search:{src_counts.get('search',0)} fill:{src_counts.get('search_fill',0)}\n", flush=True)

if __name__ == "__main__":
    main()
