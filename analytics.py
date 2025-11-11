# analytics.py
# Analytics event collection and retrieval

import os
import json
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

log = logging.getLogger("timedeck-api")

# Database path
CACHE_DIR = os.getenv("CACHE_DIR", "/data")
os.makedirs(CACHE_DIR, exist_ok=True)
ANALYTICS_DB = os.path.join(CACHE_DIR, "analytics.db")


# -------------------- Pydantic Models --------------------
class EventIn(BaseModel):
    v: int = Field(1)
    event: str
    ts: datetime
    anon_user_id: str
    session_id: str
    props: Dict[str, Any] = {}


# -------------------- Database Functions --------------------
def _conn():
    """Get a connection to the analytics database."""
    conn = sqlite3.connect(ANALYTICS_DB)
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    """Initialize the analytics database schema."""
    conn = _conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                event TEXT NOT NULL,
                anon_user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                props_json TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_event ON events(event)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_user ON events(anon_user_id)")
        conn.commit()
        log.info("Analytics database initialized.")
    except sqlite3.Error as e:
        log.error(f"Error initializing analytics database: {e}")
        raise
    finally:
        conn.close()


def save_event(e: EventIn) -> bool:
    """Save an event to the database."""
    conn = _conn()
    try:
        # Normalize timestamp: convert to UTC and store as naive datetime string
        ts = e.ts
        if ts.tzinfo is not None:
            # Convert timezone-aware to UTC and remove tzinfo
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        ts_str = ts.isoformat()

        conn.execute(
            "INSERT INTO events (ts, event, anon_user_id, session_id, props_json) VALUES (?, ?, ?, ?, ?)",
            (
                ts_str,
                e.event,
                e.anon_user_id,
                e.session_id,
                json.dumps(e.props, ensure_ascii=False),
            ),
        )
        conn.commit()
        return True
    except sqlite3.Error as e:
        log.error(f"Error saving event: {e}")
        return False
    finally:
        conn.close()


# -------------------- Analytics Queries --------------------
def get_today_stats() -> Dict[str, Any]:
    """Get today's stats: search count, export count, active users."""
    conn = _conn()
    try:
        # Get today's date
        today = datetime.utcnow().strftime("%Y-%m-%d")

        # Count searches
        searches = conn.execute(
            "SELECT COUNT(*) FROM events WHERE event='search_results' AND DATE(ts) = ?",
            (today,),
        ).fetchone()[0]

        # Count exports
        exports = conn.execute(
            "SELECT COUNT(*) FROM events WHERE event='export_success' AND DATE(ts) = ?",
            (today,),
        ).fetchone()[0]

        # Count unique active users today
        active_users = conn.execute(
            "SELECT COUNT(DISTINCT anon_user_id) FROM events WHERE DATE(ts) = ?",
            (today,),
        ).fetchone()[0]

        return {"searches": searches, "exports": exports, "active_users": active_users}
    except sqlite3.Error as e:
        log.error(f"Error getting today's stats: {e}")
        return {"searches": 0, "exports": 0, "active_users": 0}
    finally:
        conn.close()


def get_top_searches(limit: int = 10) -> List[Dict[str, Any]]:
    """Get the top search queries."""
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT json_extract(props_json, '$.query') AS query, COUNT(*) AS count FROM events WHERE event='search_results' GROUP BY query ORDER BY count DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"query": r[0], "count": r[1]} for r in rows if r[0]]
    except sqlite3.Error as e:
        log.error(f"Error getting top searches: {e}")
        return []
    finally:
        conn.close()


def get_genre_breakdown(limit: int = 20) -> List[Dict[str, Any]]:
    """Get genre breakdown from search results."""
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT json_extract(props_json, '$.genre') AS genre, COUNT(*) AS count FROM events WHERE event='search_results' GROUP BY genre ORDER BY count DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"genre": r[0] or "unknown", "count": r[1]} for r in rows]
    except sqlite3.Error as e:
        log.error(f"Error getting genre breakdown: {e}")
        return []
    finally:
        conn.close()


def get_year_breakdown(limit: int = 20) -> List[Dict[str, Any]]:
    """Get year/range breakdown from search results."""
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT json_extract(props_json, '$.year_or_range') AS year, COUNT(*) AS count FROM events WHERE event='search_results' GROUP BY year ORDER BY count DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"year": r[0] or "unknown", "count": r[1]} for r in rows]
    except sqlite3.Error as e:
        log.error(f"Error getting year breakdown: {e}")
        return []
    finally:
        conn.close()


def get_export_split() -> Dict[str, int]:
    """Get split of exports by provider."""
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT json_extract(props_json, '$.provider') AS provider, COUNT(*) AS count FROM events WHERE event='export_success' GROUP BY provider"
        ).fetchall()
        return {r[0] or "unknown": r[1] for r in rows}
    except sqlite3.Error as e:
        log.error(f"Error getting export split: {e}")
        return {}
    finally:
        conn.close()


def get_zero_result_rate() -> Dict[str, Any]:
    """Get the zero-result rate percentage."""
    conn = _conn()
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM events WHERE event='search_results'"
        ).fetchone()[0]

        if total == 0:
            return {"pct": 0.0, "zero_count": 0, "total": 0}

        zero_count = conn.execute(
            "SELECT COUNT(*) FROM events WHERE event='search_results' AND json_extract(props_json, '$.result_count') = 0"
        ).fetchone()[0]

        pct = round(100.0 * zero_count / total, 2)
        return {"pct": pct, "zero_count": zero_count, "total": total}
    except sqlite3.Error as e:
        log.error(f"Error getting zero-result rate: {e}")
        return {"pct": 0.0, "zero_count": 0, "total": 0}
    finally:
        conn.close()


def get_recent_activity(limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent activity for display."""
    conn = _conn()
    try:
        rows = conn.execute(
            """SELECT ts, event, props_json FROM events
               ORDER BY ts DESC LIMIT ?""",
            (limit,),
        ).fetchall()

        result = []
        for ts, event, props_json in rows:
            props = json.loads(props_json) if props_json else {}
            result.append({"ts": ts, "event": event, "props": props})
        return result
    except sqlite3.Error as e:
        log.error(f"Error getting recent activity: {e}")
        return []
    finally:
        conn.close()


def cleanup_old_events(days: int = 14) -> int:
    """Delete events older than N days. Returns count of deleted rows."""
    conn = _conn()
    try:
        cursor = conn.execute(
            "DELETE FROM events WHERE ts < datetime('now', '-' || ? || ' day')",
            (days,),
        )
        conn.commit()
        deleted = cursor.rowcount
        log.info(f"Cleaned up {deleted} events older than {days} days.")
        return deleted
    except sqlite3.Error as e:
        log.error(f"Error cleaning up old events: {e}")
        return 0
    finally:
        conn.close()
