"""
Persistent storage for all queries and responses using SQLite.

Every question asked through the web UI or CLI is automatically logged here.
The database file is `history.db` in the project root.
"""

import json
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "history.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # safe for concurrent reads
    return conn


def init_db() -> None:
    """Create the history table if it doesn't exist."""
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            question TEXT NOT NULL,
            mode TEXT NOT NULL,
            council_answers TEXT,
            giant_answer TEXT,
            judgment TEXT,
            elapsed_seconds REAL,
            raw_response TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_query(question: str, mode: str, response: dict) -> int:
    """
    Save a complete query + response to the database.
    Returns the row ID.
    """
    council_answers = None
    giant_answer = None
    judgment = None

    if mode == "all":
        council_answers = json.dumps(response.get("council_individual", {}))
        giant_answer = response.get("giant_answer", "")
        judgment = response.get("giant_judgment", "")
    elif mode == "council":
        council_answers = json.dumps(response.get("individual", {}))
        judgment = response.get("giant_judgment", "")
    elif mode == "giant":
        giant_answer = response.get("giant_answer", "")
        judgment = json.dumps(response.get("council_judgments", {}))
    elif mode == "single":
        giant_answer = response.get("answer", "")

    conn = _connect()
    cursor = conn.execute(
        """INSERT INTO history
           (timestamp, question, mode, council_answers, giant_answer, judgment, elapsed_seconds, raw_response)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(),
            question,
            mode,
            council_answers,
            giant_answer,
            judgment,
            response.get("elapsed"),
            json.dumps(response),
        ),
    )
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_history(limit: int = 50, offset: int = 0) -> list[dict]:
    """Fetch recent history, newest first."""
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM history ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_entry(entry_id: int) -> dict | None:
    """Fetch a single history entry by ID."""
    conn = _connect()
    row = conn.execute("SELECT * FROM history WHERE id = ?", (entry_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_stats() -> dict:
    """Return aggregate stats about stored history."""
    conn = _connect()
    total = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
    by_mode = conn.execute(
        "SELECT mode, COUNT(*) as cnt FROM history GROUP BY mode ORDER BY cnt DESC"
    ).fetchall()
    conn.close()
    return {
        "total_queries": total,
        "by_mode": {r["mode"]: r["cnt"] for r in by_mode},
    }


# Auto-create table on import
init_db()
