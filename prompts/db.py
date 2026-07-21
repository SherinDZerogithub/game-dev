"""
db.py

Simple SQLite persistence for Lumen game sessions.
Each row is one player's game: state (JSON) and turn history (JSON).
"""

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "game_states.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS games (
            session_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL,
            history_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_game(session_id: str, state: dict, history: list):
    conn = get_connection()
    now = datetime.utcnow().isoformat()
    existing = conn.execute(
        "SELECT session_id FROM games WHERE session_id = ?", (session_id,)
    ).fetchone()

    if existing:
        conn.execute(
            "UPDATE games SET state_json = ?, history_json = ?, updated_at = ? "
            "WHERE session_id = ?",
            (json.dumps(state), json.dumps(history), now, session_id),
        )
    else:
        conn.execute(
            "INSERT INTO games (session_id, state_json, history_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, json.dumps(state), json.dumps(history), now, now),
        )
    conn.commit()
    conn.close()


def load_game(session_id: str):
    conn = get_connection()
    row = conn.execute(
        "SELECT state_json, history_json FROM games WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "state": json.loads(row["state_json"]),
        "history": json.loads(row["history_json"]),
    }


def delete_game(session_id: str):
    conn = get_connection()
    conn.execute("DELETE FROM games WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


def list_sessions():
    conn = get_connection()
    rows = conn.execute(
        "SELECT session_id, updated_at FROM games ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
