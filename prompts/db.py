"""SQLite persistence for games and the small account system."""

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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            profile_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS auth_sessions (
            token_hash TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS password_resets (
            token_hash TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            expires_at TEXT NOT NULL,
            used_at TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
        """
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(games)").fetchall()}
    if "user_id" not in columns:
        conn.execute("ALTER TABLE games ADD COLUMN user_id INTEGER")
    # Anonymous games from before account support cannot be resumed safely or
    # associated with a person. Remove them during the one-time migration.
    conn.execute("DELETE FROM games WHERE user_id IS NULL")
    conn.commit()
    conn.close()


def save_game(session_id: str, state: dict, history: list, user_id: int | None = None):
    conn = get_connection()
    now = datetime.utcnow().isoformat()
    existing = conn.execute(
        "SELECT session_id FROM games WHERE session_id = ?", (session_id,)
    ).fetchone()

    if existing:
        conn.execute(
            "UPDATE games SET state_json = ?, history_json = ?, updated_at = ?, user_id = ? "
            "WHERE session_id = ?",
            (json.dumps(state), json.dumps(history), now, user_id, session_id),
        )
    else:
        conn.execute(
            "INSERT INTO games (session_id, state_json, history_json, created_at, updated_at, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, json.dumps(state), json.dumps(history), now, now, user_id),
        )
    conn.commit()
    conn.close()


def load_game(session_id: str):
    conn = get_connection()
    row = conn.execute(
        "SELECT state_json, history_json, user_id FROM games WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "state": json.loads(row["state_json"]),
        "history": json.loads(row["history_json"]),
        "user_id": row["user_id"],
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


def list_user_games(user_id: int) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT session_id, state_json, updated_at FROM games "
        "WHERE user_id = ? ORDER BY updated_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    games = []
    for row in rows:
        state = json.loads(row["state_json"])
        games.append({
            "session_id": row["session_id"],
            "updated_at": row["updated_at"],
            "turn_count": state.get("turn_count", 0),
            "world_prompt": state.get("world_prompt", ""),
            "game_over": bool(state.get("game_over", False)),
        })
    return games


def create_user(email: str, display_name: str, password_hash: str) -> dict:
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO users (email, display_name, password_hash, profile_json, created_at, updated_at) "
            "VALUES (?, ?, ?, '{}', ?, ?)",
            (email, display_name, password_hash, now, now),
        )
        conn.commit()
        return {"user_id": cursor.lastrowid, "email": email, "display_name": display_name}
    except sqlite3.IntegrityError as exc:
        raise ValueError("An account with that email already exists") from exc
    finally:
        conn.close()


def get_user_by_email(email: str):
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user(user_id: int):
    conn = get_connection()
    row = conn.execute(
        "SELECT user_id, email, display_name, profile_json FROM users WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def save_auth_session(token_hash: str, user_id: int, expires_at: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO auth_sessions (token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
        (token_hash, user_id, expires_at, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_user_by_session(token_hash: str):
    conn = get_connection()
    row = conn.execute(
        "SELECT u.user_id, u.email, u.display_name, u.profile_json "
        "FROM auth_sessions s JOIN users u ON u.user_id = s.user_id "
        "WHERE s.token_hash = ? AND s.expires_at > ?",
        (token_hash, datetime.utcnow().isoformat()),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_auth_session(token_hash: str):
    conn = get_connection()
    conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,))
    conn.commit()
    conn.close()


def create_password_reset(token_hash: str, user_id: int, expires_at: str):
    conn = get_connection()
    conn.execute("DELETE FROM password_resets WHERE user_id = ?", (user_id,))
    conn.execute(
        "INSERT INTO password_resets (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
        (token_hash, user_id, expires_at),
    )
    conn.commit()
    conn.close()


def consume_password_reset(token_hash: str):
    conn = get_connection()
    row = conn.execute(
        "SELECT user_id FROM password_resets WHERE token_hash = ? AND used_at IS NULL AND expires_at > ?",
        (token_hash, datetime.utcnow().isoformat()),
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE password_resets SET used_at = ? WHERE token_hash = ?",
            (datetime.utcnow().isoformat(), token_hash),
        )
        conn.commit()
    conn.close()
    return row["user_id"] if row else None


def update_password(user_id: int, password_hash: str):
    conn = get_connection()
    conn.execute(
        "UPDATE users SET password_hash = ?, updated_at = ? WHERE user_id = ?",
        (password_hash, datetime.utcnow().isoformat(), user_id),
    )
    conn.execute("DELETE FROM auth_sessions WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


def get_story_profile(user_id: int) -> dict:
    user = get_user(user_id)
    if not user:
        return {"sessions_started": 0, "turns_played": 0, "genres": {}, "archetypes": {}, "pacing": {}}
    try:
        profile = json.loads(user.get("profile_json") or "{}")
    except (TypeError, json.JSONDecodeError):
        profile = {}
    return {
        "sessions_started": int(profile.get("sessions_started", 0)),
        "turns_played": int(profile.get("turns_played", 0)),
        "genres": profile.get("genres", {}) if isinstance(profile.get("genres", {}), dict) else {},
        "archetypes": profile.get("archetypes", {}) if isinstance(profile.get("archetypes", {}), dict) else {},
        "pacing": profile.get("pacing", {}) if isinstance(profile.get("pacing", {}), dict) else {},
    }


def record_story_usage(user_id: int, opening_prompt: str = "", character_class: str = "", difficulty: str = "", turn: bool = False):
    profile = get_story_profile(user_id)
    profile["sessions_started"] += 0 if turn else 1
    profile["turns_played"] += 1 if turn else 0

    text = (opening_prompt or "").lower()
    genre_keywords = {
        "mystery": ("mystery", "detective", "clue", "notebook", "letter"),
        "romance": ("romance", "love", "bookshop", "vienna", "relationship"),
        "fantasy": ("kingdom", "map", "forest", "magic", "cartographer"),
        "science fiction": ("space", "signal", "galaxy", "station", "andromeda"),
        "slice of life": ("university", "campus", "café", "cafe", "study", "team"),
    }
    if not turn:
        matched = next((name for name, words in genre_keywords.items() if any(word in text for word in words)), "original")
        profile["genres"][matched] = profile["genres"].get(matched, 0) + 1
        if character_class:
            profile["archetypes"][character_class] = profile["archetypes"].get(character_class, 0) + 1
        if difficulty:
            profile["pacing"][difficulty] = profile["pacing"].get(difficulty, 0) + 1

    conn = get_connection()
    conn.execute(
        "UPDATE users SET profile_json = ?, updated_at = ? WHERE user_id = ?",
        (json.dumps(profile), datetime.utcnow().isoformat(), user_id),
    )
    conn.commit()
    conn.close()
    return profile
