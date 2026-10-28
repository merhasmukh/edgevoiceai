"""
SQLite Persistent Dialogue and Profile Store.
Lightweight, thread-safe, file-based persistence for edge devices and servers.
Uses Python standard library sqlite3 via async thread offloading.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from typing import Any
from edgevoiceai.memory.message import ChatMessage, ConversationSession
from edgevoiceai.memory.stores.base import BaseMemoryStore


class SQLiteMemoryStore(BaseMemoryStore):
    """
    Asynchronous SQLite memory store for session history and user profiles.
    """

    def __init__(self, db_path: str = "agent_memory.db") -> None:
        self.db_path = db_path
        self._init_db_sync()

    def _init_db_sync(self) -> None:
        """Initializes database tables if they do not exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # 1. Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS voice_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at REAL,
                    updated_at REAL
                )
            """)
            # 2. Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS voice_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp REAL,
                    metadata_json TEXT,
                    FOREIGN KEY (session_id) REFERENCES voice_sessions (session_id)
                )
            """)
            # 3. Profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS voice_profiles (
                    session_id TEXT PRIMARY KEY,
                    profile_json TEXT,
                    updated_at REAL,
                    FOREIGN KEY (session_id) REFERENCES voice_sessions (session_id)
                )
            """)
            conn.commit()

    async def get_or_create_session(self, session_id: str) -> ConversationSession:
        def _get() -> ConversationSession:
            now = time.time()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT created_at, updated_at FROM voice_sessions WHERE session_id = ?", (session_id,))
                row = cursor.fetchone()
                if row:
                    created_at, updated_at = row[0], row[1]
                else:
                    cursor.execute(
                        "INSERT INTO voice_sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
                        (session_id, now, now)
                    )
                    conn.commit()
                    created_at, updated_at = now, now

                # Fetch messages
                cursor.execute(
                    "SELECT role, content, timestamp, metadata_json FROM voice_messages WHERE session_id = ? ORDER BY id ASC",
                    (session_id,)
                )
                msgs = [
                    ChatMessage(
                        role=r[0],
                        content=r[1],
                        timestamp=r[2],
                        metadata=json.loads(r[3]) if r[3] else {}
                    )
                    for r in cursor.fetchall()
                ]

                # Fetch profile
                cursor.execute("SELECT profile_json FROM voice_profiles WHERE session_id = ?", (session_id,))
                p_row = cursor.fetchone()
                profile = json.loads(p_row[0]) if p_row and p_row[0] else {}

                return ConversationSession(
                    session_id=session_id,
                    messages=msgs,
                    profile=profile,
                    created_at=created_at,
                    updated_at=updated_at,
                )

        return await asyncio.to_thread(_get)

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        def _add() -> ChatMessage:
            now = time.time()
            meta_json = json.dumps(metadata or {})
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Ensure session exists
                cursor.execute(
                    "INSERT OR IGNORE INTO voice_sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
                    (session_id, now, now)
                )
                cursor.execute(
                    "INSERT INTO voice_messages (session_id, role, content, timestamp, metadata_json) VALUES (?, ?, ?, ?, ?)",
                    (session_id, role, content, now, meta_json)
                )
                cursor.execute("UPDATE voice_sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
                conn.commit()
            return ChatMessage(role=role, content=content, timestamp=now, metadata=metadata or {})

        return await asyncio.to_thread(_add)

    async def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[ChatMessage]:
        def _get() -> list[ChatMessage]:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                query = "SELECT role, content, timestamp, metadata_json FROM voice_messages WHERE session_id = ? ORDER BY id ASC"
                cursor.execute(query, (session_id,))
                rows = cursor.fetchall()
                if limit is not None and limit > 0:
                    rows = rows[-limit:]
                return [
                    ChatMessage(
                        role=r[0],
                        content=r[1],
                        timestamp=r[2],
                        metadata=json.loads(r[3]) if r[3] else {}
                    )
                    for r in rows
                ]

        return await asyncio.to_thread(_get)

    async def update_profile(
        self,
        session_id: str,
        profile_data: dict[str, Any],
    ) -> dict[str, Any]:
        def _update() -> dict[str, Any]:
            now = time.time()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Ensure session exists
                cursor.execute(
                    "INSERT OR IGNORE INTO voice_sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
                    (session_id, now, now)
                )
                cursor.execute("SELECT profile_json FROM voice_profiles WHERE session_id = ?", (session_id,))
                row = cursor.fetchone()
                existing = json.loads(row[0]) if row and row[0] else {}

                # Merge non-null data
                for k, v in profile_data.items():
                    if v is not None and v != "":
                        existing[k] = v

                cursor.execute(
                    "INSERT OR REPLACE INTO voice_profiles (session_id, profile_json, updated_at) VALUES (?, ?, ?)",
                    (session_id, json.dumps(existing), now)
                )
                cursor.execute("UPDATE voice_sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
                conn.commit()
                return existing

        return await asyncio.to_thread(_update)

    async def get_profile(self, session_id: str) -> dict[str, Any]:
        def _get() -> dict[str, Any]:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT profile_json FROM voice_profiles WHERE session_id = ?", (session_id,))
                row = cursor.fetchone()
                return json.loads(row[0]) if row and row[0] else {}

        return await asyncio.to_thread(_get)

    async def clear_session(self, session_id: str) -> None:
        def _clear() -> None:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM voice_messages WHERE session_id = ?", (session_id,))
                cursor.execute("DELETE FROM voice_profiles WHERE session_id = ?", (session_id,))
                cursor.execute("DELETE FROM voice_sessions WHERE session_id = ?", (session_id,))
                conn.commit()

        await asyncio.to_thread(_clear)
