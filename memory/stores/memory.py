"""
In-Memory Dialogue and Profile Store.
Zero-dependency in-RAM storage for ephemeral voice sessions and fast testing.
"""

from __future__ import annotations

import asyncio
from typing import Any
from edgevoiceai.memory.message import ChatMessage, ConversationSession
from edgevoiceai.memory.stores.base import BaseMemoryStore


class InMemoryMemoryStore(BaseMemoryStore):
    """
    Thread-safe in-memory store for sessions, messages, and profile attributes.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationSession] = {}
        self._lock = asyncio.Lock()

    async def get_or_create_session(self, session_id: str) -> ConversationSession:
        async with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = ConversationSession(session_id=session_id)
            return self._sessions[session_id]

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        session = await self.get_or_create_session(session_id)
        async with self._lock:
            return session.add_message(role=role, content=content, metadata=metadata)

    async def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[ChatMessage]:
        session = await self.get_or_create_session(session_id)
        async with self._lock:
            if limit is not None and limit > 0:
                return list(session.messages[-limit:])
            return list(session.messages)

    async def update_profile(
        self,
        session_id: str,
        profile_data: dict[str, Any],
    ) -> dict[str, Any]:
        session = await self.get_or_create_session(session_id)
        async with self._lock:
            session.update_profile(profile_data)
            return dict(session.profile)

    async def get_profile(self, session_id: str) -> dict[str, Any]:
        session = await self.get_or_create_session(session_id)
        async with self._lock:
            return dict(session.profile)

    async def clear_session(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)
