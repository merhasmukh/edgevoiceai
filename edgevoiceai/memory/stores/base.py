"""
Base Interface for Memory and Dialogue Stores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from edgevoiceai.memory.message import ChatMessage, ConversationSession


class BaseMemoryStore(ABC):
    """
    Abstract Base Class for managing dialogue history and persistent user profiles.
    """

    @abstractmethod
    async def get_or_create_session(self, session_id: str) -> ConversationSession:
        """Retrieves an existing session or initializes a new one."""
        pass

    @abstractmethod
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """Appends a new turn message to the session dialogue history."""
        pass

    @abstractmethod
    async def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[ChatMessage]:
        """Returns ordered messages for the given session."""
        pass

    @abstractmethod
    async def update_profile(
        self,
        session_id: str,
        profile_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Merges new profile details into the user's saved state."""
        pass

    @abstractmethod
    async def get_profile(self, session_id: str) -> dict[str, Any]:
        """Returns the current known user profile details."""
        pass

    @abstractmethod
    async def clear_session(self, session_id: str) -> None:
        """Deletes or resets a conversation session."""
        pass

    async def build_context_prompt(
        self,
        session_id: str,
        header: str = "[KNOWN USER CONTEXT]",
    ) -> str:
        """
        Constructs a clean, human-readable context block from known profile details
        for dynamic injection into voice system prompts.
        """
        profile = await self.get_profile(session_id)
        if not profile:
            return ""

        lines = [header]
        for key, val in profile.items():
            if val is not None and val != "":
                # Format key from snake_case to Title Case
                formatted_key = key.replace("_", " ").title()
                if isinstance(val, list):
                    val_str = ", ".join(str(item) for item in val)
                else:
                    val_str = str(val)
                lines.append(f"• {formatted_key}: {val_str}")

        if len(lines) == 1:
            return ""

        lines.append("(Use these known details naturally; do not repeatedly re-ask the user for them)")
        return "\n".join(lines)
