"""
Autonomous Entity & Profile Extractor.
Extracts structured user profile details (name, preferences, location, etc.)
from multi-turn conversation dialogues asynchronously in the background.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from typing import Any

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore[assignment]
    HTTPX_AVAILABLE = False

from edgevoiceai.memory.message import ChatMessage
from edgevoiceai.memory.stores.base import BaseMemoryStore



class EntityExtractor:
    """
    Configurable background entity extractor.
    Extracts structured key-value attributes from conversation turns
    according to a developer-specified schema.
    """

    DEFAULT_SCHEMA = {
        "name": "User's personal or preferred name if they explicitly introduced themselves",
        "city_or_location": "User's city, town, or location if mentioned",
        "preferences": "User's explicit preferences, requests, or interests",
    }

    def __init__(
        self,
        api_key: str | None = None,
        schema: dict[str, str] | None = None,
        model: str = "gemini-2.0-flash",
        domain_description: str = "a conversational voice dialogue between a user and an AI assistant",
    ) -> None:
        self.api_key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
        self.schema = schema or self.DEFAULT_SCHEMA
        self.model = model
        self.domain_description = domain_description
        self._client: Any = None

    def _get_client(self) -> Any:
        if not HTTPX_AVAILABLE or httpx is None:
            raise RuntimeError("httpx is required for EntityExtractor. Install with: pip install httpx")
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=20.0)
        return self._client

    def _build_prompt(self, messages: Sequence[ChatMessage | dict[str, Any]]) -> str:
        formatted_dialogue = []
        for m in messages:
            role = m.role if isinstance(m, ChatMessage) else m.get("role", "user")
            content = m.content if isinstance(m, ChatMessage) else m.get("content", "")
            formatted_dialogue.append(f"{role.upper()}: {content}")

        dialogue_str = "\n".join(formatted_dialogue)

        schema_lines = []
        for field_name, desc in self.schema.items():
            schema_lines.append(f'  "{field_name}": {desc} (or null if not stated)')

        schema_str = "{\n" + ",\n".join(schema_lines) + "\n}"

        return f"""\
You are an expert entity extraction engine analyzing {self.domain_description}.
The dialogue may be in any language (English, Gujarati, Hindi, etc.).

Your goal is to extract personal profile details that the USER EXPLICITLY stated about themselves.

STRICT EXTRACTION RULES:
1. Extract details ONLY from what the USER stated. NEVER extract or guess anything from the AI assistant's messages.
2. If the user did not explicitly state a field, set it to null. DO NOT guess, hallucinate, or assume.
3. Return ONLY a valid JSON object matching this schema:
{schema_str}

Conversation:
{dialogue_str}
"""

    async def extract(self, messages: Sequence[ChatMessage | dict[str, Any]]) -> dict[str, Any]:

        """
        Asynchronously inspects conversation messages and returns extracted profile fields.
        """
        if not messages or not self.api_key:
            return {}

        prompt = self._build_prompt(messages)
        client = self._get_client()

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0},
        }

        try:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                return {}

            data = resp.json()
            raw_text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )

            # Strip markdown code blocks if present
            clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
            extracted = json.loads(clean_json)

            # Filter out null values
            return {k: v for k, v in extracted.items() if v is not None and v != ""}
        except Exception as exc:
            print(f"⚠️  [EntityExtractor] Extraction error: {exc}")
            return {}

    async def extract_and_update(
        self,
        session_id: str,
        store: BaseMemoryStore,
        limit_messages: int = 15,
    ) -> dict[str, Any]:
        """
        Convenience background worker: retrieves recent session messages,
        extracts profile entities, and automatically persists them into the memory store.
        """
        messages = await store.get_messages(session_id, limit=limit_messages)
        if not messages:
            return {}

        entities = await self.extract(messages)
        if entities:
            print(f"🧠 [EntityExtractor] Discovered user entities for session {session_id}: {entities}")
            await store.update_profile(session_id, entities)

        return entities

    async def close(self) -> None:
        """Closes the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
