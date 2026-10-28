"""
Unit tests for the Voice AI Framework Memory subsystem.
Tests ChatMessage, ConversationSession, InMemoryMemoryStore, SQLiteMemoryStore,
and EntityExtractor.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from edgevoiceai.agent import VoiceAgent
from edgevoiceai.config import VoiceConfig
from edgevoiceai.memory import (
    ChatMessage,
    ConversationSession,
    EntityExtractor,
    InMemoryMemoryStore,
    SQLiteMemoryStore,
)


def test_chat_message_and_session_models():
    """Test ChatMessage serialization and ConversationSession state updates."""
    msg = ChatMessage(role="user", content="Hello my name is Alex")
    assert msg.role == "user"
    assert msg.content == "Hello my name is Alex"
    assert isinstance(msg.timestamp, float)

    d = msg.to_dict()
    assert d["role"] == "user"
    assert d["content"] == "Hello my name is Alex"

    restored = ChatMessage.from_dict(d)
    assert restored.role == "user"
    assert restored.content == "Hello my name is Alex"
    assert "Alex" in restored.content

    session = ConversationSession(session_id="test_sess_1")
    session.add_message("user", "My name is Alex")
    session.add_message("assistant", "Nice to meet you, Alex!")
    assert len(session.messages) == 2

    session.update_profile({"name": "Alex", "city": "Seattle", "empty": None})
    assert session.profile["name"] == "Alex"
    assert session.profile["city"] == "Seattle"
    assert "empty" not in session.profile


@pytest.mark.asyncio
async def test_in_memory_memory_store():
    """Test InMemoryMemoryStore operations."""
    store = InMemoryMemoryStore()
    session_id = "sess_mem_123"

    # 1. Add messages
    await store.add_message(session_id, "user", "I want to buy 5 tickets")
    await store.add_message(session_id, "assistant", "Sure, for which date?")
    await store.add_message(session_id, "user", "For tomorrow evening")

    msgs = await store.get_messages(session_id)
    assert len(msgs) == 3
    assert msgs[0].content == "I want to buy 5 tickets"
    assert msgs[2].content == "For tomorrow evening"

    # 2. Limit test
    recent = await store.get_messages(session_id, limit=2)
    assert len(recent) == 2
    assert recent[0].content == "Sure, for which date?"
    assert recent[1].content == "For tomorrow evening"

    # 3. Profile update & retrieval
    await store.update_profile(session_id, {"preferred_time": "evening", "tickets": 5})
    profile = await store.get_profile(session_id)
    assert profile["preferred_time"] == "evening"
    assert profile["tickets"] == 5

    # 4. Context prompt building
    prompt = await store.build_context_prompt(session_id)
    assert "[KNOWN USER CONTEXT]" in prompt
    assert "Preferred Time: evening" in prompt
    assert "Tickets: 5" in prompt

    # 5. Clear session
    await store.clear_session(session_id)
    cleared_msgs = await store.get_messages(session_id)
    assert len(cleared_msgs) == 0


@pytest.mark.asyncio
async def test_sqlite_memory_store():
    """Test SQLiteMemoryStore persistence with file database."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_agent_memory.db")
        store = SQLiteMemoryStore(db_path=db_path)
        session_id = "sess_sql_456"

        # 1. Add messages
        await store.add_message(session_id, "user", "I am looking for organic fertilizer")
        await store.add_message(session_id, "assistant", "I recommend vermicompost")

        msgs = await store.get_messages(session_id)
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[0].content == "I am looking for organic fertilizer"
        assert msgs[1].role == "assistant"

        # 2. Update profile
        await store.update_profile(session_id, {"farmer_name": "Ramesh", "crop": "Cotton"})
        profile = await store.get_profile(session_id)
        assert profile["farmer_name"] == "Ramesh"
        assert profile["crop"] == "Cotton"

        # 3. Build context prompt
        context = await store.build_context_prompt(session_id)
        assert "[KNOWN USER CONTEXT]" in context
        assert "Farmer Name: Ramesh" in context
        assert "Crop: Cotton" in context

        # 4. Clear session
        await store.clear_session(session_id)
        assert len(await store.get_messages(session_id)) == 0
        assert await store.get_profile(session_id) == {}


@pytest.mark.asyncio
async def test_entity_extractor():
    """Test EntityExtractor prompt generation, extraction mock, and update."""
    extractor = EntityExtractor(
        api_key="test_fake_api_key",
        schema={"user_name": "Name of the user", "city": "City of user"},
        domain_description="a customer support dialogue",
    )

    messages = [
        ChatMessage(role="user", content="Hi, my name is Priya and I live in Pune"),
        ChatMessage(role="assistant", content="Hello Priya from Pune! How can I help you today?"),
    ]

    prompt = extractor._build_prompt(messages)
    assert "USER: Hi, my name is Priya and I live in Pune" in prompt
    assert '"user_name": Name of the user' in prompt
    assert '"city": City of user' in prompt

    # Mock HTTP response
    fake_gemini_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": '```json\n{"user_name": "Priya", "city": "Pune", "unknown": null}\n```'}
                    ]
                }
            }
        ]
    }

    mock_client = AsyncMock()
    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 200
    mock_post_resp.json.return_value = fake_gemini_response
    mock_client.post.return_value = mock_post_resp

    with patch.object(extractor, "_get_client", return_value=mock_client):
        extracted = await extractor.extract(messages)
        assert extracted == {"user_name": "Priya", "city": "Pune"}

        # Test extract_and_update with memory store
        store = InMemoryMemoryStore()
        for m in messages:
            await store.add_message("test_sess", m.role, m.content)

        updated_profile = await extractor.extract_and_update("test_sess", store)
        assert updated_profile == {"user_name": "Priya", "city": "Pune"}

        saved_profile = await store.get_profile("test_sess")
        assert saved_profile == {"user_name": "Priya", "city": "Pune"}

    await extractor.close()


@pytest.mark.asyncio
async def test_voice_agent_memory_helpers():
    """Test VoiceAgent integration helpers for recording messages and reading context."""
    config = VoiceConfig(api_key="fake-test-key")
    memory = InMemoryMemoryStore()
    agent = VoiceAgent(config=config, memory=memory)

    session_id = "agent_session_1"
    await agent.record_user_message(session_id, "My name is John")
    await memory.update_profile(session_id, {"name": "John"})

    context = await agent.get_user_context(session_id)
    assert "[KNOWN USER CONTEXT]" in context
    assert "Name: John" in context
