"""
Tests for edgevoiceai.tools and farming_tools modular registry.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from unittest.mock import AsyncMock, patch

from edgevoiceai.tools.base import BaseTool, ToolResult, ToolExecutionContext
from edgevoiceai.tools.registry import ToolRegistry
from edgevoiceai.tools.audio_tool import AudioPlaybackTool
from edgevoiceai.tools.notification_tool import default_normalize_phone
from ai_services.farming_tools import create_farming_tool_registry, farming_tool_registry


class DummyEchoTool(BaseTool):
    @property
    def name(self) -> str:
        return "dummy_echo"

    @property
    def description(self) -> str:
        return "Echoes input back"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "OBJECT",
            "properties": {"msg": {"type": "STRING"}},
            "required": ["msg"]
        }

    async def execute(self, args: dict, context: ToolExecutionContext) -> ToolResult:
        return ToolResult(status="success", message=f"Echo: {args.get('msg')}")


@pytest.mark.asyncio
async def test_registry_registration_and_dispatch():
    registry = ToolRegistry()
    tool = DummyEchoTool()
    registry.register(tool)

    assert registry.has_tool("dummy_echo")
    assert registry.get("dummy_echo") is tool

    # Verify Gemini payload generation
    payload = registry.to_gemini_tools_payload()
    assert len(payload) == 1
    assert payload[0]["functionDeclarations"][0]["name"] == "dummy_echo"

    # Verify dispatch
    ctx = ToolExecutionContext(session_id="test_session_1")
    result = await registry.dispatch("dummy_echo", {"msg": "hello world"}, ctx)
    assert result.status == "success"
    assert result.message == "Echo: hello world"

    # Verify unregistered tool handling
    unreg_result = await registry.dispatch("non_existent_tool", {}, ctx)
    assert unreg_result.status == "error"
    assert unreg_result.message is not None and "not registered" in unreg_result.message


def test_farming_registry_initialization():
    registry = create_farming_tool_registry()
    expected_tools = {
        "search_agricultural_knowledge_base",
        "play_farmer_experience_audio",
        "play_amitshah_talk_audio",
        "send_whatsapp_answer",
        "read_soil_sensor",
    }
    registered = {t.name for t in registry.list_tools()}
    assert expected_tools.issubset(registered)

    payload = registry.to_gemini_tools_payload()
    names = {fn["name"] for fn in payload[0]["functionDeclarations"]}
    assert expected_tools == names


@pytest.mark.asyncio
async def test_farming_tools_execution_via_registry():
    ctx = ToolExecutionContext(session_id="test_sess_farming")

    # 1. Farmer experience audio fallback when not found
    result = await farming_tool_registry.dispatch(
        "play_farmer_experience_audio",
        {
            "farmer_name": "કામરીયા રમેશભાઈ રતીલાલ",
            "district": "મોરબી",
            "audio_url": "/api/experiences/audio/morbi_missing.mp3",
        },
        ctx
    )
    assert result.status == "audio_not_found"
    assert result.message is not None
    assert "કામરીયા રમેશભાઈ રતીલાલ" in result.message
    assert "તમે પોતે જ તમારા અવાજમાં" in result.message

    # 2. WhatsApp tool 2-turn confirmation gate
    # Turn 0: First time seeing phone -> confirmation_required
    wa_result_1 = await farming_tool_registry.dispatch(
        "send_whatsapp_answer",
        {"phone_number": "9825012345", "answer_text": "ટેસ્ટ જવાબ"},
        ctx
    )
    assert wa_result_1.status == "confirmation_required"
    assert wa_result_1.message is not None
    assert "9825012345" in wa_result_1.message


@pytest.mark.asyncio
async def test_tool_prompt_instruction_composition():
    """Test that registered tools dynamically contribute prompt rules to the system instruction."""
    from edgevoiceai.agent import VoiceAgent
    from edgevoiceai.config import VoiceConfig
    from edgevoiceai.memory import InMemoryMemoryStore
    from edgevoiceai.tools.rag_tool import KnowledgeBaseTool

    registry = ToolRegistry()

    # Tool 1: KB Tool with default instruction
    kb_tool = KnowledgeBaseTool(
        name="query_kb",
        description="Search knowledge",
        parameters_schema={"type": "OBJECT", "properties": {}},
        retriever_func=AsyncMock(return_value=[]),
    )
    registry.register(kb_tool)

    # Tool 2: Custom tool with customized prompt rule
    class CustomBookingTool(BaseTool):
        @property
        def name(self) -> str:
            return "book_appointment"

        @property
        def description(self) -> str:
            return "Book appointment"

        @property
        def parameters_schema(self) -> dict:
            return {"type": "OBJECT", "properties": {}}

        @property
        def prompt_instruction(self) -> str:
            return "• APPOINTMENT RULE: Only call book_appointment after collecting user date and time."

        async def execute(self, args: dict, context: ToolExecutionContext) -> ToolResult:
            return ToolResult()

    registry.register(CustomBookingTool())

    tools_instr = registry.build_tools_instruction()
    assert "[ACTIVE TOOLS & USAGE RULES]" in tools_instr
    assert "KNOWLEDGE RETRIEVAL RULE (`query_kb`)" in tools_instr
    assert "APPOINTMENT RULE" in tools_instr

    # Test effective system instruction composed by VoiceAgent
    memory = InMemoryMemoryStore()
    await memory.update_profile("sess_1", {"name": "Alex", "membership": "VIP"})

    config = VoiceConfig(
        system_instruction="You are a personal concierge."
    )
    agent = VoiceAgent(config=config, tools=registry, memory=memory)

    full_instruction = await agent.build_effective_system_instruction("sess_1")
    assert "You are a personal concierge." in full_instruction
    assert "[KNOWN USER CONTEXT]" in full_instruction
    assert "Name: Alex" in full_instruction
    assert "[ACTIVE TOOLS & USAGE RULES]" in full_instruction
    assert "APPOINTMENT RULE" in full_instruction

