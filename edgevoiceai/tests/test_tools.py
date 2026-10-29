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
from edgevoiceai.tools.notification_tool import TwoTurnConfirmationNotificationTool, default_normalize_phone
from edgevoiceai.tools.sensor_tool import SensorTelemetryTool


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


def test_audio_and_notification_tools_initialization():
    registry = ToolRegistry()
    audio_tool = AudioPlaybackTool(
        name="play_audio",
        description="Plays audio",
        parameters_schema={"type": "OBJECT", "properties": {}},
    )
    notification_tool = TwoTurnConfirmationNotificationTool(
        name="send_notification",
        description="Sends notification with 2-turn confirmation",
        parameters_schema={"type": "OBJECT", "properties": {}},
        sender_func=AsyncMock(return_value={"status": "sent"}),
    )
    sensor_tool = SensorTelemetryTool(
        name="read_sensor",
        description="Reads sensor telemetry",
        parameters_schema={"type": "OBJECT", "properties": {}},
        reader_func=lambda: {"temperature": 25.0, "status": "ok"},
    )
    registry.register(audio_tool)
    registry.register(notification_tool)
    registry.register(sensor_tool)

    assert registry.has_tool("play_audio")
    assert registry.has_tool("send_notification")
    assert registry.has_tool("read_sensor")

    payload = registry.to_gemini_tools_payload()
    names = {fn["name"] for fn in payload[0]["functionDeclarations"]}
    assert names == {"play_audio", "send_notification", "read_sensor"}


@pytest.mark.asyncio
async def test_notification_and_sensor_tool_execution():
    ctx = ToolExecutionContext(session_id="test_sess_framework")

    # 1. Notification tool 2-turn confirmation gate
    sender_mock = AsyncMock(return_value={"status": "ok"})
    wa_tool = TwoTurnConfirmationNotificationTool(
        name="send_sms",
        description="Sends SMS",
        parameters_schema={"type": "OBJECT", "properties": {}},
        sender_func=sender_mock,
    )
    # Turn 0: First time seeing phone -> confirmation_required
    wa_result_1 = await wa_tool.execute(
        {"phone_number": "9825012345", "answer_text": "Test message"},
        ctx
    )
    assert wa_result_1.status == "confirmation_required"
    assert "9825012345" in (wa_result_1.message or "")

    # 2. Sensor tool telemetry execution
    sensor_tool = SensorTelemetryTool(
        name="read_temp",
        description="Reads temperature",
        parameters_schema={"type": "OBJECT", "properties": {}},
        reader_func=lambda: {"temp": 28.5, "status": "connected"},
    )
    sensor_res = await sensor_tool.execute({}, ctx)
    assert sensor_res.status == "connected"
    assert sensor_res.data["temp"] == 28.5
    assert sensor_res.client_event is not None
    assert sensor_res.client_event["data"]["temp"] == 28.5


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

