"""
Unit tests for the WhatsApp notification subsystem in edgevoiceai.
Tests WhatsAppConfig, WhatsAppClient, WhatsAppNotificationTool, and VoiceAgent auto-registration.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from edgevoiceai.agent import VoiceAgent
from edgevoiceai.config import VoiceConfig
from edgevoiceai.notifications import (
    WhatsAppClient,
    WhatsAppConfig,
    WhatsAppNotificationTool,
)
from edgevoiceai.tools.base import ToolExecutionContext


def test_whatsapp_config_defaults():
    """Test WhatsAppConfig default initialization and properties."""
    cfg = WhatsAppConfig(
        api_key="test-key-123",
        service_url="https://api.example.com/whatsapp",
        default_template="welcome_msg",
        phone_prefix="91",
    )
    assert cfg.api_key == "test-key-123"
    assert cfg.service_url == "https://api.example.com/whatsapp"
    assert cfg.default_template == "welcome_msg"
    assert cfg.phone_prefix == "91"
    assert cfg.require_two_turn_confirmation is True


def test_whatsapp_phone_normalization():
    """Test WhatsApp phone normalization across standard, prefixed, and invalid formats."""
    client = WhatsAppClient(WhatsAppConfig(phone_prefix="91"))

    # Valid formats
    assert client.normalize_phone("9825012345") == "919825012345"
    assert client.normalize_phone("+91 98250-12345") == "919825012345"
    assert client.normalize_phone("09825012345") == "919825012345"
    assert client.normalize_phone("919825012345") == "919825012345"

    # Invalid / dummy formats
    assert client.normalize_phone("1234567890") == ""
    assert client.normalize_phone("0000000000") == ""
    assert client.normalize_phone("1111111111") == ""
    assert client.normalize_phone("98250") == ""
    assert client.normalize_phone("") == ""


def test_whatsapp_payload_building():
    """Test building payloads for generic and Meta Cloud API providers."""
    # 1. Generic provider
    gen_client = WhatsAppClient(
        WhatsAppConfig(
            api_key="key1",
            service_url="https://example.com/api",
            provider="generic",
            default_template="generic_tpl",
            default_language="en",
        )
    )

    payload_gen = gen_client._build_payload(
        to="9825012345",
        body_params=["ParamA", "ParamB"],
    )
    assert payload_gen["api_key"] == "key1"
    assert payload_gen["to"] == ["919825012345"]
    assert payload_gen["template_name"] == "generic_tpl"
    assert payload_gen["body_params"] == ["ParamA", "ParamB"]

    # 2. Meta Cloud API provider
    meta_client = WhatsAppClient(
        WhatsAppConfig(
            api_key="token_abc",
            service_url="https://graph.facebook.com/v18.0/12345/messages",
            provider="meta",
            default_template="meta_welcome",
        )
    )
    payload_meta = meta_client._build_payload(
        to="9825012345",
        body_params=["Welcome User"],
    )
    assert payload_meta["messaging_product"] == "whatsapp"
    assert payload_meta["to"] == "919825012345"
    assert payload_meta["template"]["name"] == "meta_welcome"


@pytest.mark.asyncio
async def test_whatsapp_client_send_async():
    """Test async message sending with mocked httpx client."""
    client = WhatsAppClient(
        WhatsAppConfig(
            api_key="test_api_key",
            service_url="https://api.example.com/send",
            default_template="test_tpl",
        )
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "success", "message_id": "msg_999"}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        res = await client.send_template_async(
            to="9825012345",
            body_params=["Hello Alex"],
        )
        assert res["status"] == "success"
        assert res["message_id"] == "msg_999"
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_whatsapp_notification_tool_two_turn_gate():
    """Test that WhatsAppNotificationTool enforces the strict 2-turn confirmation gate."""
    cfg = WhatsAppConfig(
        api_key="test_api_key",
        service_url="https://api.example.com/send",
        default_template="notice_tpl",
    )
    tool = WhatsAppNotificationTool(config=cfg)
    session_id = "test_wa_session_42"

    # Turn 0: First call -> Must hold and ask for confirmation
    ctx_turn_0 = ToolExecutionContext(session_id=session_id, current_turn=0)
    with patch.object(tool.whatsapp_client, "send_template_async", new_callable=AsyncMock) as mock_send:
        res_0 = await tool.execute(
            {"phone_number": "9825012345", "message_text": "Your summary details"},
            ctx_turn_0,
        )
        assert res_0.status == "confirmation_required"
        assert res_0.message is not None
        assert "recorded, but message is held" in res_0.message
        assert mock_send.call_count == 0
        assert session_id in tool.pending_confirmations

        # Turn 1: Second call after confirmation -> Must send
        mock_send.return_value = {"status": "success"}
        ctx_turn_1 = ToolExecutionContext(session_id=session_id, current_turn=1)
        res_1 = await tool.execute(
            {"phone_number": "9825012345", "message_text": "Your summary details"},
            ctx_turn_1,
        )
        assert res_1.status == "sent"
        assert res_1.message is not None
        assert "successfully sent" in res_1.message
        assert mock_send.call_count == 1
        assert session_id not in tool.pending_confirmations



def test_voice_agent_auto_registers_whatsapp_tool():
    """Test that configuring VoiceConfig.whatsapp automatically registers WhatsAppNotificationTool."""
    wa_cfg = WhatsAppConfig(
        api_key="auto_key",
        service_url="https://api.example.com/send",
        default_template="bot_reply",
    )
    voice_cfg = VoiceConfig(
        api_key="gemini_key",
        whatsapp=wa_cfg,
    )
    agent = VoiceAgent(config=voice_cfg)

    assert agent.tools.has_tool("send_whatsapp_message")
    tool = agent.tools.get("send_whatsapp_message")
    assert isinstance(tool, WhatsAppNotificationTool)


def test_whatsapp_meta_cloud_api_auto_configuration():
    """Test official Meta WhatsApp Business Cloud API setup using token and phone_number_id."""
    meta_cfg = WhatsAppConfig(
        token="EAABxyz987SecretToken",
        phone_number_id="104239859238491",
        default_template="order_confirmation",
    )

    # 1. Automatic provider detection
    assert meta_cfg.provider == "meta"

    # 2. Automatic service_url generation pointing to Meta Graph API
    assert meta_cfg.service_url == "https://graph.facebook.com/v21.0/104239859238491/messages"

    # 3. Automatic Bearer auth header generation
    client = WhatsAppClient(meta_cfg)
    headers = client._get_headers()
    assert headers["Authorization"] == "Bearer EAABxyz987SecretToken"
    assert headers["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_whatsapp_meta_direct_text_send_async():
    """Test sending direct text message via Meta WhatsApp Business Cloud API."""
    meta_cfg = WhatsAppConfig(
        token="test_bearer_token",
        phone_number_id="9988776655",
    )
    client = WhatsAppClient(meta_cfg)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "messaging_product": "whatsapp",
        "contacts": [{"input": "919825012345", "wa_id": "919825012345"}],
        "messages": [{"id": "wamid.HBgL..."}],
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        res = await client.send_text_async(
            to="9825012345",
            text="Hello from Voice AI via Meta Cloud API",
        )
        assert res["messaging_product"] == "whatsapp"
        assert res["messages"][0]["id"] == "wamid.HBgL..."

        # Verify posted payload structure
        call_kwargs = mock_post.call_args.kwargs
        posted_json = call_kwargs["json"]
        assert posted_json["messaging_product"] == "whatsapp"
        assert posted_json["type"] == "text"
        assert posted_json["text"]["body"] == "Hello from Voice AI via Meta Cloud API"
        assert posted_json["to"] == "919825012345"


def test_whatsapp_config_from_env_meta_and_custom(monkeypatch):
    """Test loading Meta token and phone number ID from environment variables."""
    monkeypatch.setenv("WHATSAPP_TOKEN", "meta_env_token_456")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "5544332211")
    monkeypatch.setenv("WHATSAPP_DEFAULT_TEMPLATE", "hello_world")

    cfg = WhatsAppConfig.from_env()
    assert cfg.token == "meta_env_token_456"
    assert cfg.phone_number_id == "5544332211"
    assert cfg.provider == "meta"
    assert cfg.service_url == "https://graph.facebook.com/v21.0/5544332211/messages"
    assert cfg.default_template == "hello_world"


