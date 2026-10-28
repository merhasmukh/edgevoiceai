"""
WhatsApp Notification Subsystem for Voice AI Framework.
Provides developer configuration, async client transport, and two-turn verbal confirmation tool.
Compatible with generic REST gateways, Meta Cloud API, and webhook providers on Python 3.10+.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore[assignment]
    HTTPX_AVAILABLE = False

from edgevoiceai.tools.notification_tool import (
    TwoTurnConfirmationNotificationTool,
)



@dataclass(slots=True)
class WhatsAppConfig:
    """
    Configuration options for WhatsApp messaging in the Voice AI Framework.
    Natively supports:
      1. Official Meta / Facebook WhatsApp Business Cloud API (token, phone_number_id)
      2. Custom REST gateways / webhook relays (service_url, api_key)
    """
    token: str = ""  # Official Meta WhatsApp Cloud API Access Token
    phone_number_id: str = ""  # Official Meta Phone Number ID (sender ID)
    business_account_id: str = ""  # Meta WABA / Business Account ID (optional)
    api_version: str = "v21.0"  # Meta Graph API version

    # Generic or custom gateway parameters
    api_key: str = ""  # API key for generic/custom gateway (or alias for token)
    service_url: str = ""  # Custom service URL (auto-generated if phone_number_id is provided)
    provider: str = "auto"  # "auto" | "meta" | "generic" | "webhook"

    # Messaging defaults
    default_template: str = ""
    default_language: str = "en"
    phone_prefix: str = "91"  # Default country calling code
    timeout: float = 15.0

    # Custom auth header overrides
    auth_header_name: str | None = None  # e.g., "Authorization" or "API-KEY"
    auth_header_prefix: str | None = None  # e.g., "Bearer "

    # Voice AI safety & guardrails
    require_two_turn_confirmation: bool = True

    # Localized / customized conversational messages
    invalid_phone_message: str | None = None
    confirmation_prompt_message: str | None = None
    success_message: str | None = None
    prompt_instruction: str | None = None


    def __post_init__(self) -> None:
        # Standardize token and api_key aliases
        if not self.token and self.api_key:
            object.__setattr__(self, "token", self.api_key)
        elif not self.api_key and self.token:
            object.__setattr__(self, "api_key", self.token)

        # Resolve provider
        if self.provider == "auto":
            if self.phone_number_id:
                object.__setattr__(self, "provider", "meta")
            elif self.service_url:
                object.__setattr__(self, "provider", "generic")
            else:
                object.__setattr__(self, "provider", "generic")

        # Auto-construct Meta Graph API endpoint URL if phone_number_id is provided
        if self.provider == "meta" and self.phone_number_id:
            constructed_url = f"https://graph.facebook.com/{self.api_version}/{self.phone_number_id}/messages"
            object.__setattr__(self, "service_url", constructed_url)


    @classmethod
    def from_env(cls) -> WhatsAppConfig:
        """
        Loads configuration from standard environment variables:
        Supports both Meta Cloud API conventions (WHATSAPP_TOKEN, WHATSAPP_PHONE_NUMBER_ID)
        and custom gateway conventions (WHATSAPP_SERVICE_URL, WHATSAPP_API_KEY).
        """
        token = (
            os.environ.get("WHATSAPP_TOKEN")
            or os.environ.get("WHATSAPP_ACCESS_TOKEN")
            or os.environ.get("WHATSAPP_API_KEY")
            or ""
        ).strip()

        phone_number_id = (
            os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
            or os.environ.get("WHATSAPP_MOBILE_NUMBER_ID")
            or os.environ.get("WHATSAPP_SENDER_PHONE_ID")
            or ""
        ).strip()

        business_account_id = (
            os.environ.get("WHATSAPP_BUSINESS_ACCOUNT_ID")
            or os.environ.get("WHATSAPP_WABA_ID")
            or ""
        ).strip()

        service_url = os.environ.get("WHATSAPP_SERVICE_URL", "").strip()
        provider = os.environ.get("WHATSAPP_PROVIDER", "auto").strip()
        api_version = os.environ.get("WHATSAPP_API_VERSION", "v21.0").strip()

        default_template = (
            os.environ.get("WHATSAPP_DEFAULT_TEMPLATE")
            or os.environ.get("WHATSAPP_TEMPLATE_NAME")
            or ""
        ).strip()

        default_language = (
            os.environ.get("WHATSAPP_DEFAULT_LANGUAGE")
            or os.environ.get("WHATSAPP_LANGUAGE")
            or "en"
        ).strip()

        phone_prefix = os.environ.get("WHATSAPP_PHONE_PREFIX", "91").strip()
        timeout = float(os.environ.get("WHATSAPP_TIMEOUT", "15.0"))

        auth_header_name = os.environ.get("WHATSAPP_AUTH_HEADER_NAME")
        auth_header_prefix = os.environ.get("WHATSAPP_AUTH_HEADER_PREFIX")

        return cls(
            token=token,
            api_key=token,
            phone_number_id=phone_number_id,
            business_account_id=business_account_id,
            service_url=service_url,
            provider=provider,
            api_version=api_version,
            default_template=default_template,
            default_language=default_language,
            phone_prefix=phone_prefix,
            timeout=timeout,
            auth_header_name=auth_header_name.strip() if auth_header_name else None,
            auth_header_prefix=auth_header_prefix if auth_header_prefix is not None else None,
        )


class WhatsAppClient:
    """
    Asynchronous and synchronous transport client for WhatsApp notification gateways.
    Seamlessly dispatches to Meta WhatsApp Business Cloud API or custom HTTP REST gateways.
    """

    def __init__(self, config: WhatsAppConfig | None = None) -> None:
        self.config = config or WhatsAppConfig.from_env()

    def normalize_phone(self, raw: str) -> str:
        """
        Validates and standardizes a phone number using the configured country prefix.
        Filters out dummy, invalid, or repeating phone sequences.
        """
        if not raw:
            return ""
        digits = re.sub(r"\D", "", raw)
        prefix = self.config.phone_prefix

        # If already starts with prefix (e.g., 919876543210 -> 9876543210)
        if prefix and digits.startswith(prefix) and len(digits) == 10 + len(prefix):
            digits = digits[len(prefix):]
        elif digits.startswith("0") and len(digits) == 11:
            digits = digits[1:]

        # Must be standard 10 mobile digits
        if len(digits) != 10:
            return ""
        # Filter trivial / dummy numbers
        if len(set(digits)) <= 2:
            return ""
        if digits in ("1234567890", "0123456789", "9876543210", "0000000000"):
            return ""
        # Valid starting digits
        if digits[0] not in "56789":
            return ""

        return (prefix + digits) if prefix else digits

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        token = self.config.token or self.config.api_key
        if not token:
            return headers

        if self.config.provider == "meta":
            h_name = self.config.auth_header_name or "Authorization"
            h_prefix = self.config.auth_header_prefix if self.config.auth_header_prefix is not None else "Bearer "
            headers[h_name] = f"{h_prefix}{token}".strip()
        else:
            h_name = self.config.auth_header_name or "API-KEY"
            h_prefix = self.config.auth_header_prefix or ""
            headers[h_name] = f"{h_prefix}{token}".strip()
        return headers

    def _build_payload(
        self,
        to: str | list[str],
        body_params: list[str] | None = None,
        button_params: list[str] | None = None,
        template_name: str | None = None,
        language: str | None = None,
        raw_text: str | None = None,
    ) -> dict[str, Any]:
        recipients = [to] if isinstance(to, str) else list(to)
        normalized = [self.normalize_phone(r) for r in recipients if self.normalize_phone(r)]
        if not normalized:
            raise ValueError("No valid recipient phone numbers provided.")

        tpl = template_name or self.config.default_template
        lang = language or self.config.default_language

        if self.config.provider == "meta":
            # Meta Graph Cloud API payload structure
            if tpl:
                params_obj = [{"type": "text", "text": p} for p in (body_params or [])]
                return {
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": normalized[0],
                    "type": "template",
                    "template": {
                        "name": tpl,
                        "language": {"code": lang},
                        "components": [{"type": "body", "parameters": params_obj}],
                    },
                }
            else:
                return {
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": normalized[0],
                    "type": "text",
                    "text": {
                        "preview_url": False,
                        "body": raw_text or "",
                    },
                }

        # Generic REST gateway payload
        payload: dict[str, Any] = {
            "api_key": self.config.token or self.config.api_key,
            "to": normalized,
            "type": "template" if tpl else "text",
            "template_name": tpl,
            "language": lang,
            "body_params": body_params or [],
        }
        if raw_text:
            payload["message"] = raw_text
        if button_params:
            payload["button_params"] = button_params
        return payload

    async def send_template_async(
        self,
        to: str | list[str],
        body_params: list[str] | None = None,
        button_params: list[str] | None = None,
        template_name: str | None = None,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Sends a WhatsApp template message asynchronously via HTTP POST."""
        if not HTTPX_AVAILABLE or httpx is None:
            raise RuntimeError("httpx is required for WhatsApp messaging. Run: pip install httpx")
        service_url = self.config.service_url
        if not service_url:
            raise ValueError(
                "WhatsApp service_url or phone_number_id is not configured. "
                "For Meta Cloud API, provide 'phone_number_id' and 'token'. "
                "For custom gateways, provide 'service_url' and 'api_key'."
            )

        payload = self._build_payload(
            to=to,
            body_params=body_params,
            button_params=button_params,
            template_name=template_name,
            language=language,
        )
        headers = self._get_headers()

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                resp = await client.post(service_url, json=payload, headers=headers)
                resp.raise_for_status()
                try:
                    return resp.json()
                except Exception:
                    return {"status": "success", "raw": resp.text}
        except httpx.HTTPStatusError as exc:
            err_text = exc.response.text
            try:
                err_json = exc.response.json()
                if "error" in err_json and isinstance(err_json["error"], dict):
                    err_text = err_json["error"].get("message", err_text)
            except Exception:
                pass
            return {
                "status": "error",
                "error": f"HTTP {exc.response.status_code}: {err_text}",
                "status_code": exc.response.status_code,
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    async def send_text_async(
        self,
        to: str | list[str],
        text: str,
    ) -> dict[str, Any]:
        """Sends a direct text WhatsApp message asynchronously."""
        if not HTTPX_AVAILABLE or httpx is None:
            raise RuntimeError("httpx is required for WhatsApp messaging. Run: pip install httpx")
        service_url = self.config.service_url
        if not service_url:
            raise ValueError(
                "WhatsApp service_url or phone_number_id is not configured. "
                "For Meta Cloud API, provide 'phone_number_id' and 'token'. "
                "For custom gateways, provide 'service_url' and 'api_key'."
            )

        payload = self._build_payload(to=to, raw_text=text)
        headers = self._get_headers()

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                resp = await client.post(service_url, json=payload, headers=headers)
                resp.raise_for_status()
                try:
                    return resp.json()
                except Exception:
                    return {"status": "success", "raw": resp.text}
        except httpx.HTTPStatusError as exc:
            err_text = exc.response.text
            try:
                err_json = exc.response.json()
                if "error" in err_json and isinstance(err_json["error"], dict):
                    err_text = err_json["error"].get("message", err_text)
            except Exception:
                pass
            return {
                "status": "error",
                "error": f"HTTP {exc.response.status_code}: {err_text}",
                "status_code": exc.response.status_code,
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}



class WhatsAppNotificationTool(TwoTurnConfirmationNotificationTool):
    """
    Ready-to-use WhatsApp Tool for Voice AI agents.
    Enforces a strict two-turn verbal confirmation protocol to ensure the agent
    never dispatches messages to hallucinated, unverified, or wrong numbers.
    """

    DEFAULT_TOOL_NAME = "send_whatsapp_message"
    DEFAULT_TOOL_DESCRIPTION = (
        "Sends a summary or answer to the user's phone via WhatsApp. "
        "STRICT TWO-TURN CONFIRMATION PROTOCOL: "
        "1. When the user asks for a message or provides their phone number, DO NOT dispatch immediately. "
        "2. First verbally confirm: 'Your number is [phone_number], right?' and stop speaking. "
        "3. Only in the next turn after the user explicitly confirms (e.g. 'yes/correct') should this tool be called."
    )

    def __init__(
        self,
        config: WhatsAppConfig | None = None,
        client: WhatsAppClient | None = None,
        name: str | None = None,
        description: str | None = None,
        phone_param: str = "phone_number",
        text_param: str = "message_text",
        format_text_func: Callable[[str], str] | None = None,
        prompt_instruction: str | None = None,
    ) -> None:
        self.whatsapp_config = config or WhatsAppConfig.from_env()
        self.whatsapp_client = client or WhatsAppClient(self.whatsapp_config)

        schema = {
            "type": "OBJECT",
            "properties": {
                phone_param: {
                    "type": "STRING",
                    "description": "User's verified mobile phone number with country prefix."
                },
                text_param: {
                    "type": "STRING",
                    "description": "The exact message or summary text to send over WhatsApp."
                },
            },
            "required": [phone_param, text_param],
        }

        async def _sender(phone: str, text: str) -> dict[str, Any]:
            if self.whatsapp_config.default_template:
                return await self.whatsapp_client.send_template_async(
                    to=phone,
                    body_params=[text],
                )
            return await self.whatsapp_client.send_text_async(to=phone, text=text)

        # Message builders
        def invalid_builder(raw_num: str) -> str:
            if self.whatsapp_config.invalid_phone_message:
                return self.whatsapp_config.invalid_phone_message.format(phone_raw=raw_num)
            return (
                f"Invalid or missing phone number: '{raw_num}'. "
                "A valid 10-digit mobile number was not detected. "
                "Please verbally ask the user for their 10-digit mobile number "
                "and do not use dummy or fictional numbers."
            )

        def conf_builder(phone: str, raw_num: str) -> str:
            if self.whatsapp_config.confirmation_prompt_message:
                return self.whatsapp_config.confirmation_prompt_message.format(
                    phone=phone, phone_raw=raw_num
                )
            return (
                f"Phone number {phone} recorded, but message is held for verbal confirmation. "
                f"Please say to the user: 'Your phone number is {raw_num}, correct?' and stop. "
                "Wait for the user to confirm in the next turn before calling this tool again."
            )

        def succ_builder(phone: str) -> str:
            if self.whatsapp_config.success_message:
                return self.whatsapp_config.success_message.format(phone=phone)
            return f"WhatsApp message successfully sent to {phone}."

        resolved_instruction = prompt_instruction or self.whatsapp_config.prompt_instruction

        super().__init__(
            name=name or self.DEFAULT_TOOL_NAME,
            description=description or self.DEFAULT_TOOL_DESCRIPTION,
            parameters_schema=schema,
            sender_func=_sender,
            normalize_phone_func=self.whatsapp_client.normalize_phone,
            format_text_func=format_text_func,
            phone_param=phone_param,
            text_param=text_param,
            invalid_phone_message_builder=invalid_builder,
            confirmation_needed_message_builder=conf_builder,
            success_message_builder=succ_builder,
            prompt_instruction=resolved_instruction,
        )

