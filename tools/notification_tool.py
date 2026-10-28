"""
Two-turn verbal confirmation notification tool for Voice AI.
Prevents hallucinated or unsolicited messaging by strictly verifying recipient
contact information across conversational turns before sending.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Coroutine, Dict, Optional
from edgevoiceai.tools.base import BaseTool, ToolResult, ToolExecutionContext


def default_normalize_phone(raw: str) -> str:
    """Standard 10-digit Indian mobile number normalizer with '91' prefix."""
    if not raw:
        return ""
    digits = re.sub(r"\D", "", raw)
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    elif digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]
    if len(digits) != 10:
        return ""
    if len(set(digits)) <= 2:
        return ""
    if digits in ("1234567890", "0123456789", "9876543210"):
        return ""
    if digits[0] not in "6789":
        return ""
    return "91" + digits


class TwoTurnConfirmationNotificationTool(BaseTool):
    """
    Guarded notification tool requiring explicit multi-turn confirmation before dispatch.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters_schema: Dict[str, Any],
        sender_func: Callable[[str, str], Coroutine[Any, Any, Dict[str, Any]]],
        normalize_phone_func: Callable[[str], str] = default_normalize_phone,
        format_text_func: Optional[Callable[[str], str]] = None,
        phone_param: str = "phone_number",
        text_param: str = "answer_text",
        invalid_phone_message_builder: Optional[Callable[[str], str]] = None,
        confirmation_needed_message_builder: Optional[Callable[[str, str], str]] = None,
        success_message_builder: Optional[Callable[[str], str]] = None,
        pending_confirmations_dict: Optional[Dict[str, Dict[str, Any]]] = None,
        turn_counter_dict: Optional[Dict[str, int]] = None,
        prompt_instruction: Optional[str] = None,
    ):
        self._name = name
        self._description = description
        self._parameters_schema = parameters_schema
        self.sender_func = sender_func
        self.normalize_phone_func = normalize_phone_func
        self.format_text_func = format_text_func or (lambda t: t)
        self.phone_param = phone_param
        self.text_param = text_param
        self.invalid_phone_message_builder = invalid_phone_message_builder
        self.confirmation_needed_message_builder = confirmation_needed_message_builder
        self.success_message_builder = success_message_builder
        self._prompt_instruction = prompt_instruction or (
            f"• NOTIFICATION & MESSAGING PROTOCOL (`{self._name}`):\n"
            f"  1. When the user asks to send an answer or provides their phone number, DO NOT call `{self._name}` in that turn.\n"
            f"  2. First say verbally: 'Your phone number is [phone_number], correct?' and stop speaking.\n"
            f"  3. Only in the subsequent turn, after the user confirms (e.g. 'yes/correct'), call `{self._name}`.\n"
            "  4. Never invent or use dummy phone numbers like '0000000000'."
        )

        # In-memory tracking of pending confirmations: session_id -> confirmation dict
        self._pending_confirmations = pending_confirmations_dict if pending_confirmations_dict is not None else {}
        self.turn_counter_dict = turn_counter_dict

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return self._parameters_schema

    @property
    def prompt_instruction(self) -> Optional[str]:
        return self._prompt_instruction

    @property
    def pending_confirmations(self) -> Dict[str, Dict[str, Any]]:
        return self._pending_confirmations


    def get_pending(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._pending_confirmations.get(session_id)

    def clear_pending(self, session_id: str) -> None:
        self._pending_confirmations.pop(session_id, None)

    async def execute(self, args: Dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        phone_raw = args.get(self.phone_param, "")
        text_raw = args.get(self.text_param, "")
        formatted_text = self.format_text_func(text_raw)

        phone = self.normalize_phone_func(phone_raw)
        if not phone:
            print(f"📲 [{self.name}] ❌ Blocked invalid/dummy phone: {phone_raw!r}")
            if self.invalid_phone_message_builder:
                msg = self.invalid_phone_message_builder(phone_raw)
            else:
                msg = (
                    f"Invalid or missing phone number: '{phone_raw}'. "
                    "A valid 10-digit mobile number was not found. "
                    "Please ask the user for their 10-digit phone number, "
                    "and verbally confirm it with them before sending. "
                    "Never use fictional or dummy numbers."
                )
            return ToolResult(status="error", message=msg)

        curr_turn = context.current_turn
        if self.turn_counter_dict is not None and context.session_id in self.turn_counter_dict:
            curr_turn = self.turn_counter_dict[context.session_id]

        pending = self._pending_confirmations.get(context.session_id)

        # Strict 2-turn confirmation gate:
        # Message is ONLY dispatched if confirmation was previously requested
        # in an earlier turn (< curr_turn) and the phone matches.
        is_confirmed = False
        if pending and pending.get("phone") == phone:
            if pending.get("asked_at_turn", -1) < curr_turn:
                is_confirmed = True

        if not is_confirmed:
            self._pending_confirmations[context.session_id] = {
                "phone": phone,
                "answer_text": formatted_text,
                "asked_at_turn": curr_turn,
                "timestamp": time.time(),
            }
            print(f"📲 [{self.name} Gate] ⏳ Confirmation requested for {phone} (Turn {curr_turn}). Holding send...")
            if self.confirmation_needed_message_builder:
                conf_msg = self.confirmation_needed_message_builder(phone, phone_raw)
            else:
                conf_msg = (
                    f"Phone number {phone} recorded, but notification is not sent yet. "
                    f"Please say to the user: 'Your phone number is {phone_raw}, right?' and stop speaking. "
                    "Wait for the user to confirm in the next turn before calling this tool again."
                )
            return ToolResult(status="confirmation_required", message=conf_msg)

        # Confirmed! Proceed to send
        self._pending_confirmations.pop(context.session_id, None)
        try:
            send_res = await self.sender_func(phone, formatted_text)
            if send_res.get("status") == "error":
                err_msg = f"Notification send failed: {send_res.get('error', 'Unknown error')}"
                print(f"📲 [{self.name}] Send failed to {phone}: {send_res}")
                return ToolResult(status="error", message=err_msg)
            else:
                if self.success_message_builder:
                    succ_msg = self.success_message_builder(phone)
                else:
                    succ_msg = f"Notification successfully sent to {phone}."
                print(f"📲 [{self.name}] Sent to {phone}: {formatted_text[:60]}...")
                return ToolResult(status="sent", message=succ_msg)

        except Exception as exc:
            print(f"📲 [{self.name}] Exception while sending: {exc}")
            return ToolResult(status="error", message=f"Notification error: {exc}")
