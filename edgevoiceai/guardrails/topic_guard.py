"""
Conversational Domain Guardrail.
Validates user queries against allowed domain boundaries, rejecting off-topic
questions before they invoke expensive tools or vector database searches.
"""

from __future__ import annotations

from typing import Callable


class TopicGuard:
    """
    Configurable topic boundary filter.
    Allows developers to set allowed keywords, off-topic keywords,
    or custom inspection functions with multilingual response builders.
    """

    def __init__(
        self,
        domain_name: str = "General",
        allowed_keywords: list[str] | None = None,
        blocked_keywords: list[str] | None = None,
        custom_validator: Callable[[str], tuple[bool, str]] | None = None,
        default_rejection_message: str | None = None,
    ) -> None:
        self.domain_name = domain_name
        self.allowed_keywords = [k.lower() for k in (allowed_keywords or [])]
        self.blocked_keywords = [k.lower() for k in (blocked_keywords or [])]
        self.custom_validator = custom_validator
        self.default_rejection_message = default_rejection_message or (
            f"I am a specialized AI assistant for {domain_name}. "
            f"Please ask questions related to {domain_name}."
        )

    def is_allowed(self, query: str) -> tuple[bool, str]:
        """
        Validates if query is within domain boundaries.

        Returns:
            (is_valid, rejection_message_or_empty)
        """
        clean_query = (query or "").strip()
        if not clean_query:
            return True, ""

        # 1. Custom validator hook if provided
        if self.custom_validator is not None:
            return self.custom_validator(clean_query)

        q_lower = clean_query.lower()

        # 2. Check explicitly blocked keywords
        if self.blocked_keywords and any(b in q_lower for b in self.blocked_keywords):
            return False, self.default_rejection_message

        # 3. Check allowed keywords if specified
        if self.allowed_keywords:
            has_match = any(a in q_lower for a in self.allowed_keywords)
            if not has_match:
                return False, self.default_rejection_message

        return True, ""
