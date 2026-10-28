"""
Base abstractions for pluggable tools in the Voice AI Framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, Optional


@dataclass
class ToolResult:
    """
    Unified return type for any tool executed within the Voice AI engine.
    """
    status: str = "success"
    message: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Just-In-Time instructions to guide the voice model's immediate response
    jit_instruction: Optional[str] = None
    
    # Event data to broadcast to connected client(s) (e.g. browser WebSocket)
    client_event: Optional[Dict[str, Any]] = None
    
    # Audio file URL to queue for physical playback (e.g. on Raspberry Pi speaker)
    queued_audio_url: Optional[str] = None

    def to_gemini_response_content(self) -> Dict[str, Any]:
        """Formats the result into the dictionary expected by Gemini's toolResponse."""
        content: Dict[str, Any] = {
            "status": self.status,
        }
        if self.message:
            content["message"] = self.message
        if self.jit_instruction:
            content["instruction"] = self.jit_instruction
        if self.data:
            content.update(self.data)
        return content


@dataclass
class ToolExecutionContext:
    """
    Context passed into every tool execution call.
    Contains session details, turn counter, and optional client message emitter.
    """
    session_id: str
    current_turn: int = 0
    client_emitter: Optional[Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    async def emit_to_client(self, event_data: Dict[str, Any]) -> None:
        """Sends an event back to the client interface (e.g., browser or device UI)."""
        if self.client_emitter:
            try:
                await self.client_emitter(event_data)
            except Exception as exc:
                print(f"⚠️ [ToolExecutionContext] Failed to emit client event: {exc}")


class BaseTool(ABC):
    """
    Abstract Base Class for any Voice Agent Tool.
    Implementations specify schema, trigger description, and async execution logic.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique function declaration name sent to Gemini."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description in language(s) optimized for LLM trigger."""
        pass

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON schema parameters definition."""
        pass

    @property
    def prompt_instruction(self) -> Optional[str]:
        """
        Optional system prompt guideline/rule for this tool.
        If provided, the framework automatically composes this rule into the AI's system prompt.
        """
        return None

    @abstractmethod
    async def execute(self, args: Dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        """
        Asynchronously execute tool with provided arguments and session context.
        """
        pass


    def to_function_declaration(self) -> Dict[str, Any]:
        """Converts this tool into a Gemini API function declaration object."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }
