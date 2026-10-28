"""
Central ToolRegistry for discovering, registering, and dynamically dispatching tool calls.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from edgevoiceai.tools.base import BaseTool, ToolResult, ToolExecutionContext

_logger = logging.getLogger("edgevoiceai.tools")


class ToolRegistry:
    """
    Manages a collection of BaseTool instances and handles dispatching tool calls.
    """

    def __init__(self, tools: Optional[List[BaseTool]] = None):
        self._tools: Dict[str, BaseTool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: BaseTool) -> None:
        """Registers a tool instance. Overwrites existing registration if name matches."""
        self._tools[tool.name] = tool
        _logger.debug("Registered tool '%s'", tool.name)

    def unregister(self, tool_name: str) -> Optional[BaseTool]:
        """Removes a tool from registry by name."""
        return self._tools.pop(tool_name, None)

    def get(self, tool_name: str) -> Optional[BaseTool]:
        """Returns the registered tool instance, if found."""
        return self._tools.get(tool_name)

    def has_tool(self, tool_name: str) -> bool:
        """Checks if a tool is registered."""
        return tool_name in self._tools

    def list_tools(self) -> List[BaseTool]:
        """Returns a list of all registered tools."""
        return list(self._tools.values())

    def to_gemini_tools_payload(self) -> List[Dict[str, Any]]:
        """
        Builds the Gemini Live `tools` array configuration:
        [{"functionDeclarations": [...]}]
        """
        if not self._tools:
            return []
        return [
            {
                "functionDeclarations": [
                    tool.to_function_declaration() for tool in self._tools.values()
                ]
            }
        ]

    def build_tools_instruction(self, header: str = "[ACTIVE TOOLS & USAGE RULES]") -> str:
        """
        Aggregates prompt instructions from all registered tools into a unified guideline block.
        Ensures the AI assistant strictly understands when and how to call active tools.
        """
        rules: List[str] = []
        for tool in self._tools.values():
            instruction = tool.prompt_instruction
            if instruction and instruction.strip():
                rules.append(instruction.strip())

        if not rules:
            return ""

        return f"{header}\n" + "\n\n".join(rules)


    async def dispatch(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """
        Dynamically dispatches a tool call to the registered handler.
        Emits client events if the tool produces any.
        """
        tool = self.get(tool_name)
        if not tool:
            _logger.warning("Attempted to call unregistered tool: %s", tool_name)
            return ToolResult(
                status="error",
                message=f"Tool '{tool_name}' is not registered or supported by this agent.",
            )

        try:
            result = await tool.execute(args, context)
            
            # Emit client event if produced by tool execution
            if result.client_event and context.client_emitter:
                await context.emit_to_client(result.client_event)

            return result
        except Exception as exc:
            _logger.exception("Error executing tool '%s': %s", tool_name, exc)
            return ToolResult(
                status="error",
                message=f"Error executing tool '{tool_name}': {exc}",
            )
