"""
Hardware telemetry and IoT sensor reader tool for Voice AI.
Supports asynchronous thread execution, live metric broadcasting to clients,
and dynamic Just-In-Time guidance prompts tailored to sensor reading conditions.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, Optional
from edgevoiceai.tools.base import BaseTool, ToolResult, ToolExecutionContext


class SensorTelemetryTool(BaseTool):
    """
    A modular tool for reading telemetry from IoT/hardware sensors
    (e.g., Modbus RS485, I2C, SPI, ADC, or MQTT).
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters_schema: Dict[str, Any],
        reader_func: Callable[[], Dict[str, Any]],
        client_event_type: str = "soil_moisture_reading",
        jit_instruction_builder: Optional[Callable[[Dict[str, Any]], str]] = None,
        terminal_formatter: Optional[Callable[[Dict[str, Any]], None]] = None,
        prompt_instruction: Optional[str] = None,
    ):
        self._name = name
        self._description = description
        self._parameters_schema = parameters_schema
        self.reader_func = reader_func
        self.client_event_type = client_event_type
        self.jit_instruction_builder = jit_instruction_builder
        self.terminal_formatter = terminal_formatter
        self._prompt_instruction = prompt_instruction or (
            f"• SENSOR TELEMETRY PROTOCOL (`{self._name}`):\n"
            f"  1. Call `{self._name}` when checking hardware readings, sensor diagnostics, or physical metrics.\n"
            "  2. Summarize key readings clearly with their appropriate measurement units."
        )

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


    async def execute(self, args: Dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        # Run blocking hardware I/O in thread pool to prevent blocking asyncio event loop
        sensor_data: Dict[str, Any] = await asyncio.to_thread(self.reader_func)
        
        # Format terminal display if helper provided
        if self.terminal_formatter:
            try:
                self.terminal_formatter(sensor_data)
            except Exception as fmt_exc:
                print(f"⚠️ [{self.name}] Terminal formatting error: {fmt_exc}")

        # Broadcast reading event to client
        client_evt = {
            "type": self.client_event_type,
            "data": sensor_data,
        }

        # Build JIT prompt for the AI model
        jit_instruction: Optional[str] = None
        if self.jit_instruction_builder:
            jit_instruction = self.jit_instruction_builder(sensor_data)

        # Merge sensor data into result
        return ToolResult(
            status=sensor_data.get("status", "success"),
            data=sensor_data,
            jit_instruction=jit_instruction,
            client_event=client_evt,
        )
