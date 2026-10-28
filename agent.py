"""
High-Level VoiceAgent Orchestrator.
Coordinates audio I/O, Gemini Live bidirectional WebSocket streaming,
voice activity detection, tool dispatch, and barge-in interruptions.
Runs across Raspberry Pi, Linux, macOS, and Windows.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable

try:
    import websockets
except ImportError:
    websockets = None  # type: ignore[assignment]

try:
    import pyaudio  # type: ignore[import-untyped]
    PYAUDIO_AVAILABLE = True
except ImportError:
    pyaudio = None  # type: ignore[assignment]
    PYAUDIO_AVAILABLE = False

from edgevoiceai.audio.device import AudioDeviceManager
from edgevoiceai.audio.media import MediaAudioPlayer
from edgevoiceai.audio.player import AudioPlayer
from edgevoiceai.audio.stream import MicrophoneStream
from edgevoiceai.audio.vad import VADConfig, VoiceActivityDetector, compute_rms
from edgevoiceai.config import VoiceConfig
from edgevoiceai.guardrails.farewell import FarewellDetector
from edgevoiceai.guardrails.topic_guard import TopicGuard
from edgevoiceai.memory.extractor import EntityExtractor
from edgevoiceai.memory.stores.base import BaseMemoryStore
from edgevoiceai.notifications.whatsapp import WhatsAppNotificationTool
from edgevoiceai.protocol.gemini import GeminiLiveProtocol
from edgevoiceai.tools.base import BaseTool, ToolExecutionContext
from edgevoiceai.tools.registry import ToolRegistry


class VoiceAgent:
    """
    Full-duplex real-time Conversational Voice AI Agent.
    """

    def __init__(
        self,
        config: VoiceConfig,
        tools: ToolRegistry | None = None,
        topic_guard: TopicGuard | None = None,
        farewell_detector: FarewellDetector | None = None,
        memory: BaseMemoryStore | None = None,
        entity_extractor: EntityExtractor | None = None,
    ) -> None:
        self.config = config
        self.tools = tools or ToolRegistry()
        self.topic_guard = topic_guard
        self.farewell_detector = farewell_detector or FarewellDetector()
        self.memory = memory
        self.entity_extractor = entity_extractor

        # Auto-register WhatsApp notification tool if configured
        if self.config.whatsapp is not None:
            wa_tool = WhatsAppNotificationTool(config=self.config.whatsapp)
            if not self.tools.has_tool(wa_tool.name):
                self.tools.register(wa_tool)

        # Optional event callbacks
        self.on_user_speech_start: Callable[[], None] | None = None
        self.on_ai_speech_start: Callable[[], None] | None = None
        self.on_turn_complete: Callable[[str], None] | None = None
        self.on_interrupted: Callable[[], None] | None = None
        self.on_tool_executed: Callable[[str, dict[str, Any], Any], None] | None = None

        self._shutdown_event = asyncio.Event()



    def register_tool(self, tool: BaseTool) -> VoiceAgent:
        """Fluent interface to register a tool with the agent."""
        self.tools.register(tool)
        return self

    def stop(self) -> None:
        """Signals the agent to stop its running loop."""
        self._shutdown_event.set()

    async def run_edge(self, session_id: str = "default") -> None:
        """
        Runs the full always-listening voice loop on edge hardware
        (Raspberry Pi, Mac, Windows, Linux laptop).
        """
        if websockets is None:
            raise RuntimeError("websockets is required. Install with: pip install websockets")

        # 1. Resolve Audio Hardware Devices
        in_dev = AudioDeviceManager.find_device_index(self.config.input_device, is_input=True)
        out_dev = AudioDeviceManager.find_device_index(self.config.output_device, is_input=False)

        print("\n" + "=" * 60)
        print("🎙️  [VOICE AGENT INITIALIZED]")
        print("=" * 60)
        print(f"  Model            : {self.config.model}")
        print(f"  Voice            : {self.config.voice}")
        print(f"  Mic Device       : {'Default' if in_dev is None else f'Index {in_dev}'}")
        print(f"  Speaker Device   : {'Default' if out_dev is None else f'Index {out_dev}'}")
        print(f"  Interrupt Thresh : {self.config.interrupt_rms_threshold}")
        print(f"  Noise Gate Thresh: {self.config.noise_gate_threshold}")
        print("=" * 60 + "\n")

        # 2. Audio Pipeline Setup
        vad_config = VADConfig(
            interrupt_rms_threshold=self.config.interrupt_rms_threshold,
            ai_interrupt_threshold=self.config.ai_interrupt_threshold,
            noise_gate_threshold=self.config.noise_gate_threshold,
            hang_frames=self.config.hang_frames,
            streak_trigger=self.config.vad_streak_trigger,
            pre_buffer_max_frames=self.config.pre_buffer_max_frames,
        )
        vad = VoiceActivityDetector(vad_config)

        # Share a single PyAudio instance across mic and speaker
        pa: Any = None
        if PYAUDIO_AVAILABLE and pyaudio is not None:
            try:
                pa = pyaudio.PyAudio()
            except Exception as pa_err:
                print(f"⚠️  [PyAudio Init Warning]: {pa_err}")

        mic_stream = MicrophoneStream(
            pa=pa,
            device_index=in_dev,
            capture_rate=self.config.mic_capture_rate,
            target_rate=self.config.gemini_input_rate,
        )

        player = AudioPlayer(
            pa=pa,
            device_index=out_dev,
            sample_rate=self.config.speaker_sample_rate,
        )

        media_player = MediaAudioPlayer(media_dir=self.config.media_directory)

        ws_url = GeminiLiveProtocol.get_ws_url(
            self.config.api_key,
            base_url=self.config.ws_base_url or "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent",
        )

        is_reconnect = False

        try:
            while not self._shutdown_event.is_set():
                try:
                    print(f"🔗 Connecting to Gemini Live ({self.config.model})...")
                    async with websockets.connect(
                        ws_url,
                        open_timeout=self.config.open_timeout,
                        close_timeout=self.config.close_timeout,
                        ping_interval=self.config.ping_interval,
                        max_size=None,
                    ) as gemini_ws:
                        # Build dynamic system instruction (base prompt + user memory + tool rules)
                        sys_instruction = await self.build_effective_system_instruction(session_id=session_id)

                        # Send handshake setup message
                        tools_schema = self.tools.to_gemini_tools_payload()
                        setup_payload = GeminiLiveProtocol.build_setup_message(
                            model=self.config.model,
                            system_instruction=sys_instruction,
                            voice_name=self.config.voice,
                            tools_declarations=tools_schema,
                            is_reconnect=is_reconnect,
                        )
                        await gemini_ws.send(json.dumps(setup_payload))

                        # Await setup confirmation
                        raw_setup = await asyncio.wait_for(gemini_ws.recv(), timeout=15.0)
                        setup_str = raw_setup.decode("utf-8") if isinstance(raw_setup, (bytes, bytearray)) else str(raw_setup)
                        setup_data = json.loads(setup_str)
                        if "setupComplete" in setup_data:
                            print("✅ [Gemini Live Connected & Ready!]")
                        else:
                            print(f"⚠️  [Unexpected Setup Response]: {setup_str[:200]}")

                        # Drain any mic frames buffered during connection
                        while not mic_stream.queue.empty():
                            try:
                                mic_stream.queue.get_nowait()
                            except Exception:
                                break

                        # Proactive greeting: trigger so the assistant speaks immediately on startup
                        if not is_reconnect and self.config.greeting_message:
                            greet_payload = {
                                "clientContent": {
                                    "turns": [{"role": "user", "parts": [{"text": self.config.greeting_message}]}],
                                    "turnComplete": True,
                                }
                            }
                            await gemini_ws.send(json.dumps(greet_payload))
                            print("🎙️ [Greeting Triggered] — Assistant is introducing itself.")

                        # Session state
                        ai_is_speaking = False
                        tool_is_executing = False
                        interrupted_at = 0.0
                        user_has_spoken = not is_reconnect
                        session_turn = 0
                        user_speaking_logged = False

                        # ── Task A: Mic -> Gemini ──────────────────────────────
                        async def mic_to_gemini() -> None:
                            nonlocal ai_is_speaking, interrupted_at, user_has_spoken, user_speaking_logged
                            while not self._shutdown_event.is_set():
                                pcm = mic_stream.get_chunk(block=False)
                                if pcm is None:
                                    await asyncio.sleep(0.005)
                                    continue

                                if tool_is_executing:
                                    continue

                                rms = compute_rms(pcm)

                                # Check barge-in on media audio file
                                if media_player.is_playing():
                                    if vad.check_barge_in(pcm, self.config.interrupt_rms_threshold):
                                        print(f"\n⚡ [Interruption detected (RMS {int(rms)}) -> Stopping audio file]")
                                        media_player.stop()
                                        if self.on_interrupted:
                                            self.on_interrupted()
                                    continue

                                # Check barge-in on AI speech
                                if ai_is_speaking or player.is_playing():
                                    if vad.check_barge_in(pcm, self.config.ai_interrupt_threshold):
                                        print(f"\n⚡ [AI Interrupted (RMS {int(rms)}) -> Clearing speaker]")
                                        player.clear()
                                        media_player.stop()
                                        ai_is_speaking = False
                                        interrupted_at = time.time()
                                        user_has_spoken = True
                                        if self.on_interrupted:
                                            self.on_interrupted()
                                    else:
                                        continue

                                # Noise gate and speech streaming
                                should_stream, frames = vad.process_chunk(pcm)
                                if should_stream:
                                    if not user_speaking_logged and vad.gate_open:
                                        print(f"\n🎤 [Voice detected: RMS {int(rms)}]", flush=True)
                                        user_speaking_logged = True
                                    user_has_spoken = True
                                    for frame in frames:
                                        try:
                                            payload = GeminiLiveProtocol.encode_audio_chunk(frame)
                                            await gemini_ws.send(payload)
                                        except Exception:
                                            break
                                else:
                                    user_speaking_logged = False

                        # ── Task B: Gemini -> Speaker + Tools ──────────────────
                        async def gemini_to_speaker() -> None:
                            nonlocal ai_is_speaking, tool_is_executing, interrupted_at, session_turn
                            ai_text_parts: list[str] = []
                            pending_audio_url: str | None = None

                            async for raw_msg in gemini_ws:
                                if self._shutdown_event.is_set():
                                    break

                                msg_str = raw_msg.decode("utf-8") if isinstance(raw_msg, (bytes, bytearray)) else str(raw_msg)
                                parsed = GeminiLiveProtocol.parse_message(msg_str)
                                if parsed.error:
                                    print(f"⚠️  [Gemini Error]: {parsed.error}")
                                    continue

                                if parsed.interrupted:
                                    player.clear()
                                    ai_is_speaking = False
                                    interrupted_at = time.time()
                                    ai_text_parts.clear()
                                    continue

                                # Handle tool calls
                                if parsed.tool_calls:
                                    tool_is_executing = True
                                    try:
                                        ctx = ToolExecutionContext(
                                            session_id=session_id,
                                            current_turn=session_turn,
                                        )
                                        fn_responses: list[dict[str, Any]] = []
                                        for tc in parsed.tool_calls:
                                            fn_name = tc.get("name", "")
                                            fn_args = tc.get("args", {})
                                            call_id = tc.get("id", "")

                                            tool_res = await self.tools.dispatch(fn_name, fn_args, ctx)
                                            if tool_res.queued_audio_url:
                                                pending_audio_url = tool_res.queued_audio_url

                                            fn_responses.append({
                                                "response": {
                                                    "name": fn_name,
                                                    "content": tool_res.to_gemini_response_content(),
                                                },
                                                "id": call_id,
                                            })
                                            if self.on_tool_executed:
                                                self.on_tool_executed(fn_name, fn_args, tool_res)

                                        if fn_responses:
                                            tool_payload = GeminiLiveProtocol.encode_tool_responses(fn_responses)
                                            await gemini_ws.send(tool_payload)
                                    finally:
                                        tool_is_executing = False

                                # Process audio output
                                if parsed.audio_pcm_chunks:
                                    if interrupted_at > 0.0 and (time.time() - interrupted_at) < 0.8:
                                        continue
                                    if is_reconnect and not user_has_spoken:
                                        continue

                                    ai_is_speaking = True
                                    for chunk in parsed.audio_pcm_chunks:
                                        player.feed(chunk)

                                if parsed.text_parts:
                                    ai_text_parts.extend(parsed.text_parts)

                                if parsed.turn_complete:
                                    ai_is_speaking = False
                                    interrupted_at = 0.0
                                    full_text = "".join(ai_text_parts).strip()
                                    ai_text_parts.clear()
                                    session_turn += 1

                                    # Notify turn complete immediately
                                    if self.on_turn_complete and full_text:
                                        self.on_turn_complete(full_text)

                                    # Record turn in memory and schedule entity extraction in background
                                    if self.memory and full_text:
                                        await self.memory.add_message(
                                            session_id, role="assistant", content=full_text
                                        )
                                        if self.entity_extractor:
                                            asyncio.create_task(
                                                self.entity_extractor.extract_and_update(
                                                    session_id, self.memory
                                                )
                                            )

                                    # Only wait for speaker if an external pre-recorded audio file is queued
                                    if pending_audio_url:
                                        while player.is_playing() and not self._shutdown_event.is_set():
                                            await asyncio.sleep(0.04)
                                        media_player.play(pending_audio_url)
                                        pending_audio_url = None

                                    if self.farewell_detector.is_farewell(full_text):
                                        print("👋 [Session concluded by farewell]")
                                        self.stop()
                                        break

                        # Run both tasks concurrently
                        t_mic = asyncio.create_task(mic_to_gemini())
                        t_spk = asyncio.create_task(gemini_to_speaker())
                        done, pending = await asyncio.wait(
                            [t_mic, t_spk], return_when=asyncio.FIRST_COMPLETED
                        )
                        for task in pending:
                            task.cancel()
                            try:
                                await task
                            except (asyncio.CancelledError, Exception):
                                pass

                        for task in done:
                            err = task.exception()
                            if err:
                                raise err

                except (asyncio.CancelledError, KeyboardInterrupt):
                    self._shutdown_event.set()
                    break
                except Exception as exc:
                    if self._shutdown_event.is_set():
                        break
                    print(f"⚠️  [Session connection error]: {exc}")
                    print(f"   Reconnecting in {self.config.reconnect_delay_seconds}s...")
                    is_reconnect = True
                    await asyncio.sleep(self.config.reconnect_delay_seconds)

        finally:
            mic_stream.close()
            player.close()
            media_player.close()
            if pa is not None:
                try:
                    pa.terminate()
                except Exception:
                    pass
            print("🛑 [Voice Agent Shutdown Complete]")

    async def record_user_message(self, session_id: str, text: str) -> None:
        """
        Records an explicit user dialogue message into memory and triggers background extraction.
        """
        if self.memory and text:
            await self.memory.add_message(session_id, role="user", content=text)
            if self.entity_extractor:
                asyncio.create_task(
                    self.entity_extractor.extract_and_update(session_id, self.memory)
                )

    async def get_user_context(self, session_id: str) -> str:
        """
        Returns the formatted user profile context prompt for dynamic prompt injection.
        """
        if self.memory:
            return await self.memory.build_context_prompt(session_id)
        return ""

    async def build_effective_system_instruction(self, session_id: str = "default") -> str:
        """
        Dynamically composes the complete system instruction for Gemini Live by merging:
          1. Base system instruction from VoiceConfig
          2. Known user profile memory context from BaseMemoryStore
          3. Dynamic per-tool prompt rules and usage instructions from registered BaseTools
        """
        instruction = self.config.system_instruction

        # 1. Inject known user profile context from memory
        if self.memory:
            user_context = await self.memory.build_context_prompt(session_id)
            if user_context:
                instruction = f"{instruction}\n\n{user_context}"

        # 2. Inject active tool instructions from tool registry
        tools_instruction = self.tools.build_tools_instruction()
        if tools_instruction:
            instruction = f"{instruction}\n\n{tools_instruction}"

        return instruction


