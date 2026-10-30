# 5-Minute Quickstart Guide

This guide walks you through building and running your first production-ready, full-duplex conversational voice assistant from scratch using `edgevoiceai`.

---

## 1. Prerequisites

- **Python**: `>= 3.10` (Python `3.10`, `3.11`, or `3.12` recommended on Windows)
- **Google Gemini API Key**: [Get an API key from Google AI Studio](https://aistudio.google.com/)
- **Microphone & Speaker**: Built-in laptop audio, USB headset, or separate USB sound cards on Raspberry Pi.

---

## 2. System Audio Dependencies

`edgevoiceai` uses **PortAudio** for cross-platform low-latency audio capture and playback.

=== "macOS"
    ```bash
    brew install portaudio
    pip install pyaudio
    ```

=== "Linux / Raspberry Pi"
    ```bash
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev libasound2-dev python3-pyaudio
    pip install pyaudio
    ```

=== "Windows"
    ```powershell
    # On Python 3.10, 3.11, or 3.12 (pre-compiled binary wheel):
    pip install pyaudio
    ```

    !!! tip "Windows Python Version Notice"
        `PyAudio` provides official pre-compiled binary wheels for **Python 3.10, 3.11, and 3.12**.

        If using **Python 3.14+**, pre-built wheels are not yet available on PyPI, and pip will try to compile from source. If you encounter `Microsoft Visual C++ 14.0 or greater is required`:
        
        - **Option A (Recommended)**: Create your virtual environment with Python 3.11 or 3.12:
          ```powershell
          py -3.11 -m venv venv
          .\venv\Scripts\Activate.ps1
          pip install edgevoiceai
          ```
        - **Option B**: Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select *"Desktop development with C++"*.

---

## 3. Environment Configuration

Create a `.env` file in your project root or export the environment variable:

```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
```

---

## 4. Complete Runnable Code

Save the following code as `assistant.py`:

```python
import asyncio
import os
from dotenv import load_dotenv

from edgevoiceai import (
    BaseTool,
    InMemoryMemoryStore,
    ToolExecutionContext,
    ToolRegistry,
    ToolResult,
    VoiceAgent,
    VoiceConfig,
)

# Load environment variables
load_dotenv()


# ── 1. Define a Custom Tool with its Own Prompt Template ─────────────────────
class BookAppointmentTool(BaseTool):
    """Schedules a customer consultation appointment."""

    @property
    def name(self) -> str:
        return "book_appointment"

    @property
    def description(self) -> str:
        return "Books a customer appointment for a requested service and date."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "OBJECT",
            "properties": {
                "service": {
                    "type": "STRING",
                    "description": "The service type requested by the user.",
                },
                "date_time": {
                    "type": "STRING",
                    "description": "The requested date or time (e.g. 'tomorrow at 3 PM').",
                },
            },
            "required": ["service", "date_time"],
        }

    @property
    def prompt_instruction(self) -> str:
        """
        Self-describing prompt template.
        The framework automatically composes this into the AI's system prompt!
        """
        return (
            "• APPOINTMENT BOOKING PROTOCOL (`book_appointment`):\n"
            "  1. Only invoke this tool when the user explicitly requests to book a visit.\n"
            "  2. Verbally ask for their preferred day and service type before booking.\n"
            "  3. Confirm the appointment in 1 concise spoken sentence."
        )

    async def execute(self, args: dict, context: ToolExecutionContext) -> ToolResult:
        service = args.get("service", "Consultation")
        date_time = args.get("date_time", "soon")
        booking_id = "APT-992"

        print(f"\n📅 [Booking System] Confirmed: {service} for {date_time} (ID: {booking_id})")

        return ToolResult(
            status="confirmed",
            message=f"Appointment for {service} successfully scheduled for {date_time}. Booking ID is {booking_id}.",
            data={"booking_id": booking_id, "service": service, "date_time": date_time},
        )


# ── 2. Main Voice Application Runner ─────────────────────────────────────────
async def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable is not set.")
        return

    # A. Register Tools
    tools = ToolRegistry()
    tools.register(BookAppointmentTool())

    # B. Configure Voice Agent
    config = VoiceConfig(
        api_key=api_key,
        model="gemini-3.1-flash-live-preview",  # Live bidirectional WebSocket model
        voice="Puck",                          # Voice name: Puck, Charon, Aoede, Fenrir, Kore
        system_instruction=(
            "You are Aria, an intelligent and courteous Voice AI assistant. "
            "Speak naturally, concisely (1-2 sentences), and warmly in English."
        ),
        # Hardware audio defaults (set None for system default mic & speaker)
        input_device=None,
        output_device=None,
    )

    # C. Instantiate Orchestrator
    agent = VoiceAgent(
        config=config,
        tools=tools,
        memory=InMemoryMemoryStore(),
    )

    # D. Register Event Callbacks
    agent.on_turn_complete = lambda text: print(f"\n🤖 [AI]: {text}")
    agent.on_interrupted = lambda: print("\n⚡ [User Interrupted AI]")
    agent.on_tool_executed = lambda name, args, res: print(f"🔧 [Tool Executed]: {name}")

    # Inspect the dynamically composed system prompt
    session_id = "user_quickstart_session"
    effective_prompt = await agent.build_effective_system_instruction(session_id)
    print("\n" + "=" * 60)
    print("📝 DYNAMICALLY COMPOSED SYSTEM PROMPT:")
    print("=" * 60)
    print(effective_prompt)
    print("=" * 60 + "\n")

    # E. Start Real-Time Voice Loop
    print("🎙️ Starting full-duplex voice loop. Speak into your microphone...")
    print("   (Press Ctrl+C to stop)")
    try:
        await agent.run_edge(session_id=session_id)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n👋 Stopping voice assistant...")
        agent.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
```

---

## 5. How It Works

### Step A: Dynamic Prompt Composition
Notice that you did not have to hardcode tool guidelines into your persona prompt. When you implement:
```python
@property
def prompt_instruction(self) -> str:
    ...
```
`ToolRegistry.build_tools_instruction()` aggregates all active tool rules and `VoiceAgent` automatically injects them into the Gemini handshake. If you add or remove tools dynamically, the system prompt stays 100% in sync.

### Step B: Low-Latency Streaming & VAD
`VoiceAgent.run_edge()` opens a native 16kHz microphone stream. The built-in `VoiceActivityDetector` monitors audio energy (RMS) and keeps syllable onsets intact so words like *"Yes"* or *"Hey"* are never clipped.

### Step C: Seamless Barge-In (Interruption)
If the assistant is speaking through your speakers and you start talking, the framework detects your voice streak and clears the speaker queue in **< 50 milliseconds**, seamlessly yielding the turn back to you.

---

## 6. Run Your Assistant

Execute the script:
```bash
python assistant.py
```

Expected output:
```text
============================================================
🎙️  [VOICE AGENT INITIALIZED]
============================================================
  Model            : gemini-3.1-flash-live-preview
  Voice            : Puck
  Mic Device       : Default
  Speaker Device   : Default
  Interrupt Thresh : 2500
  Noise Gate Thresh: 250
============================================================

🔗 Connecting to Gemini Live (gemini-3.1-flash-live-preview)...
✅ [Gemini Live Connected & Ready!]
🎙️ [Greeting Triggered] — Assistant is introducing itself.

🤖 [AI]: Hello! I am ready to assist you.

🎤 [Voice detected: RMS 410]
🤖 [AI]: I can certainly help you book an appointment. Which day and service would you prefer?
```

---

## 7. Next Steps

- **[Deep-Dive: Custom Tools & Prompt Templates](guides/custom-tools.md)**: Master the 2-turn confirmation gate and Just-In-Time instructions.
- **[Hardware Audio & Edge Routing](guides/hardware-audio.md)**: Configure USB microphones and I2S DACs on Raspberry Pi or Linux.
- **[WhatsApp Notifications](guides/whatsapp-integration.md)**: Send summaries directly to users' phones with phone verification.

