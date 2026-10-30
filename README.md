# EdgeVoiceAI

<p align="center">
  <strong>A modular, full-duplex conversational voice AI framework for Python 3.10+</strong><br>
  <em>Built for edge devices (Raspberry Pi, Linux) and laptops (macOS, Windows) powered by Google Gemini Live.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/edgevoiceai/"><img src="https://img.shields.io/pypi/v/edgevoiceai.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/edgevoiceai/"><img src="https://img.shields.io/pypi/pyversions/edgevoiceai.svg" alt="Python Versions"></a>
  <a href="https://github.com/merhasmukh/edgevoiceai/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://merhasmukh.github.io/edgevoiceai/"><img src="https://img.shields.io/badge/docs-MkDocs-teal.svg" alt="Documentation"></a>
</p>

---

## 🌟 Overview

**EdgeVoiceAI** is a developer-first Python library designed to build low-latency, full-duplex conversational voice assistants from scratch. It connects physical hardware microphones and speakers directly with Google Gemini's real-time bidirectional streaming API (`bidiGenerateContent`), enabling natural spoken interactions with instant turn-taking and conversational barge-in interruptions.

Unlike traditional voice wrappers that chain disjointed Speech-To-Text (STT) $\rightarrow$ LLM $\rightarrow$ Text-To-Speech (TTS) pipelines with noticeable latency, EdgeVoiceAI operates over a single, continuous, bidirectional audio WebSocket connection.

---

## 🏗️ Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/merhasmukh/edgevoiceai/main/docs/assets/architecture.svg" alt="EdgeVoiceAI Architecture" width="100%">
</p>

```text
+-----------------------------------------------------------------------+
|        Client Device (macOS / Linux / Windows / Raspberry Pi)         |
|                                                                       |
|   [🎙️ MicrophoneStream (16kHz Native Mono PCM)]                       |
|                │                                                      |
|                ▼                                                      |
|   [⚡ VoiceActivityDetector (Energy Gating & Syllable Onset)]          |
|                │                                                      |
|                ▼                                                      |
|   [📤 WebSockets Client (realtimeInput audio frames)]                 |
|                │                                      ▲               |
+────────────────┼──────────────────────────────────────┼───────────────+
                 │                                      │
                 │ 16kHz PCM Stream                     │
                 ▼                                      │
+───────────────────────────────────────────────────────┼───────────────+
|                  Google Gemini Live Service           │               |
|                 (gemini-2.0-flash-exp / live)         │               |
+───────────────────────────────────────────────────────┼───────────────+
                 │                                      │
                 │ 24kHz Streaming Audio & Tool Calls   │
                 ▼                                      │
+────────────────┼──────────────────────────────────────┼───────────────+
|                ▼                                                      |
|   [📥 WebSockets Receiver (serverContent)]                            |
|          │                                                            |
|          ├─────────────────────────────┐                              |
|          ▼                             ▼                              |
|   [🔊 AudioPlayer (24kHz PCM)]   [🔧 Tool Dispatcher (ToolRegistry)]  |
|    • <50ms Instant Barge-In Cancel             │                      |
|    • Acoustic Echo Suppression                 ▼                      |
|                                  [🧠 Memory & Vector Stores]          |
|                                     (SQLite / Qdrant / Pinecone)      |
+-----------------------------------------------------------------------+
```

---

## ✨ Key Features

- **⚡ Ultra-Low Latency & Instant Barge-In**: Real-time 16kHz continuous streaming. When you speak while the assistant responds, playback clears instantly (<50ms) to yield the floor naturally.
- **🧩 Self-Describing Tools with Prompt Composition**: Each tool declares its own JSON schema, execution callback, and prompt etiquette instructions (`prompt_instruction`). The framework automatically synthesizes these instructions into the agent's system prompt.
- **🎙️ Cross-Platform Audio Engine**: Powered by PortAudio with automatic hardware enumeration across macOS (CoreAudio), Linux / Raspberry Pi (ALSA/PulseAudio), and Windows (WASAPI/MME).
- **🔍 Plug-and-Play RAG Integration**: Built-in `RAGRetriever` with chunking and native vector store backends for `InMemoryVectorStore`, `QdrantVectorStore`, and `PineconeVectorStore`.
- **🛡️ Conversational Guardrails**: Turn-aware farewell detectors, topic safety guards, and entity memory management.
- **📲 Built-in Notification Tools**: Two-turn confirmation WhatsApp dispatch tool with phone number normalization.

---

## 📦 Installation

### 1. Install System Audio Dependencies (PortAudio)

EdgeVoiceAI uses PortAudio for native, cross-platform audio streaming.

- **macOS**:
  ```bash
  brew install portaudio
  ```
- **Linux / Raspberry Pi**:
  ```bash
  sudo apt-get update
  sudo apt-get install -y portaudio19-dev libasound2-dev python3-pyaudio
  ```
- **Windows**:
  - **Recommended Python**: **`3.10`, `3.11`, or `3.12`**. Official pre-compiled binary wheels for `pyaudio` are provided automatically—no compilers required.
  - **Notice for Python 3.14+**: Python 3.14 does not have pre-built wheels on PyPI yet. If you see `Microsoft Visual C++ 14.0 or greater is required`, either:
    1. *(Recommended)* Create your virtual environment with Python 3.11 or 3.12: `py -3.11 -m venv venv`
    2. Or install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (select *"Desktop development with C++"*).

### 2. Install EdgeVoiceAI via pip

```bash
# Core package
pip install edgevoiceai

# With specific optional extras
pip install "edgevoiceai[qdrant]"    # Qdrant vector store support
pip install "edgevoiceai[pinecone]"  # Pinecone vector store support
pip install "edgevoiceai[media]"     # Pygame audio sound effects

# Or install all extras
pip install "edgevoiceai[all]"
```

---

## 🚀 5-Minute Quickstart

### 1. Set Your Gemini API Key

```bash
export GEMINI_API_KEY="your-google-ai-studio-api-key"
```

### 2. Run a Full-Duplex Voice Assistant

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

# Load GEMINI_API_KEY from .env
load_dotenv()


# Define a custom tool with prompt instructions
class BookAppointmentTool(BaseTool):
    @property
    def name(self) -> str:
        return "book_appointment"

    @property
    def description(self) -> str:
        return "Schedules a consultation appointment for the user."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Customer name"},
                "date": {"type": "string", "description": "Date of appointment (YYYY-MM-DD)"},
                "time": {"type": "string", "description": "Time (e.g. 10:00 AM)"},
            },
            "required": ["name", "date", "time"],
        }

    @property
    def prompt_instruction(self) -> str:
        return (
            "When scheduling, ask for the customer's name, desired date, and preferred time. "
            "Always verbally confirm all details before executing book_appointment."
        )

    async def execute(self, params: dict, context: ToolExecutionContext) -> ToolResult:
        name = params.get("name")
        date = params.get("date")
        time = params.get("time")
        # In a real app, write to database or calendar API
        return ToolResult(
            status="success",
            message=f"Appointment booked for {name} on {date} at {time}.",
            data={"status": "confirmed", "booking_id": "APT-1001"},
        )


async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set the GEMINI_API_KEY environment variable.")

    # Configure the voice session
    config = VoiceConfig(
        api_key=api_key,
        voice="Puck",  # Puck, Charon, Aoede, Fenrir, Kore
        system_instruction=(
            "You are a friendly, concise voice assistant. "
            "Keep responses natural, conversational, and direct."
        ),
    )

    # Register tools and memory
    registry = ToolRegistry()
    registry.register(BookAppointmentTool())
    memory = InMemoryMemoryStore()

    agent = VoiceAgent(config=config, tools=registry, memory=memory)

    print("🎙️ Starting Voice Assistant... Speak into your microphone!")
    try:
        await agent.run_edge()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 📖 Documentation

Comprehensive documentation, tutorials, architecture deep-dives, and API references are available at:
**[https://merhasmukh.github.io/edgevoiceai/](https://merhasmukh.github.io/edgevoiceai/)**

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
