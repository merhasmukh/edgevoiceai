# VoiceAgent & Configuration Reference

This section documents the core runtime orchestration engine: [`VoiceAgent`](file:///Users/galaxyofai/Desktop/Github/khedut-voice-ai/edgevoiceai/agent.py) and the system-wide configuration dataclasses [`VoiceConfig`](file:///Users/galaxyofai/Desktop/Github/khedut-voice-ai/edgevoiceai/config.py) and [`AudioConfig`](file:///Users/galaxyofai/Desktop/Github/khedut-voice-ai/edgevoiceai/config.py).

---

## VoiceConfig & AudioConfig

::: edgevoiceai.config
    options:
      show_root_heading: true
      members:
        - VoiceConfig
        - AudioConfig
        - WhatsAppConfig
        - RAGConfig

---

## VoiceAgent

::: edgevoiceai.agent
    options:
      show_root_heading: true
      members:
        - VoiceAgent

