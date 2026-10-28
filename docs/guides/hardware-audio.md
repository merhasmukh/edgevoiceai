# Hardware Audio & Edge Device Routing

`edgevoiceai` is built from the ground up for edge hardware (Raspberry Pi, mini PCs, robotics) as well as development laptops (macOS, Linux, Windows).

One of its most powerful capabilities is **Independent Device Routing**: you can route audio input to one physical device (e.g. an omnidirectional USB microphone array) and audio output to a completely different physical device (e.g. an I2S DAC, 3.5mm speaker, or Bluetooth headset).

---

## 1. Inspecting Connected Audio Devices

To list all physical audio input and output devices connected to your system:

```python
from edgevoiceai import AudioDeviceManager

devices = AudioDeviceManager.list_devices()
for d in devices:
    kind = "🎙️ INPUT" if d.max_input_channels > 0 else "🔊 OUTPUT"
    print(f"[{d.index}] {kind}: {d.name} (Rate: {d.default_sample_rate}Hz)")
```

---

## 2. Configuring Audio Routing in `VoiceConfig`

You can configure hardware devices using either:
- **Device Index** (`int`): Exact hardware index from `list_devices()`.
- **Device Name Substring** (`str`): Case-insensitive fuzzy matching (e.g. `"USB PnP"`, `"Built-in"`, `"JBL"`, `"AirPods"`, `"hw:1,0"`).
- **`None`**: Uses system default audio input/output.

```python
from edgevoiceai import VoiceConfig

config = VoiceConfig(
    api_key="...",
    # Route input to external USB microphone by name substring
    input_device="USB PnP Audio Device",
    # Route output to external speakers by index or name
    output_device="Built-in Output",
    # Native 16kHz direct capture
    mic_capture_rate=16000,
    speaker_sample_rate=24000,
)
```

---

## 3. Platform-Specific Setup Guides

=== "macOS (Apple Silicon & Intel)"
    ### CoreAudio & Microphone Permissions
    1. **Microphone Access**: Grant Terminal, VS Code, or your IDE microphone access under **System Settings → Privacy & Security → Microphone**.
    2. **Lazy Audio Device Initialization**: On macOS, opening both PyAudio and Pygame's SDL audio mixer at the exact same instant can cause a low-level CoreAudio mutex lock. The framework automatically initializes `MediaAudioPlayer` lazily on playback to keep signal delivery and `Ctrl+C` handling responsive.
    3. **Device Names**: Use substring matches like `"MacBook Air Microphone"`, `"AirPods"`, or `"BlackHole"`.

=== "Raspberry Pi & Linux (Edge Devices)"
    ### System Prerequisites
    Install PortAudio and ALSA development libraries:
    ```bash
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev libasound2-dev python3-pyaudio alsa-utils
    ```

    ### Checking ALSA Device Cards
    ```bash
    arecord -l   # Lists capture cards (e.g. card 1: USB PnP Sound Device)
    aplay -l     # Lists playback cards (e.g. card 0: bcm2835 Headphones, or I2S DAC)
    ```

    ### Automatic PortAudio Negotiation
    Many cheap USB microphones on Raspberry Pi reject 16,000 Hz if opened directly in native mode because their internal DSP only supports 44,100 Hz or 48,000 Hz.
    
    `MicrophoneStream` automatically performs **Hardware Sample Rate Negotiation**:
    1. Tests native `16000 Hz` capture (fastest, zero CPU overhead).
    2. If the USB soundcard rejects 16kHz, it falls back to `48000 Hz` or `44100 Hz` and resamples automatically using `audioop.ratecv`.

    ### Autostart on Boot (Systemd Service)
    Create `/etc/systemd/system/voice-agent.service`:
    ```ini
    [Unit]
    Description=Voice AI Assistant Edge Service
    After=network-online.target sound.target

    [Service]
    Type=simple
    User=pi
    WorkingDirectory=/home/pi/voice-ai
    ExecStart=/home/pi/voice-ai/venv/bin/python main.py
    Restart=always
    RestartSec=5

    [Install]
    WantedBy=multi-user.target
    ```
    Enable and start the service:
    ```bash
    sudo systemctl daemon-reload
    sudo systemctl enable voice-agent.service
    sudo systemctl start voice-agent.service
    ```

=== "Windows 10 & 11"
    ### WASAPI & MME Audio Drivers
    1. **Microphone Privacy**: Enable microphone permissions in **Settings → Privacy & Security → Microphone**.
    2. **Device Naming**: Windows devices often appear with driver suffixes like `"Microphone (Realtek High Definition Audio)"` or `"Headset Earphone (USB Audio Device)"`.
    3. **Substrings**: Pass `"Realtek"` or `"USB Audio"` directly to `input_device` and `output_device`.

---

## 4. Latency & Buffer Calibration

| Parameter | Default | Recommended Value | Impact |
|---|---|---|---|
| `mic_capture_rate` | `16000` | `16000` | Native capture eliminates software resampling latency. |
| `speaker_sample_rate` | `24000` | `24000` | Matches Gemini Live native 24kHz PCM output. |
| `frames_per_buffer` | `512` | `512` | 32ms capture packets / 21ms playback chunks. |
| `noise_gate_threshold` | `250` | `250` (indoor) / `1200+` (noisy/outdoor) | RMS audio energy required to open microphone. |
| `ai_interrupt_threshold` | `2800` | `2800` (speakers) / `1800` (headphones) | Prevents acoustic loopback while allowing barge-in. |
| `hang_frames` | `15` | `12-18` (~380-570ms) | Ambient audio frames sent after speech ends for turn detection. |


