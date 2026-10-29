"""
Audio Device Management for Voice AI Framework.
Supports cross-platform hardware discovery and split input/output device routing
on Raspberry Pi, Linux, macOS, and Windows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Resilient pyaudio import with linter type ignore
try:
    # pyright: ignore[reportMissingModuleSource]
    import pyaudio  # type: ignore[import-untyped]
    PYAUDIO_AVAILABLE = True
except ImportError:
    pyaudio = None  # type: ignore[assignment]
    PYAUDIO_AVAILABLE = False


@dataclass(slots=True)
class AudioDeviceInfo:
    """Represents a discovered audio device with hardware characteristics."""
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    host_api: str = "Unknown"

    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        return self.max_output_channels > 0


class AudioDeviceManager:
    """
    Cross-platform manager for querying, matching, and configuring
    independent input and output audio hardware.
    """

    @classmethod
    def list_devices(cls) -> list[AudioDeviceInfo]:
        """Returns all audio devices discovered on the host system."""
        if not PYAUDIO_AVAILABLE or pyaudio is None:
            return []

        pa = pyaudio.PyAudio()
        devices: list[AudioDeviceInfo] = []
        try:
            count = pa.get_device_count()
            for i in range(count):
                try:
                    info = pa.get_device_info_by_index(i)
                    host_api_idx = int(info.get("hostApi", 0))
                    host_api_info = pa.get_host_api_info_by_index(host_api_idx)
                    api_name = host_api_info.get("name", "Unknown") if host_api_info else "Unknown"

                    devices.append(
                        AudioDeviceInfo(
                            index=i,
                            name=str(info.get("name", f"Device {i}")),
                            max_input_channels=int(info.get("maxInputChannels", 0)),
                            max_output_channels=int(info.get("maxOutputChannels", 0)),
                            default_sample_rate=float(info.get("defaultSampleRate", 16000.0)),
                            host_api=str(api_name),
                        )
                    )
                except Exception:
                    continue
        finally:
            pa.terminate()

        return devices

    @classmethod
    def find_device_index(
        cls,
        spec: int | str | None,
        is_input: bool = True,
    ) -> int | None:
        """
        Resolves an audio device specification (integer index or fuzzy name string)
        to an actual PortAudio device index.

        Supports regex and substring matching (e.g., 'USB', 'bcm2835', 'AirPods').
        Returns None to use the system default if spec is None or device not found.
        """
        if spec is None:
            return None

        # Direct integer index
        if isinstance(spec, int):
            return spec

        spec_str = str(spec).strip()
        if not spec_str:
            return None

        # Check if spec is an integer formatted as string
        if spec_str.isdigit():
            return int(spec_str)

        all_devices = cls.list_devices()
        candidates = [d for d in all_devices if (d.is_input if is_input else d.is_output)]

        # 1. Exact case-insensitive match
        for dev in candidates:
            if dev.name.lower() == spec_str.lower():
                return dev.index

        # 2. Substring match
        for dev in candidates:
            if spec_str.lower() in dev.name.lower():
                return dev.index

        # 3. Regex match
        try:
            pattern = re.compile(spec_str, re.IGNORECASE)
            for dev in candidates:
                if pattern.search(dev.name):
                    return dev.index
        except re.error:
            pass

        return None

    @classmethod
    def find_best_input_device(cls, pa: Any, preferred_index: int | None = None) -> int | None:
        """
        Finds the most suitable working audio input device index (USB mic).
        Falls back to any input-capable device if the default is not marked.
        """
        count = pa.get_device_count()
        if count == 0:
            return None

        if preferred_index is not None:
            try:
                info = pa.get_device_info_by_index(preferred_index)
                if int(info.get("maxInputChannels", 0)) > 0:
                    return preferred_index
            except Exception:
                pass

        # 1. Try system default
        try:
            default_in = pa.get_default_input_device_info()
            if int(default_in.get("maxInputChannels", 0)) > 0:
                return int(default_in["index"])
        except Exception:
            pass

        # 2. Search for USB / mic device
        usb_candidate = None
        first_input = None
        for i in range(count):
            try:
                info = pa.get_device_info_by_index(i)
                if int(info.get("maxInputChannels", 0)) > 0:
                    if first_input is None:
                        first_input = i
                    name_lower = str(info.get("name", "")).lower()
                    if "usb" in name_lower or "mic" in name_lower:
                        usb_candidate = i
                        break
            except Exception:
                continue

        return usb_candidate if usb_candidate is not None else first_input

    @classmethod
    def find_best_output_device(cls, pa: Any, preferred_index: int | None = None) -> int | None:
        """
        Finds the most suitable working audio output device index (Speaker / Bluetooth).
        """
        count = pa.get_device_count()
        if count == 0:
            return None

        if preferred_index is not None:
            try:
                info = pa.get_device_info_by_index(preferred_index)
                if int(info.get("maxOutputChannels", 0)) > 0:
                    return preferred_index
            except Exception:
                pass

        # 1. Try system default
        try:
            default_out = pa.get_default_output_device_info()
            if int(default_out.get("maxOutputChannels", 0)) > 0:
                return int(default_out["index"])
        except Exception:
            pass

        # 2. Search for preferred output devices (Pulse, bluez, default, headphone, speaker)
        first_output = None
        preferred_candidate = None
        for i in range(count):
            try:
                info = pa.get_device_info_by_index(i)
                if int(info.get("maxOutputChannels", 0)) > 0:
                    if first_output is None:
                        first_output = i
                    name_lower = str(info.get("name", "")).lower()
                    if any(k in name_lower for k in ("pulse", "default", "bluez", "headphone", "speaker", "usb", "bcm2835", "jack", "audio", "alsa")):
                        preferred_candidate = i
                        break
            except Exception:
                continue

        return preferred_candidate if preferred_candidate is not None else first_output

    @classmethod
    def print_diagnostics(cls) -> None:
        """Prints a user-friendly terminal table of available audio devices."""
        devices = cls.list_devices()
        print("\n" + "=" * 65)
        print("🔊 [VOICE FRAMEWORK - AUDIO HARDWARE DIAGNOSTICS]")
        print("=" * 65)

        inputs = [d for d in devices if d.is_input]
        outputs = [d for d in devices if d.is_output]

        print("\n🎤 Microphone / Input Devices:")
        if not inputs:
            print("  (None detected)")
        for d in inputs:
            print(f"  • [{d.index}] {d.name} ({int(d.default_sample_rate)}Hz, {d.host_api})")

        print("\n🔈 Speaker / Output Devices:")
        if not outputs:
            print("  (None detected)")
        for d in outputs:
            print(f"  • [{d.index}] {d.name} ({int(d.default_sample_rate)}Hz, {d.host_api})")
        print("=" * 65 + "\n")
