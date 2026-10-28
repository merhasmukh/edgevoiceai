"""
Audio processing and hardware I/O subsystem for Voice AI Framework.
"""

from edgevoiceai.audio.device import AudioDeviceManager, AudioDeviceInfo
from edgevoiceai.audio.vad import VADConfig, VoiceActivityDetector, compute_rms
from edgevoiceai.audio.player import AudioPlayer
from edgevoiceai.audio.media import MediaAudioPlayer
from edgevoiceai.audio.stream import MicrophoneStream

__all__ = [
    "AudioDeviceManager",
    "AudioDeviceInfo",
    "VADConfig",
    "VoiceActivityDetector",
    "compute_rms",
    "AudioPlayer",
    "MediaAudioPlayer",
    "MicrophoneStream",
]
