"""
Media Audio Player (MP3/WAV playback).
Plays pre-recorded audio files on speakers with headless Linux/Raspberry Pi support,
background thread monitoring, and instant interruption cancellation.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable

# Configure headless SDL before importing pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

try:
    # pyright: ignore[reportMissingImports]
    import pygame  # type: ignore[import-untyped, import-not-found]
    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None  # type: ignore[assignment]
    PYGAME_AVAILABLE = False


class MediaAudioPlayer:
    """
    Manages physical audio file playback (e.g. farmer experiences, music, prompts)
    with non-blocking execution and event hooks.
    """

    def __init__(self, media_dir: str = "audio_experiences") -> None:
        self.media_dir = media_dir
        self._playing_event = threading.Event()
        self._initialized = False

    def _init_pygame(self) -> None:
        if self._initialized:
            return
        if PYGAME_AVAILABLE and pygame is not None:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                self._initialized = True
            except Exception as exc:
                print(f"⚠️  [MediaAudioPlayer] pygame mixer init warning: {exc}")

    def play(
        self,
        audio_url_or_path: str,
        on_complete: Callable[[], None] | None = None,
    ) -> bool:
        """
        Plays an MP3 or WAV file asynchronously on a background thread.
        Returns True if playback started successfully.
        """
        if not PYGAME_AVAILABLE or pygame is None:
            print("⚠️  [MediaAudioPlayer] pygame unavailable — skipping audio file playback")
            return False

        self._init_pygame()

        # Extract filename if URL format (e.g., /api/experiences/audio/valsad.mp3)
        clean_path = audio_url_or_path.replace("\\", "/")
        filename = clean_path.split("/")[-1]
        
        # Check both direct path and media_dir path
        filepath = clean_path if os.path.isfile(clean_path) else os.path.join(self.media_dir, filename)

        if not os.path.isfile(filepath):
            print(f"⚠️  [MediaAudioPlayer] File not found: {filepath}")
            return False

        try:
            self._init_pygame()
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
            self._playing_event.set()
            print(f"▶️  [MediaAudioPlayer] Playing audio file: {filename}")

            def _monitor() -> None:
                while pygame is not None and pygame.mixer.music.get_busy():
                    time.sleep(0.15)
                if self._playing_event.is_set():
                    self._playing_event.clear()
                    if on_complete:
                        try:
                            on_complete()
                        except Exception as cb_err:
                            print(f"⚠️  [MediaAudioPlayer] Completion callback error: {cb_err}")

            threading.Thread(target=_monitor, daemon=True).start()
            return True
        except Exception as exc:
            print(f"⚠️  [MediaAudioPlayer] Playback error: {exc}")
            self._playing_event.clear()
            return False

    def is_playing(self) -> bool:
        """Returns True if a media file is actively playing."""
        return self._playing_event.is_set()

    def stop(self) -> None:
        """Immediately stops media audio playback."""
        if not PYGAME_AVAILABLE or pygame is None:
            return
        self._playing_event.clear()
        try:
            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                print("⏹️  [MediaAudioPlayer] Audio playback stopped.")
        except Exception:
            pass

    def close(self) -> None:
        """Closes the audio subsystem."""
        self.stop()
        if PYGAME_AVAILABLE and pygame is not None:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
