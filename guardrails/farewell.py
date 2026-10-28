"""
Conversational Farewell & Termination Detector.
Detects dialogue endings and farewell phrases across multiple languages
while distinguishing polite gratitude/confirmation from true session termination.
"""

from __future__ import annotations

import re


class FarewellDetector:
    """
    Configurable detector for session completion / farewell intents.
    Prevents false terminations when users express gratitude (e.g. 'thank you', 'thanks', 'આભાર').
    """

    DEFAULT_TERMINATION_MARKERS = [
        "[session_closed]",
        "[session_end]",
        "[end_conversation]",
    ]

    DEFAULT_FAREWELL_PHRASES = [
        "goodbye",
        "bye",
        "farewell",
        "see you",
        "see ya",
        "alvida",
        "अलविदा",
        "આવજો",
        "રામ રામ આવજો",
        "ચલો આવજો",
        "ચાલો આવજો",
        "ફરી મળીશું",
        "ફરી વાત કરીશું",
        "સત્ર પૂર્ણ",
    ]

    DEFAULT_GRATITUDE_PHRASES = [
        "thank you",
        "thanks",
        "thank u",
        "welcome",
        "you're welcome",
        "આભાર",
        "ધન્યવાદ",
        "થેન્ક યુ",
        "ઓકે આભાર",
        "શુક્રિયા",
    ]

    def __init__(
        self,
        farewell_phrases: list[str] | None = None,
        gratitude_phrases: list[str] | None = None,
        termination_markers: list[str] | None = None,
    ) -> None:
        self.farewell_phrases = [
            p.lower() for p in (farewell_phrases if farewell_phrases is not None else self.DEFAULT_FAREWELL_PHRASES)
        ]
        self.gratitude_phrases = [
            g.lower() for g in (gratitude_phrases if gratitude_phrases is not None else self.DEFAULT_GRATITUDE_PHRASES)
        ]
        self.termination_markers = [
            m.lower() for m in (termination_markers if termination_markers is not None else self.DEFAULT_TERMINATION_MARKERS)
        ]

    def is_farewell(self, text: str) -> bool:
        """
        Determines whether the given text signals a session completion.
        Returns True only if clear farewell intent is found and not overridden by gratitude.
        """
        if not text:
            return False

        t = text.lower()

        # 1. Explicit protocol termination marker
        if any(m in t for m in self.termination_markers):
            return True

        # 2. Check gratitude exclusion
        # If text contains gratitude and does not contain explicit farewells, do NOT terminate
        has_explicit_farewell = any(f in t for f in self.farewell_phrases)
        if not has_explicit_farewell:
            if any(g in t for g in self.gratitude_phrases):
                return False

        # 3. Word boundary check for short words like 'bye'
        if re.search(r"\bbye\b", t):
            return True

        # 4. Check farewell phrase list
        return has_explicit_farewell
