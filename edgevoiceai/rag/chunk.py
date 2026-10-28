"""
Document chunk representation for Voice AI RAG pipeline.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentChunk:
    """
    Standardized knowledge chunk retrieved from any vector database.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    text: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    audio_url: str | None = None
    vector_store: str = "Unknown"

    def to_dict(self) -> dict[str, Any]:
        """Converts chunk to dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
            "audio_url": self.audio_url,
            "vector_store": self.vector_store,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentChunk:
        """Constructs a DocumentChunk from dictionary data."""
        return cls(
            id=str(data.get("id", str(uuid.uuid4()))),
            title=str(data.get("title", "")),
            text=str(data.get("text", "")),
            score=float(data.get("score", 0.0)),
            metadata=dict(data.get("metadata", {})),
            audio_url=data.get("audio_url"),
            vector_store=str(data.get("vector_store", "Unknown")),
        )
