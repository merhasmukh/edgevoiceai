"""
Gemini Embeddings Generator.
Produces high-quality 768-dimensional dense vectors using Google Gemini API.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore[assignment]
    HTTPX_AVAILABLE = False

from edgevoiceai.rag.embeddings.base import BaseEmbeddingModel


class GeminiEmbeddings(BaseEmbeddingModel):
    """
    Asynchronous embedding generator powered by Google Gemini API.
    Reuses persistent HTTP connections to eliminate TLS handshake latency on edge devices.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-embedding-001",
        timeout: float = 25.0,
    ) -> None:
        self.api_key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
        self.model = model
        self.timeout = timeout
        self._client: Any = None
        self._cache: dict[str, list[float]] = {}

    def _get_client(self) -> Any:
        if not HTTPX_AVAILABLE or httpx is None:
            raise RuntimeError("httpx is required for GeminiEmbeddings. Install with: pip install httpx")
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                http2=False,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
        return self._client

    async def embed_text(self, text: str) -> list[float]:
        """Generates embedding vector for a given text."""
        clean = (text or "").strip()
        if not clean:
            return [0.0] * 768

        if clean in self._cache:
            return self._cache[clean]

        if not self.api_key:
            raise ValueError("Gemini API key is required for GeminiEmbeddings.")

        client = self._get_client()
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent"
            f"?key={self.api_key}"
        )
        payload = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": clean}]},
        }

        # Up to 2 retries on rate limits
        for attempt in range(2):
            try:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    values = data.get("embedding", {}).get("values", [])
                    if values:
                        self._cache[clean] = values
                        if len(self._cache) > 200:
                            self._cache.pop(next(iter(self._cache)))
                        return values
                elif resp.status_code == 429 and attempt == 0:
                    await asyncio.sleep(1.0)
                    continue
                else:
                    raise RuntimeError(f"Gemini embedding API error {resp.status_code}: {resp.text}")
            except Exception as exc:
                if attempt == 1:
                    raise exc
                await asyncio.sleep(0.5)

        return [0.0] * 768

    async def embed_query(self, query: str) -> list[float]:
        """Generates embedding vector for a search query."""
        return await self.embed_text(query)

    async def close(self) -> None:
        """Closes the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
