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
    from httpx import HTTPStatusError
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore[assignment]
    HTTPStatusError = Exception  # type: ignore[assignment, misc]
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
        dimension: int = 768,
        timeout: float = 25.0,
    ) -> None:
        self.api_key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
        self.model = model
        self.dimension = dimension
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
            return [0.0] * self.dimension

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
            "outputDimensionality": self.dimension,
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

    async def embed_batch(self, texts: list[str], batch_size: int = 20) -> list[list[float]]:
        """
        Generates dense vector embeddings for multiple texts using Gemini batchEmbedContents API.
        Includes automatic batch chunking, rate-limiting delays, and exponential backoff on HTTP 429.
        """
        if not texts:
            return []

        if not self.api_key:
            raise ValueError("Gemini API key is required for GeminiEmbeddings.")

        client = self._get_client()
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:batchEmbedContents"
            f"?key={self.api_key}"
        )
        all_embeddings: list[list[float]] = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for b_idx, i in enumerate(range(0, len(texts), batch_size)):
            chunk = texts[i : i + batch_size]
            requests_payload = [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": t.strip() if t.strip() else " "}]},
                    "outputDimensionality": 768,
                }
                for t in chunk
            ]

            if b_idx > 0:
                await asyncio.sleep(2.0)

            max_retries = 6
            batch_success = False

            for attempt in range(max_retries):
                try:
                    resp = await client.post(url, json={"requests": requests_payload})
                    resp.raise_for_status()
                    data = resp.json()
                    embeddings_list = data.get("embeddings", [])
                    for emb in embeddings_list:
                        all_embeddings.append(emb.get("values", [0.0] * 768))
                    batch_success = True
                    break
                except HTTPStatusError as exc:
                    resp_obj = getattr(exc, "response", None)
                    status_code = getattr(resp_obj, "status_code", 0) if resp_obj is not None else 0
                    if status_code == 429 and attempt < max_retries - 1:
                        headers = getattr(resp_obj, "headers", {}) or {}
                        retry_after = headers.get("retry-after")
                        if retry_after:
                            try:
                                wait_time = float(retry_after) + 1.0
                            except ValueError:
                                wait_time = (2 ** attempt) * 3.0 + 3.0
                        else:
                            wait_time = (2 ** attempt) * 3.0 + 3.0
                        print(f"⏳ [GeminiEmbeddings] Rate limit 429 on batch {b_idx + 1}/{total_batches}, waiting {wait_time:.1f}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    raise exc
                except Exception as exc:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2.5
                        print(f"⚠️ [GeminiEmbeddings] Network error on batch {b_idx + 1}/{total_batches} ({exc}), retrying in {wait_time:.1f}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    raise exc

            if not batch_success:
                raise RuntimeError(f"Failed to generate embeddings for batch {b_idx + 1}/{total_batches} after {max_retries} retries.")

        return all_embeddings

    async def close(self) -> None:
        """Closes the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
