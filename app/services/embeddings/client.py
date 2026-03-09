"""
Embedding service using Mistral embeddings API.
Produces 1024-dimensional vectors for pgvector storage.
"""
import logging
import httpx
from typing import List, Optional

logger = logging.getLogger("embeddings")

_EMBED_MODEL = "mistral-embed"
_EMBED_DIM = 1024


class EmbeddingClient:
    """Generate embeddings via Mistral API."""

    def __init__(self):
        from app.config import get_settings
        s = get_settings()
        self._api_key = getattr(s, "mistral_api_key", "") or ""
        self._base_url = "https://api.mistral.ai/v1"
        self._model = _EMBED_MODEL

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    @property
    def dimension(self) -> int:
        return _EMBED_DIM

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts (sync)."""
        if not self.is_configured:
            logger.warning("Embedding client not configured (no Mistral API key)")
            return []
        if not texts:
            return []

        # Mistral embed API accepts up to 16 texts per call
        all_embeddings = []
        for i in range(0, len(texts), 16):
            batch = texts[i:i + 16]
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(
                        f"{self._base_url}/embeddings",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json={"model": self._model, "input": batch},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for item in data.get("data", []):
                        all_embeddings.append(item["embedding"])
            except Exception as e:
                logger.error(f"Embedding API call failed: {e}")
                # Return empty vectors for failed batch
                all_embeddings.extend([[] for _ in batch])

        return all_embeddings

    def embed_one_sync(self, text: str) -> Optional[List[float]]:
        """Generate a single embedding (sync). Returns None on failure."""
        if not text or not self.is_configured:
            return None
        results = self.embed_sync([text])
        if results and results[0]:
            return results[0]
        return None

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings (async)."""
        if not self.is_configured or not texts:
            return []

        all_embeddings = []
        for i in range(0, len(texts), 16):
            batch = texts[i:i + 16]
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        f"{self._base_url}/embeddings",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json={"model": self._model, "input": batch},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for item in data.get("data", []):
                        all_embeddings.append(item["embedding"])
            except Exception as e:
                logger.error(f"Embedding API call failed: {e}")
                all_embeddings.extend([[] for _ in batch])

        return all_embeddings

    async def embed_one(self, text: str) -> Optional[List[float]]:
        """Generate a single embedding (async)."""
        if not text or not self.is_configured:
            return None
        results = await self.embed([text])
        if results and results[0]:
            return results[0]
        return None


# Singleton
_client = None


def get_embedding_client() -> EmbeddingClient:
    global _client
    if _client is None:
        _client = EmbeddingClient()
    return _client
