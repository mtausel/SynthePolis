"""
Shared Perplexity API client for all iB3 verticals.
Single source of truth for API configuration, error handling,
rate limiting, and response parsing.
"""
import os
import json
import logging
import asyncio
import httpx
from typing import Optional, Dict, List

logger = logging.getLogger("grounding.perplexity")

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
DEFAULT_MODEL = "sonar"


class PerplexityClient:
    """Shared Perplexity search client with retry and rate limiting."""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or PERPLEXITY_API_KEY
        self.model = model or DEFAULT_MODEL
        self._request_count = 0

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def search_sync(self, query: str, system_prompt: str = None,
                    max_retries: int = 3, timeout: float = 45.0) -> dict:
        """Synchronous search for use in sync endpoints."""
        if not self.is_configured:
            logger.warning("Perplexity API key not configured")
            return {"content": "", "citations": [], "raw": {}}

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        import time
        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(
                        PERPLEXITY_URL,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={"model": self.model, "messages": messages, "max_tokens": 2000},
                    )
                    if resp.status_code == 429:
                        wait = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait}s")
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    self._request_count += 1
                    return {
                        "content": data.get("choices", [{}])[0]
                            .get("message", {}).get("content", ""),
                        "citations": data.get("citations", []),
                        "raw": data,
                    }
            except httpx.TimeoutException:
                logger.warning(f"Perplexity timeout, attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Perplexity error: {e}")
                break

        return {"content": "", "citations": [], "raw": {}}

    async def search(self, query: str, system_prompt: str = None,
                     max_retries: int = 3, timeout: float = 45.0) -> dict:
        """Async search with exponential backoff."""
        if not self.is_configured:
            logger.warning("Perplexity API key not configured")
            return {"content": "", "citations": [], "raw": {}}

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(
                        PERPLEXITY_URL,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={"model": self.model, "messages": messages, "max_tokens": 2000},
                    )
                    if resp.status_code == 429:
                        wait = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    self._request_count += 1
                    return {
                        "content": data.get("choices", [{}])[0]
                            .get("message", {}).get("content", ""),
                        "citations": data.get("citations", []),
                        "raw": data,
                    }
            except httpx.TimeoutException:
                logger.warning(f"Perplexity timeout, attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Perplexity error: {e}")
                break

        return {"content": "", "citations": [], "raw": {}}

    def search_json_sync(self, prompt: str) -> Optional[Dict]:
        """Search and parse JSON from response (sync)."""
        result = self.search_sync(prompt, system_prompt="You are an analyst. Respond ONLY with valid JSON.")
        if not result or not result["content"]:
            return None
        import re
        try:
            match = re.search(r"\{.*\}", result["content"], re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        return None

    async def search_batch(self, queries: List[str],
                           system_prompt: str = None) -> List[dict]:
        """Search multiple queries sequentially (rate-limit safe)."""
        results = []
        for q in queries:
            r = await self.search(q, system_prompt)
            results.append(r)
        return results


# Singleton
_client: Optional[PerplexityClient] = None


def get_perplexity_client() -> PerplexityClient:
    global _client
    if _client is None:
        _client = PerplexityClient()
    return _client
