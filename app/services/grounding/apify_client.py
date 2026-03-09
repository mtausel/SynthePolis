"""
Shared Apify client for all iB3 verticals.
Handles actor runs, dataset retrieval, error handling,
and graceful degradation when no token is configured.
"""
import os
import logging
import httpx
from typing import Optional, List, Dict

logger = logging.getLogger("grounding.apify")

APIFY_TOKEN = os.getenv("APIFY_API_TOKEN", "")
APIFY_BASE = "https://api.apify.com/v2"


class ApifyClient:
    """Shared Apify client with actor run + dataset retrieval."""

    def __init__(self, token: str = None):
        self.token = token or APIFY_TOKEN

    @property
    def is_configured(self) -> bool:
        return bool(self.token)

    def run_actor_sync(self, actor_id: str, input_data: dict,
                       timeout: int = 120) -> list:
        """Synchronous actor run for sync endpoints."""
        if not self.is_configured:
            logger.info("Apify token not configured, skipping")
            return []
        try:
            url = f"{APIFY_BASE}/acts/{actor_id}/run-sync-get-dataset-items"
            with httpx.Client(timeout=float(timeout + 30)) as client:
                resp = client.post(
                    url,
                    params={"token": self.token},
                    json=input_data,
                )
                if resp.status_code == 402:
                    logger.error("Apify: payment required / insufficient credits")
                    return []
                if resp.status_code == 429:
                    logger.warning("Apify: rate limited")
                    return []
                resp.raise_for_status()
                return resp.json() if isinstance(resp.json(), list) else []
        except httpx.TimeoutException:
            logger.warning(f"Apify actor {actor_id} timed out")
        except Exception as e:
            logger.error(f"Apify error: {e}")
        return []

    async def run_actor(self, actor_id: str, input_data: dict,
                        timeout: int = 120) -> list:
        """Async actor run."""
        if not self.is_configured:
            logger.info("Apify token not configured, skipping")
            return []
        try:
            url = f"{APIFY_BASE}/acts/{actor_id}/run-sync-get-dataset-items"
            async with httpx.AsyncClient(timeout=float(timeout + 30)) as client:
                resp = await client.post(
                    url,
                    params={"token": self.token},
                    json=input_data,
                )
                if resp.status_code == 402:
                    logger.error("Apify: payment required")
                    return []
                if resp.status_code == 429:
                    logger.warning("Apify: rate limited")
                    return []
                resp.raise_for_status()
                return resp.json() if isinstance(resp.json(), list) else []
        except httpx.TimeoutException:
            logger.warning(f"Apify actor {actor_id} timed out")
        except Exception as e:
            logger.error(f"Apify error: {e}")
        return []

    def search_twitter_sync(self, query: str, max_results: int = 30,
                            actor_id: str = None) -> List[Dict]:
        """Search Twitter via Apify (sync)."""
        aid = actor_id or "apidojo/tweet-scraper"
        results = self.run_actor_sync(aid, {
            "searchTerms": [query], "maxTweets": max_results, "sort": "Latest",
        })
        return [{"source": "apify_twitter",
                 "text": t.get("full_text", t.get("text", "")),
                 "url": t.get("url", ""),
                 "author": t.get("user", {}).get("screen_name", ""),
                 } for t in results]

    def search_instagram_sync(self, hashtags: list = None,
                              profiles: list = None,
                              max_results: int = 10) -> List[Dict]:
        """Search Instagram via Apify (sync)."""
        events = []
        if hashtags:
            for tag in hashtags[:3]:
                results = self.run_actor_sync(
                    "apify/instagram-hashtag-scraper",
                    {"hashtags": [tag], "resultsLimit": max_results})
                events.extend([{"source": "apify_instagram",
                    "text": p.get("caption", ""),
                    "url": p.get("url", ""),
                    "author": p.get("ownerUsername", ""),
                } for p in results])
        if profiles:
            results = self.run_actor_sync(
                "apify/instagram-profile-scraper",
                {"usernames": profiles, "resultsLimit": max_results})
            events.extend([{"source": "apify_instagram",
                "text": p.get("caption", ""),
                "url": p.get("url", ""),
                "author": p.get("ownerUsername", ""),
            } for p in results])
        return events


# Singleton
_client: Optional[ApifyClient] = None


def get_apify_client() -> ApifyClient:
    global _client
    if _client is None:
        _client = ApifyClient()
    return _client
