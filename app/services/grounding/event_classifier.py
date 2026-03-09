"""
Shared event classifier. Verticals provide domain-specific prompts.
Falls back to a generic classification if no domain prompt given.
"""
import json
import logging
import re
from typing import Optional, Dict, List

from app.services.grounding.perplexity_client import get_perplexity_client

logger = logging.getLogger("grounding.classifier")

GENERIC_PROMPT = """
Classify this content as a structured event.
Respond ONLY with JSON:
{{
  "event_type": "string",
  "title": "brief title (max 100 chars)",
  "summary": "2-3 sentence summary",
  "sentiment": float (-1.0 to +1.0),
  "intensity": float (0.0 to 1.0)
}}

Content: {content}
"""


def classify_event_sync(content: str, domain_prompt: str = None,
                        context: dict = None) -> Optional[Dict]:
    """Classify raw content into a structured event (sync)."""
    prompt = (domain_prompt or GENERIC_PROMPT)
    # Replace placeholders from context
    if context:
        for k, v in context.items():
            prompt = prompt.replace("{" + k + "}", str(v))
    prompt = prompt.replace("{content}", content[:3000])

    client = get_perplexity_client()
    result = client.search_json_sync(prompt)
    if result:
        return result

    return {
        "event_type": "GENERAL",
        "title": content[:100],
        "summary": content[:300],
        "sentiment": 0.0,
        "intensity": 0.5,
    }


def aggregate_social_batch(events: list, source: str) -> Optional[Dict]:
    """Aggregate multiple social posts into one event for classification."""
    if not events:
        return None

    # Sort by engagement (if available)
    sorted_e = sorted(events,
        key=lambda e: sum(e.get("engagement", {}).values()) if isinstance(e.get("engagement"), dict) else 0,
        reverse=True)[:10]

    combined = "\n---\n".join(
        f"[{e.get('source', source)}] {e.get('author', '')}: {e.get('text', '')[:300]}"
        for e in sorted_e if e.get("text"))

    if not combined.strip():
        return None

    return {
        "content": combined,
        "source": source,
        "metadata": {
            "post_count": len(events),
            "top_sampled": len(sorted_e),
        },
    }
