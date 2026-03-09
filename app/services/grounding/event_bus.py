"""
Grounding Event Bus - routes external events to vertical subscribers.
Simple in-process pub/sub. No external message broker needed.
"""
import logging
from typing import Callable, Dict, List, Any
from dataclasses import dataclass, field

logger = logging.getLogger("grounding.event_bus")


@dataclass
class GroundingEvent:
    """A raw event from Perplexity or Apify before classification."""
    source: str           # 'perplexity', 'apify_twitter', etc.
    content: str          # Raw text content
    url: str = ""
    metadata: dict = field(default_factory=dict)
    timestamp: str = ""
    vertical_hint: str = ""  # 'synthecoach', 'synthepolis', 'synthesight'


@dataclass
class ClassifiedEvent:
    """An event after LLM classification."""
    raw: GroundingEvent
    event_type: str       # Domain-specific type
    title: str
    summary: str
    sentiment: float      # -1.0 to +1.0
    intensity: float      # 0.0 to 1.0
    domain_data: dict = field(default_factory=dict)


# Type alias for subscriber callback
EventHandler = Callable


class GroundingEventBus:
    """In-process event bus. Verticals register handlers for event types."""

    def __init__(self):
        self._subscribers: Dict[str, List[EventHandler]] = {}

    def subscribe(self, vertical: str, handler: EventHandler):
        """Register a handler for a vertical."""
        self._subscribers.setdefault(vertical, []).append(handler)
        logger.info(f"Subscribed {handler.__name__} to '{vertical}'")

    async def publish(self, event: ClassifiedEvent):
        """Route event to matching subscribers."""
        hint = event.raw.vertical_hint or "all"
        handlers = self._subscribers.get(hint, [])
        handlers += self._subscribers.get("all", [])

        for handler in handlers:
            try:
                result = handler(event)
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                logger.error(f"Handler {handler.__name__} failed: {e}")

    async def publish_batch(self, events: List[ClassifiedEvent]):
        """Route multiple events."""
        for e in events:
            await self.publish(e)

    @property
    def subscriber_count(self) -> int:
        return sum(len(v) for v in self._subscribers.values())

    @property
    def verticals(self) -> List[str]:
        return list(self._subscribers.keys())


# Singleton
_bus = None


def get_event_bus() -> GroundingEventBus:
    global _bus
    if _bus is None:
        _bus = GroundingEventBus()
    return _bus
