"""
KG Context helper for BDI Engine injection.
Fetches relevant KG nodes for a persona/study and formats them for the system prompt.
"""
import json
import logging
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)


def get_kg_context_for_bdi(db: Session, vertical: str, study_id: str = None,
                            persona_properties: dict = None, stimulus: str = None,
                            max_nodes: int = 10) -> str:
    """
    Fetch relevant KG context for BDI system prompt injection.
    Returns formatted string to append to system prompt.
    """
    try:
        context_parts = []

        if vertical == "synthesight" and study_id:
            # Get study stimulus node
            stimulus_nodes = db.execute(
                text("""SELECT label, properties FROM kg_nodes
                         WHERE vertical = 'synthesight' AND node_type = 'study_stimulus'
                         AND properties->>'study_id' = :sid LIMIT 1"""),
                {"sid": study_id}
            ).fetchall()

            # Get market beliefs connected to this study
            belief_nodes = db.execute(
                text("""SELECT label, properties FROM kg_nodes
                         WHERE vertical = 'synthesight' AND node_type = 'market_belief'
                         AND properties->>'study_id' = :sid
                         ORDER BY (properties->>'quality_score')::float DESC NULLS LAST
                         LIMIT :lim"""),
                {"sid": study_id, "lim": max_nodes}
            ).fetchall()

            if belief_nodes:
                beliefs_text = "\n".join(f"- {b[0]}" for b in belief_nodes)
                context_parts.append(f"MARKET INTELLIGENCE (from Knowledge Graph):\n{beliefs_text}")

            # Get market_event nodes (from grounding auto-update)
            event_nodes = db.execute(
                text("""SELECT label, properties FROM kg_nodes
                         WHERE vertical = 'synthesight' AND node_type = 'market_event'
                         AND properties->>'study_id' = :sid
                         LIMIT :lim"""),
                {"sid": study_id, "lim": 5}
            ).fetchall()

            if event_nodes:
                events_text = "\n".join(f"- {e[0]}" for e in event_nodes)
                context_parts.append(f"RECENT MARKET EVENTS:\n{events_text}")

        elif vertical == "synthepolis":
            # Get political context from KG
            if persona_properties and persona_properties.get("campaign_id"):
                campaign_id = persona_properties["campaign_id"]
                incident_nodes = db.execute(
                    text("""SELECT n.label, n.properties FROM kg_nodes n
                             WHERE n.vertical = 'synthepolis' AND n.node_type = 'political_incident'
                             ORDER BY (n.properties->>'confidence')::float DESC NULLS LAST
                             LIMIT :lim"""),
                    {"lim": max_nodes}
                ).fetchall()
                if incident_nodes:
                    incidents_text = "\n".join(f"- {n[0]}" for n in incident_nodes)
                    context_parts.append(f"POLITICAL CONTEXT (from Knowledge Graph):\n{incidents_text}")

        if not context_parts:
            return ""

        return "\n\nKNOWLEDGE GRAPH CONTEXT (verified external intelligence — use these facts to inform your evaluation):\n" + "\n\n".join(context_parts) + "\n"

    except Exception as e:
        logger.warning(f"KG context fetch failed (non-blocking): {e}")
        return ""
