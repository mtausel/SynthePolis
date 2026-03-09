"""
Ch.4 + Ch.9.4 - BDI Cognitive Engine
The core conflict resolution logic for Synthesight 2.0.
"""
import json
from typing import Optional

from app.schemas.bdi import AgentDecision, PersonaBDIContext, BDIPromptConfig
from app.services.inference.client import get_inference_client
from app.services.inference.handshake import (
    bdi_inference_handshake,
    validate_symbolic_trace,
    SymbolicTraceError,
)
import re as _re

def safe_parse_json(text: str) -> dict:
    """Parse JSON from Mistral response, handling common issues."""
    # Try direct parse first
    try:
        import json
        return json.loads(text)
    except Exception:
        pass

    # Try to extract JSON from markdown code blocks
    for pattern in [r'```json\s*(.+?)```', r'```\s*(.+?)```', r'\{.+\}']:
        match = _re.search(pattern, text, _re.DOTALL)
        if match:
            try:
                import json
                return json.loads(match.group(1) if '```' in pattern else match.group(0))
            except Exception:
                pass

    # Try to fix truncated JSON by closing open braces/brackets
    import json
    cleaned = text.strip()
    if not cleaned.endswith('}'):
        # Count open/close braces
        opens = cleaned.count('{') - cleaned.count('}')
        brackets = cleaned.count('[') - cleaned.count(']')
        # Try to find last complete value and close
        # Remove trailing incomplete string
        last_quote = cleaned.rfind('"')
        if last_quote > 0 and cleaned[last_quote-1] != '\\':
            cleaned = cleaned[:last_quote+1]
        cleaned += ']' * max(0, brackets) + '}' * max(0, opens)
        try:
            return json.loads(cleaned)
        except Exception:
            pass

    raise ValueError(f"Cannot parse JSON from response: {text[:200]}...")
from app.core.logger import worker_logger


TIER_CONFIGS = {
    "B1": BDIPromptConfig(
        tier="B1",
        instructions="Evaluate this concept based only on your grounded demographics and values. Focus on your First Impression reaction.",
        constraints="Do not simulate long-term adoption. Give an honest gut reaction based on who you are.",
        escalation="If the concept is too vague for your profile, return intention_score 0.5 and explain why in causal_trace.",
    ),
    "B3": BDIPromptConfig(
        tier="B3",
        instructions="You will be shown a messaging variation. Evaluate which specific phrases or values trigger acceptance or rejection based on your Schwartz Values.",
        constraints="You must give a clear preference. Rank based on your current Belief Store.",
        escalation="If two messages are equally appealing, cite the Symbolic Reasoning Path that broke the tie.",
    ),
    "B4": BDIPromptConfig(
        tier="B4",
        instructions="You will be shown a list of items to RANK from most preferred to least preferred. You MUST rank ALL items. Consider each item against your Values, Beliefs, and Demographics to produce a forced ordering.",
        constraints="You MUST assign a unique rank to every item (no ties). Your ranking must reflect your authentic preferences as this persona. Output your ranking as a JSON array.",
        escalation="If two items seem equally preferred, use your core Schwartz Values to break the tie and explain which value determined the ordering.",
    ),
    "B6": BDIPromptConfig(
        tier="B6",
        instructions="Access your historical memory before responding to the new stimulus. Your shift in opinion must be incremental.",
        constraints="Sudden 180-degree turns must be justified by extreme new evidence. Maintain longitudinal consistency.",
        escalation="If the new evidence contradicts a Core Belief, enter a Cognitive Dissonance state and log the logic conflict.",
    ),
}


def _build_system_prompt(persona: PersonaBDIContext, tier: str = "B1", language: str = "en", document_context: str = None, kg_context: str = None) -> str:
    config = TIER_CONFIGS.get(tier, TIER_CONFIGS["B1"])

    values_str = ", ".join(persona.schwartz_values) if persona.schwartz_values else "Not specified"

    if persona.belief_store:
        beliefs_list = persona.belief_store[:10]
        beliefs_str = "\n- ".join(beliefs_list)
    else:
        beliefs_str = "No prior beliefs loaded"

    interests_str = ", ".join(persona.digital_interests) if persona.digital_interests else "None"

    cultural_str = ""
    if persona.cultural_baggage:
        cs = persona.cultural_baggage.get("communication_style", "Standard")
        pn = persona.cultural_baggage.get("primary_norm", "None")
        vb = persona.cultural_baggage.get("value_benchmark", "None")
        cultural_str = (
            "\nCultural Context:"
            "\n- Communication style: " + cs +
            "\n- Primary norm: " + pn +
            "\n- Value benchmark: " + vb
        )

    doc_str = ""
    if document_context and document_context.strip():
        doc_str = (
            "\n\nREFERENCE MATERIAL (factual information about the product/concept being evaluated — "
            "use these details to form informed beliefs and realistic assessments):\n"
            + document_context[:4000] + "\n"
        )

    # Phase 4: KG context injection
    kg_str = ""
    if kg_context and kg_context.strip():
        kg_str = kg_context

    demographics_str = json.dumps(persona.demographics, indent=2)

    prompt = (
        "You are a BDI (Belief-Desire-Intention) Agent in a Synthetic Market Research simulation.\n\n"
        "YOUR IDENTITY:\n"
        "Demographics: " + demographics_str + "\n"
        "Region: " + persona.demographics.get("region", "Unknown") + "\n"
        + cultural_str + "\n\n"
        "YOUR INTERNAL STATE:\n"
        "VALUES / DESIRES (D): " + values_str + "\n"
        "These are your deep, stable motivational drivers from Schwartz Value Theory.\n\n"
        "CURRENT BELIEFS (B):\n- " + beliefs_str + "\n"
        "These are things you currently believe based on your information environment.\n\n"
        "DIGITAL INTERESTS (Hybrid Layer): " + interests_str + "\n"
        "These are recent digital behaviors that may or may not align with your core values.\n"
        + doc_str + kg_str + "\n"
        "REASONING INSTRUCTIONS (" + config.tier + "):\n"
        "- " + config.instructions + "\n"
        "- CONSTRAINTS: " + config.constraints + "\n"
        "- IF UNCERTAIN: " + config.escalation + "\n\n"
        "OUTPUT FORMAT:\n"
        "You MUST respond with a JSON object containing these exact fields:\n"
        '- "intention": Your final decision (e.g., "Accept", "Reject", "Neutral", or a specific choice)\n'
        '- "intention_score": A float between 0.0 (full rejection) and 1.0 (full acceptance)\n'
        '- "causal_trace": A detailed explanation of WHY you made this decision, referencing your specific Beliefs and Desires\n'
        '- "conflict_detected": true/false - whether your Beliefs conflicted with your Desires\n'
        '- "conflict_description": If conflict detected, describe the specific tension\n'
        '- "resolved_logic": The symbolic resolution (e.g., "Security > Novelty because fixed_income and risk_aversion")\n\n'
                "Think step by step. Be authentic to your demographic profile. Do not break character.\n\n"
        "LANGUAGE INSTRUCTION: You MUST respond entirely in "
        + {"en":"English","it":"Italian (Italiano)","de":"German (Deutsch)","fr":"French (Français)"}.get(language, "English")
        + ". All text fields (intention, causal_trace, conflict_description, resolved_logic) MUST be written in that language. "
        "Only JSON field names and numeric values stay in English."
    )
    return prompt


def _build_stimulus_prompt(stimulus: str) -> str:
    return "STIMULUS TO EVALUATE:\n\n" + stimulus + "\n\nNow respond as your persona. Remember to output valid JSON with all required fields."


@bdi_inference_handshake

async def resolve_bdi_intention(
    persona: PersonaBDIContext,
    stimulus: str,
    tier: str = "B1",
    language: str = "en",
    document_context: str = None,
    kg_context: str = None,
) -> AgentDecision:
    """
    Ch.4 + Ch.9.4.1 - Execute BDI conflict resolution for a single agent.
    """
    client = get_inference_client()

    system_prompt = _build_system_prompt(persona, tier, language, document_context=document_context, kg_context=kg_context)
    user_prompt = _build_stimulus_prompt(stimulus)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    worker_logger.log_action(
        user_id=persona.persona_id,
        action="BDI_INFERENCE_START",
        metadata={"tier": tier, "stimulus_preview": stimulus[:100]},
    )

    response_data = await client.generate_json(
        messages=messages,
        temperature=0.6,
        max_tokens=1500,
    )

    if not validate_symbolic_trace(response_data):
        worker_logger.log_action(
            user_id=persona.persona_id,
            action="SYMBOLIC_TRACE_MISSING",
            metadata={"raw_response": str(response_data)[:500]},
        )
        raise SymbolicTraceError(
            "Agent " + persona.persona_id + ": Response missing required symbolic trace fields"
        )

    # Normalize fields that Mistral sometimes returns as objects instead of strings
    for field in ["causal_trace", "resolved_logic", "conflict_description", "intention"]:
        if field in response_data and not isinstance(response_data[field], str):
            import json as _json
            response_data[field] = _json.dumps(response_data[field])

    decision = AgentDecision.model_validate(response_data)

    worker_logger.log_action(
        user_id=persona.persona_id,
        action="BDI_INTENTION_RESOLVED",
        metadata={
            "intention": decision.intention,
            "score": decision.intention_score,
            "conflict": decision.conflict_detected,
            "resolved_logic": decision.resolved_logic,
        },
    )

    return decision
