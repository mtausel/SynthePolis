"""
Phase 6 — SynthePolis Political News Connector.
Perplexity + Apify → political event classification → KG auto-update → BDI refresh.
Mirrors sc_grounding.py pattern but with political-specific taxonomy.
"""
import json
import logging
import os
from typing import Optional, Dict, List, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.security import get_current_user, require_role
from app.models.user import User
from app.db.session import get_db
from app.services.grounding.perplexity_client import get_perplexity_client
from app.services.grounding.apify_client import get_apify_client

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Political Event Classification Prompt ────────────────────────

POLITICAL_CLASSIFICATION_PROMPT = """
Analyze this political news content for a campaign monitoring system.
Respond ONLY with a JSON object:
{{
  "event_type": one of ["STANCE_CHANGE", "SCANDAL", "POLICY_SHIFT",
    "COALITION_MOVE", "ELECTION_RESULT", "POLL", "PROTEST", "LEGISLATION",
    "MEDIA_APPEARANCE", "ENDORSEMENT", "GAFFE", "GENERAL"],
  "sentiment": float -1.0 to +1.0,
  "intensity": float 0.0 to 1.0 (political significance),
  "title": "brief event title (max 100 chars)",
  "summary": "2-3 sentence summary with political implications",
  "affected_candidates": [
    {{"name": "candidate name mentioned",
     "relevance": 0.0-1.0,
     "impact": "positive" or "negative" or "neutral"}}
  ],
  "issues_involved": ["economy", "corruption", ...],
  "strategic_significance": "1 sentence on campaign strategy impact"
}}

Campaign context:
Region: {region}
Election type: {election_type}
Candidates being tracked: {candidate_names}

Content to analyze:
{content}
"""

STRATEGIC_REC_PROMPT = """
You are a political strategy advisor. A political event may affect a candidate's campaign.
Generate a brief, actionable strategic recommendation for the campaign team.

Candidate: {name} ({party})
Event: {summary}
Event type: {event_type}
Impact on candidate: {impact}

Write 2-3 sentences suggesting specific strategic actions.
Focus on: messaging response, media strategy, voter outreach, or coalition positioning as appropriate.
Respond with ONLY the recommendation text, no JSON.
"""


# ── Perplexity + Apify (using shared clients) ───────────────────

def _call_perplexity(query: str, system_prompt: str = None) -> Optional[Dict]:
    """Call Perplexity via shared client."""
    pplx = get_perplexity_client()
    if not pplx.is_configured:
        logger.warning("Perplexity not configured")
        return None
    return pplx.search_sync(query, system_prompt=system_prompt)


def _call_perplexity_json(prompt: str) -> Optional[Dict]:
    """Call Perplexity and parse JSON via shared client."""
    pplx = get_perplexity_client()
    return pplx.search_json_sync(prompt)


def _check_political_twitter(queries: list, max_results: int = 20) -> list:
    """Search Twitter for political keywords via shared Apify client."""
    apify = get_apify_client()
    if not apify.is_configured:
        return []
    all_results = []
    for q in queries[:5]:
        results = apify.run_actor_sync(
            "apidojo/tweet-scraper",
            {"searchTerms": [q], "maxTweets": max_results // max(len(queries[:5]), 1),
             "sort": "Latest"},
        )
        for r in results:
            all_results.append({
                "text": r.get("full_text", r.get("text", "")),
                "author": r.get("user", {}).get("screen_name", "unknown"),
                "source": "apify_twitter",
                "url": r.get("url", ""),
            })
    return all_results


def _check_political_facebook(accounts: list, max_results: int = 10) -> list:
    """Check Facebook pages via shared Apify client."""
    apify = get_apify_client()
    if not apify.is_configured:
        return []
    all_results = []
    for acct in accounts[:3]:
        results = apify.run_actor_sync(
            "apify/facebook-posts-scraper",
            {"startUrls": [{"url": f"https://www.facebook.com/{acct}"}],
             "maxPosts": max_results},
        )
        for r in results:
            all_results.append({
                "text": r.get("text", r.get("message", "")),
                "author": acct,
                "source": "apify_facebook",
                "url": r.get("url", ""),
            })
    return all_results


# ── Bilingual Processing ────────────────────────────────────────

def _translate_to_english(text: str) -> str:
    """Translate non-English text to English using Perplexity."""
    if not text or len(text.strip()) < 10:
        return text
    result = _call_perplexity(
        f"Translate the following text to English. Respond with ONLY the translation, nothing else:\n\n{text[:2000]}",
    )
    return result["content"].strip() if result else text


def _detect_language(text: str) -> str:
    """Simple heuristic language detection."""
    if not text:
        return "en"
    # Check for Cyrillic (Bulgarian, Russian, etc.)
    cyrillic_count = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    if cyrillic_count > len(text) * 0.3:
        return "bg"
    # Check for other scripts
    latin_count = sum(1 for c in text if 'a' <= c.lower() <= 'z')
    if latin_count > len(text) * 0.4:
        return "en"
    return "unknown"


# ── Event Classification ────────────────────────────────────────

def _classify_political_event(content: str, campaign: dict, candidates: list) -> Optional[Dict]:
    """Classify political content using Perplexity LLM."""
    candidate_names = ", ".join(c["name"] for c in candidates) if candidates else "none tracked"
    prompt = POLITICAL_CLASSIFICATION_PROMPT.format(
        region=campaign.get("region_code", "unknown"),
        election_type=campaign.get("election_type", "general"),
        candidate_names=candidate_names,
        content=content[:3000],
    )
    return _call_perplexity_json(prompt)


# ── Candidate Matching ──────────────────────────────────────────

def _fuzzy_match_candidate(name: str, candidates: list) -> Optional[Dict]:
    """Match a mentioned name to a tracked candidate."""
    if not name or not candidates:
        return None
    name_lower = name.lower().strip()
    for c in candidates:
        cand_name = c["name"].lower()
        # Exact or partial match
        if name_lower in cand_name or cand_name in name_lower:
            return c
        # Last name match
        parts = cand_name.split()
        if any(p in name_lower for p in parts if len(p) > 2):
            return c
    return None


def _compute_severity(sentiment: float, intensity: float, relevance: float, impact: str) -> str:
    """Compute alert severity from event metrics."""
    score = abs(sentiment) * 0.3 + intensity * 0.4 + relevance * 0.3
    if impact == "negative" and abs(sentiment) > 0.7:
        score += 0.2  # Boost for negative high-sentiment
    if score >= 0.8:
        return "critical"
    elif score >= 0.6:
        return "high"
    elif score >= 0.4:
        return "medium"
    return "low"


def _map_event_to_alert_type(event_type: str, impact: str) -> str:
    """Map event type + impact to alert type."""
    mapping = {
        ("SCANDAL", "negative"): "CRISIS",
        ("SCANDAL", "positive"): "COMPETITOR_MOVE",
        ("STANCE_CHANGE", "negative"): "STANCE_SHIFT",
        ("STANCE_CHANGE", "positive"): "OPPORTUNITY",
        ("POLICY_SHIFT", "negative"): "COMPETITOR_MOVE",
        ("POLICY_SHIFT", "positive"): "OPPORTUNITY",
        ("COALITION_MOVE", "negative"): "COMPETITOR_MOVE",
        ("COALITION_MOVE", "positive"): "OPPORTUNITY",
        ("POLL", "negative"): "POLL_CHANGE",
        ("POLL", "positive"): "POLL_CHANGE",
        ("ENDORSEMENT", "positive"): "ENDORSEMENT_ALERT",
        ("ENDORSEMENT", "negative"): "COMPETITOR_MOVE",
        ("GAFFE", "negative"): "VULNERABILITY_DETECTED",
    }
    return mapping.get((event_type, impact), "COMPETITOR_MOVE" if impact == "negative" else "OPPORTUNITY")


# ── Strategic Recommendation ────────────────────────────────────

def _generate_strategic_recommendation(candidate_name: str, party: str,
                                        event_summary: str, event_type: str,
                                        impact: str) -> str:
    """Generate AI-powered strategic recommendation via shared client."""
    prompt = STRATEGIC_REC_PROMPT.format(
        name=candidate_name, party=party or "Independent",
        summary=event_summary, event_type=event_type, impact=impact,
    )
    pplx = get_perplexity_client()
    if not pplx.is_configured:
        return f"Monitor {event_type.lower()} event closely. Prepare response strategy."
    result = pplx.search_sync(prompt)
    if result and result.get("content"):
        return result["content"].strip()[:500]
    return f"Monitor {event_type.lower()} event closely. Prepare response strategy."


# ── KG Integration ──────────────────────────────────────────────

def _add_political_event_to_kg(db: Session, event_id: int, event_data: dict,
                                campaign_id: int, affected: list) -> Optional[int]:
    """Create political_event KG node + AFFECTS_CANDIDATE edges."""
    try:
        # Create political_event node
        entity_id = 500000 + event_id  # offset for political events
        result = db.execute(
            text("""INSERT INTO kg_nodes (vertical, node_type, entity_id, label, properties)
                     VALUES ('synthepolis', 'political_event', :eid, :label, :props)
                     ON CONFLICT (vertical, node_type, entity_id, node_subtype) DO UPDATE
                     SET label = EXCLUDED.label, properties = EXCLUDED.properties
                     RETURNING id"""),
            {"eid": entity_id,
             "label": (event_data.get("title", "Political event"))[:200],
             "props": json.dumps({
                 "event_type": event_data.get("event_type", "GENERAL"),
                 "sentiment": event_data.get("sentiment_score", 0),
                 "intensity": event_data.get("intensity", 0.5),
                 "source": event_data.get("source", "perplexity"),
                 "event_id": event_id,
             })}
        )
        node_id = result.fetchone()[0]

        # Update event with KG node reference
        db.execute(
            text("UPDATE sp_political_events SET kg_node_id = :nid WHERE id = :eid"),
            {"nid": node_id, "eid": event_id}
        )

        # AFFECTS_CANDIDATE edges
        for af in affected:
            cand_entity_id = 100000 + af["candidate_id"]
            cand_node = db.execute(
                text("SELECT id FROM kg_nodes WHERE vertical = 'synthepolis' AND node_type = 'political_actor' AND entity_id = :eid"),
                {"eid": cand_entity_id}
            ).fetchone()
            if cand_node:
                db.execute(
                    text("""INSERT INTO kg_edges (vertical, source_node_id, target_node_id, edge_type, weight, properties)
                             VALUES ('synthepolis', :src, :tgt, 'AFFECTS_CANDIDATE', :w, :props)"""),
                    {"src": node_id, "tgt": cand_node[0],
                     "w": af.get("relevance", 0.5),
                     "props": json.dumps({"impact": af.get("impact", "neutral"),
                                          "event_type": event_data.get("event_type", "GENERAL")})}
                )

        # OCCURRED_IN edge to campaign
        camp_node = db.execute(
            text("SELECT id FROM kg_nodes WHERE vertical = 'synthepolis' AND node_type = 'campaign' AND entity_id = :eid"),
            {"eid": campaign_id}
        ).fetchone()
        if camp_node:
            db.execute(
                text("""INSERT INTO kg_edges (vertical, source_node_id, target_node_id, edge_type, weight, properties)
                         VALUES ('synthepolis', :src, :tgt, 'OCCURRED_IN', :w, :props)"""),
                {"src": node_id, "tgt": camp_node[0],
                 "w": event_data.get("intensity", 0.5),
                 "props": json.dumps({"event_type": event_data.get("event_type", "GENERAL")})}
            )

        return node_id
    except Exception as e:
        logger.warning(f"KG event insertion failed (non-blocking): {e}")
        return None


# ── BDI Auto-Refresh ────────────────────────────────────────────

def _maybe_refresh_bdi(db: Session, candidate_id: int, event_data: dict,
                        auto_bdi_refresh: bool = True) -> bool:
    """Refresh candidate BDI if event is significant enough."""
    if not auto_bdi_refresh:
        return False
    sentiment = abs(event_data.get("sentiment_score", 0))
    intensity = event_data.get("intensity", 0)
    if sentiment < 0.5 or intensity < 0.6:
        return False

    try:
        from app.api.v2_8.endpoints.synthepolis_kg import extract_synthepolis_bdi
        extract_synthepolis_bdi(db, candidate_id)
        db.execute(
            text("""UPDATE sp_political_alerts SET bdi_refresh_triggered = TRUE
                     WHERE candidate_id = :cid ORDER BY created_at DESC LIMIT 1"""),
            {"cid": candidate_id}
        )
        logger.info(f"BDI auto-refresh triggered for candidate {candidate_id}")
        return True
    except Exception as e:
        logger.warning(f"BDI refresh failed for candidate {candidate_id}: {e}")
        return False


# ── Alert Generation ────────────────────────────────────────────

def _generate_political_alerts(db: Session, event_id: int, event_data: dict,
                                campaign_id: int, candidates: list,
                                config: dict) -> list:
    """Generate strategic alerts when events affect monitored candidates."""
    affected_list = event_data.get("affected_candidates", [])
    if not affected_list:
        return []

    alerts = []
    for af in affected_list:
        matched = _fuzzy_match_candidate(af.get("name", ""), candidates)
        if not matched:
            continue

        severity = _compute_severity(
            event_data.get("sentiment_score", 0),
            event_data.get("intensity", 0.5),
            af.get("relevance", 0.5),
            af.get("impact", "neutral"),
        )
        alert_type = _map_event_to_alert_type(
            event_data.get("event_type", "GENERAL"),
            af.get("impact", "neutral"),
        )

        # Generate strategic recommendation
        recommendation = _generate_strategic_recommendation(
            candidate_name=matched["name"],
            party=matched.get("party", ""),
            event_summary=event_data.get("summary", ""),
            event_type=event_data.get("event_type", "GENERAL"),
            impact=af.get("impact", "neutral"),
        )

        db.execute(
            text("""INSERT INTO sp_political_alerts
                     (event_id, campaign_id, candidate_id, alert_type, severity, title, recommended_action)
                     VALUES (:eid, :cid, :candid, :at, :sev, :title, :rec)"""),
            {"eid": event_id, "cid": campaign_id, "candid": matched["id"],
             "at": alert_type, "sev": severity,
             "title": f'{event_data.get("event_type", "EVENT")}: {event_data.get("title", "")[:80]}',
             "rec": recommendation}
        )
        alerts.append({"candidate": matched["name"], "type": alert_type,
                        "severity": severity, "candidate_id": matched["id"]})

        # BDI auto-refresh if significant
        _maybe_refresh_bdi(db, matched["id"], event_data,
                           config.get("auto_bdi_refresh", True))

    return alerts


# ── Main Check Pipeline ─────────────────────────────────────────

class PoliticalCheckRequest(BaseModel):
    campaign_id: int


def run_political_check(db: Session, campaign_id: int) -> Dict[str, Any]:
    """
    Main political news check pipeline.
    1. Load campaign config + candidates
    2. Query Perplexity with search terms
    3. Query Apify for social media (if token set)
    4. Classify events
    5. Generate alerts
    6. Update KG
    7. Auto-refresh BDI if needed
    """
    # Load campaign
    campaign = db.execute(
        text("SELECT id, name, code, region_code, country_code, election_type FROM sp_campaigns WHERE id = :cid"),
        {"cid": campaign_id}
    ).fetchone()
    if not campaign:
        raise ValueError(f"Campaign {campaign_id} not found")

    campaign_dict = {
        "id": campaign[0], "name": campaign[1], "code": campaign[2],
        "region_code": campaign[3] or "", "country_code": campaign[4] or "",
        "election_type": campaign[5] or "general",
    }

    # Load candidates
    candidates = db.execute(
        text("SELECT id, name, party, political_stance FROM sp_candidates WHERE campaign_id = :cid"),
        {"cid": campaign_id}
    ).fetchall()
    candidates_list = [{"id": c[0], "name": c[1], "party": c[2], "stance": c[3]} for c in candidates]

    # Load config
    config_row = db.execute(
        text("SELECT search_queries, social_accounts, monitored_candidates, sentiment_threshold, event_intensity_threshold, auto_bdi_refresh, source_lang FROM sp_grounding_config WHERE campaign_id = :cid AND is_active = TRUE"),
        {"cid": campaign_id}
    ).fetchone()

    if not config_row:
        # Auto-generate config
        queries = _auto_generate_queries(db, campaign_id, campaign_dict, candidates_list)
        db.execute(
            text("""INSERT INTO sp_grounding_config (campaign_id, search_queries)
                     VALUES (:cid, :q)
                     ON CONFLICT (campaign_id) DO UPDATE SET search_queries = EXCLUDED.search_queries"""),
            {"cid": campaign_id, "q": json.dumps(queries)}
        )
        db.commit()
        config = {"search_queries": queries, "social_accounts": {},
                  "sentiment_threshold": -0.3, "event_intensity_threshold": 0.4,
                  "auto_bdi_refresh": True, "source_lang": "en"}
    else:
        config = {
            "search_queries": config_row[0] if isinstance(config_row[0], list) else json.loads(config_row[0] or "[]"),
            "social_accounts": config_row[1] if isinstance(config_row[1], dict) else json.loads(config_row[1] or "{}"),
            "monitored_candidates": config_row[2] if isinstance(config_row[2], list) else json.loads(config_row[2] or "[]"),
            "sentiment_threshold": config_row[3] or -0.3,
            "event_intensity_threshold": config_row[4] or 0.4,
            "auto_bdi_refresh": config_row[5] if config_row[5] is not None else True,
            "source_lang": config_row[6] or "en",
        }

    events_found = 0
    alerts_generated = 0
    kg_nodes_created = 0
    sources_checked = []

    # ── Perplexity Search ────────────────────────────────────
    search_queries = config.get("search_queries", [])
    for query in search_queries[:5]:
        raw = _call_perplexity(f"Latest political news: {query}")
        if not raw:
            continue
        sources_checked.append("perplexity")

        content = raw["content"]
        source_lang = _detect_language(content)

        # Translate if needed
        english_content = content
        summary_en = None
        if source_lang != "en":
            english_content = _translate_to_english(content)
            summary_en = english_content

        # Classify
        classified = _classify_political_event(english_content, campaign_dict, candidates_list)
        if not classified:
            continue

        # Store event
        result = db.execute(
            text("""INSERT INTO sp_political_events
                     (campaign_id, event_type, source, title, summary, summary_en,
                      source_lang, sentiment_score, intensity, affected_candidates, raw_json)
                     VALUES (:cid, :et, 'perplexity', :title, :summary, :summary_en,
                             :sl, :sent, :intens, :ac, :raw)
                     RETURNING id"""),
            {"cid": campaign_id,
             "et": classified.get("event_type", "GENERAL"),
             "title": classified.get("title", query)[:200],
             "summary": classified.get("summary", "")[:1000],
             "summary_en": summary_en,
             "sl": source_lang,
             "sent": classified.get("sentiment", 0),
             "intens": classified.get("intensity", 0.5),
             "ac": json.dumps(classified.get("affected_candidates", [])),
             "raw": json.dumps({"query": query, "citations": raw.get("citations", [])})}
        )
        event_id = result.fetchone()[0]
        events_found += 1

        event_data = {
            "event_type": classified.get("event_type", "GENERAL"),
            "title": classified.get("title", ""),
            "summary": classified.get("summary", ""),
            "sentiment_score": classified.get("sentiment", 0),
            "intensity": classified.get("intensity", 0.5),
            "affected_candidates": classified.get("affected_candidates", []),
            "source": "perplexity",
        }

        # Generate alerts
        alerts = _generate_political_alerts(db, event_id, event_data,
                                            campaign_id, candidates_list, config)
        alerts_generated += len(alerts)

        # KG update
        affected_for_kg = []
        for af in classified.get("affected_candidates", []):
            matched = _fuzzy_match_candidate(af.get("name", ""), candidates_list)
            if matched:
                affected_for_kg.append({
                    "candidate_id": matched["id"],
                    "relevance": af.get("relevance", 0.5),
                    "impact": af.get("impact", "neutral"),
                })
        node_id = _add_political_event_to_kg(db, event_id, event_data,
                                              campaign_id, affected_for_kg)
        if node_id:
            kg_nodes_created += 1

    # ── Apify Social Media ───────────────────────────────────
    social_accounts = config.get("social_accounts", {})

    # Twitter
    if search_queries:
        tweets = _check_political_twitter(search_queries[:3])
        if tweets:
            sources_checked.append("apify_twitter")
            # Aggregate tweets for classification
            combined_text = "\n".join(t.get("text", "")[:200] for t in tweets[:10])
            if combined_text.strip():
                classified = _classify_political_event(combined_text, campaign_dict, candidates_list)
                if classified:
                    result = db.execute(
                        text("""INSERT INTO sp_political_events
                                 (campaign_id, event_type, source, title, summary,
                                  sentiment_score, intensity, affected_candidates, raw_json)
                                 VALUES (:cid, :et, 'apify_twitter', :title, :summary,
                                         :sent, :intens, :ac, :raw)
                                 RETURNING id"""),
                        {"cid": campaign_id,
                         "et": classified.get("event_type", "GENERAL"),
                         "title": classified.get("title", "Twitter activity")[:200],
                         "summary": classified.get("summary", "")[:1000],
                         "sent": classified.get("sentiment", 0),
                         "intens": classified.get("intensity", 0.5),
                         "ac": json.dumps(classified.get("affected_candidates", [])),
                         "raw": json.dumps({"tweet_count": len(tweets)})}
                    )
                    event_id = result.fetchone()[0]
                    events_found += 1

    # Facebook
    fb_accounts = social_accounts.get("facebook", [])
    if fb_accounts:
        posts = _check_political_facebook(fb_accounts)
        if posts:
            sources_checked.append("apify_facebook")
            combined_text = "\n".join(p.get("text", "")[:200] for p in posts[:10])
            if combined_text.strip():
                classified = _classify_political_event(combined_text, campaign_dict, candidates_list)
                if classified:
                    result = db.execute(
                        text("""INSERT INTO sp_political_events
                                 (campaign_id, event_type, source, title, summary,
                                  sentiment_score, intensity, affected_candidates, raw_json)
                                 VALUES (:cid, :et, 'apify_facebook', :title, :summary,
                                         :sent, :intens, :ac, :raw)
                                 RETURNING id"""),
                        {"cid": campaign_id,
                         "et": classified.get("event_type", "GENERAL"),
                         "title": classified.get("title", "Facebook activity")[:200],
                         "summary": classified.get("summary", "")[:1000],
                         "sent": classified.get("sentiment", 0),
                         "intens": classified.get("intensity", 0.5),
                         "ac": json.dumps(classified.get("affected_candidates", [])),
                         "raw": json.dumps({"post_count": len(posts)})}
                    )
                    events_found += 1

    # Update last check timestamp
    db.execute(
        text("UPDATE sp_grounding_config SET last_check_at = NOW() WHERE campaign_id = :cid"),
        {"cid": campaign_id}
    )
    db.commit()

    return {
        "status": "success",
        "campaign_id": campaign_id,
        "campaign_name": campaign_dict["name"],
        "events_found": events_found,
        "alerts_generated": alerts_generated,
        "kg_nodes_created": kg_nodes_created,
        "sources_checked": list(set(sources_checked)),
        "search_queries_used": len(search_queries[:5]),
    }


# ── Auto-Config Generator ───────────────────────────────────────

def _auto_generate_queries(db: Session, campaign_id: int, campaign: dict, candidates: list) -> list:
    """Build search queries from campaign candidates + region."""
    queries = []
    if campaign.get("region_code"):
        queries.append(f"{campaign['region_code']} {campaign.get('election_type', 'elections')} elections")
    if campaign.get("country_code"):
        queries.append(f"{campaign['country_code']} politics")
    if campaign.get("name"):
        queries.append(campaign["name"])

    for c in candidates:
        queries.append(c["name"])
        if c.get("party"):
            queries.append(c["party"])

    # Add region-specific parties if available
    try:
        region_data = db.execute(
            text("SELECT data FROM sp_region_political_data WHERE region_code = :rc AND data_type = 'party_support'"),
            {"rc": campaign.get("region_code", "")}
        ).fetchone()
        if region_data and region_data[0]:
            data = region_data[0] if isinstance(region_data[0], dict) else json.loads(region_data[0])
            parties = data.get("parties", [])
            for p in parties[:5]:
                name = p.get("name", "") if isinstance(p, dict) else str(p)
                if name:
                    queries.append(name)
    except Exception:
        pass

    return list(set(q for q in queries if q and len(q) > 1))[:15]


# ── API Endpoints ────────────────────────────────────────────────


@router.post("/grounding/check")
def political_check(
    request: PoliticalCheckRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("sysadmin", "supervisor", "psychologist")),
):
    """Trigger political news check for a campaign."""
    try:
        result = run_political_check(db, request.campaign_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Political check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Political check failed: {str(e)}")


@router.get("/grounding/events/{campaign_id}")
def get_political_events(
    campaign_id: int,
    limit: int = Query(20, ge=1, le=100),
    event_type: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List recent political events for a campaign."""
    query = "SELECT id, event_type, source, title, summary, summary_en, source_lang, sentiment_score, intensity, affected_candidates, kg_node_id, created_at FROM sp_political_events WHERE campaign_id = :cid"
    params = {"cid": campaign_id, "lim": limit}
    if event_type:
        query += " AND event_type = :et"
        params["et"] = event_type
    query += " ORDER BY created_at DESC LIMIT :lim"

    rows = db.execute(text(query), params).fetchall()
    return {
        "campaign_id": campaign_id,
        "events": [
            {"id": r[0], "event_type": r[1], "source": r[2], "title": r[3],
             "summary": r[4], "summary_en": r[5], "source_lang": r[6],
             "sentiment_score": r[7], "intensity": r[8],
             "affected_candidates": r[9], "kg_node_id": r[10],
             "created_at": str(r[11])}
            for r in rows
        ],
        "count": len(rows),
    }


@router.get("/grounding/alerts/{campaign_id}")
def get_political_alerts(
    campaign_id: int,
    unread_only: bool = Query(False),
    severity: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List political alerts for a campaign."""
    query = "SELECT a.id, a.event_id, a.candidate_id, a.alert_type, a.severity, a.title, a.recommended_action, a.bdi_refresh_triggered, a.is_read, a.created_at, c.name as candidate_name FROM sp_political_alerts a LEFT JOIN sp_candidates c ON a.candidate_id = c.id WHERE a.campaign_id = :cid"
    params = {"cid": campaign_id, "lim": limit}
    if unread_only:
        query += " AND a.is_read = FALSE"
    if severity:
        query += " AND a.severity = :sev"
        params["sev"] = severity
    query += " ORDER BY a.created_at DESC LIMIT :lim"

    rows = db.execute(text(query), params).fetchall()
    return {
        "campaign_id": campaign_id,
        "alerts": [
            {"id": r[0], "event_id": r[1], "candidate_id": r[2],
             "alert_type": r[3], "severity": r[4], "title": r[5],
             "recommended_action": r[6], "bdi_refresh_triggered": r[7],
             "is_read": r[8], "created_at": str(r[9]),
             "candidate_name": r[10]}
            for r in rows
        ],
        "count": len(rows),
    }


@router.put("/grounding/alerts/{alert_id}/read")
def mark_alert_read(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Mark a political alert as read."""
    db.execute(
        text("UPDATE sp_political_alerts SET is_read = TRUE WHERE id = :aid"),
        {"aid": alert_id}
    )
    db.commit()
    return {"status": "ok", "alert_id": alert_id}


@router.get("/grounding/config/{campaign_id}")
def get_grounding_config(
    campaign_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get grounding configuration for a campaign."""
    row = db.execute(
        text("SELECT id, campaign_id, search_queries, social_accounts, monitored_candidates, check_interval_hours, sentiment_threshold, event_intensity_threshold, auto_bdi_refresh, source_lang, is_active, last_check_at FROM sp_grounding_config WHERE campaign_id = :cid"),
        {"cid": campaign_id}
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"No config for campaign {campaign_id}")
    return {
        "id": row[0], "campaign_id": row[1],
        "search_queries": row[2], "social_accounts": row[3],
        "monitored_candidates": row[4], "check_interval_hours": row[5],
        "sentiment_threshold": row[6], "event_intensity_threshold": row[7],
        "auto_bdi_refresh": row[8], "source_lang": row[9],
        "is_active": row[10], "last_check_at": str(row[11]) if row[11] else None,
    }


@router.put("/grounding/config/{campaign_id}")
def update_grounding_config(
    campaign_id: int,
    search_queries: Optional[list] = None,
    social_accounts: Optional[dict] = None,
    sentiment_threshold: Optional[float] = None,
    auto_bdi_refresh: Optional[bool] = None,
    source_lang: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("sysadmin", "supervisor")),
):
    """Update grounding config for a campaign."""
    updates = []
    params = {"cid": campaign_id}
    if search_queries is not None:
        updates.append("search_queries = :sq")
        params["sq"] = json.dumps(search_queries)
    if social_accounts is not None:
        updates.append("social_accounts = :sa")
        params["sa"] = json.dumps(social_accounts)
    if sentiment_threshold is not None:
        updates.append("sentiment_threshold = :st")
        params["st"] = sentiment_threshold
    if auto_bdi_refresh is not None:
        updates.append("auto_bdi_refresh = :abr")
        params["abr"] = auto_bdi_refresh
    if source_lang is not None:
        updates.append("source_lang = :sl")
        params["sl"] = source_lang

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    db.execute(
        text(f"UPDATE sp_grounding_config SET {', '.join(updates)} WHERE campaign_id = :cid"),
        params
    )
    db.commit()
    return {"status": "updated", "campaign_id": campaign_id}


@router.post("/grounding/auto-config/{campaign_id}")
def auto_config(
    campaign_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("sysadmin", "supervisor")),
):
    """Auto-generate search queries from campaign candidates + region."""
    campaign = db.execute(
        text("SELECT id, name, code, region_code, country_code, election_type FROM sp_campaigns WHERE id = :cid"),
        {"cid": campaign_id}
    ).fetchone()
    if not campaign:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found")

    campaign_dict = {
        "id": campaign[0], "name": campaign[1], "code": campaign[2],
        "region_code": campaign[3] or "", "country_code": campaign[4] or "",
        "election_type": campaign[5] or "general",
    }

    candidates = db.execute(
        text("SELECT id, name, party, political_stance FROM sp_candidates WHERE campaign_id = :cid"),
        {"cid": campaign_id}
    ).fetchall()
    candidates_list = [{"id": c[0], "name": c[1], "party": c[2], "stance": c[3]} for c in candidates]

    queries = _auto_generate_queries(db, campaign_id, campaign_dict, candidates_list)

    # Upsert config
    db.execute(
        text("""INSERT INTO sp_grounding_config (campaign_id, search_queries)
                 VALUES (:cid, :q)
                 ON CONFLICT (campaign_id) DO UPDATE SET search_queries = EXCLUDED.search_queries"""),
        {"cid": campaign_id, "q": json.dumps(queries)}
    )
    db.commit()

    return {
        "status": "success",
        "campaign_id": campaign_id,
        "queries_generated": len(queries),
        "queries": queries,
    }
