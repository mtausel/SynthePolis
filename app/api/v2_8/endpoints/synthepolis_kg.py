"""
Phase 5 — SynthePolis BDI + Knowledge Graph Transfer.
Builds Knowledge Graphs from SynthePolis campaign data:
candidates, political incidents, documents, parties, issues, studies.
"""
import json
import hashlib
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional

from app.core.security import get_current_user, require_role
from app.models.user import User
from app.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Helper: Add KG Node ──────────────────────────────────────────


def _add_node(db: Session, vertical: str, node_type: str, entity_id: int,
              label: str, properties: dict, node_subtype: str = None) -> int:
    """Insert a KG node and return its ID."""
    result = db.execute(
        text("""INSERT INTO kg_nodes (vertical, node_type, node_subtype, entity_id, label, properties)
                 VALUES (:v, :nt, :ns, :eid, :label, :props)
                 ON CONFLICT (vertical, node_type, entity_id, node_subtype) DO UPDATE
                 SET label = EXCLUDED.label, properties = EXCLUDED.properties
                 RETURNING id"""),
        {"v": vertical, "nt": node_type, "ns": node_subtype or "",
         "eid": entity_id, "label": label[:200], "props": json.dumps(properties)}
    )
    return result.fetchone()[0]


def _add_edge(db: Session, vertical: str, source_id: int, target_id: int,
              edge_type: str, weight: float = 1.0, properties: dict = None) -> int:
    """Insert a KG edge and return its ID."""
    result = db.execute(
        text("""INSERT INTO kg_edges (vertical, source_node_id, target_node_id, edge_type, weight, properties)
                 VALUES (:v, :src, :tgt, :et, :w, :props)
                 RETURNING id"""),
        {"v": vertical, "src": source_id, "tgt": target_id,
         "et": edge_type, "w": weight, "props": json.dumps(properties or {})}
    )
    return result.fetchone()[0]


def _find_node(db: Session, vertical: str, node_type: str, entity_id: int) -> Optional[int]:
    """Find a KG node by type and entity_id, return its ID."""
    row = db.execute(
        text("SELECT id FROM kg_nodes WHERE vertical = :v AND node_type = :nt AND entity_id = :eid"),
        {"v": vertical, "nt": node_type, "eid": entity_id}
    ).fetchone()
    return row[0] if row else None


def _str_hash(s: str) -> int:
    """Deterministic hash for string -> integer entity_id."""
    return int(hashlib.md5(s.encode()).hexdigest()[:7], 16) % 10_000_000


# ── BDI Data Loader ──────────────────────────────────────────────


def _load_synthepolis_data(db: Session, entity_id: str, entity_type: str = "political_actor") -> dict:
    """
    Load SynthePolis data for BDI extraction.
    Supports political_actor (candidate) entity type.
    """
    if entity_type == "political_actor":
        # entity_id format: 'candidate_42' or just '42'
        cand_id = int(entity_id.split("_")[-1]) if "_" in entity_id else int(entity_id)

        cand = db.execute(
            text("SELECT id, name, party, political_stance, key_policies, biography FROM sp_candidates WHERE id = :cid"),
            {"cid": cand_id}
        ).fetchone()
        if not cand:
            return {}

        # Get documents
        docs = db.execute(
            text("SELECT doc_type, title, content FROM sp_candidate_documents WHERE candidate_id = :cid"),
            {"cid": cand_id}
        ).fetchall()

        # Get incidents (if table exists)
        incidents = []
        try:
            incidents = db.execute(
                text("""SELECT tag, title, COALESCE(excerpt_en, excerpt) as excerpt,
                         markers, confidence FROM sp_political_incidents
                         WHERE candidate_id = :cid AND review_status = 'approved'
                         ORDER BY confidence DESC"""),
                {"cid": cand_id}
            ).fetchall()
        except Exception:
            pass  # Table may not exist

        # Get persona
        persona = db.execute(
            text("SELECT persona_json FROM sp_candidate_personas WHERE candidate_id = :cid ORDER BY generated_at DESC LIMIT 1"),
            {"cid": cand_id}
        ).fetchone()

        # Aggregate markers from incidents
        marker_agg = {}
        for inc in incidents:
            ms = inc[3] if isinstance(inc[3], list) else []
            for m in ms:
                k = m if isinstance(m, str) else (m.get("type", "") if isinstance(m, dict) else str(m))
                marker_agg.setdefault(k, []).append(float(inc[4] or 0.5))

        return {
            "interviews": [f"[{d[0]}] {d[1]}: {(d[2] or '')[:500]}" for d in docs],
            "incidents": [{"type": i[0], "description": i[2] or "", "title": i[1] or "",
                           "confidence": float(i[4] or 0.5)} for i in incidents],
            "markers": {k: sum(v) / len(v) for k, v in marker_agg.items()} if marker_agg else {},
            "profile": {
                "name": cand[1],
                "party": cand[2],
                "stance": cand[3],
                "policies": cand[4] or [],
                "biography": (cand[5] or "")[:500],
                "persona": persona[0] if persona else None,
            },
        }

    return {}


# ── Build SynthePolis KG ─────────────────────────────────────────


def build_synthepolis_graph(db: Session, campaign_id: int) -> dict:
    """
    Build a Knowledge Graph for a SynthePolis campaign.
    Creates nodes for candidates, incidents, issues, parties, documents.
    """
    campaign = db.execute(
        text("SELECT id, name, code, election_type, region_code, status FROM sp_campaigns WHERE id = :cid"),
        {"cid": campaign_id}
    ).fetchone()
    if not campaign:
        raise ValueError(f"Campaign {campaign_id} not found")

    nodes_created = 0
    edges_created = 0

    # 1. Campaign node
    camp_node_id = _add_node(db, "synthepolis", "campaign", campaign_id,
        campaign[1] or f"Campaign #{campaign_id}", {
            "code": campaign[2], "election_type": campaign[3],
            "region_code": campaign[4], "status": campaign[5],
        })
    nodes_created += 1

    # 2. Candidate nodes + party nodes
    candidates = db.execute(
        text("SELECT id, name, party, political_stance, key_policies, biography FROM sp_candidates WHERE campaign_id = :cid"),
        {"cid": campaign_id}
    ).fetchall()

    parties = {}
    candidate_node_map = {}

    for c in candidates:
        cand_entity_id = 100000 + c[0]  # offset to avoid collisions
        cand_node_id = _add_node(db, "synthepolis", "political_actor", cand_entity_id,
            f"{c[1]} ({c[2]})" if c[2] else c[1], {
                "candidate_id": c[0], "party": c[2], "stance": c[3],
                "policies": c[4] or [],
            })
        candidate_node_map[c[0]] = cand_node_id
        nodes_created += 1

        # Edge: candidate -> campaign (PARTICIPATES_IN)
        _add_edge(db, "synthepolis", cand_node_id, camp_node_id, "PARTICIPATES_IN", 1.0)
        edges_created += 1

        # Party node
        if c[2] and c[2] not in parties:
            party_entity_id = _str_hash(f"party_{c[2]}")
            parties[c[2]] = _add_node(db, "synthepolis", "party", party_entity_id,
                c[2], {"party_name": c[2]})
            nodes_created += 1

        if c[2] and c[2] in parties:
            _add_edge(db, "synthepolis", cand_node_id, parties[c[2]], "MEMBER_OF", 1.0)
            edges_created += 1

    # 3. Political incidents (if table exists)
    tag_map = {
        "policy_position": "policy", "promise_commitment": "policy",
        "vulnerability": "credibility", "strength_signal": "leadership",
        "coalition_signal": "alliances", "voter_appeal": "electability",
        "media_behavior": "media_relations", "contradiction": "credibility",
        "rhetorical_pattern": "communication",
    }
    issues = {}

    try:
        incidents = db.execute(
            text("""SELECT i.id, i.candidate_id, i.tag, i.title, i.excerpt, i.confidence
                     FROM sp_political_incidents i
                     JOIN sp_candidates c ON i.candidate_id = c.id
                     WHERE c.campaign_id = :cid AND i.review_status = 'approved'"""),
            {"cid": campaign_id}
        ).fetchall()

        for inc in incidents:
            inc_entity_id = 200000 + inc[0]
            inc_node_id = _add_node(db, "synthepolis", "political_incident", inc_entity_id,
                (inc[3] or inc[2] or "Incident")[:200], {
                    "tag": inc[2], "confidence": float(inc[5] or 0),
                    "candidate_id": inc[1],
                })
            nodes_created += 1

            # Edge: candidate -> incident (HAS_INCIDENT)
            if inc[1] in candidate_node_map:
                _add_edge(db, "synthepolis", candidate_node_map[inc[1]], inc_node_id,
                    "HAS_INCIDENT", float(inc[5] or 0.5))
                edges_created += 1

            # Issue node (from tag mapping)
            issue_name = tag_map.get(inc[2], inc[2] or "general")
            if issue_name and issue_name not in issues:
                issue_entity_id = _str_hash(f"issue_{issue_name}")
                issues[issue_name] = _add_node(db, "synthepolis", "political_issue",
                    issue_entity_id, issue_name, {"issue_category": issue_name})
                nodes_created += 1

            if issue_name and issue_name in issues:
                _add_edge(db, "synthepolis", inc_node_id, issues[issue_name],
                    "RELATED_TO", float(inc[5] or 0.5))
                edges_created += 1

    except Exception as e:
        logger.warning(f"sp_political_incidents not available: {e}")

    # 4. Documents -> media_coverage nodes
    docs = db.execute(
        text("""SELECT d.id, d.candidate_id, d.doc_type, d.title
                 FROM sp_candidate_documents d
                 JOIN sp_candidates c ON d.candidate_id = c.id
                 WHERE c.campaign_id = :cid"""),
        {"cid": campaign_id}
    ).fetchall()

    for d in docs:
        doc_entity_id = 300000 + d[0]
        doc_node_id = _add_node(db, "synthepolis", "media_coverage", doc_entity_id,
            (d[3] or f"{d[2]} document")[:200], {"doc_type": d[2], "candidate_id": d[1]})
        nodes_created += 1

        if d[1] in candidate_node_map:
            _add_edge(db, "synthepolis", candidate_node_map[d[1]], doc_node_id, "COVERED_BY", 1.0)
            edges_created += 1

    # 5. Completed studies -> study_result nodes
    studies = db.execute(
        text("SELECT id, study_type, title, status FROM sp_studies WHERE campaign_id = :cid AND status = 'completed'"),
        {"cid": campaign_id}
    ).fetchall()

    for s in studies:
        study_entity_id = 400000 + s[0]
        _add_node(db, "synthepolis", "study_result", study_entity_id,
            (s[2] or f"{s[1]} #{s[0]}")[:200], {"study_type": s[1], "study_id": s[0]})
        nodes_created += 1

    db.commit()

    return {
        "status": "success",
        "campaign_id": campaign_id,
        "campaign_name": campaign[1],
        "nodes_created": nodes_created,
        "edges_created": edges_created,
        "candidates": len(candidates),
        "parties": len(parties),
        "incidents": len(incidents) if 'incidents' in dir() else 0,
        "documents": len(docs),
    }


# ── BDI Extraction ───────────────────────────────────────────────


def extract_synthepolis_bdi(db: Session, candidate_id: int) -> dict:
    """Extract a BDI profile for a SynthePolis political actor."""
    data = _load_synthepolis_data(db, str(candidate_id), "political_actor")
    if not data:
        raise ValueError(f"No data found for candidate {candidate_id}")

    beliefs = []
    desires = []
    intentions = []

    # Beliefs from documents (interviews)
    for interview in data.get("interviews", []):
        beliefs.append({"content": interview[:300], "confidence": 0.7, "source": "document"})

    # Beliefs from incidents
    for inc in data.get("incidents", []):
        beliefs.append({"content": f"[{inc['type']}] {inc.get('title', '')}: {inc['description'][:200]}",
                        "confidence": inc.get("confidence", 0.5), "source": "incident"})

    # Desires from profile
    profile = data.get("profile", {})
    if profile.get("policies"):
        policies = profile["policies"]
        if isinstance(policies, list):
            for p in policies:
                desires.append({"content": str(p), "strength": 0.8, "source": "key_policy"})
        elif isinstance(policies, dict):
            for k, v in policies.items():
                desires.append({"content": f"{k}: {v}", "strength": 0.8, "source": "key_policy"})
    if profile.get("stance"):
        desires.append({"content": f"Political stance: {profile['stance']}", "strength": 0.9, "source": "stance"})

    # Intentions from markers
    for k, v in data.get("markers", {}).items():
        intentions.append({"content": k, "score": v, "source": "incident_markers"})

    context_summary = f"Political actor: {profile.get('name', '?')}, party: {profile.get('party', '?')}. " \
                      f"Stance: {profile.get('stance', 'unknown')}."

    entity_id = 100000 + candidate_id

    existing = db.execute(
        text("SELECT id FROM bdi_profiles WHERE vertical = 'synthepolis' AND entity_id = :eid"),
        {"eid": entity_id}
    ).fetchone()

    if existing:
        db.execute(
            text("""UPDATE bdi_profiles SET beliefs = :b, desires = :d, intentions = :i,
                     context_summary = :cs, extraction_model = 'rule-based', updated_at = NOW()
                     WHERE id = :pid"""),
            {"b": json.dumps(beliefs), "d": json.dumps(desires), "i": json.dumps(intentions),
             "cs": context_summary, "pid": existing[0]}
        )
        profile_id = existing[0]
    else:
        result = db.execute(
            text("""INSERT INTO bdi_profiles (vertical, entity_type, entity_id, beliefs, desires, intentions,
                     context_summary, extraction_model, extraction_version, created_at, updated_at)
                     VALUES ('synthepolis', 'political_actor', :eid, :b, :d, :i, :cs, 'rule-based', 1, NOW(), NOW())
                     RETURNING id"""),
            {"eid": entity_id, "b": json.dumps(beliefs), "d": json.dumps(desires),
             "i": json.dumps(intentions), "cs": context_summary}
        )
        profile_id = result.fetchone()[0]

    db.commit()

    return {
        "profile_id": profile_id,
        "entity_id": entity_id,
        "candidate_id": candidate_id,
        "candidate_name": profile.get("name"),
        "beliefs_count": len(beliefs),
        "desires_count": len(desires),
        "intentions_count": len(intentions),
        "context_summary": context_summary,
    }


# ── API Endpoints ────────────────────────────────────────────────


@router.post("/build-campaign/{campaign_id}")
def build_campaign_kg(
    campaign_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("sysadmin", "supervisor")),
):
    """Build Knowledge Graph from a SynthePolis campaign."""
    try:
        result = build_synthepolis_graph(db, campaign_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/campaign-graph/{campaign_id}")
def get_campaign_graph(
    campaign_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get the full KG subgraph for a SynthePolis campaign."""
    # Find all nodes connected to this campaign
    camp_node = db.execute(
        text("SELECT id FROM kg_nodes WHERE vertical = 'synthepolis' AND node_type = 'campaign' AND entity_id = :eid"),
        {"eid": campaign_id}
    ).fetchone()

    if not camp_node:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} KG not built yet. POST /build-campaign/{campaign_id} first.")

    nodes = db.execute(
        text("SELECT id, node_type, node_subtype, entity_id, label, properties FROM kg_nodes WHERE vertical = 'synthepolis' ORDER BY node_type, entity_id")
    ).fetchall()

    node_ids = [n[0] for n in nodes]
    edges = []
    if node_ids:
        edges = db.execute(
            text("""SELECT id, source_node_id, target_node_id, edge_type, weight, properties
                    FROM kg_edges WHERE vertical = 'synthepolis' ORDER BY edge_type""")
        ).fetchall()

    return {
        "campaign_id": campaign_id,
        "nodes": [
            {"id": n[0], "type": n[1], "subtype": n[2], "entity_id": n[3],
             "label": n[4], "properties": n[5]}
            for n in nodes
        ],
        "edges": [
            {"id": e[0], "source": e[1], "target": e[2], "type": e[3],
             "weight": e[4], "properties": e[5]}
            for e in edges
        ],
        "summary": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "node_types": {nt: sum(1 for n in nodes if n[1] == nt) for nt in set(n[1] for n in nodes)},
        },
    }


@router.get("/candidate/{candidate_id}")
def get_candidate_kg(
    candidate_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get the KG neighborhood of a specific candidate."""
    cand_entity_id = 100000 + candidate_id
    cand_node = db.execute(
        text("SELECT id, label, properties FROM kg_nodes WHERE vertical = 'synthepolis' AND node_type = 'political_actor' AND entity_id = :eid"),
        {"eid": cand_entity_id}
    ).fetchone()

    if not cand_node:
        raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found in KG")

    edges = db.execute(
        text("""SELECT e.id, e.source_node_id, e.target_node_id, e.edge_type, e.weight, e.properties,
                       n.node_type, n.label, n.properties as node_props
                FROM kg_edges e
                JOIN kg_nodes n ON (n.id = CASE WHEN e.source_node_id = :nid THEN e.target_node_id ELSE e.source_node_id END)
                WHERE e.vertical = 'synthepolis'
                AND (e.source_node_id = :nid OR e.target_node_id = :nid)"""),
        {"nid": cand_node[0]}
    ).fetchall()

    bdi_profile = db.execute(
        text("SELECT beliefs, desires, intentions, context_summary FROM bdi_profiles WHERE vertical = 'synthepolis' AND entity_id = :eid"),
        {"eid": cand_entity_id}
    ).fetchone()

    return {
        "candidate": {
            "node_id": cand_node[0],
            "label": cand_node[1],
            "properties": cand_node[2],
        },
        "connections": [
            {"edge_id": e[0], "edge_type": e[3], "weight": e[4],
             "direction": "outgoing" if e[1] == cand_node[0] else "incoming",
             "connected_node": {"type": e[6], "label": e[7], "properties": e[8]}}
            for e in edges
        ],
        "bdi_profile": {
            "beliefs": bdi_profile[0],
            "desires": bdi_profile[1],
            "intentions": bdi_profile[2],
            "context_summary": bdi_profile[3],
        } if bdi_profile else None,
    }


@router.post("/extract-bdi-candidate/{candidate_id}")
def extract_bdi_candidate(
    candidate_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("sysadmin", "supervisor", "psychologist")),
):
    """Extract BDI profile for a SynthePolis political actor."""
    try:
        result = extract_synthepolis_bdi(db, candidate_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/synthepolis-stats")
def synthepolis_kg_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get SynthePolis KG statistics."""
    nodes = db.execute(
        text("SELECT node_type, COUNT(*) FROM kg_nodes WHERE vertical = 'synthepolis' GROUP BY node_type ORDER BY 1")
    ).fetchall()
    edges = db.execute(
        text("SELECT edge_type, COUNT(*) FROM kg_edges WHERE vertical = 'synthepolis' GROUP BY edge_type ORDER BY 1")
    ).fetchall()
    profiles = db.execute(
        text("SELECT COUNT(*) FROM bdi_profiles WHERE vertical = 'synthepolis'")
    ).fetchone()

    return {
        "vertical": "synthepolis",
        "nodes": {r[0]: r[1] for r in nodes},
        "edges": {r[0]: r[1] for r in edges},
        "total_nodes": sum(r[1] for r in nodes),
        "total_edges": sum(r[1] for r in edges),
        "bdi_profiles": profiles[0] if profiles else 0,
    }
