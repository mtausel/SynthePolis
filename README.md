# SynthePolis

**Political Intelligence & Campaign Simulation Platform**

Part of the Synthesight ecosystem. SynthePolis provides political campaign monitoring, candidate analysis, and BDI (Belief-Desire-Intention) simulation for political actors.

## Components

### Knowledge Graph (`synthepolis_kg.py`)
- Campaign KG builder: political_actor, party, political_incident, political_issue, media_coverage nodes
- BDI extraction for political actors
- Endpoints: build-campaign, campaign-graph, candidate, extract-bdi-candidate, synthepolis-stats

### Political News Connector (`sp_grounding.py`)
- Perplexity API: real-time political news monitoring
- Apify integration: Twitter/Facebook social media monitoring
- Political event classification (12 types: STANCE_CHANGE, SCANDAL, POLICY_SHIFT, etc.)
- Candidate-aware alert generation with AI strategic recommendations
- Auto BDI refresh when significant events affect candidates
- KG auto-update: political_event nodes + AFFECTS_CANDIDATE edges
- Bilingual support: non-English content auto-translated

### BDI Integration (`kg_integration.py`)
- KG context injection into BDI system prompts
- Political incident context for simulation

## Database Tables

| Table | Purpose |
|-------|---------|
| sp_campaigns | Campaign definitions (region, election type) |
| sp_candidates | Political actors/candidates |
| sp_candidate_documents | Candidate documents/interviews |
| sp_political_incidents | Political incidents affecting candidates |
| sp_candidate_personas | AI-generated candidate personas |
| sp_political_events | Raw events from Perplexity/Apify monitoring |
| sp_political_alerts | Strategic alerts for campaign teams |
| sp_grounding_config | Per-campaign monitoring configuration |

## API Endpoints

### KG Endpoints (`/api/v2.8/kg/pol/`)
- `POST /build-campaign/{id}` — Build KG from campaign data
- `GET /campaign-graph/{id}` — Full campaign subgraph
- `GET /candidate/{id}` — Candidate KG neighborhood
- `POST /extract-bdi-candidate/{id}` — Extract BDI profile
- `GET /synthepolis-stats` — KG statistics

### Grounding Endpoints (`/api/v2.8/sp/`)
- `POST /grounding/check` — Trigger political news check
- `GET /grounding/events/{campaign_id}` — List political events
- `GET /grounding/alerts/{campaign_id}` — List political alerts
- `PUT /grounding/alerts/{alert_id}/read` — Mark alert read
- `GET /grounding/config/{campaign_id}` — Get monitoring config
- `PUT /grounding/config/{campaign_id}` — Update config
- `POST /grounding/auto-config/{campaign_id}` — Auto-generate search queries

## Version History

- **v1.3.22** — Phase 6: Political news monitoring (Perplexity + Apify)
- **v1.3.21** — Phase 5: SynthePolis KG builder + BDI extraction
