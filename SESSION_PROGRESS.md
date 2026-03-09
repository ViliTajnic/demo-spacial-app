# Session Progress (Saved: 2026-03-06)

## Completed
- Built Sentinel EU demo web app with:
  - synthetic EU telemetry generation
  - Kaggle GPX upload support
  - alert detection (zone/device)
  - simultaneous co-location detection
  - participant risk scoring
  - map/dashboard UI in Streamlit
- Added Oracle schema and optional persistence integration.
- Added LLM copilot abstraction:
  - default local Ollama
  - optional OCI GenAI
- Added runtime web configuration panel for:
  - DB connection (user/password/dsn)
  - LLM provider/model/endpoint
  - test buttons for DB and LLM
- Updated Docker setup:
  - removed Ollama container
  - app in Docker calls native macOS Ollama at `host.docker.internal:11434`

## Current Key Files
- `app/main.py`
- `app/simulator.py`
- `app/kaggle_loader.py`
- `app/detection.py`
- `app/oracle_repo.py`
- `app/llm_client.py`
- `app/oci_genai.py`
- `sql/schema_oracle_26ai.sql`
- `docker-compose.yml`
- `Dockerfile`
- `README.md`

## Run Locally (Recommended)
```bash
cd "/Users/vili/Applications/sentinel demo"
source .venv/bin/activate
streamlit run app/main.py
```
Open: http://localhost:8501

## Run With Docker (Oracle + App, host Ollama)
```bash
cd "/Users/vili/Applications/sentinel demo"
docker compose up -d --build
```
Then ensure on macOS host:
```bash
ollama pull gpt-oss:20b
```
In app sidebar settings, set:
- LLM Provider: `ollama`
- Ollama URL: `http://host.docker.internal:11434`
- Model: `gpt-oss:20b`

## Oracle Notes
- Create tables from:
  - `sql/schema_oracle_26ai.sql`
- In app sidebar set DB credentials and use:
  - `Test DB`
  - `Push Data To Oracle`

## Next Suggested Work
1. Add incident lifecycle for co-location (`OPEN/IN_PROGRESS/CLOSED`).
2. Add sequential co-location detection logic.
3. Add replay mode for near-real-time simulation.
4. Add stricter EU geofence presets and policy packs.
5. Add API layer (FastAPI) for external integrations.
