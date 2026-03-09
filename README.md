# Sentinel EU Demo App

Demo app for offender/location monitoring with:
- EU-focused location telemetry simulation (or Kaggle GPX upload)
- alert and risk scoring pipeline
- map/dashboard UI
- local LLM copilot via Ollama (`gpt-oss:20b`) or OCI GenAI
- optional Oracle DB persistence
- template-based synthetic generation from uploaded GPX waypoint/track files

## Version Documentation

- `v0.1.0`: see `docs/v0.1.0.md`

## Project Structure

- `app/main.py`: Streamlit UI
- `app/simulator.py`: EU telemetry/data generator
- `app/kaggle_loader.py`: GPX parser for Kaggle files
- `app/detection.py`: alert/co-location/risk logic
- `app/oracle_repo.py`: Oracle DB connector
- `app/llm_client.py`: local Ollama / OCI GenAI switch
- `sql/schema_oracle_26ai.sql`: baseline Oracle persistence tables
- `sql/schema_oracle_26ai_converged.sql`: Oracle 26ai Spatial + Vector schema for the presentation demo
- `sql/create_sentinel_user.sql`: dedicated Oracle app user/schema bootstrap

## Option A: Run Locally (No Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/main.py
```

Open: `http://localhost:8502`

In the web UI sidebar (`Scenario`), you can choose:
- `Presentation scenarios` for the three-slide demo cases (routine / risk / outside zone)
- `EU synthetic` for the built-in random demo
- `Template-based synthetic` to clone the movement pattern of one uploaded GPX across EU cities
- `GPX upload` to visualize uploaded GPX files directly

The app is now organized into workspaces:
- `Operations Center` for live monitoring, anomaly queue, and map focus
- `Subject Investigation` for one-person drill-down, track review, and copilot
- `Hybrid Intelligence` for vector matches and decision outcomes
- `Data & Scenarios` for raw dataset inspection and scenario-driven runs
- `Oracle Lab` for schema setup, Oracle loading, and hybrid SQL execution

In the web UI sidebar (`Runtime Settings`), configure:
- LLM provider and model (`ollama` or `oci`)
- Oracle DB credentials/DSN
- Use `Test LLM` and `Test DB` buttons before running copilot or data push

## Option B: Docker Compose (Oracle + App, Ollama on macOS host)

1) Start stack

```bash
docker compose up -d --build
```

2) Ensure Ollama is running on your Mac and pull the model locally

```bash
ollama pull gpt-oss:20b
```

3) Open UI

- `http://localhost:8502`
- In sidebar settings, keep Ollama URL as `http://host.docker.internal:11434` when app runs in Docker.

## Oracle Setup

Create the dedicated app user/schema first:

```bash
docker exec -i oracle26ai sqlplus system/OraclePwd123@localhost:1521/FREEPDB1 < sql/create_sentinel_user.sql
```

Then create the app tables inside the `SENTINEL` schema:

```bash
docker exec -i oracle26ai sqlplus sentinel/SentinelPwd123@localhost:1521/FREEPDB1 < sql/schema_oracle_26ai.sql
```

For the full presentation demo with Oracle 26ai Spatial + Vector tables:

```bash
docker exec -i oracle26ai sqlplus sentinel/SentinelPwd123@localhost:1521/FREEPDB1 < sql/schema_oracle_26ai_converged.sql
```

Or use the Streamlit buttons:
- `Setup Converged Schema`
- `Push Hybrid Demo To Oracle`
- `Run Oracle Hybrid Query`

If your Oracle image/service differs, update:
- `ORACLE_IMAGE`
- `ORACLE_DSN`
- `ORACLE_USER` (default: `sentinel`)
- `ORACLE_PASSWORD` (default: `SentinelPwd123`)

## LLM Provider Configuration

You can configure this either with environment variables or directly in the web UI.
Default is local Ollama.

- `LLM_PROVIDER=ollama`
- `OLLAMA_URL=http://localhost:11434`
- `OLLAMA_MODEL=gpt-oss:20b`

For OCI GenAI instead:

- `LLM_PROVIDER=oci`
- `OCI_GENAI_MODEL_ID=...`
- `OCI_COMPARTMENT_OCID=...`
- `OCI_CONFIG_FILE=~/.oci/config`
- `OCI_CONFIG_PROFILE=DEFAULT`

## Kaggle Dataset Usage

- Download GPX files from: <https://www.kaggle.com/datasets/thomasnibb/criminal-location-tracking>
- In the app sidebar, upload one or more `.gpx` files.
- App auto-assigns participant IDs from filenames and runs the same detection pipeline.

