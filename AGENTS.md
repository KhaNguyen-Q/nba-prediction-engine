# AGENTS.md

## Cursor Cloud specific instructions

This repo is an NBA prediction engine with two runnable services plus a batch pipeline:
- FastAPI backend: `api/main.py` (default port 8000).
- Streamlit dashboard: `streamlit_app.py` (default port 8501), which is the primary dev app (see `.devcontainer/devcontainer.json`).
- Pipeline/training scripts in `scripts/` (see `CONTRIBUTING.md` for the documented flows).

Python dependencies are installed into a virtualenv at `.venv/` (gitignored) by the Cloud Agent update script. Run tools via `.venv/bin/...` (e.g. `.venv/bin/uvicorn`, `.venv/bin/streamlit`). `requirements.txt` pulls in `xgboost` and related serving deps; the first install can take a bit.

Non-obvious gotchas:
- The FastAPI app runs startup checks that **raise and abort startup** when model/data artifacts are missing, unless you set `STRICT_STARTUP_CHECKS=0`. In a fresh dev environment (no generated data/models) you must start it as:
  `STRICT_STARTUP_CHECKS=0 .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000`
- Start the API before the dashboard. The dashboard reads the backend URL from the `API_URL` env var (default `http://127.0.0.1:8000`); it also exposes an "API Base URL" field in the sidebar.
- Run the dashboard with `.venv/bin/streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true`.
- The `data/`, `models/`, and `reports/` directories are gitignored and empty by default. They are produced by `scripts/` (`run_pipeline.py`, `update_pipeline.py`, `train_*.py`) which fetch from the external NBA API (`stats.nba.com`) and other sources over the network. Until that data exists, the prediction/upcoming-games endpoints and dashboard prediction panels return expected "unavailable"/`503`/`not found` warnings — this is normal for a fresh environment, not a setup failure. Health/features endpoints and the dashboard shell still work.
- `docker-compose.yml` defines the full stack (api + scheduler + streamlit + postgres) but is not required for local dev; the devcontainer flow runs only Streamlit + API.
