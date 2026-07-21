<div align="center">

<img src="docs/assets/banner.png" alt="NBA Prediction Engine" width="100%" />

# 🏀 NBA Prediction Engine

An end-to-end machine-learning platform that ingests NBA data, engineers leakage-safe features, trains multiple win-probability and player-projection models, and serves predictions through a **FastAPI** service and an interactive **Streamlit** dashboard — fully orchestrated with **Docker**.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Deploy-Docker%20Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![PostgreSQL](https://img.shields.io/badge/DB-PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![ML](https://img.shields.io/badge/ML-scikit--learn%20·%20XGBoost%20·%20PyTorch-F7931E?style=for-the-badge)](#-models)

</div>

---

## 📘 Overview

This repository is a production-shaped NBA prediction engine. It is intentionally structured as a real ML product — from **data ingestion → feature engineering → model training → a served API → a dashboard**, with monitoring, a model registry, and champion-model promotion along the way.

What it does today:

- **Ingests** raw NBA games, current-season rosters, player game logs, injuries/availability, and sportsbook odds.
- **Engineers** leakage-safe rolling team stats, rest / back-to-back / fatigue features, market-odds features, and a custom **team-market quality** rating.
- **Trains** several team win-probability models (baseline, tree, sequential, ensemble) plus a dedicated player-projection model.
- **Serves** normalized home/away win probabilities, fair moneylines, per-feature explanations, and availability-adjusted player projections via FastAPI.
- **Visualizes** everything in a Streamlit dashboard with live API health, monitoring/drift, and prediction-quality panels.

> ⚠️ **Data & models are generated, not committed.** The `data/`, `models/`, and `reports/` directories are produced by the pipeline scripts (which pull from the external NBA API and other sources) and are gitignored due to size. A fresh checkout serves the API/UI shell immediately; prediction endpoints become fully populated after you run the pipeline.

---

## 🎬 Demo

The Streamlit dashboard connects to the FastAPI backend and surfaces model status, pipeline health, monitoring, and prediction-quality panels. Pick an upcoming game and model in the sidebar to generate a win-probability report with player projections and per-feature explanations.

<div align="center">
  <img src="docs/assets/dashboard.webp" alt="Streamlit dashboard connected to the FastAPI backend" width="80%" />
  <br/><sub>Dashboard shell connected to the API (fresh environment, before the data pipeline is run).</sub>
  <br/><br/>
  <img src="docs/assets/health_smoke_test.webp" alt="In-app /health smoke test returning status ok" width="60%" />
  <br/><sub>Built-in smoke test calling the live backend <code>/health</code> endpoint.</sub>
</div>

---

## 🧠 Pipeline

```mermaid
flowchart LR
    subgraph Ingest
        A1[nba_api games] --> RAW[(data/raw)]
        A2[rosters & player logs] --> RAW
        A3[injuries / availability] --> RAW
        A4[sportsbook odds] --> RAW
    end

    RAW --> FE[Feature engineering<br/>rolling stats · rest/fatigue<br/>market · team-market quality]
    FE --> PROC[(data/processed)]

    PROC --> TR[Model training<br/>baseline · tree · sequential<br/>ensemble · player projection]
    TR --> REG[(Model registry<br/>+ champion promotion)]

    REG --> API[FastAPI service]
    PROC --> API
    API --> UI[Streamlit dashboard]

    API --> MON[Monitoring & quality<br/>drift / PSI · prediction log]
    SCH[Scheduler] -. periodic refresh .-> Ingest
    DB[(PostgreSQL)] --- API
```

### Features

- **Leakage-safe rolling stats** — points/rebounds/assists form computed strictly from prior games.
- **Rest & fatigue** — rest days, back-to-back flags, travel/timezone context, and a fatigue index.
- **Availability** — injury severity and out/questionable/probable status folded into team and player projections.
- **Market features** — sportsbook odds → implied team totals and market signals, with backtesting utilities.
- **Team-market quality rating** — a custom per-team quality index used as an engineered feature (maintained via `scripts/`).

### Models

| Model | Type | Purpose |
| --- | --- | --- |
| Baseline | Logistic regression (scaled) | Interpretable team win probability + linear feature contributions |
| Tree | XGBoost | Non-linear win probability + SHAP-style contributions |
| Sequential | PyTorch | Sequence/form-aware team model |
| Ensemble | Blended | Combines model signals |
| Player projection | Two-stage (minutes → per-minute rates) | Availability-adjusted points/rebounds/assists with confidence bands |

### Results

No metrics are reported here because none should be fabricated. After training on your ingested dataset, populate this table from `scripts/generate_prediction_quality_report.py` and the model registry:

| Model | Accuracy | Log Loss | Brier | ROC-AUC |
| --- | :---: | :---: | :---: | :---: |
| Baseline | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Tree | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Ensemble | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

---

## 🚀 How to Run / Install

### Option 1 — Docker (recommended)

Brings up the full stack (PostgreSQL, API, scheduler, and dashboard) in one synchronized environment.

```bash
git clone https://github.com/KhaNguyen-Q/nba-prediction-engine
cd nba-prediction-engine
docker-compose up --build
```

- **Dashboard:** http://localhost:8501
- **API docs:** http://localhost:8000/docs

### Option 2 — Local Python

```bash
pip install -r requirements.txt          # torch/xgboost/lightgbm — first install is large

python scripts/update_pipeline.py        # ingest → features → train (needs network for NBA data)

uvicorn api.main:app --host 0.0.0.0 --port 8000     # terminal 1
streamlit run streamlit_app.py                      # terminal 2
```

Set the sidebar **API Base URL** to `http://127.0.0.1:8000` when running locally. To boot the API before any models/data exist (shell only), set `STRICT_STARTUP_CHECKS=0`.

---

## 📁 Folder Structure

```
nba-prediction-engine/
├─ api/                 # FastAPI service — win-probability & player-projection endpoints
│  └─ main.py
├─ scripts/             # Ingestion, feature engineering, training, monitoring, utilities
│  ├─ update_pipeline.py        # Orchestrates ingest → features → train
│  ├─ get_data.py / fetch_*.py  # Data ingestion (games, players, injuries, odds, schedule)
│  ├─ build_features.py         # Leakage-safe feature engineering
│  ├─ build_inference_features.py
│  ├─ train_baseline.py / train_tree_model.py / train_sequential.py
│  ├─ train_ensemble.py / train_player_model.py / train_automl_challenger.py
│  ├─ model_utils.py / model_promotion.py   # Registry & champion promotion
│  ├─ generate_monitoring_report.py         # Freshness + drift (PSI)
│  ├─ generate_prediction_quality_report.py # Logged-prediction quality
│  └─ smoke_test.py
├─ config/             # Canonical NBA team metadata & static config
├─ streamlit_app.py    # Streamlit dashboard
├─ docker-compose.yml  # api + scheduler + streamlit + postgres
├─ Dockerfile
├─ requirements.txt
├─ data/               # (generated) raw & processed datasets — gitignored
├─ models/             # (generated) trained artifacts & registry — gitignored
└─ reports/            # (generated) monitoring & quality reports — gitignored
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the documented step-by-step pipeline flows.

---

## 💡 Lessons Learned

- **Leakage safety is a feature, not an afterthought.** Rolling stats and labels are computed strictly from past games so validation reflects real forecasting.
- **Separate the shell from the data.** Keeping generated `data/`/`models/` out of the repo — and letting the API boot in a degraded-but-honest state — makes the project easy to clone and reason about.
- **Explainability builds trust.** Linear contributions (baseline) and tree contributions (XGBoost) turn a probability into an inspectable decision.
- **Operate the model, don't just train it.** A registry, champion promotion, drift/PSI monitoring, and a prediction log turn a notebook model into a maintainable service.

---

## 🔭 Future Improvements

- Player-level regression beyond the current two-stage projection (opponent-adjusted matchup priors).
- Calibrated probability outputs and richer backtesting/betting-edge analysis.
- CI that runs `scripts/smoke_test.py` and publishes the results table automatically.
- Managed deployment (container registry + hosted Postgres) and scheduled retraining.
- Expanded market features and automated data-quality gating before promotion.

---

<div align="center"><sub>FastAPI · Streamlit · scikit-learn · XGBoost · PyTorch · PostgreSQL · Docker</sub></div>
