# Contributing to NBA Prediction Project

Thanks for helping improve this project. This guide explains how to contribute, maintain consistency, and keep documentation aligned with code changes.

## 1. Project purpose

This repository is a modern NBA prediction engine focused on:
- team win-probability prediction
- NBA data ingestion (games, players, injuries, odds, schedule)
- leakage-safe feature engineering (rolling form, rest/fatigue, travel)
- baseline logistic regression, XGBoost, AutoML challenger, and player projection
- FastAPI serving + Streamlit dashboard + Docker Compose orchestration

## 2. Working with the repository

### Recommended workflow

1. Fork or clone the repository.
2. Create a new branch for every feature or fix.
   - Example: `feature/team-location-metadata`
   - Example: `fix/nba-team-filter`
3. Make small, focused commits.
4. Update documentation for any architecture, feature, or pipeline change.
5. Run the relevant scripts and verify behavior.
6. Open a pull request with a short summary and testing notes.

### Branch naming

- `feature/...` for new capabilities
- `fix/...` for bug fixes
- `docs/...` for documentation updates
- `refactor/...` for code organization or style improvements

## 3. Required documentation updates

This project uses documentation as a source of truth.
Whenever any of the following change, update documentation immediately:
- new scripts or major refactors in `scripts/`
- new config files in `config/`
- new model artifacts in `models/`
- new API routes or service patterns in `api/`
- data flow changes or new feature engineering logic

Primary docs to update:
- `README.md`
- `CONTRIBUTING.md`
- file-level docstrings and comments

## 4. Project layout summary

- `api/` — FastAPI service layer
- `config/` — canonical team metadata and location/timezone maps
- `scripts/` — ingestion, feature engineering, training, monitoring, utilities
- `streamlit_app.py` — interactive dashboard
- `data/raw/` — raw ingestion outputs (generated, gitignored)
- `data/processed/` — engineered datasets (generated, gitignored)
- `models/` — saved model artifacts + registry (generated, gitignored)
- `reports/` — monitoring / quality reports (generated, gitignored)

## 5. How to run core flows

### 1. Ingest games

```bash
python scripts/get_data.py
```

This fetches raw NBA games and saves them to `data/raw/games_raw.csv`.

### 2. Ingest additional sources

```bash
python scripts/fetch_players.py
python scripts/fetch_injuries.py
python scripts/fetch_odds.py
python scripts/fetch_schedule.py
```

These scripts ingest players, injuries, odds, and upcoming schedule (with cache/empty-file fallbacks on failure).

### 3. Build features

```bash
python scripts/build_features.py
python scripts/build_inference_features.py
```

Training features are leakage-safe historical rows. Inference features recompute upcoming-matchup fields for serving.

If `stats.nba.com` is unreachable, you can generate an offline synthetic dataset first:

```bash
python scripts/simulate_offline_dataset.py
```

Then run the feature/train scripts as usual (README metrics table was produced this way when live ingest timed out).

### 4. Train models

```bash
python scripts/train_baseline.py
python scripts/train_tree_model.py
python scripts/train_automl_challenger.py
python scripts/train_player_model.py
python scripts/model_promotion.py
```

### 5. Run the full pipeline

```bash
python scripts/run_pipeline.py
# or production-style refresh:
python scripts/update_pipeline.py
```

## 6. Coding conventions

- Keep code readable and modular.
- Use `config/` for shared static data.
- Use `scripts/team_utils.py` for any NBA team validation or lookup.
- Avoid hard-coding NBA team IDs or abbreviations in multiple places.
- Prefer explicit feature names and comment non-obvious logic.
- Do not reintroduce unfinished experiment stubs into the serve/promotion path without registry + holdout metrics + API wiring.

## 7. Testing and validation

- Run scripts locally after any code change.
- Confirm `data/processed/games_with_features.csv` regenerates cleanly.
- Confirm baseline/tree scripts train and save artifacts under `models/`.
- Prefer game-level evaluation helpers in `scripts/model_utils.py` for classification metrics.
- If you add new data columns, describe them in `README.md`.

## 8. Adding new features

When adding a new feature or data source:
1. Add any config metadata to `config/`.
2. Add helper functions to `scripts/team_utils.py` or a new util module.
3. Read and filter raw data in the relevant `scripts/fetch_*.py` / `get_data.py`.
4. Add feature engineering logic in `scripts/build_features.py` (and mirror in inference if needed).
5. Add model training or evaluation in the relevant `train_*.py` script.
6. Update docs.

## 9. Notes for future maintainers

- `config/nba_teams.json` is the project’s canonical NBA team source.
- `scripts/team_utils.py` is the shared team validation API.
- Keep `README.md` and `CONTRIBUTING.md` synchronized.
- Postgres in Compose is provisioned for future persistence; current serving is file-artifact based.
- Production path today: baseline / tree / AutoML challenger / player projection → FastAPI → Streamlit.

---

_Last updated with the experimental-code cleanup (LSTM/ensemble stubs + speculative adult-entertainment feature removed)._
