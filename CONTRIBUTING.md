# Contributing to NBA Prediction Project

Thanks for helping improve this project. This guide explains how to contribute, maintain consistency, and keep documentation aligned with code changes.

## 1. Project purpose

This repository is a modern NBA prediction engine. The current focus is:
- team win probability prediction
- NBA-only data ingestion
- feature engineering for fatigue and rest
- a baseline logistic regression model

Future phases include player stat regression, market features, deep learning models, a FastAPI service, and a dashboard.

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

- `api/` — planned FastAPI service layer
- `config/` — canonical team metadata and future location/timezone maps
- `dashboard/` — dashboard assets and prototypes
- `data/raw/` — raw ingestion outputs
- `data/processed/` — engineered datasets
- `models/` — saved model artifacts
- `notebooks/` — analysis and prototyping
- `scripts/` — ingestion, processing, utilities, training

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
```

These scripts provide placeholders for player, injury, and odds ingestion.

### 3. Build features

```bash
python scripts/build_features.py
```

This loads raw data, filters NBA teams, labels games, adds rolling stats, and computes fatigue features.

### 4. Train tree model

```bash
python scripts/train_tree_model.py
```

This trains an XGBoost model and compares it to the baseline logistic model.

### 5. Run the full pipeline

```bash
python scripts/run_pipeline.py
```

This executes ingestion, feature engineering, and model training phases together.

### 5. Train baseline model

```bash
python scripts/train_baseline.py
```

This trains a logistic regression baseline and saves the model artifact.

## 6. Coding conventions

- Keep code readable and modular.
- Use `config/` for shared static data.
- Use `scripts/team_utils.py` for any NBA team validation or lookup.
- Avoid hard-coding NBA team IDs or abbreviations in multiple places.
- Prefer explicit feature names and comment non-obvious logic.

## 7. Testing and validation

- Run scripts locally after any code change.
- Confirm `data/processed/games_with_features.csv` regenerates cleanly.
- Confirm the baseline script trains and saves `models/logistic_baseline.pkl`.
- If you add new data columns, describe them in `README.md`.

## 8. Adding new features

When adding a new feature or data source:
1. Add any config metadata to `config/`.
2. Add helper functions to `scripts/team_utils.py` or a new util module.
3. Read and filter raw data in `scripts/get_data.py`.
4. Add feature engineering logic in `scripts/build_features.py`.
5. Add model training or evaluation in `scripts/train_baseline.py` or a new script.
6. Update docs.

## 9. Notes for future maintainers

- `config/nba_teams.json` is the project’s canonical NBA team source.
- `scripts/team_utils.py` is the shared team validation API.
- Keep `README.md` and `CONTRIBUTING.md` synchronized.
- This project is intentionally built to grow from ingestion → features → model → API.

---

_Last updated by the agent on March 27, 2026._
