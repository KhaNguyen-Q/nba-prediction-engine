<<<<<<< HEAD
# nba-prediction-engine
=======
# NBA Prediction Project

## Project Overview

This repository is an NBA prediction engine designed to evolve from a data pipeline into a full production model service. The initial scope is focused on team win probability, with a foundation for future player-level regression, advanced fatigue metrics, market features, and deployment.

The current implementation includes:
- raw NBA game ingestion from `nba_api`
- NBA-only team filtering via a shared config and utility API
- feature engineering for leakage-safe rolling stats, rest/back-to-back, adult entertainment factors and fatigue score
- current-season roster ingestion, player game-log ingestion, and sportsbook odds ingestion
- market odds feature engineering and backtesting utilities
- baseline/tree/sequential/ensemble team modeling pipelines
- a dedicated player projection model pipeline
- FastAPI prediction service and Streamlit dashboard

## Architecture

### Folder structure

- `api/` — FastAPI service layer for predictions and feature access
- `config/` — canonical project configuration data
- `dashboard/` — applicable later
- `data/`
  - `raw/` — raw ingestion output CSV files
  - `processed/` — cleaned and feature-engineered datasets
- `models/` — trained model artifacts
- `notebooks/` — exploratory analysis and prototyping notebooks
- `scripts/` — data ingestion, feature engineering, utilities, and model training scripts


### Script map

Core runtime scripts:
- `scripts/update_pipeline.py`
- `scripts/get_data.py`
- `scripts/fetch_schedule.py`
- `scripts/fetch_availability.py`
- `scripts/build_features.py`
- `scripts/build_inference_features.py`
- `scripts/train_baseline.py`
- `scripts/train_tree_model.py`
- `scripts/train_automl_challenger.py`
- `scripts/train_player_model.py`
- `scripts/train_ensemble.py`
- `scripts/train_sequential.py`
- `scripts/smoke_test.py`
- `scripts/model_utils.py` (shared validation, leakage-guard feature selection, model registry helpers)
- `scripts/generate_monitoring_report.py` (freshness + drift report writer)
- `scripts/generate_prediction_quality_report.py` (logged prediction quality report writer)
- `scripts/model_promotion.py` (champion model promotion from registry)

Feature-config maintenance:
- `scripts/add_adult_scores.py` (maintains `adult_quality_rating` values in `config/nba_teams.json`)

Convenience ingestion wrappers:
- `scripts/fetch_players.py`
- `scripts/fetch_player_logs.py`
- `scripts/fetch_odds.py`
- `scripts/fetch_injuries.py`
- `scripts/fetch_availability.py` (upcoming-game scoped availability snapshot)

Evaluation/analysis tools:
- `scripts/backtest.py`

### Current data flow

1. `python scripts/get_data.py`
   - fetches raw game data from `nba_api`
   - filters to the 30 NBA franchises only
   - saves `data/raw/games_raw.csv`
   - fetches current roster data and writes `data/raw/players_raw.csv`
   - fetches player game logs and writes `data/raw/player_game_logs_raw.csv`
   - fetches current NBA injury reports and writes `data/raw/injuries_raw.csv`
   - builds latest player availability snapshot at `data/raw/injuries_latest.csv`
   - fetches sportsbook odds when `ODDS_API_KEY` is configured and writes `data/raw/odds_raw.csv`

2. `python scripts/fetch_schedule.py`
   - fetches the next 7 days of NBA games
   - saves `data/raw/upcoming_games.csv`

3. `python scripts/fetch_availability.py`
   - fetches latest injury/availability news
   - scopes availability to teams in `upcoming_games.csv` when available
   - writes `data/raw/injuries_raw.csv` and `data/raw/injuries_latest.csv`

4. `python scripts/build_features.py`
   - loads `data/raw/games_raw.csv`
   - filters NBA teams again as a safety check
   - creates labels and prior-game-only rolling team statistics
   - merges sportsbook odds into processed game features when available
5. `python scripts/build_inference_features.py`
   - builds leakage-safe inference features for future scheduled games
   - saves `data/processed/upcoming_inference_features.csv`
6. `python scripts/train_baseline.py`
   - trains a logistic regression baseline on processed features
   - evaluates accuracy, ROC-AUC, and log loss
   - saves the model to `models/logistic_baseline.pkl`

7. `python scripts/train_tree_model.py`
   - trains an XGBoost classifier on processed features
   - compares XGBoost metrics to the baseline model when available
   - saves the model to `models/xgb_tree_model.pkl`

8. `python scripts/train_player_model.py`
   - trains a two-stage player projection model: minutes model, then per-minute rate model
   - computes final `PTS`, `REB`, and `AST` as projected minutes × projected per-minute rates
   - uses rolling player game-log form, rest, and injury-severity features
   - saves the model to `models/player_projection_model.pkl`

## Key modules and responsibilities

### `config/nba_teams.json` and `config/team_locations.json`

A single source of truth for NBA team metadata and location data.
- `config/nba_teams.json` contains the 30 current NBA teams with `team_id`, `abbreviation`, and `name`.
- `config/team_locations.json` contains the same 30 teams with `city`, `state`, `lat`, `lon`, and `timezone`.

This config layer prevents repeated hardcoding across multiple scripts and enables travel/timezone feature engineering.

### `scripts/team_utils.py`

A central utility module for NBA team logic. Current helpers include:
- `load_nba_teams()`
- `get_nba_team_ids()`
- `get_nba_abbreviations()`
- `is_nba_team_by_id(team_id)`
- `is_nba_team_by_abbreviation(abbr)`
- `find_team_profile(...)`
- `get_team_adult_quality(...)`

This canonical API is the shared entry point for team validation and profile lookup.

### `scripts/get_data.py`

Raw ingestion from `nba_api.stats.endpoints.leaguegamefinder`. It:
- creates `data/raw/`
- fetches game rows
- filters out non-NBA affiliate/overseas teams
- writes `games_raw.csv`
- fetches current NBA roster data and writes `data/raw/players_raw.csv`
- fetches current NBA injury reports from ESPN and writes `data/raw/injuries_raw.csv`
- optionally fetches sportsbook odds when `ODDS_API_KEY` is configured and writes `data/raw/odds_raw.csv`

> Odds ingestion uses `ODDS_API_KEY` for The Odds API; if not configured, a schema-compatible placeholder file is created.

Future work should extend this to fetch richer player logs and market signals.

### New ingestion helpers

- `scripts/fetch_players.py` — current NBA roster ingestion helper
- `scripts/fetch_player_logs.py` — player game-log ingestion helper
- `scripts/fetch_odds.py` — sportsbook odds ingestion helper
- `scripts/fetch_schedule.py` — upcoming-games ingestion helper (next 7 days)

### `scripts/build_features.py`

Feature engineering for team-level modeling. It currently:
- loads raw game data
- filters NBA rows
- adds a binary `WIN` label
- extracts matchup metadata
- computes `HOME`, `OPPONENT`, and `WIN_STREAK`
- computes an `ADULT_ENTERTAINMENT_INDEX` feature from the opponent profile
- adds rolling statistics for `PTS`, `REB`, `AST`, and shooting percentages using only games before the current row
- computes `REST_DAYS`, `BACK_TO_BACK`, `TRAVEL_KM`, `TZ_DIFF`, rolling `INJURY_IMPACT`, and `fatigue_index`
- computes travel distance, timezone shift, and improved fatigue features using `config/team_locations.json`

### `scripts/train_baseline.py`

Baseline model training using scikit-learn logistic regression. It:
- loads processed features
- validates required baseline schema before training
- trains and evaluates a baseline model with a time-based holdout split when game dates are available
- saves the artifact to `models/logistic_baseline.pkl`

### `scripts/train_player_model.py`

Dedicated player performance projection training. It:
- loads `data/raw/player_game_logs_raw.csv`
- engineers rolling player-form features (last 5/10), rest, and injury severity context
- trains a dedicated minutes regressor (`MIN`)
- trains a multi-output regressor for per-minute `PTS`, `REB`, and `AST` rates
- converts rates to box-score projections using predicted minutes
- saves artifact to `models/player_projection_model.pkl`

### `scripts/build_inference_features.py`

Future-only inference feature builder. It:
- loads `data/raw/upcoming_games.csv`
- uses only latest historical rolling/team context features
- computes future game/team inference rows with no outcome leakage
- writes `data/processed/upcoming_inference_features.csv`

### `scripts/update_pipeline.py`

Unified update orchestrator. It:
- runs ingestion and feature steps with retries (3 attempts per step)
- runs schedule refresh before availability refresh so player status is linked to the exact upcoming team slate
- logs to `logs/pipeline.log`
- writes timestamped run metadata to `data/raw/pipeline_run_*.json`
- supports APScheduler mode (every 6 hours) using `PIPELINE_MODE=scheduler`
- emits stale-data warnings when schedule/inference files exceed freshness threshold
- runs monitoring + prediction-quality report generation each cycle
- supports trigger-based retraining and champion promotion:
  - drift trigger from `reports/monitoring_report.json` (`PIPELINE_RETRAIN_MAX_PSI`)
  - stale-data trigger (`PIPELINE_RETRAIN_ON_STALE=1`)
  - manual override (`PIPELINE_FORCE_RETRAIN=1`)
  - retrain sequence: baseline -> tree -> player -> automl challenger -> promote champion

### Availability troubleshooting

If player availability appears stale or empty:
1. Verify endpoint connectivity:
   - `Test-NetConnection site.api.espn.com -Port 443`
   - `Invoke-WebRequest "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries" -UseBasicParsing`
2. If network checks pass, re-run:
   - `python scripts/fetch_availability.py`
3. Confirm files are populated:
   - `data/raw/injuries_raw.csv`
   - `data/raw/injuries_latest.csv`
4. Rebuild inference features and restart API.

When outbound network is blocked (`WinError 10013`), availability snapshots can be empty and lineup-shock redistribution will not activate.

### `scripts/smoke_test.py`

Reproducible reliability checks. It validates:
- required raw/processed files exist
- model artifacts load successfully
- key datasets are readable with row/column counts
- API `/health` responds

## Dependencies

The repository uses the following Python packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `xgboost`
- `lightgbm`
- `torch`
- `fastapi`
- `streamlit`
- `uvicorn`
- `psycopg2-binary`
- `nba-api`
- `python-dotenv`
- `requests`
- `beautifulsoup4`
- `apscheduler`

Install with:

```bash
pip install -r requirements.txt
```

Interpreter consistency note:
- Use the same Python interpreter for training, scripts, and API runtime.
- If you have multiple Python versions installed, prefer:
  - `py -3.14 -m uvicorn api.main:app --reload`
  - `py -3.14 scripts/update_pipeline.py`
- Mixed interpreters can cause runtime import errors like `ModuleNotFoundError: joblib`.

See `CONTRIBUTING.md` for developer workflow, branch policies, and documentation rules.

## Current status

The project has completed Phase 1, Phase 2, and Phase 3 with sportsbook market features and backtesting utilities.

Current active phase: **Phase 3.5 - Optimization and Hardening (IN PROGRESS)**.

### Implemented features

- NBA team filtering from `nba_api` output
- canonical NBA metadata config
- reusable team utilities
- rolling team statistics (last 5 / last 10 games)
- rest day and back-to-back features
- sportsbook odds ingestion for NBA events
- market odds feature engineering and betting probability signals
- fatigue index and active injury impact features
- baseline/tree/sequential/ensemble team models
- dedicated player projection model training pipeline
- production-style team prediction endpoint (`POST /predict/team`) with player projections
- interactive Streamlit dashboard (`streamlit_app.py`)
- upcoming schedule ingestion (`scripts/fetch_schedule.py`)
- future-only inference feature generation (`scripts/build_inference_features.py`)
- automated update orchestrator with retry + logs (`scripts/update_pipeline.py`)
- player availability guardrails: projected stats are forced to `0` for players marked out/inactive or with return date after the selected game date
- context-aware player projections: available players are adjusted by teammate-absence redistribution and opponent defensive factors
- prediction explainability panel in dashboard (baseline linear contributions, tree SHAP-style contributions)
- color-coded explainability bars with tooltips (green=positive, red=negative impact)
- AI Game Brief panel with recent matchup headlines, short quote snippets, and top confidence stat recommendation
- dashboard-side baseline vs tree comparison panel for model divergence checks on the same upcoming game
- leakage-safe training guardrails for tree/AutoML to prevent post-game feature contamination
- rolling time-series validation metrics in training scripts
- model registry metadata in `models/registry/` for reproducibility and auditability
- AutoML challenger training script (`scripts/train_automl_challenger.py`) using time-aware folds

### Prediction policy (intended behavior)

The intended production logic is:
1. Train models on recent historical data only (team form, injuries, fatigue, market/hype features).
2. Generate predictions only for future upcoming games (for example the next 7 days).
3. Keep training and inference windows separated to prevent leakage and stale-data reuse.

At the moment, this policy is partially implemented:
- `POST /predict/team` is now gated to games listed in `data/raw/upcoming_games.csv`
- `POST /predict` is disabled for production-style usage
- `POST /predict/team` now requires future inference rows for the selected game and no longer falls back to historical rows
- schedule and inference updates still depend on upstream NBA endpoint reliability

### Known gaps (must be addressed)

1. Real-time upcoming games ingestion reliability:
   - next-7-days fetch exists, but external API/network failures can still interrupt updates.
2. Future-only inference coverage:
   - enforcement exists for team endpoint; predictions are rejected if upcoming inference features are missing.
   - full reliability still depends on successful schedule refresh + inference feature refresh.
3. Player projection team-context disconnect: (PARTIALLY ADDRESSED)
   - completed: out/inactive players (or players returning after game date) are now hard-zeroed in API projections.
   - remaining: projections can still look too similar across some matchups when player-model coverage is limited and fallback logic dominates.
   - remaining: add stronger opponent-conditioned player features and explicit minutes modeling.

### Next development areas

- production deployment and API monitoring
- expanded market signal and betting strategy analysis
- model calibration and drift monitoring
- robust retry/cache strategy for player log ingestion when upstream NBA endpoints are slow

### Immediate next steps (recommended order)

1. Harden schedule and inference refresh reliability: (PARTIALLY COMPLETE)
   - completed: added odds-based secondary schedule fallback when NBA schedule API is unavailable
   - completed: added stale-file alerts for `upcoming_games.csv` and `upcoming_inference_features.csv` in `scripts/update_pipeline.py`
   - remaining: add an additional non-odds schedule source fallback
2. Keep future-only prediction flow healthy: (PARTIALLY COMPLETE)
   - completed: dashboard game choices can be sourced from `/upcoming-games`
   - completed: API team endpoint is gated to `upcoming_games.csv`
   - completed: dashboard now runs in upcoming-game-only selection mode
   - keep model training strictly on recent historical windows
3. Fix player projection quality and team-context behavior:
   - completed: hard-zero out players unavailable for selected game date
   - increase player log coverage
   - ensure projections vary by team/opponent context and injury-driven minutes assumptions
4. Package and run consistently: (COMPLETE)
   - completed: added `Dockerfile` and `docker-compose.yml`
   - completed: container startup runs `uvicorn api.main:app`
5. Add reproducible smoke tests: (COMPLETE)
   - completed: added `scripts/smoke_test.py` for files, model loads, data shape, and API `/health` checks
6. Improve evaluation reliability: (IN PROGRESS)
   - completed: added Brier score reporting in baseline/tree training
   - completed: added rolling time-split metrics in baseline/tree training
   - completed: calibration curve artifacts now exported to `reports/calibration_baseline.csv` and `reports/calibration_tree.csv`
   - completed: monthly rolling backtest summary now exported to `reports/backtest_summary.csv` (writes status row when odds are unavailable)
7. Expand dashboard analytics:
   - completed: added backtest trend chart panel
   - completed: added calibration plot panel
   - completed: added prediction confidence-band label (Low/Medium/High)

## How to continue

This section describes the most valuable next workstreams for the repository. Follow the order below to keep the project progressive and avoid scope creep.

### Phase 1: complete the data pipeline (COMPLETE)

The data pipeline has been validated end-to-end:
- `python scripts/get_data.py` writes `data/raw/games_raw.csv`, `data/raw/players_raw.csv`, `data/raw/player_game_logs_raw.csv`, `data/raw/injuries_raw.csv`, and `data/raw/odds_raw.csv`
- `python scripts/fetch_schedule.py` writes `data/raw/upcoming_games.csv`
- `python scripts/build_features.py` creates `data/processed/games_with_features.csv`
- `python scripts/build_inference_features.py` creates `data/processed/upcoming_inference_features.csv`

Completed Phase 1 work:
1. Added `config/team_locations.json` with NBA arena city, latitude, longitude, and timezone.
2. Extended `scripts/team_utils.py` with travel and timezone helpers:
   - `find_team_location(team_id|abbreviation)`
   - `get_team_timezone_offset(team_id|abbreviation)`
   - `haversine_distance(lat1, lon1, lat2, lon2)`
3. Updated `scripts/build_features.py` to compute:
   - `TRAVEL_DISTANCE` per game
   - `TIMEZONE_SHIFT` between consecutive games
   - improved `fatigue_index` using travel and timezone cost
4. Implemented ingestion helpers for additional raw sources:
   - `scripts/fetch_odds.py`
   - `scripts/fetch_players.py`
   - `scripts/fetch_player_logs.py`
   - `scripts/fetch_schedule.py`

### `api/main.py`

A FastAPI service is available for health checks, feature metadata, sample predictions, generic feature-based predictions, and team-ID prediction reports.

Run it with:

```bash
py -3.14 -m uvicorn api.main:app --reload
```

Primary endpoints:
- `GET /health`
- `GET /features`
- `GET /monitoring`
- `GET /prediction-quality`
- `GET /upcoming-games`
- `GET /predict/sample`
- `POST /predict` (disabled for production-style usage)
- `POST /predict/team` (future games only; requires upcoming inference features)
  - logs each prediction to `reports/prediction_log.csv`
  - adds availability freshness guardrails using `data/raw/injuries_latest.csv`
  - reduces recommendation confidence when availability data is stale/empty
  - supports `model=champion` after promotion

### `streamlit_app.py`

A Streamlit dashboard is included for interactive game predictions and player projection review.

Run it with:

```bash
streamlit run streamlit_app.py
```

The dashboard now supports upcoming-game driven selection via API schedule endpoint:
- `GET /upcoming-games`
- manual historical-team selection fallback has been removed to enforce future-game flow
- optional baseline vs tree comparison panel is available for the same selected game to inspect prediction divergence
- game selection is now team-locked: prediction requests always use the selected upcoming game's exact home/away team IDs (no fallback remapping)

### Phase 2: improve modeling (COMPLETE)

Phase 1, Phase 2, and Phase 3 are complete. The project includes baseline, tree, sequential, and ensemble training pipelines, plus sportsbook market feature engineering and backtesting.

Completed Phase 2 work:
1. Implemented tree-based model training in `scripts/train_tree_model.py`.
2. Added sequence model training in `scripts/train_sequential.py`.
3. Added ensemble training in `scripts/train_ensemble.py` with out-of-fold stacking signals and holdout evaluation.
4. Updated baseline and tree training scripts to use time-based holdout splits when possible and strict schema checks.

### Phase 3: market and evaluation layer (COMPLETE)

Completed Phase 3 work:
1. Added market comparison features in `scripts/build_features.py`, including `HOME_ML_PRICE`, `AWAY_ML_PRICE`, `HOME_ML_PROB`, `AWAY_ML_PROB`, `TEAM_IMPLIED_PROB`, `ODDS_PROB_DIFF`, and `PUBLIC_BIAS_INDICATOR`.
2. Built backtesting utilities in `scripts/backtest.py` for strategy simulation and ROI/profit outcomes.
3. Added metrics reporting for accuracy, ROC-AUC, and log loss across training pipelines.

### Phase 3.5: optimization and hardening (IN PROGRESS)

Current focus:
1. Leakage-safe model training:
   - completed: tree and AutoML challengers now remove known post-game leakage columns before fitting.
2. Time-aware model validation:
   - completed: baseline and tree training now report rolling time-split metrics.
3. Model registry and experiment traceability:
   - completed: training scripts now write run metadata and metrics to `models/registry/`.
4. AutoML challenger lane:
   - completed: added `scripts/train_automl_challenger.py` with leaderboard output over leakage-safe features.
5. Calibration and evaluation artifacts:
   - completed: baseline/tree training writes calibration reports in `reports/`.
6. Backtest summary robustness:
   - completed: `scripts/backtest.py` writes `reports/backtest_summary.csv` with monthly/overall summaries and Sharpe-like score.
7. Monitoring and drift visibility:
   - completed: `scripts/generate_monitoring_report.py` writes freshness + drift summary to `reports/monitoring_report.json`.
   - completed: API exposes monitoring via `GET /monitoring`.
   - completed: Streamlit dashboard shows drift status and top PSI features.
8. Trigger-based retraining orchestration:
   - completed: `scripts/update_pipeline.py` now supports drift/staleness/forced retrain triggers.
   - completed: retraining now runs baseline/tree/player/automl and then promotes a champion model.
9. Prediction quality tracking:
   - completed: prediction requests are logged to `reports/prediction_log.csv`.
   - completed: `scripts/generate_prediction_quality_report.py` writes `reports/prediction_quality_report.json` and `reports/prediction_quality_summary.csv`.
   - completed: API exposes quality report via `GET /prediction-quality`.
10. Opponent-conditioned player modeling:
   - completed: `scripts/train_player_model.py` now adds opponent defensive context, pace proxy, and vegas implied team total features.
   - completed: `api/main.py` inference player features now include matching opponent/team context features.
11. Champion/challenger promotion:
   - completed: `scripts/model_promotion.py` promotes best registry candidate to `models/champion_team_model.pkl`.
   - completed: API can serve champion model using `model=champion`.

Run commands for this phase:
1. `python scripts/train_baseline.py`
2. `python scripts/train_tree_model.py`
3. `python scripts/train_player_model.py`
4. `python scripts/train_automl_challenger.py`
5. `python scripts/model_promotion.py`
6. `python scripts/backtest.py --model baseline --threshold 0.05 --stake 1.0`
7. `python scripts/generate_prediction_quality_report.py`
8. inspect `models/registry/index.json` and latest entries in `models/registry/`
9. inspect `reports/calibration_baseline.csv`, `reports/calibration_tree.csv`, `reports/backtest_summary.csv`, `reports/monitoring_report.json`, and `reports/prediction_quality_report.json`

Recommended refresh order for live projections:
1. `python scripts/fetch_schedule.py`
2. `python scripts/fetch_availability.py`
3. `python scripts/build_inference_features.py`
4. restart API (`py -3.14 -m uvicorn api.main:app --reload`)

Exit criteria for Phase 3.5:
1. Team models show stable rolling time-split performance (no leakage artifacts).
2. Registry entries are generated for each training run with holdout + rolling metrics.
3. Challenger training can be rerun reproducibly on the same dataset snapshot.
4. Calibration and backtest summary artifacts are generated each cycle.
5. Monitoring report is generated and visible in API/dashboard.

### Phase 4: deployment and dashboard (NEXT)

Current focus:
1. Runtime packaging:
   - add `Dockerfile` and optional `docker-compose.yml`.
2. Production hardening:
   - add robust smoke tests and startup checks for API + dashboard.
3. Evaluation and monitoring:
   - add calibration, drift tracking, and backtest monitoring panels.
4. Data reliability:
   - harden schedule and inference refresh when NBA upstream endpoints are unstable.

Exit criteria for Phase 4:
1. API can be started and validated via documented smoke tests.
2. Containerized run works on a clean machine.
3. Dashboard displays live predictions and player projections from saved model artifacts.

### Good extension points

- `config/team_locations.json`
- `scripts/fetch_odds.py`
- `scripts/fetch_injuries.py`
- `scripts/fetch_players.py`
- `scripts/fetch_player_logs.py`
- `scripts/fetch_schedule.py`
- `scripts/train_tree_model.py`
- `scripts/train_sequential.py`
- `scripts/train_ensemble.py`
- `scripts/train_player_model.py`
- `scripts/build_inference_features.py`
- `scripts/update_pipeline.py`
- `scripts/backtest.py`
- `api/main.py`
- `streamlit_app.py`
- `Dockerfile`
- `dashboard/` or `streamlit_app.py`

## Important conventions

- Keep all NBA team metadata in `config/nba_teams.json`.
- Use `scripts/team_utils.py` for validation and team profile lookup.
- Keep ingestion, processing, and modeling separated by folder.
- Use the root project path when executing scripts directly; the scripts already patch `sys.path` at runtime for direct execution.

## Change log policy

This README is the living architectural document for the repository. Whenever new functionality is added, updated, or refactored, update this file immediately so future engineers can understand the design and intent.

---

_Last updated on March 31, 2026 (Phase 4 in progress)._
>>>>>>> 7f294d3 (Phase 3.5 optimization + README update)
