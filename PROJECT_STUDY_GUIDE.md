# 🏀 NBA ML Prediction Engine: The Complete Plain-English Guide

> **Who this is for:** You — preparing for a technical interview after ~4 months away from this repo.  
> **How to use it:** Read sections 1 → 7 in order once. Then drill the elevator pitch and the “Architectural Trade-offs” section until you can say them out loud without looking.

This guide translates every important Machine Learning and Software Engineering idea in this repository into plain English — while staying **100% accurate to the real code paths**.

---

## 1. 🎡 The Big Picture (How Everything Works Together)

### The Factory Assembly Line Analogy

Imagine this project is a **sports prediction factory**.

1. **Trucks arrive** with raw NBA boxes of data (games, players, injuries, odds).
2. **Workers sort and clean** the boxes into useful “clues” (features).
3. **Several student robots** practice guessing who wins (model training).
4. **A tournament judge** picks the best robot (champion promotion).
5. **The kitchen chef** uses that robot to cook predictions when someone orders (FastAPI).
6. **The waiter** shows the dish on a nice menu screen (Streamlit).
7. **Shipping containers** keep every room of the factory packed the same way wherever you move it (Docker Compose).

That is the whole system.

### Step-by-step: Raw NBA stats → prediction on a web screen

```text
Raw Ingestion
    → Feature Engine
        → Model Training & Tournament
            → Model Registry
                → FastAPI Service
                    → Streamlit Dashboard
                        → Docker Containers (how the whole factory ships)
```

#### Step A — Raw Ingestion (“Trucks unload boxes”)

Scripts pull data from the outside world and save CSV files under `data/raw/`.

| What arrives | Main script(s) | Typical output file |
|---|---|---|
| Historical NBA games | `scripts/get_data.py` | `data/raw/games_raw.csv` |
| Players / rosters | `scripts/fetch_players.py` | under `data/raw/` |
| Player game logs | `scripts/fetch_player_logs.py` | under `data/raw/` |
| Injuries / availability | `scripts/fetch_injuries.py`, `scripts/fetch_availability.py` | `data/raw/injuries_raw.csv`, `data/raw/injuries_latest.csv` |
| Sportsbook odds | `scripts/fetch_odds.py` (also orchestrated via `get_data.py`) | `data/raw/odds_raw.csv` |
| Upcoming schedule | `scripts/fetch_schedule.py` | `data/raw/upcoming_games.csv` |

**Orchestrators (the factory managers):**
- `scripts/run_pipeline.py` — bootstrap / one-shot full run
- `scripts/update_pipeline.py` — production-style refresh (also used by the Docker **scheduler**)

#### Step B — Feature Engine (“Turn boxes into clues”)

| Job | File | Output |
|---|---|---|
| Build historical training features (no cheating) | `scripts/build_features.py` | `data/processed/games_with_features.csv` |
| Build today’s upcoming-game feature rows | `scripts/build_inference_features.py` | `data/processed/upcoming_inference_features.csv` |
| Shared team / city / timezone helpers | `scripts/team_utils.py` + `config/nba_teams.json` + `config/team_locations.json` | config lookups used by features |

#### Step C — Model Training & Tournament (“Student robots practice”)

| Model | Script | Artifact (typical) |
|---|---|---|
| Baseline logistic regression | `scripts/train_baseline.py` | `models/logistic_baseline.pkl` |
| XGBoost tree | `scripts/train_tree_model.py` | `models/xgb_tree_model.pkl` |
| AutoML challenger bake-off | `scripts/train_automl_challenger.py` | challenger artifact + registry entry |
| Ensemble stacker | `scripts/train_ensemble.py` | ensemble artifact (not champion-eligible) |
| LSTM sequential | `scripts/train_sequential.py` | experimental only |
| Player projection | `scripts/train_player_model.py` | `models/player_projection_model.pkl` |

Shared helpers: `scripts/model_utils.py` (metrics, time splits, leakage-safe column filter, registry writer).

#### Step D — Model Registry (“Hall of Fame binder”)

| Job | File / path |
|---|---|
| Write registry entries | `scripts/model_utils.py` → `models/registry/*.json` + `models/registry/index.json` |
| Promote champion | `scripts/model_promotion.py` → `models/champion_team_model.pkl` + `models/champion_team_model_meta.json` |

Eligible for champion: `logistic_baseline`, `xgb_tree_model`, `automl_challenger`.

#### Step E — FastAPI Service (“The kitchen / chef”)

| Job | File |
|---|---|
| Load models, validate upcoming games, score predictions, log results | `api/main.py` |

Important routes:
- `GET /health`, `GET /features`, `GET /upcoming-games`
- `POST /predict/team` ← the real prediction endpoint
- `POST /predict` ← intentionally disabled (HTTP 410)
- `GET /monitoring`, `GET /prediction-quality`, `GET /pipeline-status`

#### Step F — Streamlit Dashboard (“The menu / waiter”)

| Job | File |
|---|---|
| Call the API and show win probabilities, explainability, player projections, monitoring | `streamlit_app.py` |

Default API URL: env var `API_URL` (or sidebar field), usually `http://127.0.0.1:8000`.

#### Step G — Docker Containers (“Shipping the factory”)

| Piece | File |
|---|---|
| Image recipe | `Dockerfile` |
| Four-service stack | `docker-compose.yml` |

Services:
1. **api** — FastAPI on port `8000`
2. **streamlit** — dashboard on port `8501`
3. **scheduler** — runs `scripts/update_pipeline.py` on a loop
4. **postgres** — database container (configured, but not used by Python yet — see Part 4)

### One-picture mental model

```mermaid
flowchart LR
    A[Raw CSVs<br/>data/raw/] --> B[Features<br/>data/processed/]
    B --> C[Train models<br/>scripts/train_*.py]
    C --> D[Registry + Champion<br/>models/]
    D --> E[FastAPI<br/>api/main.py]
    B --> E
    E --> F[Streamlit<br/>streamlit_app.py]
    G[Docker Compose] -. packages .-> E
    G -. packages .-> F
    G -. packages .-> H[Scheduler<br/>update_pipeline.py]
```

---

## 2. 🛡️ Part 1: Data Leakage & Feature Engineering (“No Cheating!”)

### The Analogy: Data Leakage = tomorrow’s newspaper

**Data leakage** means your model accidentally sees information from the future when it is supposed to be predicting the past or the present.

If you are guessing who wins **today’s** game, but your notebook already includes **tonight’s final score**, that is cheating. You are reading tomorrow’s newspaper to “predict” today.

In interviews, managers love this topic because it separates “I trained a model” from “I built an honest forecasting system.”

### How we stop cheating in code (`scripts/build_features.py`)

#### Rolling stats = “look backwards over past games only”

For each team, the code sorts games by date, then does this idea:

```python
series.shift(1).rolling(window).mean()
```

Plain English:
1. Line the team’s games up in time order.
2. `shift(1)` = **ignore the current game’s box score**.
3. `rolling(...)` = average the previous few games (windows of 5 and 10).

So for Game 6, `pts_last5` uses Games 1–5 — never Game 6’s points.

Comment in the real code:

> “Use only prior games to avoid leakage from the current game's stats.”

#### Rest days = “how long since the last game?”

```python
PREV_GAME_DATE = previous game date for that team
REST_DAYS = current game date − previous game date
```

Again: only looking **backward**.

#### Travel / timezone / fatigue

- Travel distance uses the team’s **previous** city vs **current** city.
- Fatigue index mixes:
  - back-to-back flag
  - inverse rest days
  - travel distance
  - timezone shift

Think of fatigue as a handmade “tiredness score,” not a neural network.

#### Injury impact

Injuries are built as an “active burden” over time, then attached to each game with a **backward** as-of merge (`direction='backward'`). A game only sees injury events dated on or before that game.

#### Second safety net in training

`scripts/model_utils.py` has `leakage_safe_team_features()`, which drops obvious outcome columns (like raw `PTS`, `WIN`, `FG_PCT`, etc.) before tree/AutoML training. Belt **and** suspenders.

### Inference vs Training: why `build_inference_features.py` exists

**Training** (`build_features.py`) answers:

> “For every *past* game, what clues would we have known *before* tip-off?”

**Inference** (`build_inference_features.py`) answers:

> “For *tonight’s upcoming* games, build the clue sheets the API needs right now.”

How inference works in plain English:
1. Read historical processed features: `data/processed/games_with_features.csv`
2. Read upcoming schedule: `data/raw/upcoming_games.csv`
3. For each team, start from their latest processed row (schema-compatible base).
4. Create one home row + one away row for each upcoming game.
5. **Recompute** the matchup-critical fields for tip-off:
   - rolling form from completed games (including the latest box score)
   - rest / back-to-back
   - travel + timezone from last venue → upcoming venue
   - fatigue index
   - adult-entertainment away index
   - injury impact as-of the upcoming date
   - upcoming odds when available (otherwise clear stale odds from the copied base row)
6. Save: `data/processed/upcoming_inference_features.csv`

The API (`POST /predict/team`) then loads those inference rows and scores them. If a team’s inference row is missing → **503** (“kitchen can’t cook; missing ingredients”).

### Architectural Trade-offs (Interview Critique Preparedness)

These are the kinds of points interviewers want — **design choices and remaining hard problems**, not “I forgot an easy line of code.”

#### 1) Odds Timestamp Correctness (still the real data-integrity risk)

**What the code does:** Odds are aggregated by home team / away team / commence date and turned into implied probabilities and spreads. Inference now merges upcoming odds when present, and clears stale historical odds when not.

**What’s still hard:** There is **no hard “T−1 hour before tip-off” snapshot rule**. Closing lines, live betting updates, or post-tip revisions can make market features look unfairly strong in backtests.

**How to say it:**
> “Odds are a powerful signal. The remaining risk isn’t ‘did we join odds at serve time?’ — it’s point-in-time correctness: freezing a pregame snapshot so training never sees information that wasn’t knowable before tip.”

#### 2) Offline Feature Materialization vs Online Recompute (train/serve parity — addressed)

**What we fixed in code:** `scripts/build_inference_features.py` no longer blindly trusts photocopied rolling/travel/fatigue/odds from the last historical row. It recomputes those fields for the upcoming matchup while keeping a schema-compatible base row for any extra tree-model columns.

**What to talk about architecturally (the insight, not the bug):**
This is the classic MLOps **offline vs online feature** problem:
- Batch training likes a big historical table.
- Serving needs “as of now, for a future game.”
- Options are: recompute on a schedule (what we do), recompute per request (higher latency), or a feature store with point-in-time joins.

**How to say it:**
> “I treat train/serve parity as an architecture requirement. Inference materializes upcoming features on the pipeline schedule using the same definitions as training — rolling windows from completed games, venue travel, fatigue, and live odds when available — rather than shipping a stale historical row unchanged.”

#### 3) Team-row training vs Game-level evaluation (pairing — addressed)

**Why two rows exist on purpose:** Team-centric features (form, rest, travel for *this* team) naturally produce a home row and an away row. That design matches how the kitchen scores both sides, then normalizes `home + away = 1.0`.

**What we fixed in code:** Holdout / rolling-CV metrics in `train_baseline.py`, `train_tree_model.py`, and `train_automl_challenger.py` now collapse to **one row per `GAME_ID`** (home perspective) via `select_game_level_rows()` / `classification_metrics_game_level()` in `scripts/model_utils.py`. Training can still use both team-rows; evaluation no longer double-counts anti-correlated labels.

**How to say it:**
> “The unit of modeling is team-centric, but the unit of the product is a game. So I train on team-rows for feature richness, and I evaluate at game-level so paired home/away outcomes aren’t treated as independent samples.”

#### 4) Remaining deeper topics (if they want senior-level discussion)

- **Effective sample size / dependence:** even with game-level metrics, seasons are autocorrelated; rolling time splits matter more than random K-fold.
- **Champion promotion without significance tests:** metric gates help, but a bootstrap or paired test would harden promotion further.
- **Postgres provisioned but unused:** file artifacts are demo-friendly; a DB/feature store is the natural next persistence layer.
---

## 3. 🤖 Part 2: The Model Tournament (“Choosing the Champion”)

### Baseline (Logistic Regression) — “The Simple Scale”

**Script:** `scripts/train_baseline.py`  
**Artifact:** `models/logistic_baseline.pkl`

Analogy: a kitchen scale with numbered dials.

- Each clue (recent points, rest days, fatigue, etc.) gets a weight.
- The model adds weighted clues and turns the total into a probability between 0 and 1.
- Features are scaled (`StandardScaler`) so big numbers don’t bully small ones.

**Why it exists:** Interpretable baseline. Easy to explain with linear feature contributions in the API.

### Tree / XGBoost — “The Decision Tree Flowchart”

**Script:** `scripts/train_tree_model.py`  
**Artifact:** `models/xgb_tree_model.pkl`

Analogy: a giant flowchart of yes/no questions:

> “Is rest days < 2?  
> Is opponent pts_last10 high?  
> Is injury impact big?  
> Then home win probability ≈ …”

Trees are great for messy tabular sports data with nonlinear interactions. The API can show SHAP-like contribution explanations for tree predictions.

### Ensemble / Stacker — “The Panel of Experts”

**Script:** `scripts/train_ensemble.py`

Analogy: Baseline Expert and Tree Expert each cast a vote (a probability). A small meta-model (another logistic regression) learns how to blend those votes.

**Honest status:** Trained in the bootstrap pipeline, but **not** registered for champion promotion and **not** exposed as a first-class API model choice. Mention it as architecture exploration, not the production path.

### LSTM — “The Memory Bot” (experiment only)

**Script:** `scripts/train_sequential.py`

Analogy: a robot that reads the last 8 team game-steps in order, trying to remember “form over time.”

**Why it is NOT production:**
- Trains briefly (small epoch count)
- No proper holdout tournament metrics like the others
- Not champion-eligible
- Not served by FastAPI model selection

**Interview line:**
> “I included an LSTM as a sequence experiment. For noisy tabular NBA data, trees/logistic usually win on reliability. The production path promotes on probability metrics, not model fashion.”

### Player projection model — bonus robot

**Script:** `scripts/train_player_model.py`  
**Artifact:** `models/player_projection_model.pkl`

Two-stage idea:
1. Predict minutes.
2. Predict per-minute scoring/rebounds/assists rates.
3. Multiply → projected box-score stats.

This is separate from the team win-probability champion.

### The Promotion Gate (`scripts/model_promotion.py`)

Eligible fighters only:
- `logistic_baseline`
- `xgb_tree_model`
- `automl_challenger`

**We do NOT crown a champion by raw win/loss accuracy alone.**

Why? A model can say “home wins” and be right, while being wildly overconfident (90% when it should be 55%). For a prediction product, **probability quality** matters.

Promotion score (lower is better):

```text
0.45 * log_loss
+ 0.30 * brier_score
+ 0.15 * rolling_log_loss
+ 0.10 * rolling_brier
```

Plain English:
- **Log Loss** — “How badly wrong were your probability guesses, with big punishments for cocky wrong answers?”
- **Brier Score** — “On average, how far were your probabilities from the true 0/1 outcomes?”

A challenger must beat the incumbent by at least `PROMOTION_MIN_IMPROVEMENT` (default `0.005`) or the old champion stays.

Winner files:
- `models/champion_team_model.pkl`
- `models/champion_team_model_meta.json`

**Interview line:**
> “Promotion is metric-driven and prioritizes calibrated probabilistic quality — log loss and Brier — with a minimum improvement gate so we don’t thrash champions on noise.”

### Monitoring after the tournament

| Report | Script | What it checks |
|---|---|---|
| Drift / freshness | `scripts/generate_monitoring_report.py` | file age + PSI feature drift |
| Prediction quality | `scripts/generate_prediction_quality_report.py` | logged predictions vs actual outcomes |

---

## 4. 🚀 Part 3: The Serving Engine & UI (“The Restaurant Kitchen & Menu”)

### FastAPI = Kitchen / Chef (`api/main.py`)

Customers (the dashboard, tests, or any client) place an order:

> “Predict Boston vs Lakers with the tree model.”

The kitchen:
1. Checks the order is a **real upcoming scheduled game**
2. Pulls each team’s inference feature sheet
3. Runs the selected model (`baseline` / `tree` / `champion`)
4. Returns a full plate: probabilities, fair moneylines, explanations, data-quality warnings, optional player projections
5. Writes a receipt to `reports/prediction_log.csv`

### Streamlit = Menu / Waiter (`streamlit_app.py`)

The waiter does **not** cook. It:
- Asks the kitchen for health / features / upcoming games
- Lets you pick a matchup and model in the sidebar
- Calls `POST /predict/team`
- Displays the dish with charts, alerts, and downloads

### Probability normalization (`Home + Away = 1.0`)

The kitchen scores home and away a bit independently first (two team-view estimates). Those two raw numbers might add up to 1.05 or 0.92 — weird for a two-team game.

So the code renormalizes:

```text
home_prob = home_raw / (home_raw + away_raw)
away_prob = away_raw / (home_raw + away_raw)
```

Now they always add to **1.0**. The API even labels this with a calculation note about normalizing independent team-win estimates.

Fair moneylines are then derived from those probabilities (the “sportsbook translation” of the model’s belief).

### Fallback Mechanics: kitchen keeps serving when suppliers flake

#### If the Odds API crashes / key missing

In ingestion (`scripts/get_data.py` odds path):
1. Log a warning
2. Keep the previous non-empty `odds_raw.csv` if it exists
3. Otherwise write an empty but schema-compatible odds CSV

Serving does **not** call odds live during `/predict/team`. As long as schedule + inference features + a model exist, food still goes out.

#### If the Injury / ESPN feed crashes

Same pattern: keep cached injuries, or write empty schema CSV.  
At serve time:
- missing/stale availability → warnings in `data_quality`
- injury impact can default toward “no known impact”
- advisory confidence gets **penalized** (the kitchen still cooks, but admits the fridge thermometer is broken)

#### Schedule fallbacks

Upcoming games try multiple sources: NBA API → ESPN → odds-derived schedule → retain previous file.

#### Startup strictness

- `STRICT_STARTUP_CHECKS=1` (Docker API default): kitchen refuses to open if required models/data are missing (fail fast).
- `STRICT_STARTUP_CHECKS=0`: kitchen opens in degraded mode for local/dev demos; individual endpoints return 404/503 when ingredients are missing.

### Why `/predict` is off (HTTP 410) and `/predict/team` is used

`POST /predict` is intentionally disabled:

> Direct feature-vector prediction is disabled in production mode. Use `POST /predict/team`.

**Why that matters:**
- Stops random clients from stuffing arbitrary feature vectors into the model (easy to misuse / leak / spoof).
- Forces predictions through **scheduled-game validation** + the real inference feature path.
- Cleaner product contract: “predict this upcoming matchup,” not “score this mystery vector.”

---

## 5. 🐳 Part 4: Docker & Infrastructure (“Shipping Containers”)

### Docker Compose = four standardized shipping containers

File: `docker-compose.yml`  
Image recipe: `Dockerfile` (Python 3.11, installs `requirements.txt`, default CMD is uvicorn)

| Container | Role | Port / command |
|---|---|---|
| `api` | FastAPI kitchen | `8000` → `uvicorn api.main:app` |
| `streamlit` | Dashboard waiter | `8501` → `streamlit run streamlit_app.py` |
| `scheduler` | Night-shift factory manager | runs `python scripts/update_pipeline.py` |
| `postgres` | Database box | `5432` |

Analogy: instead of saying “install 40 libraries on your laptop and hope,” you ship the whole factory in labeled containers that start the same way on any machine.

### Honest code flaw: Postgres is configured but unused

**Facts:**
- Compose sets `DATABASE_URL=postgresql://nba:nba@postgres:5432/nba`
- `psycopg2-binary` is in `requirements.txt`
- Current Python serving/pipeline code reads/writes **local CSV / JSON / pickle files**, not Postgres tables

**Professional framing (memorize):**
> “Postgres is provisioned in the deployment topology for future persistence of predictions, pipeline metadata, or a feature store. Today the serving path is intentionally file-artifact based — simple to inspect and demo. That’s a clear next step, not a hidden dependency.”

Do **not** claim you have a database-backed feature store.

Other infrastructure honesty points (if asked):
- Models load at API import time → scheduler retrain won’t hot-reload a running API without restart.
- `/health` is a shallow liveness check, not a deep readiness probe of every artifact.

---

## 6. 🌉 Part 5: The IBM Developer Skills Network (DSN) Bridge

IBM DSN (Markham, ON) builds developer education: labs, tutorials, APIs, reproducible environments, and increasingly LLM/foundation-model content. Map **your engineering choices**, not NBA trivia.

| This repository | Exact talking point for DSN |
|---|---|
| **Streamlit dashboard** (`streamlit_app.py`) | “I built an interactive front door so non-experts can explore model outputs — the same pattern as developer labs and guided demos.” |
| **Docker Compose** (`docker-compose.yml`) | “I containerized API + UI + scheduler so the environment is reproducible — critical when thousands of learners need identical lab setups.” |
| **FastAPI** (`api/main.py`) | “I exposed a clean REST contract with typed requests, OpenAPI docs, and clear failure codes — easy to teach against and easy to integrate.” |
| **Modular `scripts/` + `config/`** | “Config-driven team/location metadata keeps the pipeline portable across environments instead of hardcoding constants.” |
| **Monitoring + prediction log** | “I didn’t stop at training — freshness, PSI drift, and logged prediction quality show I think about operating models in the wild.” |
| **Explainability (linear + tree contribs)** | “Predictions come with ‘why’ — essential for teaching trust and debugging, not just accuracy flexing.” |
| **Future LLM matchup summaries** | “Next feature: an LLM agent that summarizes matchups using API outputs — aligning with DSN’s foundation-model and educational AI focus.” |

**Pivot sentence you can reuse:**
> “DSN’s job is making advanced tech understandable and runnable for developers worldwide. This project is my proof that I can take messy ML and package it as a teachable, deployable product — API, UI, containers, and honest operational caveats included.”

---

## 7. 🎯 The 60-Second Elevator Pitch & Cheat Sheet

### Opening response: “Tell me about this project.” (3 bullets — memorize)

1. **End-to-end ML product, not a notebook:** I built a factory that ingests NBA games/injuries/odds, engineers leakage-safe features, trains multiple models, promotes a champion on log loss/Brier, and serves predictions through FastAPI + a Streamlit dashboard — all Dockerized.
2. **Designed for honest forecasting:** Rolling features use prior-game-only windows; inference recomputes upcoming matchup features; metrics are evaluated at game-level; and the remaining hard problem I call out is point-in-time odds snapshots.
3. **Built like something you’d teach with:** Clean REST contract, interactive UI, monitoring/drift reports, and containerized reproducibility — the same muscles needed for developer labs and educational platforms.

### Ultra cheat sheet (pocket card)

| Topic | One-liner |
|---|---|
| Leakage | `shift(1).rolling(...)` = look only at past games |
| Inference | Recomputes rolling/travel/fatigue/odds for upcoming matchups |
| Eval unit | Train on team-rows; score holdout at game-level |
| Hard remaining risk | Pregame odds snapshot / point-in-time market features |
| Champion rule | Best probabilistic score (log loss + Brier), not flashy architecture |
| LSTM | Experiment only; not served / not promoted |
| Kitchen / Menu | FastAPI cooks; Streamlit waits tables |
| Odds fail | Cache or empty CSV; serving continues with warnings |
| `/predict` | Disabled (410); use `/predict/team` |
| Postgres | In Compose, unused in Python — file artifacts today |
| DSN bridge | Teachable API + lab UI + reproducible Docker |

### Files to be able to screen-share

1. `scripts/build_features.py` — the no-cheating factory
2. `scripts/build_inference_features.py` — upcoming feature materialization (recomputed fields)
3. `scripts/model_utils.py` — `select_game_level_rows` / promotion metrics helpers
4. `scripts/model_promotion.py` — the tournament judge
5. `api/main.py` — `/predict/team` + normalization + startup checks
6. `docker-compose.yml` — four containers, including unused Postgres (own it)

### Practice question drill

**Q: How do you prevent leakage?**  
A: Prior-game-only rolling features via `shift(1)`, backward injury as-of merges, and a leakage column guard. Odds still need a stricter pregame freeze for point-in-time correctness.

**Q: How do you avoid train/serve skew?**  
A: Inference rebuilds matchup features for upcoming games — rolling form from completed games, venue travel, fatigue, injuries, and odds when available — instead of shipping an untouched historical row.

**Q: Why two rows per game if you evaluate once?**  
A: Team-centric features want both perspectives for training/serving each side; game-level metrics avoid treating paired anti-correlated labels as independent samples.

**Q: Did the LSTM win?**  
A: It wasn’t in the production tournament path. Champion promotion is among baseline / tree / AutoML challenger using probability metrics.

**Q: What if an upstream API fails?**  
A: Fetchers fail soft (cache / empty schema), API serves with data-quality warnings and confidence penalties; hard stop only when schedule, inference features, or models are missing.

**Q: Why FastAPI + Streamlit?**  
A: Separation of concerns — reusable inference service + rapid educational dashboard on top.

---

_Last tip: In the interview, lead with software design decisions (modularity, failure modes, promotion gates, offline/online features, game-level evaluation). Use NBA stats only as seasoning, not the main course._
