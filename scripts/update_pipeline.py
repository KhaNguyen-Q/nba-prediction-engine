import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

from apscheduler.schedulers.blocking import BlockingScheduler

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.get_data import (
    fetch_games_data,
    fetch_players_data,
    fetch_player_game_logs_data,
    fetch_odds_data,
)
from scripts.fetch_schedule import fetch_upcoming_schedule
from scripts.fetch_availability import fetch_availability_for_upcoming
from scripts.build_features import main as build_features_main
from scripts.build_inference_features import main as build_inference_features_main
from scripts.generate_monitoring_report import generate_monitoring_report
from scripts.generate_prediction_quality_report import generate_prediction_quality_report
from scripts.model_promotion import promote_champion_model
from scripts.train_automl_challenger import run_automl_challenger
from scripts.train_baseline import train_baseline
from scripts.train_player_model import train_player_model
from scripts.train_tree_model import train_tree_model


LOG_DIR = "logs"
PIPELINE_LOG_PATH = os.path.join(LOG_DIR, "pipeline.log")
RUN_META_DIR = "data/raw"
PIPELINE_LATEST_STATUS_PATH = os.path.join(RUN_META_DIR, "pipeline_status_latest.json")
STALE_THRESHOLD_HOURS = float(os.environ.get("PIPELINE_STALE_THRESHOLD_HOURS", "8"))
RETRAIN_MAX_PSI = float(os.environ.get("PIPELINE_RETRAIN_MAX_PSI", "0.18"))
RETRAIN_ON_STALE = os.environ.get("PIPELINE_RETRAIN_ON_STALE", "1").strip().lower() in {"1", "true", "yes"}
FORCE_RETRAIN = os.environ.get("PIPELINE_FORCE_RETRAIN", "0").strip().lower() in {"1", "true", "yes"}
MIN_PROCESSED_ROWS_FOR_RETRAIN = int(os.environ.get("MIN_PROCESSED_ROWS_FOR_RETRAIN", "3000"))
MIN_UPCOMING_ROWS_FOR_RETRAIN = int(os.environ.get("MIN_UPCOMING_ROWS_FOR_RETRAIN", "2"))
MIN_PLAYER_LOG_ROWS_FOR_RETRAIN = int(os.environ.get("MIN_PLAYER_LOG_ROWS_FOR_RETRAIN", "1000"))


def _setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("update_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(PIPELINE_LOG_PATH, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


LOGGER = _setup_logger()


def _run_with_retry(step_name, fn, max_attempts=3):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            LOGGER.info(f"Starting step '{step_name}' (attempt {attempt}/{max_attempts})")
            started = time.time()
            result = fn()
            elapsed = time.time() - started
            LOGGER.info(f"Finished step '{step_name}' in {elapsed:.2f}s")
            return {"step": step_name, "status": "success", "elapsed_sec": round(elapsed, 2)}
        except Exception as exc:
            last_error = str(exc)
            LOGGER.warning(f"Step '{step_name}' failed on attempt {attempt}: {exc}")
            time.sleep(min(5 * attempt, 15))
    LOGGER.error(f"Step '{step_name}' failed after {max_attempts} attempts: {last_error}")
    return {"step": step_name, "status": "failed", "error": last_error}


def run_update_once():
    run_ts = datetime.now(timezone.utc)
    run_id = run_ts.strftime("%Y%m%d_%H%M%S")
    LOGGER.info(f"=== Pipeline update run {run_id} started ===")

    steps = [
        ("get_data", fetch_games_data),
        ("fetch_players", fetch_players_data),
        ("fetch_player_logs", fetch_player_game_logs_data),
        ("fetch_odds", fetch_odds_data),
        ("fetch_schedule", lambda: fetch_upcoming_schedule(days_ahead=7)),
        ("fetch_availability", fetch_availability_for_upcoming),
        ("build_features", build_features_main),
        ("build_inference_features", build_inference_features_main),
        ("generate_monitoring_report", generate_monitoring_report),
        ("generate_prediction_quality_report", generate_prediction_quality_report),
    ]

    results = []
    for step_name, fn in steps:
        results.append(_run_with_retry(step_name, fn, max_attempts=3))

    retrain_info = _maybe_retrain_and_promote()
    if retrain_info.get("triggered"):
        for item in retrain_info.get("steps", []):
            results.append(item)

    ok = all(item["status"] == "success" for item in results)
    staleness = _compute_staleness_warnings(threshold_hours=STALE_THRESHOLD_HOURS)
    for warning in staleness:
        LOGGER.warning(warning)
    run_record = {
        "run_id": run_id,
        "started_at_utc": run_ts.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "success" if ok else "partial_failure",
        "steps": results,
        "staleness_warnings": staleness,
        "retrain": retrain_info,
    }

    os.makedirs(RUN_META_DIR, exist_ok=True)
    meta_path = os.path.join(RUN_META_DIR, f"pipeline_run_{run_id}.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(run_record, fh, indent=2)
    with open(PIPELINE_LATEST_STATUS_PATH, "w", encoding="utf-8") as fh:
        json.dump(run_record, fh, indent=2)
    LOGGER.info(f"Wrote run metadata to {meta_path}")
    LOGGER.info(f"Updated latest pipeline status at {PIPELINE_LATEST_STATUS_PATH}")
    LOGGER.info(f"=== Pipeline update run {run_id} completed with status={run_record['status']} ===")
    return run_record


def _load_monitoring_payload():
    report_path = "reports/monitoring_report.json"
    if not os.path.exists(report_path):
        return {}
    try:
        with open(report_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _retrain_reasons_from_monitoring(payload):
    reasons = []
    drift = payload.get("drift", {}) if isinstance(payload, dict) else {}
    max_psi = drift.get("max_psi")
    if max_psi is not None:
        try:
            max_psi = float(max_psi)
            if max_psi >= RETRAIN_MAX_PSI:
                reasons.append(
                    f"drift max_psi={max_psi:.4f} exceeds threshold={RETRAIN_MAX_PSI:.4f}"
                )
        except Exception:
            pass

    if RETRAIN_ON_STALE:
        freshness = payload.get("freshness", {}) if isinstance(payload, dict) else {}
        for name, section in freshness.items():
            if not isinstance(section, dict):
                continue
            if not section.get("exists", False):
                reasons.append(f"freshness source missing: {name}")
                continue
            age = section.get("age_hours")
            try:
                age_f = float(age)
                if age_f > STALE_THRESHOLD_HOURS:
                    reasons.append(
                        f"freshness source stale: {name} age={age_f:.2f}h threshold={STALE_THRESHOLD_HOURS:.2f}h"
                    )
            except Exception:
                continue
    return reasons


def _maybe_retrain_and_promote():
    monitoring_payload = _load_monitoring_payload()
    reasons = _retrain_reasons_from_monitoring(monitoring_payload)
    data_ready, data_reason = _minimum_data_ready_for_retrain()
    triggered = FORCE_RETRAIN or bool(reasons)
    if FORCE_RETRAIN and "forced via PIPELINE_FORCE_RETRAIN=1" not in reasons:
        reasons.insert(0, "forced via PIPELINE_FORCE_RETRAIN=1")
    if not data_ready and not FORCE_RETRAIN:
        reasons.append(f"retrain blocked: {data_reason}")
        triggered = False

    retrain_record = {
        "triggered": bool(triggered),
        "reasons": reasons,
        "minimum_data_ready": bool(data_ready),
        "minimum_data_reason": data_reason,
        "steps": [],
        "champion": None,
    }
    if not triggered:
        LOGGER.info("Retrain skipped: trigger conditions not met.")
        # Still try champion promotion in case new challenger exists from manual runs.
        retrain_record["steps"].append(_run_with_retry("promote_champion_model", promote_champion_model, max_attempts=1))
        return retrain_record

    LOGGER.info("Retrain triggered. Reasons: %s", "; ".join(reasons))
    train_steps = [
        ("train_baseline", train_baseline),
        ("train_tree_model", train_tree_model),
        ("train_player_model", train_player_model),
        ("train_automl_challenger", run_automl_challenger),
        ("generate_monitoring_report_post_retrain", generate_monitoring_report),
        ("generate_prediction_quality_report_post_retrain", generate_prediction_quality_report),
        ("promote_champion_model", promote_champion_model),
    ]
    for step_name, fn in train_steps:
        retrain_record["steps"].append(_run_with_retry(step_name, fn, max_attempts=1))

    champion_meta_path = "models/champion_team_model_meta.json"
    if os.path.exists(champion_meta_path):
        try:
            with open(champion_meta_path, "r", encoding="utf-8") as fh:
                retrain_record["champion"] = json.load(fh)
        except Exception:
            retrain_record["champion"] = None
    return retrain_record


def _safe_row_count(path):
    if not os.path.exists(path):
        return 0
    try:
        import pandas as pd
        return int(len(pd.read_csv(path)))
    except Exception:
        return 0


def _minimum_data_ready_for_retrain():
    processed_rows = _safe_row_count("data/processed/games_with_features.csv")
    upcoming_rows = _safe_row_count("data/raw/upcoming_games.csv")
    player_log_rows = _safe_row_count("data/raw/player_game_logs_raw.csv")
    if processed_rows < MIN_PROCESSED_ROWS_FOR_RETRAIN:
        return False, (
            f"processed rows too low ({processed_rows} < {MIN_PROCESSED_ROWS_FOR_RETRAIN})"
        )
    if upcoming_rows < MIN_UPCOMING_ROWS_FOR_RETRAIN:
        return False, (
            f"upcoming schedule rows too low ({upcoming_rows} < {MIN_UPCOMING_ROWS_FOR_RETRAIN})"
        )
    if player_log_rows < MIN_PLAYER_LOG_ROWS_FOR_RETRAIN:
        return False, (
            f"player log rows too low ({player_log_rows} < {MIN_PLAYER_LOG_ROWS_FOR_RETRAIN})"
        )
    return True, "minimum data thresholds satisfied"


def _compute_staleness_warnings(threshold_hours=8.0):
    warnings = []
    now = datetime.now(timezone.utc).timestamp()
    targets = [
        "data/raw/upcoming_games.csv",
        "data/processed/upcoming_inference_features.csv",
        "data/raw/injuries_latest.csv",
        "reports/monitoring_report.json",
    ]
    for path in targets:
        if not os.path.exists(path):
            warnings.append(f"Staleness alert: missing required file {path}")
            continue
        age_hours = (now - os.path.getmtime(path)) / 3600.0
        if age_hours > threshold_hours:
            warnings.append(
                f"Staleness alert: {path} is {age_hours:.2f} hours old "
                f"(threshold={threshold_hours:.2f}h)"
            )
    return warnings


def run_scheduler():
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(run_update_once, "interval", hours=6, id="pipeline_update")
    LOGGER.info("Starting scheduler: pipeline update every 6 hours (UTC).")
    run_update_once()
    scheduler.start()


def main():
    mode = os.environ.get("PIPELINE_MODE", "once").strip().lower()
    if mode == "scheduler":
        run_scheduler()
    else:
        run_update_once()


if __name__ == "__main__":
    main()
