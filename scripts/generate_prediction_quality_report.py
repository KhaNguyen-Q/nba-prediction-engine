import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


PREDICTION_LOG_PATH = "reports/prediction_log.csv"
PROCESSED_PATH = "data/processed/games_with_features.csv"
REPORT_JSON_PATH = "reports/prediction_quality_report.json"
REPORT_CSV_PATH = "reports/prediction_quality_summary.csv"


def _load_predictions():
    if not os.path.exists(PREDICTION_LOG_PATH):
        return pd.DataFrame()
    df = pd.read_csv(PREDICTION_LOG_PATH)
    if df.empty:
        return df
    for col in ["home_team_id", "away_team_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "predicted_at_utc" in df.columns:
        df["predicted_at_utc"] = pd.to_datetime(df["predicted_at_utc"], errors="coerce", utc=True).dt.tz_convert(None)
    return df


def _load_actuals():
    if not os.path.exists(PROCESSED_PATH):
        return pd.DataFrame()
    df = pd.read_csv(PROCESSED_PATH)
    required = {"GAME_ID", "TEAM_ID", "WIN"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    df = df[list(required)].copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str)
    df["TEAM_ID"] = pd.to_numeric(df["TEAM_ID"], errors="coerce")
    df["WIN"] = pd.to_numeric(df["WIN"], errors="coerce")
    df = df.dropna(subset=["TEAM_ID", "WIN"])
    df["TEAM_ID"] = df["TEAM_ID"].astype(int)
    df["WIN"] = df["WIN"].astype(int)
    return df


def _compute_metrics(frame):
    if frame.empty or frame["actual_home_win"].nunique() < 2:
        return {
            "rows": int(len(frame)),
            "accuracy": None,
            "log_loss": None,
            "brier_score": None,
        }
    y_true = frame["actual_home_win"].astype(int)
    y_score = frame["home_win_probability"].astype(float).clip(1e-6, 1 - 1e-6)
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "rows": int(len(frame)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_score)),
        "brier_score": float(brier_score_loss(y_true, y_score)),
    }


def _append_quality_history_row(payload):
    row = {
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": payload.get("status"),
        "scored_rows": payload.get("scored_rows"),
        "logged_rows": payload.get("logged_rows"),
        "overall_accuracy": (payload.get("overall") or {}).get("accuracy"),
        "overall_log_loss": (payload.get("overall") or {}).get("log_loss"),
        "overall_brier_score": (payload.get("overall") or {}).get("brier_score"),
        "recent_7d_accuracy": (payload.get("recent_7d") or {}).get("accuracy"),
        "recent_7d_log_loss": (payload.get("recent_7d") or {}).get("log_loss"),
        "recent_7d_brier_score": (payload.get("recent_7d") or {}).get("brier_score"),
        "recent_30d_accuracy": (payload.get("recent_30d") or {}).get("accuracy"),
        "recent_30d_log_loss": (payload.get("recent_30d") or {}).get("log_loss"),
        "recent_30d_brier_score": (payload.get("recent_30d") or {}).get("brier_score"),
    }
    row_df = pd.DataFrame([row])
    if os.path.exists(REPORT_CSV_PATH):
        hist = pd.read_csv(REPORT_CSV_PATH)
        out = pd.concat([hist, row_df], ignore_index=True)
    else:
        out = row_df
    out.to_csv(REPORT_CSV_PATH, index=False)


def generate_prediction_quality_report():
    pred = _load_predictions()
    actual = _load_actuals()
    os.makedirs(os.path.dirname(REPORT_JSON_PATH), exist_ok=True)

    if pred.empty or actual.empty:
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "unavailable",
            "detail": "Missing prediction logs or actual outcomes.",
        }
        with open(REPORT_JSON_PATH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        _append_quality_history_row(payload)
        print(f"Wrote prediction quality report to {REPORT_JSON_PATH}")
        return payload

    pred["game_id"] = pred["game_id"].astype(str)
    merged = pred.merge(
        actual.rename(columns={"GAME_ID": "game_id", "TEAM_ID": "home_team_id", "WIN": "actual_home_win"}),
        how="left",
        on=["game_id", "home_team_id"],
    )
    scored = merged.dropna(subset=["actual_home_win", "home_win_probability"]).copy()
    if scored.empty:
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "unavailable",
            "detail": "No scored predictions yet (actual outcomes not available for logged predictions).",
        }
        with open(REPORT_JSON_PATH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        _append_quality_history_row(payload)
        print(f"Wrote prediction quality report to {REPORT_JSON_PATH}")
        return payload

    now_naive_utc = pd.Timestamp.now(tz="UTC").tz_convert(None)
    recent_7 = scored[scored["predicted_at_utc"] >= (now_naive_utc - pd.Timedelta(days=7))] if "predicted_at_utc" in scored.columns else scored.iloc[0:0]
    recent_30 = scored[scored["predicted_at_utc"] >= (now_naive_utc - pd.Timedelta(days=30))] if "predicted_at_utc" in scored.columns else scored.iloc[0:0]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "overall": _compute_metrics(scored),
        "recent_7d": _compute_metrics(recent_7),
        "recent_30d": _compute_metrics(recent_30),
        "scored_rows": int(len(scored)),
        "logged_rows": int(len(pred)),
    }
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    _append_quality_history_row(payload)
    print(f"Wrote prediction quality report to {REPORT_JSON_PATH}")
    return payload


def main():
    generate_prediction_quality_report()


if __name__ == "__main__":
    main()
