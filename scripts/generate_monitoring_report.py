import json
import os
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd


PROCESSED_PATH = "data/processed/games_with_features.csv"
UPCOMING_PATH = "data/raw/upcoming_games.csv"
INFERENCE_PATH = "data/processed/upcoming_inference_features.csv"
INJURIES_LATEST_PATH = "data/raw/injuries_latest.csv"
REPORT_PATH = "reports/monitoring_report.json"
DAILY_SUMMARY_PATH = "reports/monitoring_daily_summary.csv"
ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL")
STALE_ALERT_HOURS = float(os.environ.get("STALE_ALERT_HOURS", "8"))
HIGH_DRIFT_PSI = float(os.environ.get("HIGH_DRIFT_PSI", "0.25"))

CORE_FEATURES = [
    "pts_last10",
    "reb_last10",
    "ast_last10",
    "REST_DAYS",
    "BACK_TO_BACK",
    "TRAVEL_DISTANCE",
    "TIMEZONE_SHIFT",
    "fatigue_index",
    "INJURY_IMPACT",
    "ADULT_ENTERTAINMENT_INDEX",
]


def _file_freshness(path):
    if not os.path.exists(path):
        return {"exists": False, "age_hours": None, "modified_utc": None}
    mtime = os.path.getmtime(path)
    now = datetime.now(timezone.utc).timestamp()
    age_hours = (now - mtime) / 3600.0
    modified = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    return {"exists": True, "age_hours": round(age_hours, 2), "modified_utc": modified}


def _psi(expected, actual, bins=10):
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.empty or actual.empty:
        return None
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 3:
        return None
    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)
    exp_pct = np.clip(exp_counts / max(exp_counts.sum(), 1), 1e-6, 1)
    act_pct = np.clip(act_counts / max(act_counts.sum(), 1), 1e-6, 1)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _drift_section(df):
    if df.empty or "GAME_DATE" not in df.columns:
        return {"status": "unavailable", "reason": "missing data or GAME_DATE"}
    work = df.copy()
    work["GAME_DATE"] = pd.to_datetime(work["GAME_DATE"], errors="coerce")
    work = work.dropna(subset=["GAME_DATE"])
    if work.empty:
        return {"status": "unavailable", "reason": "no valid GAME_DATE rows"}
    latest_date = work["GAME_DATE"].max()
    recent_start = latest_date - timedelta(days=14)
    baseline_start = latest_date - timedelta(days=74)
    baseline_end = latest_date - timedelta(days=14)
    recent = work[(work["GAME_DATE"] > recent_start) & (work["GAME_DATE"] <= latest_date)]
    baseline = work[(work["GAME_DATE"] > baseline_start) & (work["GAME_DATE"] <= baseline_end)]
    if recent.empty or baseline.empty:
        return {
            "status": "unavailable",
            "reason": "insufficient windows",
            "recent_rows": int(len(recent)),
            "baseline_rows": int(len(baseline)),
        }

    feature_psi = {}
    for feature in CORE_FEATURES:
        if feature not in work.columns:
            continue
        psi = _psi(baseline[feature], recent[feature])
        if psi is not None:
            feature_psi[feature] = round(float(psi), 4)

    avg_psi = float(np.mean(list(feature_psi.values()))) if feature_psi else None
    max_psi = float(np.max(list(feature_psi.values()))) if feature_psi else None
    status = "ok"
    if max_psi is not None and max_psi >= 0.25:
        status = "high_drift"
    elif max_psi is not None and max_psi >= 0.10:
        status = "moderate_drift"

    return {
        "status": status,
        "latest_game_date": str(latest_date.date()),
        "recent_rows": int(len(recent)),
        "baseline_rows": int(len(baseline)),
        "avg_psi": round(avg_psi, 4) if avg_psi is not None else None,
        "max_psi": round(max_psi, 4) if max_psi is not None else None,
        "feature_psi": feature_psi,
    }


def generate_monitoring_report(
    processed_path=PROCESSED_PATH,
    report_path=REPORT_PATH,
):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    processed = pd.read_csv(processed_path) if os.path.exists(processed_path) else pd.DataFrame()
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "freshness": {
            "upcoming_games": _file_freshness(UPCOMING_PATH),
            "inference_features": _file_freshness(INFERENCE_PATH),
            "injuries_latest": _file_freshness(INJURIES_LATEST_PATH),
            "processed_features": _file_freshness(processed_path),
        },
        "drift": _drift_section(processed),
    }
    report["alerts"] = _alerts_section(report)
    _append_daily_summary(report)
    _send_webhook_if_needed(report)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Wrote monitoring report to {report_path}")
    return report


def _alerts_section(report):
    alerts = []
    freshness = report.get("freshness", {})
    drift = report.get("drift", {})

    up = freshness.get("upcoming_games", {})
    inf = freshness.get("inference_features", {})
    inj = freshness.get("injuries_latest", {})

    for name, section in [("upcoming_games", up), ("inference_features", inf), ("injuries_latest", inj)]:
        if not section.get("exists", False):
            alerts.append({"name": f"{name}_missing", "severity": "fail", "message": f"{name} file missing"})
            continue
        age = section.get("age_hours")
        try:
            age_f = float(age)
            if age_f > STALE_ALERT_HOURS:
                alerts.append({
                    "name": f"{name}_stale",
                    "severity": "warn",
                    "message": f"{name} stale: {age_f:.2f}h > {STALE_ALERT_HOURS:.2f}h",
                })
        except Exception:
            pass

    if os.path.exists(UPCOMING_PATH):
        try:
            up_df = pd.read_csv(UPCOMING_PATH)
            if up_df.empty:
                alerts.append({"name": "upcoming_games_empty", "severity": "fail", "message": "upcoming_games.csv has 0 rows"})
        except Exception:
            alerts.append({"name": "upcoming_games_unreadable", "severity": "fail", "message": "upcoming_games.csv unreadable"})

    max_psi = drift.get("max_psi")
    if max_psi is not None:
        try:
            max_psi_f = float(max_psi)
            if max_psi_f >= HIGH_DRIFT_PSI:
                alerts.append({
                    "name": "high_drift",
                    "severity": "warn",
                    "message": f"max_psi={max_psi_f:.4f} >= {HIGH_DRIFT_PSI:.4f}",
                })
        except Exception:
            pass
    overall = "pass"
    if any(a["severity"] == "fail" for a in alerts):
        overall = "fail"
    elif any(a["severity"] == "warn" for a in alerts):
        overall = "warn"
    return {"overall_status": overall, "items": alerts}


def _append_daily_summary(report):
    row = {
        "generated_at_utc": report.get("generated_at_utc"),
        "alert_status": (report.get("alerts") or {}).get("overall_status"),
        "drift_status": (report.get("drift") or {}).get("status"),
        "max_psi": (report.get("drift") or {}).get("max_psi"),
        "avg_psi": (report.get("drift") or {}).get("avg_psi"),
        "upcoming_age_hours": ((report.get("freshness") or {}).get("upcoming_games") or {}).get("age_hours"),
        "inference_age_hours": ((report.get("freshness") or {}).get("inference_features") or {}).get("age_hours"),
        "injuries_age_hours": ((report.get("freshness") or {}).get("injuries_latest") or {}).get("age_hours"),
        "num_alerts": len(((report.get("alerts") or {}).get("items") or [])),
    }
    row_df = pd.DataFrame([row])
    os.makedirs(os.path.dirname(DAILY_SUMMARY_PATH), exist_ok=True)
    if os.path.exists(DAILY_SUMMARY_PATH):
        try:
            hist = pd.read_csv(DAILY_SUMMARY_PATH)
            out = pd.concat([hist, row_df], ignore_index=True)
        except Exception:
            out = row_df
    else:
        out = row_df
    out.to_csv(DAILY_SUMMARY_PATH, index=False)


def _send_webhook_if_needed(report):
    if not ALERT_WEBHOOK_URL:
        return
    alerts = (report.get("alerts") or {}).get("items") or []
    if not alerts:
        return
    payload = {
        "text": (
            f"[NBA Monitoring] status={(report.get('alerts') or {}).get('overall_status')} "
            f"alerts={len(alerts)} generated_at={report.get('generated_at_utc')}"
        ),
        "alerts": alerts[:10],
    }
    try:
        import requests
        requests.post(ALERT_WEBHOOK_URL, json=payload, timeout=6)
    except Exception:
        pass


def main():
    generate_monitoring_report()


if __name__ == "__main__":
    main()
