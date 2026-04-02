import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


REGISTRY_DIR = "models/registry"
REGISTRY_INDEX_PATH = os.path.join(REGISTRY_DIR, "index.json")

LEAKAGE_EXACT_COLUMNS = {
    "WIN",
    "WL",
    "PTS",
    "REB",
    "AST",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "PLUS_MINUS",
    "OREB",
    "DREB",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "MIN",
}
LEAKAGE_PREFIXES = (
    "TEAM_",
    "OPPONENT_",
    "HOME_TEAM_",
    "AWAY_TEAM_",
    "CURRENT_",
    "PREV_GAME_",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value, default=0.0) -> float:
    try:
        if pd.isna(value):
            return float(default)
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return float(default)


def time_aware_train_test_split(
    df: pd.DataFrame,
    date_col: str = "GAME_DATE",
    test_size: float = 0.2,
    min_unique_dates: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    if date_col in df.columns:
        dated = df.copy()
        dated[date_col] = pd.to_datetime(dated[date_col], errors="coerce")
        dated = dated.dropna(subset=[date_col]).sort_values(date_col)
        unique_dates = dated[date_col].drop_duplicates().sort_values().tolist()
        if len(unique_dates) >= min_unique_dates:
            split_idx = max(1, int(len(unique_dates) * (1 - test_size)))
            split_idx = min(split_idx, len(unique_dates) - 1)
            split_date = unique_dates[split_idx]
            train_df = dated[dated[date_col] < split_date]
            test_df = dated[dated[date_col] >= split_date]
            if not train_df.empty and not test_df.empty:
                return train_df, test_df, f"time-based split at {split_date.date()}"

    split_idx = max(1, int(len(df) * (1 - test_size)))
    split_idx = min(split_idx, len(df) - 1)
    if split_idx <= 0:
        split_idx = 1
    return (
        df.iloc[:split_idx].copy(),
        df.iloc[split_idx:].copy(),
        "row-order split fallback",
    )


def rolling_time_splits(
    df: pd.DataFrame,
    date_col: str = "GAME_DATE",
    n_splits: int = 4,
    min_train_dates: int = 20,
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    if date_col not in df.columns:
        return []

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values(date_col)
    if work.empty:
        return []

    unique_dates = work[date_col].drop_duplicates().sort_values().tolist()
    if len(unique_dates) < max(min_train_dates + n_splits, min_train_dates + 1):
        return []

    split_points = np.linspace(min_train_dates, len(unique_dates) - 1, n_splits + 1, dtype=int)[1:]
    folds = []
    for idx, point in enumerate(split_points, start=1):
        split_date = unique_dates[int(point)]
        train_idx = work.index[work[date_col] < split_date].to_numpy()
        test_idx = work.index[work[date_col] >= split_date].to_numpy()
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        label = f"fold_{idx}_split_{split_date.date()}"
        folds.append((train_idx, test_idx, label))
    return folds


def classification_metrics(y_true, y_pred, y_score) -> Dict[str, float]:
    y_true = pd.Series(y_true).astype(int)
    out = {
        "accuracy": safe_float(accuracy_score(y_true, y_pred)),
        "log_loss": safe_float(log_loss(y_true, y_score)),
        "brier_score": safe_float(brier_score_loss(y_true, y_score)),
    }
    if y_true.nunique() > 1:
        out["roc_auc"] = safe_float(roc_auc_score(y_true, y_score))
    else:
        out["roc_auc"] = float("nan")
    return out


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    mae = safe_float(mean_absolute_error(y_true, y_pred))
    rmse = safe_float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"mae": mae, "rmse": rmse}


def write_registry_entry(
    model_name: str,
    model_path: str,
    task_type: str,
    dataset_path: str,
    feature_columns: List[str],
    metrics: Dict,
    split_description: str,
    extra: Dict = None,
) -> str:
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    entry = {
        "model_name": model_name,
        "model_path": model_path,
        "task_type": task_type,
        "dataset_path": dataset_path,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "metrics": metrics,
        "split": split_description,
        "trained_at": utc_now_iso(),
    }
    if extra:
        entry.update(extra)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    entry_path = os.path.join(REGISTRY_DIR, f"{model_name}_{stamp}.json")
    with open(entry_path, "w", encoding="utf-8") as fh:
        json.dump(entry, fh, indent=2)

    index = {"updated_at": utc_now_iso(), "entries": []}
    if os.path.exists(REGISTRY_INDEX_PATH):
        try:
            with open(REGISTRY_INDEX_PATH, "r", encoding="utf-8") as fh:
                index = json.load(fh)
        except Exception:
            index = {"updated_at": utc_now_iso(), "entries": []}

    entries = index.get("entries", [])
    entries = [e for e in entries if e.get("model_name") != model_name]
    entries.insert(0, {"model_name": model_name, "entry_path": entry_path, "updated_at": utc_now_iso()})
    index["entries"] = entries[:50]
    index["updated_at"] = utc_now_iso()

    with open(REGISTRY_INDEX_PATH, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)

    return entry_path


def leakage_safe_team_features(df: pd.DataFrame) -> List[str]:
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in LEAKAGE_EXACT_COLUMNS]
    safe_cols = []
    for col in numeric_cols:
        if col in LEAKAGE_EXACT_COLUMNS:
            continue
        if any(col.startswith(prefix) for prefix in LEAKAGE_PREFIXES):
            continue
        safe_cols.append(col)
    return safe_cols


def calibration_table(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({
        "y_true": pd.Series(y_true).astype(float),
        "y_score": pd.Series(y_score).astype(float).clip(lower=0.0, upper=1.0),
    }).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["bin", "count", "predicted_mean", "observed_rate", "abs_gap"])

    # rank-based bins are robust when probabilities are concentrated
    frame["bin"] = pd.qcut(frame["y_score"], q=min(n_bins, max(2, frame["y_score"].nunique())), duplicates="drop")
    grouped = frame.groupby("bin", observed=False).agg(
        count=("y_true", "size"),
        predicted_mean=("y_score", "mean"),
        observed_rate=("y_true", "mean"),
    ).reset_index()
    grouped["abs_gap"] = (grouped["predicted_mean"] - grouped["observed_rate"]).abs()
    return grouped


def write_calibration_report(
    model_name: str,
    y_true,
    y_score,
    out_dir: str = "reports",
    n_bins: int = 10,
) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    table = calibration_table(y_true, y_score, n_bins=n_bins)
    csv_path = os.path.join(out_dir, f"calibration_{model_name}.csv")
    table.to_csv(csv_path, index=False)
    if table.empty:
        ece = float("nan")
    else:
        ece = float((table["abs_gap"] * table["count"]).sum() / max(float(table["count"].sum()), 1.0))
    return {"path": csv_path, "ece": ece, "bins": int(len(table))}
