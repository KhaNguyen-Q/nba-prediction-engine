import os
import sys
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.model_utils import (
    classification_metrics,
    leakage_safe_team_features,
    rolling_time_splits,
    time_aware_train_test_split,
    write_registry_entry,
)


PROCESSED_PATH = "data/processed/games_with_features.csv"
MODEL_PATH = "models/automl_challenger.pkl"


def load_data(path=PROCESSED_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Processed dataset is empty.")
    if "WIN" not in df.columns:
        raise ValueError("Processed dataset must contain WIN label.")
    df = df.dropna(subset=["WIN"]).copy()
    if df["WIN"].nunique() < 2:
        raise ValueError("WIN label must contain both classes for training.")
    return df


def candidate_models():
    candidates = {
        "logreg_l2": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=4000, C=1.0, random_state=42)),
        ]),
        "rf_400": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=1)),
        ]),
    }
    if HAS_XGB:
        candidates["xgb_default"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                n_jobs=1,
                random_state=42,
            )),
        ])
    return candidates


def score_for_ranking(metric: dict) -> float:
    # Lower is better. Prioritize probabilistic quality.
    return float(metric["log_loss"]) + 0.6 * float(metric["brier_score"])


def data_has_values(series: pd.Series) -> bool:
    return series.notna().any()


def run_automl_challenger():
    df = load_data()
    numeric_features = leakage_safe_team_features(df)
    if not numeric_features:
        raise ValueError("No numeric features available for AutoML challenger.")
    numeric_features = [c for c in numeric_features if data_has_values(df[c])]
    if not numeric_features:
        raise ValueError("No non-empty numeric features available for AutoML challenger.")
    dropped_numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in numeric_features and c != "WIN"]
    if dropped_numeric:
        print(f"Leakage guard removed {len(dropped_numeric)} numeric columns.")

    data = df[numeric_features + ["WIN"] + ([ "GAME_DATE"] if "GAME_DATE" in df.columns else [])].dropna(subset=["WIN"]).copy()
    X_all = data[numeric_features]
    y_all = data["WIN"].astype(int)
    candidates = candidate_models()

    folds = rolling_time_splits(data, date_col="GAME_DATE", n_splits=4, min_train_dates=20)
    if not folds:
        train_df, test_df, desc = time_aware_train_test_split(data, date_col="GAME_DATE", test_size=0.2)
        folds = [(train_df.index.to_numpy(), test_df.index.to_numpy(), desc)]

    leaderboard = []
    for name, model in candidates.items():
        fold_metrics = []
        for train_idx, test_idx, fold_label in folds:
            X_train = X_all.loc[train_idx]
            y_train = y_all.loc[train_idx]
            X_test = X_all.loc[test_idx]
            y_test = y_all.loc[test_idx]
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                continue
            est = deepcopy(model)
            est.fit(X_train, y_train)
            y_pred = est.predict(X_test)
            y_score = est.predict_proba(X_test)[:, 1]
            m = classification_metrics(y_test, y_pred, y_score)
            m["fold"] = fold_label
            fold_metrics.append(m)
        if not fold_metrics:
            continue
        avg = {
            "accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
            "roc_auc": float(np.nanmean([m["roc_auc"] for m in fold_metrics])),
            "log_loss": float(np.mean([m["log_loss"] for m in fold_metrics])),
            "brier_score": float(np.mean([m["brier_score"] for m in fold_metrics])),
        }
        leaderboard.append({
            "model_name": name,
            "avg_metrics": avg,
            "score": score_for_ranking(avg),
            "fold_metrics": fold_metrics,
        })
        print(f"[{name}] avg log_loss={avg['log_loss']:.4f}, brier={avg['brier_score']:.4f}, auc={avg['roc_auc']:.4f}")

    if not leaderboard:
        raise RuntimeError("No valid AutoML challenger candidates were trainable.")

    leaderboard = sorted(leaderboard, key=lambda x: x["score"])
    winner = leaderboard[0]
    winner_name = winner["model_name"]
    print(f"AutoML winner: {winner_name}")

    train_df, test_df, split_desc = time_aware_train_test_split(data, date_col="GAME_DATE", test_size=0.2)
    X_train = train_df[numeric_features]
    y_train = train_df["WIN"].astype(int)
    X_test = test_df[numeric_features]
    y_test = test_df["WIN"].astype(int)

    final_model = deepcopy(candidates[winner_name])
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    y_score = final_model.predict_proba(X_test)[:, 1]
    holdout = classification_metrics(y_test, y_pred, y_score)

    artifact = {
        "model": final_model,
        "model_name": winner_name,
        "feature_columns": numeric_features,
        "leaderboard": leaderboard,
        "holdout_metrics": holdout,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved AutoML challenger model to {MODEL_PATH}")

    registry_path = write_registry_entry(
        model_name="automl_challenger",
        model_path=MODEL_PATH,
        task_type="team_classification",
        dataset_path=PROCESSED_PATH,
        feature_columns=numeric_features,
        metrics={
            "winner": winner_name,
            "winner_holdout": holdout,
            "leaderboard": leaderboard,
        },
        split_description=split_desc,
        extra={
            "candidate_count": len(candidates),
            "xgboost_available": HAS_XGB,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
    )
    print(f"Wrote registry entry to {registry_path}")


if __name__ == "__main__":
    run_automl_challenger()
