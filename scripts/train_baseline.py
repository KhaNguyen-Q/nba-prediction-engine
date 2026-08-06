import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.model_utils import (
    classification_metrics_game_level,
    rolling_time_splits,
    select_game_level_rows,
    time_aware_train_test_split,
    write_calibration_report,
    write_registry_entry,
)

PROCESSED_PATH = "data/processed/games_with_features.csv"
MODEL_PATH = "models/logistic_baseline.pkl"
REQUIRED_FEATURES = [
    'pts_last5', 'reb_last5', 'ast_last5',
    'pts_last10', 'reb_last10', 'ast_last10',
    'REST_DAYS', 'BACK_TO_BACK', 'TRAVEL_DISTANCE', 'TIMEZONE_SHIFT',
    'fatigue_index',
]


def _time_split_dataframe(df, test_size=0.2):
    train_df, test_df, split_desc = time_aware_train_test_split(df, date_col='GAME_DATE', test_size=test_size)
    if not train_df.empty and not test_df.empty:
        return train_df, test_df, split_desc

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=42,
        stratify=df['WIN'].astype(int),
    )
    return df.iloc[train_idx], df.iloc[test_idx], "stratified random split (fallback)"


def train_baseline():
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(f"Processed dataset not found: {PROCESSED_PATH}")

    df = pd.read_csv(PROCESSED_PATH)
    if 'WIN' not in df.columns:
        raise ValueError("Processed dataset must contain WIN label.")
    if df.empty:
        raise ValueError("Processed dataset is empty.")

    missing_features = [feature for feature in REQUIRED_FEATURES if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required baseline features: {missing_features}")

    features = REQUIRED_FEATURES.copy()
    df = df.dropna(subset=features + ['WIN']).copy()
    if df.empty:
        raise ValueError("No rows left after dropping missing features/WIN.")
    if df['WIN'].nunique() < 2:
        raise ValueError("WIN label must contain both classes for training.")

    train_df, test_df, split_desc = _time_split_dataframe(df, test_size=0.2)
    print(f"Using {split_desc}.")

    X_train = train_df[features]
    y_train = train_df['WIN'].astype(int)
    X_test = test_df[features]
    y_test = test_df['WIN'].astype(int)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=5000, solver='lbfgs', random_state=42)),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    holdout_metrics = classification_metrics_game_level(test_df, y_pred, y_score)
    eval_test = select_game_level_rows(test_df.assign(_y_score=y_score))

    print("Baseline metrics (game-level holdout)")
    print("Accuracy:", holdout_metrics.get("accuracy"))
    print("ROC-AUC:", holdout_metrics.get("roc_auc"))
    print("Log loss:", holdout_metrics.get("log_loss"))
    print("Brier score:", holdout_metrics.get("brier_score"))
    print(f"Eval unit: {holdout_metrics.get('eval_unit')} ({holdout_metrics.get('eval_rows')} rows)")
    calibration_report = write_calibration_report(
        "baseline",
        eval_test['WIN'],
        eval_test['_y_score'],
        out_dir="reports",
        n_bins=10,
    )
    print(f"Calibration report: {calibration_report['path']} (ECE={calibration_report['ece']:.4f})")

    rolling_results = []
    folds = rolling_time_splits(df, date_col='GAME_DATE', n_splits=4, min_train_dates=20)
    if folds:
        X_all = df[features].fillna(0.0)
        y_all = df['WIN'].astype(int)
        for train_idx, test_idx, fold_label in folds:
            X_fold_train = X_all.loc[train_idx]
            y_fold_train = y_all.loc[train_idx]
            fold_test_df = df.loc[test_idx]
            if y_fold_train.nunique() < 2:
                continue
            fold_model = Pipeline([
                ('scaler', StandardScaler()),
                ('logreg', LogisticRegression(max_iter=5000, solver='lbfgs', random_state=42)),
            ])
            fold_model.fit(X_fold_train, y_fold_train)
            y_fold_pred = fold_model.predict(fold_test_df[features].fillna(0.0))
            y_fold_score = fold_model.predict_proba(fold_test_df[features].fillna(0.0))[:, 1]
            fold_metrics = classification_metrics_game_level(fold_test_df, y_fold_pred, y_fold_score)
            if select_game_level_rows(fold_test_df)['WIN'].nunique() < 2:
                continue
            fold_metrics['fold'] = fold_label
            rolling_results.append(fold_metrics)
    if rolling_results:
        avg_log_loss = sum(m['log_loss'] for m in rolling_results) / len(rolling_results)
        avg_brier = sum(m['brier_score'] for m in rolling_results) / len(rolling_results)
        print(f"Rolling CV folds: {len(rolling_results)}")
        print(f"Rolling CV avg log loss: {avg_log_loss:.4f}")
        print(f"Rolling CV avg brier: {avg_brier:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    registry_path = write_registry_entry(
        model_name="logistic_baseline",
        model_path=MODEL_PATH,
        task_type="team_classification",
        dataset_path=PROCESSED_PATH,
        feature_columns=features,
        metrics={
            "holdout": holdout_metrics,
            "rolling_cv": rolling_results,
            "calibration": {
                "ece": calibration_report["ece"],
                "bins": calibration_report["bins"],
                "path": calibration_report["path"],
            },
        },
        split_description=f"{split_desc}; metrics evaluated at game-level",
        extra={
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_games": int(holdout_metrics.get("eval_rows", len(X_test))),
            "eval_unit": holdout_metrics.get("eval_unit", "row"),
        },
    )
    print(f"Wrote registry entry to {registry_path}")


if __name__ == '__main__':
    train_baseline()
