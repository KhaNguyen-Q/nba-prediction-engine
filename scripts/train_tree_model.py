import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.model_utils import (
    classification_metrics,
    leakage_safe_team_features,
    rolling_time_splits,
    time_aware_train_test_split,
    write_calibration_report,
    write_registry_entry,
)

PROCESSED_PATH = "data/processed/games_with_features.csv"
LOGISTIC_MODEL_PATH = "models/logistic_baseline.pkl"
TREE_MODEL_PATH = "models/xgb_tree_model.pkl"


def load_processed_data(path=PROCESSED_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Processed dataset not found: {path}')
    df = pd.read_csv(path)
    if 'WIN' not in df.columns:
        raise ValueError('Processed dataset must contain WIN label.')
    if df.empty:
        raise ValueError('Processed dataset is empty.')
    return df


BASELINE_FEATURES = [
    'pts_last5', 'reb_last5', 'ast_last5',
    'pts_last10', 'reb_last10', 'ast_last10',
    'REST_DAYS', 'BACK_TO_BACK', 'TRAVEL_DISTANCE', 'TIMEZONE_SHIFT',
    'fatigue_index', 'ADULT_ENTERTAINMENT_INDEX'
]


def get_baseline_feature_set(df):
    return [f for f in BASELINE_FEATURES if f in df.columns]


def _time_split_dataframe(df, test_size=0.2):
    train_df, test_df, split_desc = time_aware_train_test_split(df, date_col='GAME_DATE', test_size=test_size)
    if not train_df.empty and not test_df.empty:
        return train_df, test_df, split_desc

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=42,
        stratify=df['WIN'].astype(int)
    )
    return df.iloc[train_idx], df.iloc[test_idx], "stratified random split (fallback)"


def train_tree_model():
    df = load_processed_data()
    df = df.dropna(subset=['WIN'])
    if df['WIN'].nunique() < 2:
        raise ValueError('WIN label must contain both classes for training.')

    feature_columns = leakage_safe_team_features(df)
    if not feature_columns:
        raise ValueError('No numeric features found for tree model training.')
    dropped_numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in feature_columns and c != 'WIN']
    if dropped_numeric:
        print(f"Leakage guard removed {len(dropped_numeric)} numeric columns.")

    train_df, test_df, split_desc = _time_split_dataframe(df, test_size=0.2)
    print(f'Using {split_desc}.')
    if train_df.empty or test_df.empty:
        raise ValueError('Split produced empty train or test set.')

    X_train = train_df[feature_columns].fillna(0)
    y_train = train_df['WIN'].astype(int)
    X_test = test_df[feature_columns].fillna(0)
    y_test = test_df['WIN'].astype(int)

    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    holdout_metrics = classification_metrics(y_test, y_pred, y_score)

    print('XGBoost model metrics')
    print('Accuracy:', holdout_metrics.get('accuracy'))
    print('ROC-AUC:', holdout_metrics.get('roc_auc'))
    print('Log loss:', holdout_metrics.get('log_loss'))
    print('Brier score:', holdout_metrics.get('brier_score'))
    calibration_report = write_calibration_report("tree", y_test, y_score, out_dir="reports", n_bins=10)
    print(f"Calibration report: {calibration_report['path']} (ECE={calibration_report['ece']:.4f})")

    rolling_results = []
    folds = rolling_time_splits(df, date_col='GAME_DATE', n_splits=4, min_train_dates=20)
    if folds:
        X_all = df[feature_columns].fillna(0.0)
        y_all = df['WIN'].astype(int)
        for train_idx, test_idx, fold_label in folds:
            X_fold_train = X_all.loc[train_idx]
            y_fold_train = y_all.loc[train_idx]
            X_fold_test = X_all.loc[test_idx]
            y_fold_test = y_all.loc[test_idx]
            if y_fold_train.nunique() < 2 or y_fold_test.nunique() < 2:
                continue
            fold_model = XGBClassifier(eval_metric='logloss', random_state=42)
            fold_model.fit(X_fold_train, y_fold_train)
            y_fold_pred = fold_model.predict(X_fold_test)
            y_fold_score = fold_model.predict_proba(X_fold_test)[:, 1]
            fold_metrics = classification_metrics(y_fold_test, y_fold_pred, y_fold_score)
            fold_metrics['fold'] = fold_label
            rolling_results.append(fold_metrics)
    if rolling_results:
        avg_log_loss = sum(m['log_loss'] for m in rolling_results) / len(rolling_results)
        avg_brier = sum(m['brier_score'] for m in rolling_results) / len(rolling_results)
        print(f"Rolling CV folds: {len(rolling_results)}")
        print(f"Rolling CV avg log loss: {avg_log_loss:.4f}")
        print(f"Rolling CV avg brier: {avg_brier:.4f}")

    if os.path.exists(LOGISTIC_MODEL_PATH):
        try:
            baseline_model = joblib.load(LOGISTIC_MODEL_PATH)
            if set(BASELINE_FEATURES).issubset(set(df.columns)):
                X_test_base = test_df[BASELINE_FEATURES].fillna(0)
                y_test_base = test_df['WIN'].astype(int)
                y_pred_base = baseline_model.predict(X_test_base)
                y_score_base = baseline_model.predict_proba(X_test_base)[:, 1]
                baseline_metrics = classification_metrics(y_test_base, y_pred_base, y_score_base)
                print('Logistic baseline metrics on same holdout split')
                print('Logistic Accuracy:', baseline_metrics.get('accuracy'))
                print('Logistic ROC-AUC:', baseline_metrics.get('roc_auc'))
                print('Logistic Log loss:', baseline_metrics.get('log_loss'))
                print('Logistic Brier score:', baseline_metrics.get('brier_score'))
            else:
                missing = [f for f in BASELINE_FEATURES if f not in df.columns]
                print(f'Cannot evaluate logistic baseline on same feature set, missing columns: {missing}')
        except Exception as exc:
            print(f'Could not evaluate logistic baseline: {exc}')

    os.makedirs(os.path.dirname(TREE_MODEL_PATH), exist_ok=True)
    joblib.dump(model, TREE_MODEL_PATH)
    print(f'Saved XGBoost model to {TREE_MODEL_PATH}')

    registry_path = write_registry_entry(
        model_name="xgb_tree_model",
        model_path=TREE_MODEL_PATH,
        task_type="team_classification",
        dataset_path=PROCESSED_PATH,
        feature_columns=feature_columns,
        metrics={
            "holdout": holdout_metrics,
            "rolling_cv": rolling_results,
            "calibration": {
                "ece": calibration_report["ece"],
                "bins": calibration_report["bins"],
                "path": calibration_report["path"],
            },
        },
        split_description=split_desc,
        extra={
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
    )
    print(f'Wrote registry entry to {registry_path}')


def main():
    train_tree_model()


if __name__ == '__main__':
    main()
