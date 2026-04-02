import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit, train_test_split

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

BASELINE_PATH = "models/logistic_baseline.pkl"
TREE_PATH = "models/xgb_tree_model.pkl"
ENSEMBLE_PATH = "models/ensemble_meta.pkl"
PROCESSED_PATH = "data/processed/games_with_features.csv"


BASELINE_FEATURES = [
    'pts_last5', 'reb_last5', 'ast_last5',
    'pts_last10', 'reb_last10', 'ast_last10',
    'REST_DAYS', 'BACK_TO_BACK', 'TRAVEL_DISTANCE', 'TIMEZONE_SHIFT',
    'fatigue_index', 'ADULT_ENTERTAINMENT_INDEX'
]


def load_meta_features(path=PROCESSED_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Processed dataset not found: {path}')
    df = pd.read_csv(path)
    if 'WIN' not in df.columns:
        raise ValueError('Processed dataset must contain WIN label.')
    if df.empty:
        raise ValueError('Processed dataset is empty.')
    return df


def _time_split_dataframe(df, test_size=0.2):
    dated = df.copy()
    dated['GAME_DATE'] = pd.to_datetime(dated['GAME_DATE'], errors='coerce') if 'GAME_DATE' in dated.columns else pd.NaT
    if 'GAME_DATE' in dated.columns:
        dated = dated.dropna(subset=['GAME_DATE']).sort_values('GAME_DATE')
        unique_dates = dated['GAME_DATE'].drop_duplicates().sort_values().tolist()
        if len(unique_dates) >= 10:
            split_idx = max(1, int(len(unique_dates) * (1 - test_size)))
            split_idx = min(split_idx, len(unique_dates) - 1)
            split_date = unique_dates[split_idx]
            train_df = dated[dated['GAME_DATE'] < split_date]
            test_df = dated[dated['GAME_DATE'] >= split_date]
            if not train_df.empty and not test_df.empty:
                return train_df, test_df, f"time-based split at {split_date.date()}"

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=42,
        stratify=df['WIN'].astype(int)
    )
    return df.iloc[train_idx], df.iloc[test_idx], "stratified random split (fallback)"


def _resolve_model_features(model, fallback_features, df, model_name):
    features = model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else fallback_features
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(f'Missing {model_name} features in dataset: {missing}')
    return features


def _build_oof_stack_features(baseline_template, tree_template, X_base_train, X_tree_train, y_train, game_dates):
    order = np.argsort(game_dates.values)
    n_samples = len(order)
    if n_samples < 8:
        raise ValueError('Not enough training rows to build OOF stack features.')

    n_splits = min(5, n_samples - 1)
    if n_splits < 2:
        raise ValueError('Unable to create enough time-series folds for OOF predictions.')

    splitter = TimeSeriesSplit(n_splits=n_splits)
    baseline_oof = np.full(n_samples, np.nan)
    tree_oof = np.full(n_samples, np.nan)
    y_sorted = y_train.iloc[order]

    for fold_id, (train_pos, val_pos) in enumerate(splitter.split(order), start=1):
        train_idx = order[train_pos]
        val_idx = order[val_pos]

        y_fold_train = y_train.iloc[train_idx]
        if y_fold_train.nunique() < 2:
            continue

        base_fold = clone(baseline_template)
        tree_fold = clone(tree_template)
        base_fold.fit(X_base_train.iloc[train_idx], y_fold_train)
        tree_fold.fit(X_tree_train.iloc[train_idx], y_fold_train)

        baseline_oof[val_idx] = base_fold.predict_proba(X_base_train.iloc[val_idx])[:, 1]
        tree_oof[val_idx] = tree_fold.predict_proba(X_tree_train.iloc[val_idx])[:, 1]
        print(f'Built OOF predictions for fold {fold_id}/{n_splits}.')

    valid_mask = (~np.isnan(baseline_oof)) & (~np.isnan(tree_oof))
    if valid_mask.sum() < max(50, int(0.3 * n_samples)):
        raise ValueError('Too few valid OOF rows to train a reliable ensemble meta-model.')

    stack_features = pd.DataFrame({
        'baseline_prob': baseline_oof[valid_mask],
        'tree_prob': tree_oof[valid_mask],
    })
    stack_targets = y_train.iloc[valid_mask]
    return stack_features, stack_targets


def train_ensemble():
    df = load_meta_features()
    df = df.dropna(subset=['WIN'])
    df['WIN'] = df['WIN'].astype(int)
    if df['WIN'].nunique() < 2:
        raise ValueError('WIN label must contain both classes for training.')

    try:
        baseline = joblib.load(BASELINE_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f'Baseline model not found at {BASELINE_PATH}')

    try:
        tree = joblib.load(TREE_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f'Tree model not found at {TREE_PATH}')

    if not hasattr(baseline, 'predict_proba'):
        raise ValueError('Baseline model does not support predict_proba.')
    if not hasattr(tree, 'predict_proba'):
        raise ValueError('Tree model does not support predict_proba.')

    train_df, test_df, split_desc = _time_split_dataframe(df, test_size=0.2)
    print(f'Using {split_desc}.')
    if train_df.empty or test_df.empty:
        raise ValueError('Split produced empty train or test set.')

    numeric = df.select_dtypes(include=['number']).drop(columns=['WIN'], errors='ignore')
    if numeric.empty:
        raise ValueError('No numeric metadata available for ensemble training.')

    if not set(BASELINE_FEATURES).issubset(set(df.columns)):
        missing = [feature for feature in BASELINE_FEATURES if feature not in df.columns]
        raise ValueError(f'Missing baseline features for ensemble: {missing}')

    baseline_features = _resolve_model_features(baseline, BASELINE_FEATURES, df, 'baseline model')
    tree_fallback_features = numeric.columns.tolist()
    tree_features = _resolve_model_features(tree, tree_fallback_features, df, 'tree model')

    X_base_train = train_df[baseline_features].fillna(0)
    X_tree_train = train_df[tree_features].fillna(0)
    y_train = train_df['WIN'].astype(int).reset_index(drop=True)
    X_base_train = X_base_train.reset_index(drop=True)
    X_tree_train = X_tree_train.reset_index(drop=True)
    train_dates = pd.to_datetime(train_df['GAME_DATE'], errors='coerce').reset_index(drop=True)
    if train_dates.isna().any():
        raise ValueError('Train split contains invalid GAME_DATE values.')

    stack_train_X, stack_train_y = _build_oof_stack_features(
        baseline, tree, X_base_train, X_tree_train, y_train, train_dates
    )

    ensemble_model = LogisticRegression(max_iter=1000, random_state=42)
    ensemble_model.fit(stack_train_X[['baseline_prob', 'tree_prob']], stack_train_y)

    baseline_refit = clone(baseline)
    tree_refit = clone(tree)
    baseline_refit.fit(X_base_train, y_train)
    tree_refit.fit(X_tree_train, y_train)

    X_base_test = test_df[baseline_features].fillna(0)
    X_tree_test = test_df[tree_features].fillna(0)
    y_test = test_df['WIN'].astype(int)
    stack_test_X = pd.DataFrame({
        'baseline_prob': baseline_refit.predict_proba(X_base_test)[:, 1],
        'tree_prob': tree_refit.predict_proba(X_tree_test)[:, 1],
    })

    y_pred = ensemble_model.predict(stack_test_X[['baseline_prob', 'tree_prob']])
    y_score = ensemble_model.predict_proba(stack_test_X[['baseline_prob', 'tree_prob']])[:, 1]

    print('Ensemble metrics')
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('ROC-AUC:', roc_auc_score(y_test, y_score))
    print('Log loss:', log_loss(y_test, y_score))

    os.makedirs(os.path.dirname(ENSEMBLE_PATH), exist_ok=True)
    artifact = {
        'meta_model': ensemble_model,
        'baseline_model': baseline_refit,
        'tree_model': tree_refit,
        'baseline_features': baseline_features,
        'tree_features': tree_features,
        'split': split_desc,
    }
    joblib.dump(artifact, ENSEMBLE_PATH)
    print(f'Saved ensemble model to {ENSEMBLE_PATH}')


def main():
    train_ensemble()


if __name__ == '__main__':
    main()
