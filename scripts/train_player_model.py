import os
import sys
import re
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.model_utils import regression_metrics, write_registry_entry
from scripts.team_utils import find_team_profile


PLAYER_LOGS_PATH = "data/raw/player_game_logs_raw.csv"
INJURIES_PATH = "data/raw/injuries_raw.csv"
PROCESSED_PATH = "data/processed/games_with_features.csv"
MODEL_PATH = "models/player_projection_model.pkl"


TARGETS = ['PTS', 'REB', 'AST']
RATE_TARGETS = ['PTS_PER_MIN', 'REB_PER_MIN', 'AST_PER_MIN']
BASE_FEATURES = [
    'MIN_LAST5', 'PTS_LAST5', 'REB_LAST5', 'AST_LAST5',
    'MIN_LAST10', 'PTS_LAST10', 'REB_LAST10', 'AST_LAST10',
    'FG_PCT_LAST10', 'FG3_PCT_LAST10', 'FT_PCT_LAST10',
    'HOME', 'REST_DAYS', 'INJURY_SEVERITY', 'GAME_NUMBER',
    'OPP_DEF_PTS_ALLOWED_30', 'OPP_DEF_REB_ALLOWED_30', 'OPP_DEF_AST_ALLOWED_30',
    'OPP_PACE_30', 'VEGAS_IMPLIED_TEAM_TOTAL_10',
]


def _utc_naive(series):
    return pd.to_datetime(series, errors='coerce', utc=True).dt.tz_convert(None)


def load_player_logs(path=PLAYER_LOGS_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Player logs file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Player logs file is empty.")
    if 'PLAYER_ID' not in df.columns:
        raise ValueError("Player logs must contain PLAYER_ID.")
    df['PLAYER_ID'] = pd.to_numeric(df['PLAYER_ID'], errors='coerce')
    df = df.dropna(subset=['PLAYER_ID']).copy()
    df['PLAYER_ID'] = df['PLAYER_ID'].astype(int)
    df['GAME_DATE'] = _utc_naive(df.get('GAME_DATE'))
    df = df.dropna(subset=['GAME_DATE'])
    return df


def add_player_features(df):
    work = df.copy()
    work = work.sort_values(['PLAYER_ID', 'GAME_DATE'])
    for target in TARGETS + ['MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT']:
        if target in work.columns:
            work[target] = pd.to_numeric(work[target], errors='coerce')

    work['HOME'] = work.get('MATCHUP', '').astype(str).str.contains(r'vs\.').astype(int)
    work['PREV_GAME_DATE'] = work.groupby('PLAYER_ID')['GAME_DATE'].shift(1)
    work['REST_DAYS'] = (work['GAME_DATE'] - work['PREV_GAME_DATE']).dt.days.fillna(5).clip(lower=0, upper=14)
    work['GAME_NUMBER'] = work.groupby('PLAYER_ID').cumcount() + 1

    for metric in ['MIN', 'PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'FT_PCT']:
        if metric not in work.columns:
            work[metric] = np.nan
        work[f'{metric}_LAST5'] = work.groupby('PLAYER_ID')[metric].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )
        work[f'{metric}_LAST10'] = work.groupby('PLAYER_ID')[metric].transform(
            lambda s: s.shift(1).rolling(10, min_periods=1).mean()
        )

    return work


def merge_injury_severity(player_df, injuries_path=INJURIES_PATH):
    if not os.path.exists(injuries_path):
        player_df['INJURY_SEVERITY'] = 0.0
        return player_df

    try:
        injuries = pd.read_csv(injuries_path)
    except Exception:
        player_df['INJURY_SEVERITY'] = 0.0
        return player_df
    if injuries.empty:
        player_df['INJURY_SEVERITY'] = 0.0
        return player_df

    if 'PLAYER_ID' not in injuries.columns:
        player_df['INJURY_SEVERITY'] = 0.0
        return player_df

    injuries['PLAYER_ID'] = pd.to_numeric(injuries['PLAYER_ID'], errors='coerce')
    injuries = injuries.dropna(subset=['PLAYER_ID']).copy()
    injuries['PLAYER_ID'] = injuries['PLAYER_ID'].astype(int)
    injuries['GAME_DATE'] = _utc_naive(injuries.get('GAME_DATE'))
    injuries['INJURY_SEVERITY'] = pd.to_numeric(injuries.get('INJURY_SEVERITY'), errors='coerce').fillna(1.0)
    injuries = injuries.dropna(subset=['GAME_DATE'])

    injury_daily = injuries.groupby(['PLAYER_ID', 'GAME_DATE'], as_index=False)['INJURY_SEVERITY'].max()
    merged = player_df.merge(
        injury_daily,
        how='left',
        on=['PLAYER_ID', 'GAME_DATE']
    )
    merged['INJURY_SEVERITY'] = merged['INJURY_SEVERITY'].fillna(0.0)
    return merged


def _abbr_to_team_id(abbr):
    if not isinstance(abbr, str):
        return None
    team = find_team_profile(abbreviation=abbr.strip().upper())
    if not team:
        return None
    try:
        return int(team.get('team_id'))
    except Exception:
        return None


def _extract_opponent_id_from_matchup(matchup):
    if not isinstance(matchup, str):
        return None
    m = re.search(r"(?:vs\.|@)\s+([A-Z]{2,3})", matchup.upper())
    if not m:
        return None
    return _abbr_to_team_id(m.group(1))


def merge_opponent_team_context(player_df, processed_path=PROCESSED_PATH):
    fallback_defaults = {
        'OPP_DEF_PTS_ALLOWED_30': 112.0,
        'OPP_DEF_REB_ALLOWED_30': 44.0,
        'OPP_DEF_AST_ALLOWED_30': 25.0,
        'OPP_PACE_30': 99.0,
        'VEGAS_IMPLIED_TEAM_TOTAL_10': 110.0,
    }
    if not os.path.exists(processed_path):
        for col, val in fallback_defaults.items():
            player_df[col] = float(val)
        return player_df

    processed = pd.read_csv(processed_path)
    if processed.empty:
        for col, val in fallback_defaults.items():
            player_df[col] = float(val)
        return player_df
    needed = {'GAME_DATE', 'TEAM_ID', 'PTS', 'REB', 'AST'}
    if not needed.issubset(processed.columns):
        for col, val in fallback_defaults.items():
            player_df[col] = float(val)
        return player_df
    if 'HOME_TEAM' not in processed.columns or 'AWAY_TEAM' not in processed.columns:
        for col, val in fallback_defaults.items():
            player_df[col] = float(val)
        return player_df

    work = processed.copy()
    work['GAME_DATE'] = _utc_naive(work.get('GAME_DATE'))
    work['TEAM_ID'] = pd.to_numeric(work['TEAM_ID'], errors='coerce')
    work['HOME_TEAM'] = pd.to_numeric(work['HOME_TEAM'], errors='coerce')
    work['AWAY_TEAM'] = pd.to_numeric(work['AWAY_TEAM'], errors='coerce')
    for stat in ['PTS', 'REB', 'AST', 'FGA', 'FTA', 'OREB', 'TOV', 'TOTAL_POINT', 'TEAM_SPREAD_POINT', 'pts_last10']:
        if stat in work.columns:
            work[stat] = pd.to_numeric(work[stat], errors='coerce')
    work = work.dropna(subset=['GAME_DATE', 'TEAM_ID', 'HOME_TEAM', 'AWAY_TEAM']).copy()
    work['OPPONENT_TEAM_ID'] = np.where(
        work['TEAM_ID'] == work['HOME_TEAM'],
        work['AWAY_TEAM'],
        np.where(work['TEAM_ID'] == work['AWAY_TEAM'], work['HOME_TEAM'], np.nan)
    )
    work = work.dropna(subset=['OPPONENT_TEAM_ID']).copy()
    work['OPPONENT_TEAM_ID'] = work['OPPONENT_TEAM_ID'].astype(int)

    if {'FGA', 'FTA', 'OREB', 'TOV'}.issubset(work.columns):
        possessions = work['FGA'] + 0.44 * work['FTA'] - work['OREB'] + work['TOV']
        work['PACE_PROXY'] = possessions
    else:
        work['PACE_PROXY'] = np.nan

    defensive_daily = work.groupby(['OPPONENT_TEAM_ID', 'GAME_DATE'], as_index=False).agg({
        'PTS': 'mean',
        'REB': 'mean',
        'AST': 'mean',
        'PACE_PROXY': 'mean',
    }).sort_values(['OPPONENT_TEAM_ID', 'GAME_DATE'])
    defensive_daily['OPP_DEF_PTS_ALLOWED_30'] = defensive_daily.groupby('OPPONENT_TEAM_ID')['PTS'].transform(
        lambda s: s.shift(1).rolling(30, min_periods=1).mean()
    )
    defensive_daily['OPP_DEF_REB_ALLOWED_30'] = defensive_daily.groupby('OPPONENT_TEAM_ID')['REB'].transform(
        lambda s: s.shift(1).rolling(30, min_periods=1).mean()
    )
    defensive_daily['OPP_DEF_AST_ALLOWED_30'] = defensive_daily.groupby('OPPONENT_TEAM_ID')['AST'].transform(
        lambda s: s.shift(1).rolling(30, min_periods=1).mean()
    )
    defensive_daily['OPP_PACE_30'] = defensive_daily.groupby('OPPONENT_TEAM_ID')['PACE_PROXY'].transform(
        lambda s: s.shift(1).rolling(30, min_periods=1).mean()
    )
    defensive_daily = defensive_daily[['OPPONENT_TEAM_ID', 'GAME_DATE', 'OPP_DEF_PTS_ALLOWED_30', 'OPP_DEF_REB_ALLOWED_30', 'OPP_DEF_AST_ALLOWED_30', 'OPP_PACE_30']]

    if 'TOTAL_POINT' in work.columns and 'TEAM_SPREAD_POINT' in work.columns:
        work['VEGAS_IMPLIED_TEAM_TOTAL'] = (work['TOTAL_POINT'] / 2.0) - (work['TEAM_SPREAD_POINT'] / 2.0)
    else:
        work['VEGAS_IMPLIED_TEAM_TOTAL'] = np.nan
    if 'pts_last10' in work.columns:
        work['VEGAS_IMPLIED_TEAM_TOTAL'] = work['VEGAS_IMPLIED_TEAM_TOTAL'].fillna(work['pts_last10'])
    work['VEGAS_IMPLIED_TEAM_TOTAL'] = work['VEGAS_IMPLIED_TEAM_TOTAL'].fillna(work['PTS'])

    vegas_daily = work.groupby(['TEAM_ID', 'GAME_DATE'], as_index=False)['VEGAS_IMPLIED_TEAM_TOTAL'].mean()
    vegas_daily = vegas_daily.sort_values(['TEAM_ID', 'GAME_DATE'])
    vegas_daily['VEGAS_IMPLIED_TEAM_TOTAL_10'] = vegas_daily.groupby('TEAM_ID')['VEGAS_IMPLIED_TEAM_TOTAL'].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )
    vegas_daily = vegas_daily[['TEAM_ID', 'GAME_DATE', 'VEGAS_IMPLIED_TEAM_TOTAL_10']]

    out = player_df.copy()
    matchup_series = out['MATCHUP'] if 'MATCHUP' in out.columns else pd.Series(index=out.index, dtype=object)
    out['OPPONENT_TEAM_ID'] = matchup_series.apply(_extract_opponent_id_from_matchup)
    out['OPPONENT_TEAM_ID'] = pd.to_numeric(out['OPPONENT_TEAM_ID'], errors='coerce').fillna(-1).astype(int)
    out = out.sort_values(['OPPONENT_TEAM_ID', 'GAME_DATE'])
    defensive_daily = defensive_daily.sort_values(['OPPONENT_TEAM_ID', 'GAME_DATE'])
    out = pd.merge_asof(
        out,
        defensive_daily,
        on='GAME_DATE',
        by='OPPONENT_TEAM_ID',
        direction='backward',
    )
    out = out.sort_values(['TEAM_ID', 'GAME_DATE'])
    vegas_daily = vegas_daily.sort_values(['TEAM_ID', 'GAME_DATE'])
    out = pd.merge_asof(
        out,
        vegas_daily,
        on='GAME_DATE',
        by='TEAM_ID',
        direction='backward',
    )

    for col, val in fallback_defaults.items():
        out[col] = pd.to_numeric(out.get(col), errors='coerce').fillna(float(val))
    return out


def _time_split(df, test_ratio=0.2):
    work = df.sort_values('GAME_DATE').copy()
    unique_dates = work['GAME_DATE'].drop_duplicates().sort_values().tolist()
    if len(unique_dates) < 20:
        split_idx = int(len(work) * (1 - test_ratio))
        split_idx = max(1, min(split_idx, len(work) - 1))
        return work.iloc[:split_idx], work.iloc[split_idx:], 'row-based fallback split'

    split_date = unique_dates[max(1, min(int(len(unique_dates) * (1 - test_ratio)), len(unique_dates) - 1))]
    train_df = work[work['GAME_DATE'] < split_date]
    test_df = work[work['GAME_DATE'] >= split_date]
    if train_df.empty or test_df.empty:
        split_idx = int(len(work) * (1 - test_ratio))
        split_idx = max(1, min(split_idx, len(work) - 1))
        return work.iloc[:split_idx], work.iloc[split_idx:], 'row-based fallback split'
    return train_df, test_df, f'time-based split at {split_date.date()}'


def train_player_model():
    logs = load_player_logs()
    logs = add_player_features(logs)
    logs = merge_injury_severity(logs)
    logs = merge_opponent_team_context(logs)

    missing_targets = [target for target in TARGETS if target not in logs.columns]
    if missing_targets:
        raise ValueError(f"Player logs missing targets: {missing_targets}")

    required = BASE_FEATURES + TARGETS + ['MIN', 'GAME_DATE']
    data = logs.dropna(subset=required).copy()
    if data.empty:
        raise ValueError("No player rows available after feature/target filtering.")
    data['MIN'] = pd.to_numeric(data['MIN'], errors='coerce')
    data = data[data['MIN'] > 0].copy()
    if data.empty:
        raise ValueError("No player rows with positive MIN available for training.")
    min_denom = data['MIN'].clip(lower=1.0)
    data['PTS_PER_MIN'] = pd.to_numeric(data['PTS'], errors='coerce') / min_denom
    data['REB_PER_MIN'] = pd.to_numeric(data['REB'], errors='coerce') / min_denom
    data['AST_PER_MIN'] = pd.to_numeric(data['AST'], errors='coerce') / min_denom
    data = data.dropna(subset=RATE_TARGETS).copy()
    if data.empty:
        raise ValueError("No rows left after building per-minute targets.")

    train_df, test_df, split_desc = _time_split(data, test_ratio=0.2)
    print(f"Using {split_desc}")

    X_train = train_df[BASE_FEATURES].fillna(0.0)
    y_train_minutes = train_df['MIN'].astype(float)
    y_train_rates = train_df[RATE_TARGETS].astype(float)
    X_test = test_df[BASE_FEATURES].fillna(0.0)
    y_test = test_df[TARGETS].astype(float)
    y_test_minutes = test_df['MIN'].astype(float)

    minutes_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=1,
        min_samples_leaf=2,
    )
    minutes_model.fit(X_train, y_train_minutes)
    pred_minutes = np.clip(minutes_model.predict(X_test), 0.0, 48.0)

    rate_model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=1,
            min_samples_leaf=2,
        )
    )
    rate_model.fit(X_train, y_train_rates)
    pred_rates = np.clip(rate_model.predict(X_test), 0.0, None)
    pred = pred_rates * pred_minutes.reshape(-1, 1)

    print("Player projection model metrics")
    metrics_by_target = {}
    for idx, target in enumerate(TARGETS):
        mae = mean_absolute_error(y_test[target], pred[:, idx])
        rmse = float(np.sqrt(mean_squared_error(y_test[target], pred[:, idx])))
        print(f"{target} MAE: {mae:.3f}")
        print(f"{target} RMSE: {rmse:.3f}")
        metrics_by_target[target] = regression_metrics(y_test[target], pred[:, idx])
    minutes_metrics = regression_metrics(y_test_minutes, pred_minutes)
    print(f"MIN MAE: {minutes_metrics['mae']:.3f}")
    print(f"MIN RMSE: {minutes_metrics['rmse']:.3f}")

    artifact = {
        'projection_version': 'two_stage_minutes_rates',
        'minutes_model': minutes_model,
        'rate_model': rate_model,
        'feature_columns': BASE_FEATURES,
        'target_columns': TARGETS,
        'rate_target_columns': RATE_TARGETS,
        'train_rows': len(train_df),
        'test_rows': len(test_df),
        'split': split_desc,
        'trained_at': pd.Timestamp.utcnow().isoformat(),
    }
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved player projection model to {MODEL_PATH}")

    registry_path = write_registry_entry(
        model_name="player_projection_model",
        model_path=MODEL_PATH,
        task_type="player_regression",
        dataset_path=PLAYER_LOGS_PATH,
        feature_columns=BASE_FEATURES,
        metrics={
            "holdout_minutes": minutes_metrics,
            "holdout_by_target": metrics_by_target,
        },
        split_description=split_desc,
        extra={
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "targets": TARGETS,
            "rate_targets": RATE_TARGETS,
            "projection_version": "two_stage_minutes_rates",
        },
    )
    print(f"Wrote registry entry to {registry_path}")


if __name__ == '__main__':
    train_player_model()
