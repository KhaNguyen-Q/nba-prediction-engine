import os
import json
import csv
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
import re
import html
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
from email.utils import parsedate_to_datetime

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
try:
    import xgboost as xgb
except Exception:
    xgb = None

from scripts.team_utils import find_team_profile

app = FastAPI(title='NBA Prediction API', version='0.1.0')

MODEL_PATH = os.path.join('models', 'logistic_baseline.pkl')
TREE_MODEL_PATH = os.path.join('models', 'xgb_tree_model.pkl')
PLAYER_MODEL_PATH = os.path.join('models', 'player_projection_model.pkl')
PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'games_with_features.csv')
PLAYER_LOGS_PATH = os.path.join('data', 'raw', 'player_game_logs_raw.csv')
PLAYERS_RAW_PATH = os.path.join('data', 'raw', 'players_raw.csv')
INJURIES_RAW_PATH = os.path.join('data', 'raw', 'injuries_raw.csv')
INJURIES_LATEST_PATH = os.path.join('data', 'raw', 'injuries_latest.csv')
UPCOMING_GAMES_PATH = os.path.join('data', 'raw', 'upcoming_games.csv')
INFERENCE_FEATURES_PATH = os.path.join('data', 'processed', 'upcoming_inference_features.csv')
MONITORING_REPORT_PATH = os.path.join('reports', 'monitoring_report.json')
PREDICTION_LOG_PATH = os.path.join('reports', 'prediction_log.csv')
PREDICTION_QUALITY_REPORT_PATH = os.path.join('reports', 'prediction_quality_report.json')
CHAMPION_MODEL_PATH = os.path.join('models', 'champion_team_model.pkl')
SCHEDULE_SOURCE_STATUS_PATH = os.path.join('data', 'raw', 'schedule_source_status.json')
PIPELINE_STATUS_PATH = os.path.join('data', 'raw', 'pipeline_status_latest.json')
AVAILABILITY_STALE_HOURS = float(os.environ.get("AVAILABILITY_STALE_HOURS", "12"))
STRICT_STARTUP_CHECKS = os.environ.get("STRICT_STARTUP_CHECKS", "1").strip().lower() in {"1", "true", "yes"}
TEAM_CONFIDENCE_Z = float(os.environ.get("TEAM_CONFIDENCE_Z", "1.28"))
PLAYER_CONFIDENCE_Z = float(os.environ.get("PLAYER_CONFIDENCE_Z", "1.28"))
NEWS_MAX_AGE_DAYS = int(os.environ.get("NEWS_MAX_AGE_DAYS", "3"))

baseline_model = None
tree_model = None
champion_model = None
player_projection_artifact = None

if os.path.exists(MODEL_PATH):
    try:
        baseline_model = joblib.load(MODEL_PATH)
    except Exception:
        baseline_model = None

if os.path.exists(TREE_MODEL_PATH):
    try:
        tree_model = joblib.load(TREE_MODEL_PATH)
    except Exception:
        tree_model = None

if os.path.exists(CHAMPION_MODEL_PATH):
    try:
        champion_model = joblib.load(CHAMPION_MODEL_PATH)
    except Exception:
        champion_model = None

if os.path.exists(PLAYER_MODEL_PATH):
    try:
        player_projection_artifact = joblib.load(PLAYER_MODEL_PATH)
    except Exception:
        player_projection_artifact = None


class PredictionRequest(BaseModel):
    features: Dict[str, float]
    model: Optional[str] = 'baseline'


class TeamPredictionRequest(BaseModel):
    home_team_id: int
    away_team_id: int
    game_id: Optional[str] = None
    model: Optional[str] = 'baseline'
    include_player_projection: bool = True
    include_headlines: bool = True


def _resolve_model(model_name: str):
    model_choice = (model_name or 'baseline').lower()
    if model_choice == 'tree':
        model = tree_model
    elif model_choice == 'champion':
        model = champion_model
    else:
        model = baseline_model
        model_choice = 'baseline'
    if model is None:
        raise HTTPException(status_code=503, detail='Requested model is not available')
    return model_choice, model


def _predict_from_features(model_name: str, features: Dict[str, float]):
    model_choice, model = _resolve_model(model_name)
    feature_names = model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else None
    if feature_names is not None:
        missing = [name for name in feature_names if name not in features]
        if missing:
            raise HTTPException(status_code=400, detail=f'Missing features: {missing}')
        X = pd.DataFrame([{name: _safe_float(features[name], default=0.0) for name in feature_names}], columns=feature_names)
        probability = float(model.predict_proba(X)[0][1])
    else:
        feature_vector = [float(value) for value in features.values()]
        probability = float(model.predict_proba([feature_vector])[0][1])
    prediction = int(probability >= 0.5)
    return {
        'model': model_choice,
        'probability': probability,
        'prediction': prediction,
    }


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return float(default)
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_player_name(name: str):
    if not isinstance(name, str):
        return ""
    text = name.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if "," in name:
        # convert "last, first" -> "first last" for roster formats
        parts = [p.strip().lower() for p in name.split(",") if p.strip()]
        if len(parts) >= 2:
            text = re.sub(r"[^\w\s]", " ", f"{parts[1]} {parts[0]}")
            text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_date_only(value):
    if value is None:
        return None
    ts = pd.to_datetime(value, errors='coerce')
    if pd.isna(ts):
        return None
    try:
        return ts.date()
    except Exception:
        return None


def _is_player_out_for_game(injury_status: str, injury_return_date, game_date, severity: float = 0.0, is_unavailable: bool = False):
    status = str(injury_status or "").strip().lower()
    game_dt = _parse_date_only(game_date)
    return_dt = _parse_date_only(injury_return_date)
    status_out = any(
        token in status for token in
        ["out", "inactive", "suspended", "doubtful", "out for season", "ruled out", "will not play", "not expected to play"]
    )
    return_after_game = (return_dt is not None and game_dt is not None and return_dt > game_dt)
    severe_out = float(severity or 0.0) >= 2.0
    return bool(is_unavailable) or status_out or return_after_game or severe_out


def _load_injuries_projection_frame():
    source = INJURIES_LATEST_PATH if os.path.exists(INJURIES_LATEST_PATH) else INJURIES_RAW_PATH
    if not os.path.exists(source):
        return pd.DataFrame()
    try:
        injuries = pd.read_csv(source)
    except Exception:
        return pd.DataFrame()
    if injuries.empty:
        return injuries
    if 'TEAM_ID' in injuries.columns:
        injuries['TEAM_ID'] = pd.to_numeric(injuries['TEAM_ID'], errors='coerce')
    if 'PLAYER_ID' in injuries.columns:
        injuries['PLAYER_ID'] = pd.to_numeric(injuries['PLAYER_ID'], errors='coerce')
    if 'GAME_DATE' in injuries.columns:
        injuries['GAME_DATE'] = _utc_naive(injuries.get('GAME_DATE'))
    if 'INJURY_RETURN_DATE' in injuries.columns:
        injuries['INJURY_RETURN_DATE'] = _utc_naive(injuries.get('INJURY_RETURN_DATE'))
    if 'INJURY_SEVERITY' in injuries.columns:
        injuries['INJURY_SEVERITY'] = pd.to_numeric(injuries.get('INJURY_SEVERITY'), errors='coerce').fillna(0.0)
    if 'IS_UNAVAILABLE' in injuries.columns:
        injuries['IS_UNAVAILABLE'] = injuries['IS_UNAVAILABLE'].astype(str).str.lower().isin(['1', 'true', 'yes'])
    else:
        injuries['IS_UNAVAILABLE'] = False
    return injuries


def _utc_naive(series):
    return pd.to_datetime(series, errors='coerce', utc=True).dt.tz_convert(None)


def _american_price_from_probability(probability: float):
    p = max(min(float(probability), 0.999), 0.001)
    if p >= 0.5:
        return int(round(-(100 * p / (1 - p))))
    return int(round((100 * (1 - p) / p)))


def _team_name_from_id(team_id: int, fallback: str):
    profile = find_team_profile(team_id=int(team_id))
    if profile and profile.get('name'):
        return str(profile['name'])
    return fallback


def _load_processed_df():
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise HTTPException(status_code=404, detail='Processed dataset not found')
    df = pd.read_csv(PROCESSED_DATA_PATH)
    if df.empty:
        raise HTTPException(status_code=404, detail='Processed dataset is empty')
    if 'TEAM_ID' not in df.columns:
        raise HTTPException(status_code=500, detail='Processed dataset missing TEAM_ID')
    if 'GAME_DATE' in df.columns:
        df['GAME_DATE_SORT'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
    else:
        df['GAME_DATE_SORT'] = pd.NaT
    return df


def _load_upcoming_games_df():
    if not os.path.exists(UPCOMING_GAMES_PATH):
        raise HTTPException(
            status_code=503,
            detail='upcoming_games.csv not found. Run scripts/fetch_schedule.py or scripts/update_pipeline.py first.'
        )
    upcoming = pd.read_csv(UPCOMING_GAMES_PATH)
    if upcoming.empty:
        raise HTTPException(status_code=503, detail='upcoming_games.csv is empty.')
    required = {'GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID'}
    missing = [c for c in required if c not in upcoming.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f'upcoming_games.csv missing columns: {missing}')
    upcoming['GAME_DATE'] = pd.to_datetime(upcoming['GAME_DATE'], errors='coerce')
    upcoming['HOME_TEAM_ID'] = pd.to_numeric(upcoming['HOME_TEAM_ID'], errors='coerce')
    upcoming['AWAY_TEAM_ID'] = pd.to_numeric(upcoming['AWAY_TEAM_ID'], errors='coerce')
    upcoming = upcoming.dropna(subset=['GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']).copy()
    if upcoming.empty:
        raise HTTPException(status_code=503, detail='No valid upcoming rows in upcoming_games.csv')
    upcoming['HOME_TEAM_ID'] = upcoming['HOME_TEAM_ID'].astype(int)
    upcoming['AWAY_TEAM_ID'] = upcoming['AWAY_TEAM_ID'].astype(int)
    upcoming['GAME_DATE_ONLY'] = upcoming['GAME_DATE'].dt.date
    today_utc = datetime.now(timezone.utc).date()
    upcoming = upcoming[upcoming['GAME_DATE_ONLY'] >= today_utc]
    if upcoming.empty:
        raise HTTPException(status_code=503, detail='No future upcoming games available.')
    return upcoming


def _resolve_upcoming_match(home_team_id: int, away_team_id: int, game_id: Optional[str] = None):
    upcoming = _load_upcoming_games_df()
    matches = upcoming[
        (upcoming['HOME_TEAM_ID'] == int(home_team_id)) &
        (upcoming['AWAY_TEAM_ID'] == int(away_team_id))
    ].copy()
    if game_id is not None:
        matches = matches[matches['GAME_ID'].astype(str) == str(game_id)]
    if matches.empty:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Requested matchup {home_team_id} vs {away_team_id} is not in upcoming_games.csv. "
                "Predictions are restricted to future scheduled games."
            )
        )
    matches = matches.sort_values('GAME_DATE')
    return matches.iloc[0]


def _load_inference_features_df():
    if not os.path.exists(INFERENCE_FEATURES_PATH):
        return pd.DataFrame()
    try:
        inf = pd.read_csv(INFERENCE_FEATURES_PATH)
    except Exception:
        return pd.DataFrame()
    if inf.empty:
        return pd.DataFrame()
    inf['GAME_ID'] = inf['GAME_ID'].astype(str)
    inf['TEAM_ID'] = pd.to_numeric(inf['TEAM_ID'], errors='coerce')
    inf = inf.dropna(subset=['TEAM_ID'])
    inf['TEAM_ID'] = inf['TEAM_ID'].astype(int)
    return inf


def _upcoming_games_payload(limit=50):
    upcoming = _load_upcoming_games_df().sort_values('GAME_DATE').head(limit)
    items = []
    for _, row in upcoming.iterrows():
        items.append({
            'game_id': str(row['GAME_ID']),
            'game_date': str(pd.to_datetime(row['GAME_DATE'], errors='coerce').date()),
            'home_team_id': int(row['HOME_TEAM_ID']),
            'away_team_id': int(row['AWAY_TEAM_ID']),
            'home_team_abbr': row.get('HOME_TEAM_ABBR'),
            'away_team_abbr': row.get('AWAY_TEAM_ABBR'),
        })
    return items


def _latest_team_row(df: pd.DataFrame, team_id: int):
    team_df = df[df['TEAM_ID'] == team_id].copy()
    if team_df.empty:
        raise HTTPException(status_code=404, detail=f'No processed rows found for team_id={team_id}')
    team_df = team_df.sort_values('GAME_DATE_SORT')
    return team_df.iloc[-1]


def _inference_team_row(inference_df: pd.DataFrame, game_id: str, team_id: int):
    if inference_df.empty:
        return None
    rows = inference_df[
        (inference_df['GAME_ID'] == str(game_id)) &
        (inference_df['TEAM_ID'] == int(team_id))
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def _current_team_injury_impact(team_id: int):
    injuries = _load_injuries_projection_frame()
    if injuries.empty or 'TEAM_ID' not in injuries.columns:
        return 0.0
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    team_inj = injuries[injuries['TEAM_ID'] == team_id].dropna(subset=['GAME_DATE']).copy()
    if team_inj.empty:
        return 0.0
    active_mask = (
        (team_inj['GAME_DATE'] <= now) &
        (team_inj['INJURY_RETURN_DATE'].isna() | (team_inj['INJURY_RETURN_DATE'] >= now))
    )
    return float(team_inj.loc[active_mask, 'INJURY_SEVERITY'].sum())


def _model_features_from_team_row(
    model,
    team_row: pd.Series,
    team_id: int,
    home_team_id: int,
    away_team_id: int,
    is_home: bool
):
    feature_names = model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else []
    if not feature_names:
        raise HTTPException(status_code=500, detail='Model has no feature names; cannot build team-id prediction payload')

    injury_impact = _current_team_injury_impact(team_id)
    base_features = {}
    for feature in feature_names:
        base_features[feature] = _safe_float(team_row.get(feature, 0.0), default=0.0)

    overrides = {
        'TEAM_ID': float(team_id),
        'HOME': 1.0 if is_home else 0.0,
        'IS_AWAY': 0.0 if is_home else 1.0,
        'BACK_TO_BACK': 1.0 if base_features.get('REST_DAYS', 2.0) <= 1.0 else 0.0,
        'ADULT_ENTERTAINMENT_INDEX': 0.0 if is_home else base_features.get('ADULT_ENTERTAINMENT_INDEX', 5.0),
        'TRAVEL_DISTANCE': 0.0 if is_home else base_features.get('TRAVEL_DISTANCE', 0.0),
        'TIMEZONE_SHIFT': 0.0 if is_home else base_features.get('TIMEZONE_SHIFT', 0.0),
        'INJURY_IMPACT': injury_impact,
        'HOME_TEAM': float(home_team_id),
        'AWAY_TEAM': float(away_team_id),
    }
    for key, value in overrides.items():
        if key in base_features:
            base_features[key] = float(value)

    return base_features


def _player_projection_for_team(team_id: int, team_row: pd.Series, game_date=None):
    players_path = PLAYERS_RAW_PATH
    injuries_path = INJURIES_RAW_PATH
    if not os.path.exists(players_path):
        return {'players': [], 'projection_method': 'unavailable: players_raw.csv not found'}

    players = pd.read_csv(players_path)
    if 'TEAM_ID' in players.columns:
        players['TEAM_ID'] = pd.to_numeric(players['TEAM_ID'], errors='coerce')
        players = players[players['TEAM_ID'] == int(team_id)].copy()
    else:
        players = pd.DataFrame()
    if players.empty:
        return {'players': [], 'projection_method': 'unavailable: no roster rows for team'}

    if 'PLAYER_NAME' not in players.columns and 'PLAYER_NAME.1' in players.columns:
        players['PLAYER_NAME'] = players['PLAYER_NAME.1']
    if 'PLAYER_NAME' not in players.columns:
        players['PLAYER_NAME'] = 'Unknown'
    players['PLAYER_NAME_NORMALIZED'] = players['PLAYER_NAME'].astype(str).apply(_normalize_player_name)

    injuries = _load_injuries_projection_frame()
    injury_lookup_by_id = {}
    injury_lookup_by_name = {}
    if not injuries.empty and 'TEAM_ID' in injuries.columns and 'PLAYER_NAME' in injuries.columns:
        injuries = injuries[injuries['TEAM_ID'] == team_id].copy()
        injuries = injuries.sort_values('GAME_DATE')
        for _, row in injuries.iterrows():
            name_key = _normalize_player_name(str(row.get('PLAYER_NAME', '')))
            player_id = row.get('PLAYER_ID')
            entry = {
                'status': str(row.get('INJURY_STATUS', 'Available')),
                'severity': float(row.get('INJURY_SEVERITY', 0.0)),
                'return_date': str(row.get('INJURY_RETURN_DATE')) if pd.notna(row.get('INJURY_RETURN_DATE')) else None,
                'is_unavailable': bool(row.get('IS_UNAVAILABLE', False)),
            }
            if pd.notna(player_id):
                try:
                    injury_lookup_by_id[int(player_id)] = entry
                except Exception:
                    pass
            if name_key:
                injury_lookup_by_name[name_key] = entry

    team_pts = _safe_float(team_row.get('pts_last10', 110.0), default=110.0)
    team_reb = _safe_float(team_row.get('reb_last10', 44.0), default=44.0)
    team_ast = _safe_float(team_row.get('ast_last10', 25.0), default=25.0)

    projections = []
    multipliers = []
    for _, player in players.iterrows():
        name = str(player.get('PLAYER_NAME', '')).strip() or 'Unknown'
        player_id = int(player['PLAYER_ID']) if 'PLAYER_ID' in player and pd.notna(player['PLAYER_ID']) else None
        normalized_name = _normalize_player_name(name)
        info = (
            injury_lookup_by_id.get(player_id)
            if player_id is not None else None
        ) or injury_lookup_by_name.get(normalized_name, {'status': 'Available', 'severity': 0.0, 'return_date': None, 'is_unavailable': False})
        severity = float(info['severity'])
        is_out_for_game = _is_player_out_for_game(
            injury_status=info.get('status'),
            injury_return_date=info.get('return_date'),
            game_date=game_date,
            severity=severity,
            is_unavailable=bool(info.get('is_unavailable', False)),
        )
        if is_out_for_game:
            availability_mult = 0.0
            availability = 'Out'
        elif severity >= 2.0:
            availability_mult = 0.1
            availability = 'Out'
        elif severity >= 1.2:
            availability_mult = 0.6
            availability = 'Questionable'
        elif severity >= 0.5:
            availability_mult = 0.85
            availability = 'Probable'
        else:
            availability_mult = 1.0
            availability = 'Available'
        multipliers.append(availability_mult)
        projections.append({
            'player_id': player_id,
            'player_name': name,
            'team_id': int(team_id),
            'availability': availability,
            'injury_status': info['status'],
            'injury_return_date': info['return_date'],
            'availability_multiplier': availability_mult,
            'projection_source': 'heuristic_fallback',
        })

    denom = max(sum(multipliers), 1.0)
    base_min = 240.0 / denom
    base_pts = team_pts / denom
    base_reb = team_reb / denom
    base_ast = team_ast / denom

    for player in projections:
        mult = player['availability_multiplier']
        player['projected_minutes'] = round(base_min * mult, 2)
        player['projected_points'] = round(base_pts * mult, 2)
        player['projected_rebounds'] = round(base_reb * mult, 2)
        player['projected_assists'] = round(base_ast * mult, 2)

    projections = sorted(projections, key=lambda x: x['projected_points'], reverse=True)
    return {
        'players': projections,
        'projection_method': 'Heuristic availability-adjusted split of recent team averages; upgrade with player game logs for production player-level accuracy.',
    }


def _latest_team_vegas_context(team_id: int):
    if not os.path.exists(PROCESSED_DATA_PATH):
        return {"VEGAS_IMPLIED_TEAM_TOTAL_10": 110.0}
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except Exception:
        return {"VEGAS_IMPLIED_TEAM_TOTAL_10": 110.0}
    if df.empty or "TEAM_ID" not in df.columns:
        return {"VEGAS_IMPLIED_TEAM_TOTAL_10": 110.0}
    df["TEAM_ID"] = pd.to_numeric(df["TEAM_ID"], errors="coerce")
    team = df[df["TEAM_ID"] == int(team_id)].copy()
    if team.empty:
        return {"VEGAS_IMPLIED_TEAM_TOTAL_10": 110.0}
    if "GAME_DATE" in team.columns:
        team["GAME_DATE"] = pd.to_datetime(team["GAME_DATE"], errors="coerce")
        team = team.sort_values("GAME_DATE")

    implied = None
    if "TOTAL_POINT" in team.columns and "TEAM_SPREAD_POINT" in team.columns:
        total = pd.to_numeric(team["TOTAL_POINT"], errors="coerce")
        spread = pd.to_numeric(team["TEAM_SPREAD_POINT"], errors="coerce")
        implied = (total / 2.0) - (spread / 2.0)
    if implied is None or implied.dropna().empty:
        implied = pd.to_numeric(team.get("pts_last10"), errors="coerce")
    if implied is None or implied.dropna().empty:
        return {"VEGAS_IMPLIED_TEAM_TOTAL_10": 110.0}
    return {"VEGAS_IMPLIED_TEAM_TOTAL_10": float(implied.dropna().tail(10).mean())}


def _latest_opponent_context(opponent_team_id: Optional[int]):
    fallback = {
        "OPP_DEF_PTS_ALLOWED_30": 112.0,
        "OPP_DEF_REB_ALLOWED_30": 44.0,
        "OPP_DEF_AST_ALLOWED_30": 25.0,
        "OPP_PACE_30": 99.0,
    }
    if opponent_team_id is None or not os.path.exists(PROCESSED_DATA_PATH):
        return fallback
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except Exception:
        return fallback
    required = {"TEAM_ID", "PTS", "REB", "AST"}
    if df.empty or not required.issubset(df.columns):
        return fallback
    home_col = "HOME_TEAM" if "HOME_TEAM" in df.columns else None
    away_col = "AWAY_TEAM" if "AWAY_TEAM" in df.columns else None
    if home_col is None or away_col is None:
        return fallback

    work = df.copy()
    work["TEAM_ID"] = pd.to_numeric(work["TEAM_ID"], errors="coerce")
    work["_HOME_TEAM"] = pd.to_numeric(work[home_col], errors="coerce")
    work["_AWAY_TEAM"] = pd.to_numeric(work[away_col], errors="coerce")
    work["PTS"] = pd.to_numeric(work["PTS"], errors="coerce")
    work["REB"] = pd.to_numeric(work["REB"], errors="coerce")
    work["AST"] = pd.to_numeric(work["AST"], errors="coerce")
    work = work.dropna(subset=["TEAM_ID", "_HOME_TEAM", "_AWAY_TEAM", "PTS", "REB", "AST"]).copy()
    if work.empty:
        return fallback
    work["OPPONENT_ID"] = np.where(
        work["TEAM_ID"] == work["_HOME_TEAM"],
        work["_AWAY_TEAM"],
        np.where(work["TEAM_ID"] == work["_AWAY_TEAM"], work["_HOME_TEAM"], np.nan),
    )
    work = work.dropna(subset=["OPPONENT_ID"]).copy()
    if work.empty:
        return fallback
    vs = work[work["OPPONENT_ID"] == int(opponent_team_id)].copy()
    if vs.empty:
        return fallback

    pace_series = None
    if {"FGA", "FTA", "OREB", "TOV"}.issubset(vs.columns):
        fga = pd.to_numeric(vs["FGA"], errors="coerce")
        fta = pd.to_numeric(vs["FTA"], errors="coerce")
        oreb = pd.to_numeric(vs["OREB"], errors="coerce")
        tov = pd.to_numeric(vs["TOV"], errors="coerce")
        pace_series = fga + 0.44 * fta - oreb + tov
    if pace_series is None or pace_series.dropna().empty:
        pace_val = fallback["OPP_PACE_30"]
    else:
        pace_val = float(pace_series.dropna().tail(30).mean())

    return {
        "OPP_DEF_PTS_ALLOWED_30": float(vs["PTS"].tail(30).mean()),
        "OPP_DEF_REB_ALLOWED_30": float(vs["REB"].tail(30).mean()),
        "OPP_DEF_AST_ALLOWED_30": float(vs["AST"].tail(30).mean()),
        "OPP_PACE_30": pace_val,
    }


def _team_absence_context(team_id: int, game_date=None):
    injuries = _load_injuries_projection_frame()
    if injuries.empty or "TEAM_ID" not in injuries.columns:
        return {"TEAM_ABSENT_SEVERITY": 0.0, "TEAM_ABSENT_COUNT": 0.0}
    work = injuries.copy()
    work["TEAM_ID"] = pd.to_numeric(work["TEAM_ID"], errors="coerce")
    work = work.dropna(subset=["TEAM_ID"]).copy()
    work["TEAM_ID"] = work["TEAM_ID"].astype(int)
    work = work[work["TEAM_ID"] == int(team_id)]
    if work.empty:
        return {"TEAM_ABSENT_SEVERITY": 0.0, "TEAM_ABSENT_COUNT": 0.0}
    if "GAME_DATE" in work.columns:
        work["GAME_DATE"] = pd.to_datetime(work.get("GAME_DATE"), errors="coerce")
    date_limit = pd.to_datetime(game_date, errors="coerce")
    if pd.notna(date_limit) and "GAME_DATE" in work.columns:
        work = work[work["GAME_DATE"] <= date_limit]
    if work.empty:
        return {"TEAM_ABSENT_SEVERITY": 0.0, "TEAM_ABSENT_COUNT": 0.0}
    sev = pd.to_numeric(work.get("INJURY_SEVERITY"), errors="coerce").fillna(0.0)
    count = float((sev >= 1.0).sum())
    return {"TEAM_ABSENT_SEVERITY": float(sev.sum()), "TEAM_ABSENT_COUNT": count}


def _player_feature_frame_for_inference(
    player_logs: pd.DataFrame,
    player_id: int,
    team_id: int,
    opponent_team_id: Optional[int] = None,
):
    logs = player_logs[player_logs['PLAYER_ID'] == player_id].copy()
    if logs.empty:
        return None
    logs = logs.sort_values('GAME_DATE')
    latest = logs.iloc[-1]

    def _recent_mean(column, n):
        if column not in logs.columns:
            return 0.0
        values = pd.to_numeric(logs[column], errors='coerce').dropna()
        if values.empty:
            return 0.0
        return float(values.tail(n).mean())

    prev_date = logs['GAME_DATE'].iloc[-2] if len(logs) > 1 else logs['GAME_DATE'].iloc[-1]
    rest_days = (logs['GAME_DATE'].iloc[-1] - prev_date).days if pd.notna(prev_date) else 5
    rest_days = float(max(0, min(rest_days, 14)))
    home = 1.0 if 'MATCHUP' in latest and isinstance(latest['MATCHUP'], str) and 'vs.' in latest['MATCHUP'] else 0.0

    role_guard = _recent_mean('AST', 10) / max(_recent_mean('AST', 10) + _recent_mean('REB', 10), 1e-6)
    role_big = _recent_mean('REB', 10) / max(_recent_mean('AST', 10) + _recent_mean('REB', 10), 1e-6)
    min_share = _recent_mean('MIN', 10) / max(240.0, 1.0)
    features = {
        'MIN_LAST5': _recent_mean('MIN', 5),
        'PTS_LAST5': _recent_mean('PTS', 5),
        'REB_LAST5': _recent_mean('REB', 5),
        'AST_LAST5': _recent_mean('AST', 5),
        'MIN_LAST10': _recent_mean('MIN', 10),
        'PTS_LAST10': _recent_mean('PTS', 10),
        'REB_LAST10': _recent_mean('REB', 10),
        'AST_LAST10': _recent_mean('AST', 10),
        'FG_PCT_LAST10': _recent_mean('FG_PCT', 10),
        'FG3_PCT_LAST10': _recent_mean('FG3_PCT', 10),
        'FT_PCT_LAST10': _recent_mean('FT_PCT', 10),
        'HOME': home,
        'REST_DAYS': rest_days,
        'INJURY_SEVERITY': 0.0,  # set from injuries below
        'GAME_NUMBER': float(len(logs) + 1),
        'MIN_ROLE_SHARE': float(max(0.01, min(0.9, min_share))),
        'PLAYER_ROLE_GUARD_SCORE': float(max(0.0, min(1.0, role_guard))),
        'PLAYER_ROLE_BIG_SCORE': float(max(0.0, min(1.0, role_big))),
    }
    features.update(_latest_opponent_context(opponent_team_id))
    features.update(_latest_team_vegas_context(team_id))
    features.update(_team_absence_context(team_id, game_date=latest.get('GAME_DATE')))
    features['OPP_VS_GUARD_AST_ALLOWED_30'] = float(features.get('OPP_DEF_AST_ALLOWED_30', 25.0))
    features['OPP_VS_BIG_REB_ALLOWED_30'] = float(features.get('OPP_DEF_REB_ALLOWED_30', 44.0))
    return features


_OPPONENT_FACTOR_CACHE: Dict[int, Dict[str, float]] = {}


def _opponent_adjustment_factors(opponent_team_id: Optional[int]):
    if opponent_team_id is None:
        return {'points_factor': 1.0, 'rebounds_factor': 1.0, 'assists_factor': 1.0, 'source': 'none'}
    try:
        opp_id = int(opponent_team_id)
    except Exception:
        return {'points_factor': 1.0, 'rebounds_factor': 1.0, 'assists_factor': 1.0, 'source': 'none'}
    if opp_id in _OPPONENT_FACTOR_CACHE:
        return _OPPONENT_FACTOR_CACHE[opp_id]
    if not os.path.exists(PROCESSED_DATA_PATH):
        return {'points_factor': 1.0, 'rebounds_factor': 1.0, 'assists_factor': 1.0, 'source': 'missing_processed'}
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except Exception:
        return {'points_factor': 1.0, 'rebounds_factor': 1.0, 'assists_factor': 1.0, 'source': 'read_error'}
    home_col = 'HOME_TEAM' if 'HOME_TEAM' in df.columns else ('HOME_TEAM_ID' if 'HOME_TEAM_ID' in df.columns else None)
    away_col = 'AWAY_TEAM' if 'AWAY_TEAM' in df.columns else ('AWAY_TEAM_ID' if 'AWAY_TEAM_ID' in df.columns else None)
    needed = {'TEAM_ID', 'PTS', 'REB', 'AST'}
    if df.empty or home_col is None or away_col is None or not needed.issubset(df.columns):
        return {'points_factor': 1.0, 'rebounds_factor': 1.0, 'assists_factor': 1.0, 'source': 'missing_columns'}
    work = df.copy()
    work['TEAM_ID'] = pd.to_numeric(work['TEAM_ID'], errors='coerce')
    work['_HOME_TEAM'] = pd.to_numeric(work[home_col], errors='coerce')
    work['_AWAY_TEAM'] = pd.to_numeric(work[away_col], errors='coerce')
    for stat in ['PTS', 'REB', 'AST']:
        work[stat] = pd.to_numeric(work[stat], errors='coerce')
    work = work.dropna(subset=['TEAM_ID', 'PTS', 'REB', 'AST'])
    if work.empty:
        return {'points_factor': 1.0, 'rebounds_factor': 1.0, 'assists_factor': 1.0, 'source': 'empty'}

    if work['_HOME_TEAM'].notna().any() and work['_AWAY_TEAM'].notna().any():
        work = work.dropna(subset=['_HOME_TEAM', '_AWAY_TEAM']).copy()
        work['OPPONENT_ID'] = np.where(
            work['TEAM_ID'] == work['_HOME_TEAM'],
            work['_AWAY_TEAM'],
            np.where(work['TEAM_ID'] == work['_AWAY_TEAM'], work['_HOME_TEAM'], np.nan)
        )
        source = 'home_away_columns'
    elif 'GAME_ID' in work.columns:
        work['GAME_ID'] = work['GAME_ID'].astype(str)
        work = work.dropna(subset=['GAME_ID']).copy()
        game_to_teams = work.groupby('GAME_ID')['TEAM_ID'].agg(lambda s: sorted(set(s.astype(int).tolist())))
        opponent_map = {}
        for gid, teams in game_to_teams.items():
            if len(teams) != 2:
                continue
            opponent_map[(gid, teams[0])] = teams[1]
            opponent_map[(gid, teams[1])] = teams[0]
        work['OPPONENT_ID'] = work.apply(
            lambda r: opponent_map.get((str(r['GAME_ID']), int(r['TEAM_ID']))),
            axis=1
        )
        source = 'game_pairing'
    else:
        return {'points_factor': 1.0, 'rebounds_factor': 1.0, 'assists_factor': 1.0, 'source': 'missing_opponent_map'}

    work = work.dropna(subset=['OPPONENT_ID'])
    if work.empty:
        return {'points_factor': 1.0, 'rebounds_factor': 1.0, 'assists_factor': 1.0, 'source': 'empty_opponent_map'}
    vs = work[work['OPPONENT_ID'] == opp_id]
    if vs.empty:
        factors = {'points_factor': 1.0, 'rebounds_factor': 1.0, 'assists_factor': 1.0, 'source': 'no_history'}
        _OPPONENT_FACTOR_CACHE[opp_id] = factors
        return factors

    def _factor(stat):
        league_avg = float(work[stat].mean()) if not work.empty else 0.0
        allowed = float(vs[stat].mean()) if not vs.empty else league_avg
        if league_avg <= 0:
            return 1.0
        return float(max(0.9, min(1.1, allowed / league_avg)))

    factors = {
        'points_factor': _factor('PTS'),
        'rebounds_factor': _factor('REB'),
        'assists_factor': _factor('AST'),
        'source': f'opponent_{opp_id}_history_{source}',
    }
    _OPPONENT_FACTOR_CACHE[opp_id] = factors
    return factors


def _redistribute_absence_impact(players):
    if not players:
        return players
    available = [p for p in players if str(p.get('availability', '')).lower() != 'out']
    out_players = [p for p in players if str(p.get('availability', '')).lower() == 'out']
    if not available or not out_players:
        return players

    missing_minutes = sum(float(p.get('_raw_projected_minutes', 0.0)) for p in out_players)
    missing_points = sum(float(p.get('_raw_projected_points', 0.0)) for p in out_players)
    missing_rebounds = sum(float(p.get('_raw_projected_rebounds', 0.0)) for p in out_players)
    missing_assists = sum(float(p.get('_raw_projected_assists', 0.0)) for p in out_players)

    if missing_minutes <= 0 and missing_points <= 0 and missing_rebounds <= 0 and missing_assists <= 0:
        return players

    def _role_scores(player):
        ast = float(player.get('_ast_last10', 0.0))
        reb = float(player.get('_reb_last10', 0.0))
        denom = max(ast + reb, 1e-6)
        guard = max(0.0, min(1.0, ast / denom))
        big = max(0.0, min(1.0, reb / denom))
        return guard, big

    out_guard = 0.0
    out_big = 0.0
    for p in out_players:
        g, b = _role_scores(p)
        role_weight = max(float(p.get('_raw_projected_minutes', 0.0)), 1.0)
        out_guard += g * role_weight
        out_big += b * role_weight
    out_role_total = max(out_guard + out_big, 1e-6)
    out_guard_share = out_guard / out_role_total
    out_big_share = out_big / out_role_total

    def _weights(keys, role_alignment=None):
        weights = []
        for p in available:
            val = 1.0
            for k, floor in keys:
                val *= max(float(p.get(k, floor)), floor)
            if role_alignment:
                g, b = _role_scores(p)
                align = (
                    out_guard_share * (g if role_alignment == 'guard' else 0.0) +
                    out_big_share * (b if role_alignment == 'big' else 0.0)
                )
                # Keep a floor to avoid zeroing non-matching players entirely.
                val *= max(0.35, align * 2.0)
            weights.append(val)
        total = sum(weights) or 1.0
        return [w / total for w in weights]

    w_min = _weights([('_min_last10', 8.0)])
    w_pts = _weights([('_min_last10', 8.0), ('_pts_last10', 2.0)], role_alignment='guard')
    # Rebounds should flow more to "big-like" profiles when a big is out.
    w_reb = _weights([('_min_last10', 8.0), ('_reb_last10', 1.0)], role_alignment='big')
    w_ast = _weights([('_min_last10', 8.0), ('_ast_last10', 1.0)], role_alignment='guard')

    for idx, p in enumerate(available):
        p['projected_minutes'] = round(min(44.0, float(p.get('projected_minutes', 0.0)) + missing_minutes * 0.9 * w_min[idx]), 2)
        p['projected_points'] = round(max(0.0, float(p.get('projected_points', 0.0)) + missing_points * 0.9 * w_pts[idx]), 2)
        p['projected_rebounds'] = round(max(0.0, float(p.get('projected_rebounds', 0.0)) + missing_rebounds * 0.9 * w_reb[idx]), 2)
        p['projected_assists'] = round(max(0.0, float(p.get('projected_assists', 0.0)) + missing_assists * 0.9 * w_ast[idx]), 2)
        for key in [
            'projected_minutes_ci_low', 'projected_minutes_ci_high',
            'projected_points_ci_low', 'projected_points_ci_high',
            'projected_rebounds_ci_low', 'projected_rebounds_ci_high',
            'projected_assists_ci_low', 'projected_assists_ci_high',
        ]:
            if key in p:
                p[key] = round(max(0.0, float(p[key])), 2)
    return players


def _apply_opponent_context(players, opponent_factors):
    if not players:
        return players
    pf = float(opponent_factors.get('points_factor', 1.0))
    rf = float(opponent_factors.get('rebounds_factor', 1.0))
    af = float(opponent_factors.get('assists_factor', 1.0))
    pace_like = max(0.95, min(1.05, (pf + 1.0) / 2.0))

    for p in players:
        if str(p.get('availability', '')).lower() == 'out':
            p['projected_minutes'] = 0.0
            p['projected_points'] = 0.0
            p['projected_rebounds'] = 0.0
            p['projected_assists'] = 0.0
            for key in [
                'projected_minutes_ci_low', 'projected_minutes_ci_high',
                'projected_points_ci_low', 'projected_points_ci_high',
                'projected_rebounds_ci_low', 'projected_rebounds_ci_high',
                'projected_assists_ci_low', 'projected_assists_ci_high',
            ]:
                if key in p:
                    p[key] = 0.0
            continue
        p['projected_minutes'] = round(max(0.0, float(p.get('projected_minutes', 0.0)) * pace_like), 2)
        p['projected_points'] = round(max(0.0, float(p.get('projected_points', 0.0)) * pf), 2)
        p['projected_rebounds'] = round(max(0.0, float(p.get('projected_rebounds', 0.0)) * rf), 2)
        p['projected_assists'] = round(max(0.0, float(p.get('projected_assists', 0.0)) * af), 2)
        if 'projected_minutes_ci_low' in p:
            p['projected_minutes_ci_low'] = round(max(0.0, float(p['projected_minutes_ci_low']) * pace_like), 2)
            p['projected_minutes_ci_high'] = round(max(0.0, float(p['projected_minutes_ci_high']) * pace_like), 2)
        if 'projected_points_ci_low' in p:
            p['projected_points_ci_low'] = round(max(0.0, float(p['projected_points_ci_low']) * pf), 2)
            p['projected_points_ci_high'] = round(max(0.0, float(p['projected_points_ci_high']) * pf), 2)
        if 'projected_rebounds_ci_low' in p:
            p['projected_rebounds_ci_low'] = round(max(0.0, float(p['projected_rebounds_ci_low']) * rf), 2)
            p['projected_rebounds_ci_high'] = round(max(0.0, float(p['projected_rebounds_ci_high']) * rf), 2)
        if 'projected_assists_ci_low' in p:
            p['projected_assists_ci_low'] = round(max(0.0, float(p['projected_assists_ci_low']) * af), 2)
            p['projected_assists_ci_high'] = round(max(0.0, float(p['projected_assists_ci_high']) * af), 2)
    return players


def _model_based_player_projection(team_id: int, game_date=None, opponent_team_id: Optional[int] = None):
    if not isinstance(player_projection_artifact, dict):
        return None
    if not os.path.exists(PLAYERS_RAW_PATH) or not os.path.exists(PLAYER_LOGS_PATH):
        return None

    model = player_projection_artifact.get('model')
    minutes_model = player_projection_artifact.get('minutes_model')
    rate_model = player_projection_artifact.get('rate_model')
    feature_columns = player_projection_artifact.get('feature_columns') or []
    target_columns = player_projection_artifact.get('target_columns') or ['PTS', 'REB', 'AST']
    if (model is None and (minutes_model is None or rate_model is None)) or not feature_columns:
        return None

    players = pd.read_csv(PLAYERS_RAW_PATH)
    if players.empty or 'TEAM_ID' not in players.columns:
        return None
    players['TEAM_ID'] = pd.to_numeric(players['TEAM_ID'], errors='coerce')
    players = players[players['TEAM_ID'] == int(team_id)].copy()
    if players.empty or 'PLAYER_ID' not in players.columns:
        return None
    players['PLAYER_ID'] = pd.to_numeric(players['PLAYER_ID'], errors='coerce')
    players = players.dropna(subset=['PLAYER_ID'])
    players['PLAYER_ID'] = players['PLAYER_ID'].astype(int)
    if players.empty:
        return None
    if 'PLAYER_NAME' not in players.columns and 'PLAYER_NAME.1' in players.columns:
        players['PLAYER_NAME'] = players['PLAYER_NAME.1']
    players['PLAYER_NAME'] = players.get('PLAYER_NAME', 'Unknown').astype(str)
    players['PLAYER_NAME_NORMALIZED'] = players['PLAYER_NAME'].apply(_normalize_player_name)

    logs = pd.read_csv(PLAYER_LOGS_PATH)
    if logs.empty or 'PLAYER_ID' not in logs.columns:
        return None
    logs['PLAYER_ID'] = pd.to_numeric(logs['PLAYER_ID'], errors='coerce')
    logs = logs.dropna(subset=['PLAYER_ID']).copy()
    logs['PLAYER_ID'] = logs['PLAYER_ID'].astype(int)
    logs['GAME_DATE'] = _utc_naive(logs.get('GAME_DATE'))
    logs = logs.dropna(subset=['GAME_DATE'])
    if logs.empty:
        return None

    injuries = _load_injuries_projection_frame()
    injury_map = {}
    injury_name_map = {}
    if not injuries.empty and 'PLAYER_ID' in injuries.columns:
        injuries = injuries.dropna(subset=['PLAYER_ID']).copy()
        injuries['PLAYER_ID'] = injuries['PLAYER_ID'].astype(int)
        injuries['INJURY_STATUS'] = injuries.get('INJURY_STATUS', '').astype(str)
        injuries['GAME_DATE_SORT'] = injuries.get('GAME_DATE')
        injuries = injuries.sort_values('GAME_DATE_SORT')
        for _, row in injuries.iterrows():
            injury_map[int(row['PLAYER_ID'])] = {
                'severity': float(row.get('INJURY_SEVERITY', 0.0)),
                'status': str(row.get('INJURY_STATUS', 'Available')) or 'Available',
                'return_date': str(row.get('INJURY_RETURN_DATE')) if pd.notna(row.get('INJURY_RETURN_DATE')) else None,
                'is_unavailable': bool(row.get('IS_UNAVAILABLE', False)),
            }
            normalized = _normalize_player_name(str(row.get('PLAYER_NAME', '')))
            if normalized:
                injury_name_map[normalized] = injury_map[int(row['PLAYER_ID'])]

    rows = []
    meta = []
    for _, player in players.iterrows():
        player_id = int(player['PLAYER_ID'])
        features = _player_feature_frame_for_inference(
            logs,
            player_id,
            team_id,
            opponent_team_id=opponent_team_id,
        )
        if not features:
            continue
        normalized_name = _normalize_player_name(str(player.get('PLAYER_NAME', '')))
        injury = injury_map.get(player_id) or injury_name_map.get(normalized_name) or {'severity': 0.0, 'status': 'Available', 'return_date': None, 'is_unavailable': False}
        is_out_for_game = _is_player_out_for_game(
            injury_status=injury.get('status'),
            injury_return_date=injury.get('return_date'),
            game_date=game_date,
            severity=float(injury.get('severity', 0.0)),
            is_unavailable=bool(injury.get('is_unavailable', False)),
        )
        features['INJURY_SEVERITY'] = float(injury['severity'])
        row = {col: _safe_float(features.get(col, 0.0), default=0.0) for col in feature_columns}
        rows.append(row)
        meta.append({
            'player_id': player_id,
            'player_name': str(player.get('PLAYER_NAME', 'Unknown')),
            'team_id': int(team_id),
            'injury_status': injury['status'],
            'injury_severity': float(injury['severity']),
            'injury_return_date': injury.get('return_date'),
            'is_out_for_game': bool(is_out_for_game),
            'min_last10': float(features.get('MIN_LAST10', 0.0)),
            'pts_last10': float(features.get('PTS_LAST10', 0.0)),
            'reb_last10': float(features.get('REB_LAST10', 0.0)),
            'ast_last10': float(features.get('AST_LAST10', 0.0)),
        })

    if not rows:
        return None

    X = pd.DataFrame(rows)[feature_columns].fillna(0.0)
    is_two_stage = minutes_model is not None and rate_model is not None
    uncertainty = player_projection_artifact.get('uncertainty', {}) if isinstance(player_projection_artifact, dict) else {}
    z_value = float(uncertainty.get('z_value', PLAYER_CONFIDENCE_Z))
    min_sigma = float(uncertainty.get('minutes_rmse', 3.0))
    pts_sigma = float(uncertainty.get('PTS_rmse', 4.0))
    reb_sigma = float(uncertainty.get('REB_rmse', 2.5))
    ast_sigma = float(uncertainty.get('AST_rmse', 2.0))
    if is_two_stage:
        pred_minutes = np.clip(minutes_model.predict(X), 0.0, 48.0)
        pred_rates = np.clip(rate_model.predict(X), 0.0, None)
        pred = pred_rates * np.asarray(pred_minutes).reshape(-1, 1)
    else:
        pred = model.predict(X)
        pred_minutes = np.asarray([_safe_float(r.get('MIN_LAST10', 24.0), default=24.0) for r in rows], dtype=float)
        pred_minutes = np.clip(pred_minutes, 0.0, 48.0)
    projections = []
    for idx, m in enumerate(meta):
        pts = max(0.0, float(pred[idx][0])) if len(target_columns) > 0 else 0.0
        reb = max(0.0, float(pred[idx][1])) if len(target_columns) > 1 else 0.0
        ast = max(0.0, float(pred[idx][2])) if len(target_columns) > 2 else 0.0
        minutes = max(0.0, float(pred_minutes[idx]))
        if m.get('is_out_for_game'):
            pts, reb, ast, minutes = 0.0, 0.0, 0.0, 0.0
        projections.append({
            'player_id': m['player_id'],
            'player_name': m['player_name'],
            'team_id': m.get('team_id'),
            'injury_status': m['injury_status'],
            'injury_severity': m['injury_severity'],
            'injury_return_date': m.get('injury_return_date'),
            'availability': (
                'Out' if m.get('is_out_for_game') else (
                    'Out' if m['injury_severity'] >= 2.0 else (
                        'Questionable' if m['injury_severity'] >= 1.2 else (
                            'Probable' if m['injury_severity'] >= 0.5 else 'Available'
                        )
                    )
                )
            ),
            'projected_minutes': round(minutes, 2),
            'projected_points': round(pts, 2),
            'projected_rebounds': round(reb, 2),
            'projected_assists': round(ast, 2),
            'projected_minutes_ci_low': round(max(0.0, minutes - z_value * min_sigma), 2),
            'projected_minutes_ci_high': round(max(0.0, minutes + z_value * min_sigma), 2),
            'projected_points_ci_low': round(max(0.0, pts - z_value * pts_sigma), 2),
            'projected_points_ci_high': round(max(0.0, pts + z_value * pts_sigma), 2),
            'projected_rebounds_ci_low': round(max(0.0, reb - z_value * reb_sigma), 2),
            'projected_rebounds_ci_high': round(max(0.0, reb + z_value * reb_sigma), 2),
            'projected_assists_ci_low': round(max(0.0, ast - z_value * ast_sigma), 2),
            'projected_assists_ci_high': round(max(0.0, ast + z_value * ast_sigma), 2),
            '_raw_projected_minutes': float(minutes),
            '_raw_projected_points': float(pts),
            '_raw_projected_rebounds': float(reb),
            '_raw_projected_assists': float(ast),
            '_min_last10': float(m.get('min_last10', 0.0)),
            '_pts_last10': float(m.get('pts_last10', 0.0)),
            '_reb_last10': float(m.get('reb_last10', 0.0)),
            '_ast_last10': float(m.get('ast_last10', 0.0)),
        })

    projections = _redistribute_absence_impact(projections)
    opp_factors = _opponent_adjustment_factors(opponent_team_id)
    projections = _apply_opponent_context(projections, opp_factors)

    for p in projections:
        for key in [
            '_raw_projected_minutes', '_raw_projected_points', '_raw_projected_rebounds', '_raw_projected_assists',
            '_min_last10', '_pts_last10', '_reb_last10', '_ast_last10'
        ]:
            p.pop(key, None)

    projections = sorted(projections, key=lambda x: x['projected_points'], reverse=True)
    return {
        'players': projections,
        'projection_method': (
            'Two-stage player model (minutes -> per-minute rates -> box score)'
            if is_two_stage else
            'Dedicated player model using player game logs (rolling form + rest + injury severity).'
        ),
        'context_adjustment': {
            'absence_redistribution': True,
            'opponent_factors': opp_factors,
            'uncertainty_model': {
                'z_value': z_value,
                'minutes_rmse': min_sigma,
                'PTS_rmse': pts_sigma,
                'REB_rmse': reb_sigma,
                'AST_rmse': ast_sigma,
            },
        },
        'covered_player_ids': [p.get('player_id') for p in projections if p.get('player_id') is not None],
    }


def _merged_player_projection_for_team(
    team_id: int,
    team_row: pd.Series,
    game_date=None,
    opponent_team_id: Optional[int] = None,
):
    fallback = _player_projection_for_team(team_id, team_row, game_date=game_date)
    model_based = _model_based_player_projection(team_id, game_date=game_date, opponent_team_id=opponent_team_id)

    if not model_based:
        opp_factors = _opponent_adjustment_factors(opponent_team_id)
        fallback_players = _apply_opponent_context(fallback.get('players', []), opp_factors)
        fallback['players'] = sorted(fallback_players, key=lambda x: float(x.get('projected_points', 0.0)), reverse=True)
        fallback['context_adjustment'] = {
            'absence_redistribution': False,
            'opponent_factors': opp_factors,
        }
        fallback['coverage_note'] = 'Model unavailable for this team; using heuristic fallback for all players.'
        return fallback

    model_players = model_based.get('players', [])
    model_method = str(model_based.get('projection_method', 'player model'))
    model_by_id = {
        p.get('player_id'): p for p in model_players
        if p.get('player_id') is not None
    }
    merged_players = []
    model_count = 0
    fallback_count = 0

    for p in fallback.get('players', []):
        pid = p.get('player_id')
        if pid in model_by_id:
            model_pred = model_by_id[pid]
            merged = p.copy()
            merged['projected_points'] = model_pred.get('projected_points', merged.get('projected_points'))
            merged['projected_rebounds'] = model_pred.get('projected_rebounds', merged.get('projected_rebounds'))
            merged['projected_assists'] = model_pred.get('projected_assists', merged.get('projected_assists'))
            if 'projected_minutes' in model_pred:
                merged['projected_minutes'] = model_pred.get('projected_minutes')
            merged['projection_source'] = 'player_model'
            model_count += 1
            merged_players.append(merged)
        else:
            p2 = p.copy()
            p2['projection_source'] = 'heuristic_fallback'
            fallback_count += 1
            merged_players.append(p2)

    merged_players = sorted(
        merged_players,
        key=lambda x: float(x.get('projected_points', 0.0)),
        reverse=True
    )
    return {
        'players': merged_players,
        'projection_method': (
            f'Hybrid output: {model_method} where coverage exists, '
            'heuristic fallback for uncovered players.'
        ),
        'coverage_note': (
            f"Model-covered players: {model_count}; fallback players: {fallback_count}."
        ),
        'context_adjustment': model_based.get('context_adjustment', {}) if isinstance(model_based, dict) else {},
    }


def _model_feature_names(model):
    if model is None:
        return []
    if hasattr(model, 'feature_names_in_'):
        try:
            return model.feature_names_in_.tolist()
        except Exception:
            return []
    return []


def _processed_numeric_features():
    if not os.path.exists(PROCESSED_DATA_PATH):
        return []
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH, nrows=1000)
    except Exception:
        return []
    if df.empty:
        return []
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    return [col for col in numeric_cols if col != 'WIN']


def _percent_contribution_rows(feature_contrib_map: Dict[str, float], top_n: int = 10):
    if not feature_contrib_map:
        return []
    items = [(k, float(v)) for k, v in feature_contrib_map.items()]
    abs_sum = sum(abs(v) for _, v in items) or 1.0
    ranked = sorted(items, key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    rows = []
    for feat, contrib in ranked:
        rows.append({
            'feature': feat,
            'contribution': round(contrib, 6),
            'impact_pct': round(abs(contrib) / abs_sum * 100.0, 2),
            'direction': 'up' if contrib >= 0 else 'down',
        })
    return rows


def _baseline_linear_explanation(model, features: Dict[str, float], top_n: int = 10):
    if not hasattr(model, 'named_steps'):
        return {'available': False, 'detail': 'Baseline explanation unavailable for this model type.'}
    scaler = model.named_steps.get('scaler')
    logreg = model.named_steps.get('logreg')
    if scaler is None or logreg is None or not hasattr(logreg, 'coef_'):
        return {'available': False, 'detail': 'Baseline scaler/logreg steps missing.'}
    feature_names = model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else list(features.keys())
    X = pd.DataFrame([{name: _safe_float(features.get(name, 0.0), default=0.0) for name in feature_names}], columns=feature_names)
    x_scaled = scaler.transform(X)[0]
    coef = np.asarray(logreg.coef_)[0]
    contrib = x_scaled * coef
    contrib_map = {name: float(val) for name, val in zip(feature_names, contrib)}
    intercept = float(np.asarray(logreg.intercept_)[0]) if hasattr(logreg, 'intercept_') else 0.0
    logit = intercept + float(np.sum(contrib))
    probability = 1.0 / (1.0 + np.exp(-logit))
    return {
        'available': True,
        'method': 'linear_contributions',
        'base_value': round(intercept, 6),
        'logit': round(float(logit), 6),
        'probability_from_explanation': round(float(probability), 6),
        'top_features': _percent_contribution_rows(contrib_map, top_n=top_n),
    }


def _tree_shap_explanation(model, features: Dict[str, float], top_n: int = 10):
    if xgb is None:
        return {'available': False, 'detail': 'xgboost not available for SHAP explanation.'}
    if not hasattr(model, 'get_booster'):
        return {'available': False, 'detail': 'Tree model does not expose booster.'}
    feature_names = model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else list(features.keys())
    X = pd.DataFrame([{name: _safe_float(features.get(name, 0.0), default=0.0) for name in feature_names}], columns=feature_names)
    try:
        dm = xgb.DMatrix(X, feature_names=feature_names)
        shap_vals = model.get_booster().predict(dm, pred_contribs=True)[0]
        base_value = float(shap_vals[-1])
        contrib_map = {name: float(val) for name, val in zip(feature_names, shap_vals[:-1])}
        logit = base_value + float(sum(contrib_map.values()))
        probability = 1.0 / (1.0 + np.exp(-logit))
        return {
            'available': True,
            'method': 'tree_shap_pred_contribs',
            'base_value': round(base_value, 6),
            'logit': round(float(logit), 6),
            'probability_from_explanation': round(float(probability), 6),
            'top_features': _percent_contribution_rows(contrib_map, top_n=top_n),
        }
    except Exception as exc:
        return {'available': False, 'detail': f'Tree SHAP explanation failed: {exc}'}


def _prediction_explanation(model_choice: str, model, features: Dict[str, float], top_n: int = 10):
    if model_choice == 'baseline':
        return _baseline_linear_explanation(model, features, top_n=top_n)
    if model_choice == 'tree':
        return _tree_shap_explanation(model, features, top_n=top_n)
    return {'available': False, 'detail': f'No explanation implementation for model={model_choice}'}


def _short_quote(text: str, limit: int = 120):
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + "..."
    return f"\"{text}\"" if text else ""


def _fetch_game_headlines(home_team_name: str, away_team_name: str, max_items: int = 5):
    def _pub_dt(pub_text: str):
        if not pub_text:
            return None
        try:
            dt = parsedate_to_datetime(pub_text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, NEWS_MAX_AGE_DAYS))
    query = f"{away_team_name} vs {home_team_name} NBA injuries"
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    try:
        resp = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        items = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            source = (item.findtext("source") or "News").strip()
            desc = (item.findtext("description") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            if not title or not link:
                continue
            pub_dt = _pub_dt(pub)
            # Enforce strict recency so game briefs stay relevant.
            if pub_dt is None or pub_dt < cutoff:
                continue
            items.append({
                "source": source,
                "title": title,
                "url": link,
                "quote": _short_quote(desc or title, limit=110),
                "published_at": pub,
                "published_at_utc": pub_dt.isoformat(),
            })
            if len(items) >= max_items:
                break
        return items
    except Exception:
        return []


def _top_stat_recommendation(projected_player_performance):
    if not isinstance(projected_player_performance, dict):
        return None
    candidates = []
    for side, block in projected_player_performance.items():
        for p in block.get("players", []):
            if str(p.get("availability", "")).lower() == "out":
                continue
            pts = _safe_float(p.get("projected_points", 0.0))
            reb = _safe_float(p.get("projected_rebounds", 0.0))
            ast = _safe_float(p.get("projected_assists", 0.0))
            mins = _safe_float(p.get("projected_minutes", 0.0))
            best_stat = max([("PTS", pts), ("REB", reb), ("AST", ast)], key=lambda x: x[1])
            confidence = min(99.0, max(55.0, 55.0 + mins * 0.8 + best_stat[1] * 0.5))
            candidates.append({
                "team_side": side,
                "player_name": p.get("player_name"),
                "stat": best_stat[0],
                "projected_value": round(float(best_stat[1]), 2),
                "projected_minutes": round(float(mins), 2),
                "confidence_pct": round(confidence, 1),
                "reason": (
                    f"{p.get('player_name')} projects for {best_stat[1]:.1f} {best_stat[0]} "
                    f"in {mins:.1f} minutes with availability={p.get('availability','Available')}."
                ),
            })
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda x: (x["confidence_pct"], x["projected_value"]), reverse=True)
    return candidates[0]


def _advisory_narrative(game, probability_report, top_recommendation, headlines):
    home_name = game.get("home_team_name", "Home")
    away_name = game.get("away_team_name", "Away")
    home_pct = float(probability_report.get("home_win_probability", 0.5)) * 100.0
    away_pct = float(probability_report.get("away_win_probability", 0.5)) * 100.0
    edge_team = home_name if home_pct >= away_pct else away_name
    edge = abs(home_pct - away_pct)
    tone = "strong edge" if edge >= 12 else ("moderate edge" if edge >= 6 else "tight matchup")
    line1 = (
        f"Model lean: {edge_team} ({home_pct:.1f}% vs {away_pct:.1f}%) with a {tone} "
        f"of {edge:.1f} percentage points."
    )
    line2 = ""
    if top_recommendation:
        line2 = (
            f"Top stat angle: {top_recommendation['player_name']} {top_recommendation['stat']} "
            f"at {top_recommendation['projected_value']:.1f} (confidence {top_recommendation['confidence_pct']:.1f}%)."
        )
    line3 = "News context unavailable from feed; rely more on injury snapshot + model features." if not headlines else (
        f"News pulse: pulled {len(headlines)} recent headlines relevant to this matchup."
    )
    ci_low = probability_report.get("home_win_ci_low")
    ci_high = probability_report.get("home_win_ci_high")
    line4 = ""
    if ci_low is not None and ci_high is not None:
        line4 = f"Confidence band (home win): {float(ci_low)*100:.1f}% to {float(ci_high)*100:.1f}%."
    return " ".join([line1, line2, line3, line4]).strip()


def _load_monitoring_report():
    if not os.path.exists(MONITORING_REPORT_PATH):
        return {
            'available': False,
            'detail': 'Monitoring report not found. Run scripts/generate_monitoring_report.py or scripts/update_pipeline.py.',
        }
    try:
        import json
        with open(MONITORING_REPORT_PATH, 'r', encoding='utf-8') as fh:
            payload = json.load(fh)
        payload['available'] = True
        return payload
    except Exception as exc:
        return {'available': False, 'detail': f'Failed to read monitoring report: {exc}'}


def _availability_quality_status(stale_hours: float = AVAILABILITY_STALE_HOURS):
    status = {
        "path": INJURIES_LATEST_PATH,
        "exists": os.path.exists(INJURIES_LATEST_PATH),
        "rows": 0,
        "age_hours": None,
        "stale_hours_threshold": float(stale_hours),
        "is_stale": True,
        "is_empty": True,
        "warning": None,
    }
    if not status["exists"]:
        status["warning"] = "Availability snapshot missing (injuries_latest.csv not found)."
        return status
    try:
        frame = pd.read_csv(INJURIES_LATEST_PATH)
        status["rows"] = int(len(frame))
        status["is_empty"] = bool(frame.empty)
    except Exception:
        status["warning"] = "Availability snapshot unreadable."
        return status

    age_hours = (datetime.now(timezone.utc).timestamp() - os.path.getmtime(INJURIES_LATEST_PATH)) / 3600.0
    status["age_hours"] = round(float(age_hours), 2)
    status["is_stale"] = bool(age_hours > stale_hours)
    if status["is_empty"]:
        status["warning"] = "Availability snapshot is empty; player availability adjustments may be unreliable."
    elif status["is_stale"]:
        status["warning"] = (
            f"Availability snapshot is stale ({age_hours:.2f}h old > {stale_hours:.2f}h threshold)."
        )
    return status


def _load_prediction_quality_report():
    if not os.path.exists(PREDICTION_QUALITY_REPORT_PATH):
        return {
            "available": False,
            "detail": "Prediction quality report not found. Run scripts/generate_prediction_quality_report.py.",
        }
    try:
        with open(PREDICTION_QUALITY_REPORT_PATH, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        payload["available"] = True
        return payload
    except Exception as exc:
        return {"available": False, "detail": f"Failed to read prediction quality report: {exc}"}


def _load_schedule_source_status():
    if not os.path.exists(SCHEDULE_SOURCE_STATUS_PATH):
        return {"available": False, "detail": "schedule_source_status.json not found."}
    try:
        with open(SCHEDULE_SOURCE_STATUS_PATH, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        payload["available"] = True
        return payload
    except Exception as exc:
        return {"available": False, "detail": f"Failed to read schedule source status: {exc}"}


def _load_pipeline_status():
    if not os.path.exists(PIPELINE_STATUS_PATH):
        return {
            "available": False,
            "detail": "pipeline_status_latest.json not found. Run scripts/update_pipeline.py.",
        }
    try:
        with open(PIPELINE_STATUS_PATH, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        payload["available"] = True
        return payload
    except Exception as exc:
        return {"available": False, "detail": f"Failed to read pipeline status: {exc}"}


def _append_prediction_log_row(row: Dict[str, object]):
    os.makedirs(os.path.dirname(PREDICTION_LOG_PATH), exist_ok=True)
    headers = [
        "predicted_at_utc",
        "game_id",
        "game_date",
        "model",
        "home_team_id",
        "away_team_id",
        "home_win_probability",
        "away_win_probability",
        "availability_rows",
        "availability_age_hours",
        "availability_is_stale",
        "availability_is_empty",
    ]
    needs_header = not os.path.exists(PREDICTION_LOG_PATH)
    with open(PREDICTION_LOG_PATH, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        if needs_header:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in headers})


def _team_probability_interval(home_prob: float, request, resolved_game_id: str):
    model_choice = (request.model or "baseline").lower()
    candidates = [float(home_prob)]
    for alt in ["baseline", "tree", "champion"]:
        if (request.model or "baseline").lower() == alt:
            continue
        try:
            _, alt_model = _resolve_model(alt)
        except Exception:
            continue
        try:
            inference_df = _load_inference_features_df()
            home_row = _inference_team_row(inference_df, resolved_game_id, request.home_team_id)
            away_row = _inference_team_row(inference_df, resolved_game_id, request.away_team_id)
            if home_row is None or away_row is None:
                continue
            hf = _model_features_from_team_row(
                model=alt_model, team_row=home_row, team_id=request.home_team_id,
                home_team_id=request.home_team_id, away_team_id=request.away_team_id, is_home=True
            )
            af = _model_features_from_team_row(
                model=alt_model, team_row=away_row, team_id=request.away_team_id,
                home_team_id=request.home_team_id, away_team_id=request.away_team_id, is_home=False
            )
            h = float(_predict_from_features(alt, hf)['probability'])
            a = float(_predict_from_features(alt, af)['probability'])
            den = h + a
            if den > 0:
                candidates.append(float(h / den))
        except Exception:
            continue
    spread = float(np.std(candidates)) if len(candidates) > 1 else 0.03
    calibrated_halfwidth, calibration_rows = _historical_calibrated_halfwidth(model_choice=model_choice, default_halfwidth=TEAM_CONFIDENCE_Z * spread)
    halfwidth = max(TEAM_CONFIDENCE_Z * spread, calibrated_halfwidth)
    low = max(0.01, float(home_prob) - halfwidth)
    high = min(0.99, float(home_prob) + halfwidth)
    return {
        "home_win_ci_low": round(low, 6),
        "home_win_ci_high": round(high, 6),
        "model_spread_std": round(spread, 6),
        "num_models": len(candidates),
        "calibrated_halfwidth": round(float(halfwidth), 6),
        "calibration_rows": int(calibration_rows),
    }


def _historical_calibrated_halfwidth(model_choice: str, default_halfwidth: float = 0.08):
    if not os.path.exists(PREDICTION_LOG_PATH) or not os.path.exists(PROCESSED_DATA_PATH):
        return float(default_halfwidth), 0
    try:
        pred = pd.read_csv(PREDICTION_LOG_PATH)
        actual = pd.read_csv(PROCESSED_DATA_PATH)
    except Exception:
        return float(default_halfwidth), 0
    if pred.empty or actual.empty:
        return float(default_halfwidth), 0
    needed_pred = {"game_id", "home_team_id", "home_win_probability", "model"}
    needed_actual = {"GAME_ID", "TEAM_ID", "WIN"}
    if not needed_pred.issubset(pred.columns) or not needed_actual.issubset(actual.columns):
        return float(default_halfwidth), 0

    pred = pred.copy()
    pred["game_id"] = pred["game_id"].astype(str)
    pred["home_team_id"] = pd.to_numeric(pred["home_team_id"], errors="coerce")
    pred["home_win_probability"] = pd.to_numeric(pred["home_win_probability"], errors="coerce")
    pred["model"] = pred["model"].astype(str).str.lower()
    pred = pred.dropna(subset=["home_team_id", "home_win_probability"])
    if pred.empty:
        return float(default_halfwidth), 0
    pred["home_team_id"] = pred["home_team_id"].astype(int)
    pred = pred[pred["model"] == str(model_choice).lower()]
    if pred.empty:
        return float(default_halfwidth), 0

    actual = actual.copy()
    actual["GAME_ID"] = actual["GAME_ID"].astype(str)
    actual["TEAM_ID"] = pd.to_numeric(actual["TEAM_ID"], errors="coerce")
    actual["WIN"] = pd.to_numeric(actual["WIN"], errors="coerce")
    actual = actual.dropna(subset=["TEAM_ID", "WIN"])
    if actual.empty:
        return float(default_halfwidth), 0
    actual["TEAM_ID"] = actual["TEAM_ID"].astype(int)

    merged = pred.merge(
        actual.rename(columns={"GAME_ID": "game_id", "TEAM_ID": "home_team_id", "WIN": "actual_home_win"}),
        how="inner",
        on=["game_id", "home_team_id"],
    )
    merged = merged.dropna(subset=["actual_home_win"])
    if merged.empty:
        return float(default_halfwidth), 0
    y_true = pd.to_numeric(merged["actual_home_win"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    y_prob = pd.to_numeric(merged["home_win_probability"], errors="coerce").fillna(0.5).clip(0.001, 0.999)
    abs_err = (y_true - y_prob).abs()
    if abs_err.empty:
        return float(default_halfwidth), 0
    q = float(abs_err.quantile(0.80))
    q = float(max(0.03, min(0.35, q)))
    return q, int(len(abs_err))


def _confidence_drivers(home_prob: float, away_prob: float, availability_status: Dict[str, object], model_spread_std: float = 0.0):
    drivers = []
    edge = abs(float(home_prob) - float(away_prob))
    drivers.append({
        "factor": "model_edge_strength",
        "impact": "positive" if edge >= 0.08 else "neutral",
        "detail": f"home-away probability gap={edge:.3f}",
    })

    monitoring = _load_monitoring_report()
    drift_status = (monitoring.get("drift") or {}).get("status")
    if drift_status in {"high_drift", "moderate_drift"}:
        drivers.append({
            "factor": "data_drift",
            "impact": "negative",
            "detail": f"monitoring drift status={drift_status}",
        })
    else:
        drivers.append({
            "factor": "data_drift",
            "impact": "neutral",
            "detail": f"monitoring drift status={drift_status or 'unknown'}",
        })

    if availability_status.get("is_stale") or availability_status.get("is_empty"):
        drivers.append({
            "factor": "availability_freshness",
            "impact": "negative",
            "detail": availability_status.get("warning") or "availability stale/empty",
        })
    else:
        drivers.append({
            "factor": "availability_freshness",
            "impact": "positive",
            "detail": "availability snapshot fresh and non-empty",
        })
    drivers.append({
        "factor": "model_divergence",
        "impact": "negative" if model_spread_std >= 0.06 else ("neutral" if model_spread_std >= 0.03 else "positive"),
        "detail": f"cross-model std={float(model_spread_std):.3f}",
    })
    return drivers


def _data_quality_alerts(availability_status: Dict[str, object], monitoring_payload: Dict[str, object], schedule_source: Dict[str, object]):
    items = []
    if availability_status.get("is_empty"):
        items.append({
            "name": "availability_empty",
            "severity": "warn",
            "message": availability_status.get("warning") or "Availability snapshot is empty",
        })
    elif availability_status.get("is_stale"):
        items.append({
            "name": "availability_stale",
            "severity": "warn",
            "message": availability_status.get("warning") or "Availability snapshot is stale",
        })

    mon_alerts = (monitoring_payload.get("alerts") or {}).get("items") or []
    for a in mon_alerts:
        items.append({
            "name": a.get("name"),
            "severity": a.get("severity", "warn"),
            "message": a.get("message", ""),
        })

    source_used = schedule_source.get("source_used") if isinstance(schedule_source, dict) else None
    if source_used in {"retained_previous", "odds_fallback", "espn_fallback"}:
        sev = "warn" if source_used != "retained_previous" else "fail"
        items.append({
            "name": "schedule_fallback_active",
            "severity": sev,
            "message": f"Schedule source used: {source_used}",
        })

    overall = "pass"
    if any(i.get("severity") == "fail" for i in items):
        overall = "fail"
    elif any(i.get("severity") == "warn" for i in items):
        overall = "warn"
    return {"overall_status": overall, "items": items}


def _startup_required_paths():
    return [
        PROCESSED_DATA_PATH,
        UPCOMING_GAMES_PATH,
        INFERENCE_FEATURES_PATH,
        MODEL_PATH,
        TREE_MODEL_PATH,
        PLAYER_MODEL_PATH,
    ]


def _run_startup_checks():
    missing = [p for p in _startup_required_paths() if not os.path.exists(p)]
    model_loaded = any(m is not None for m in [baseline_model, tree_model, champion_model])
    if missing or not model_loaded:
        message = (
            f"Startup checks failed. missing_paths={missing}; "
            f"baseline_loaded={baseline_model is not None}; "
            f"tree_loaded={tree_model is not None}; "
            f"champion_loaded={champion_model is not None}"
        )
        if STRICT_STARTUP_CHECKS:
            raise RuntimeError(message)
        print(f"Warning: {message}")


@app.on_event("startup")
def startup_event():
    _run_startup_checks()


@app.get('/')
def root():
    return {
        'service': app.title,
        'version': app.version,
        'status': 'ok',
        'routes': ['/health', '/sample-features', '/features', '/monitoring', '/prediction-quality', '/pipeline-status', '/upcoming-games', '/predict', '/predict/sample', '/predict/team', '/docs'],
    }


@app.get('/health')
def health():
    pipeline = _load_pipeline_status()
    return {
        'status': 'ok',
        'pipeline_status_available': bool(pipeline.get("available", False)),
        'pipeline_last_run_id': pipeline.get("run_id"),
        'pipeline_last_status': pipeline.get("status"),
    }


@app.get('/sample-features')
def sample_features():
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise HTTPException(status_code=404, detail='Processed dataset not found')
    df = pd.read_csv(PROCESSED_DATA_PATH)
    if df.empty:
        raise HTTPException(status_code=404, detail='Processed dataset is empty')
    sample = df.head(1).to_dict(orient='records')[0]
    return {'sample': sample}


@app.get('/features')
def features():
    baseline_features = _model_feature_names(baseline_model)
    tree_features = _model_feature_names(tree_model)
    processed_numeric = _processed_numeric_features()
    upcoming_count = 0
    try:
        upcoming_count = len(_load_upcoming_games_df())
    except Exception:
        upcoming_count = 0

    return {
        'baseline_model_loaded': baseline_model is not None,
        'tree_model_loaded': tree_model is not None,
        'champion_model_loaded': champion_model is not None,
        'player_projection_model_loaded': player_projection_artifact is not None,
        'upcoming_games_count': int(upcoming_count),
        'inference_features_available': os.path.exists(INFERENCE_FEATURES_PATH),
        'baseline_features': baseline_features,
        'tree_features': tree_features,
        'processed_numeric_features': processed_numeric,
        'monitoring_report_available': os.path.exists(MONITORING_REPORT_PATH),
        'prediction_quality_report_available': os.path.exists(PREDICTION_QUALITY_REPORT_PATH),
        'schedule_source_status_available': os.path.exists(SCHEDULE_SOURCE_STATUS_PATH),
        'pipeline_status_available': os.path.exists(PIPELINE_STATUS_PATH),
    }


@app.get('/monitoring')
def monitoring():
    return _load_monitoring_report()


@app.get('/prediction-quality')
def prediction_quality():
    return _load_prediction_quality_report()


@app.get('/pipeline-status')
def pipeline_status():
    return _load_pipeline_status()


@app.get('/upcoming-games')
def upcoming_games(limit: int = 50):
    limit = max(1, min(int(limit), 300))
    return {'games': _upcoming_games_payload(limit=limit)}


@app.post('/predict')
def predict(request: PredictionRequest):
    raise HTTPException(
        status_code=410,
        detail="Direct feature-vector prediction is disabled in production mode. Use POST /predict/team for future scheduled games."
    )


@app.get('/predict/sample')
def predict_sample(model: str = 'baseline'):
    upcoming = _load_upcoming_games_df().sort_values('GAME_DATE')
    sample_game = upcoming.iloc[0]
    req = TeamPredictionRequest(
        home_team_id=int(sample_game['HOME_TEAM_ID']),
        away_team_id=int(sample_game['AWAY_TEAM_ID']),
        game_id=str(sample_game['GAME_ID']),
        model=model,
        include_player_projection=False,
    )
    return predict_team(req)


@app.post('/predict/team')
def predict_team(request: TeamPredictionRequest):
    if request.home_team_id == request.away_team_id:
        raise HTTPException(status_code=400, detail='home_team_id and away_team_id must be different')

    model_choice, model = _resolve_model(request.model or 'baseline')
    upcoming_game = _resolve_upcoming_match(
        home_team_id=request.home_team_id,
        away_team_id=request.away_team_id,
        game_id=request.game_id,
    )
    resolved_game_id = str(upcoming_game['GAME_ID'])
    inference_df = _load_inference_features_df()
    availability_status = _availability_quality_status()

    home_row = _inference_team_row(inference_df, resolved_game_id, request.home_team_id)
    away_row = _inference_team_row(inference_df, resolved_game_id, request.away_team_id)
    if home_row is None or away_row is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Inference features missing for GAME_ID={resolved_game_id}. "
                "Run scripts/build_inference_features.py (or scripts/update_pipeline.py) to generate future-game features."
            ),
        )

    home_features = _model_features_from_team_row(
        model=model,
        team_row=home_row,
        team_id=request.home_team_id,
        home_team_id=request.home_team_id,
        away_team_id=request.away_team_id,
        is_home=True
    )
    away_features = _model_features_from_team_row(
        model=model,
        team_row=away_row,
        team_id=request.away_team_id,
        home_team_id=request.home_team_id,
        away_team_id=request.away_team_id,
        is_home=False
    )

    home_raw = float(_predict_from_features(model_choice, home_features)['probability'])
    away_raw = float(_predict_from_features(model_choice, away_features)['probability'])
    normalizer = home_raw + away_raw
    if normalizer <= 0:
        home_prob = home_raw
        away_prob = 1.0 - home_prob
    else:
        home_prob = home_raw / normalizer
        away_prob = away_raw / normalizer

    probability_report = {
        'model': model_choice,
        'home_win_probability': round(home_prob, 6),
        'away_win_probability': round(away_prob, 6),
        'home_fair_moneyline': _american_price_from_probability(home_prob),
        'away_fair_moneyline': _american_price_from_probability(away_prob),
        'raw_home_probability': round(home_raw, 6),
        'raw_away_probability': round(away_raw, 6),
        'calculation_note': 'Normalized from independent team-win estimates to enforce home+away=1.0.',
    }
    probability_report.update(_team_probability_interval(home_prob, request, resolved_game_id))
    explain_home = _prediction_explanation(model_choice, model, home_features, top_n=10)
    explain_away = _prediction_explanation(model_choice, model, away_features, top_n=10)
    monitoring_payload = _load_monitoring_report()
    schedule_source = _load_schedule_source_status()

    response = {
        'game': {
            'home_team_id': request.home_team_id,
            'away_team_id': request.away_team_id,
            'game_id': resolved_game_id,
            'home_team_name': _team_name_from_id(request.home_team_id, str(home_row.get('TEAM_NAME', 'Home Team'))),
            'away_team_name': _team_name_from_id(request.away_team_id, str(away_row.get('TEAM_NAME', 'Away Team'))),
            'game_date': str(pd.to_datetime(upcoming_game.get('GAME_DATE'), errors='coerce').date()),
            'feature_as_of_date': max(str(home_row.get('GAME_DATE', '')), str(away_row.get('GAME_DATE', ''))),
            'source': 'upcoming_schedule',
        },
        'probability_report': probability_report,
        'explainability': {
            'home': explain_home,
            'away': explain_away,
        },
        'feature_snapshot': {
            'home': {
                'pts_last10': _safe_float(home_row.get('pts_last10', 0.0)),
                'reb_last10': _safe_float(home_row.get('reb_last10', 0.0)),
                'ast_last10': _safe_float(home_row.get('ast_last10', 0.0)),
                'rest_days': _safe_float(home_row.get('REST_DAYS', 0.0)),
                'injury_impact': _safe_float(home_features.get('INJURY_IMPACT', 0.0)),
                'fatigue_index': _safe_float(home_row.get('fatigue_index', 0.0)),
            },
            'away': {
                'pts_last10': _safe_float(away_row.get('pts_last10', 0.0)),
                'reb_last10': _safe_float(away_row.get('reb_last10', 0.0)),
                'ast_last10': _safe_float(away_row.get('ast_last10', 0.0)),
                'rest_days': _safe_float(away_row.get('REST_DAYS', 0.0)),
                'injury_impact': _safe_float(away_features.get('INJURY_IMPACT', 0.0)),
                'fatigue_index': _safe_float(away_row.get('fatigue_index', 0.0)),
            }
        },
        'data_quality': {
            'availability': availability_status,
            'monitoring_alerts': (monitoring_payload.get("alerts") or {}),
            'schedule_source': schedule_source,
            'alerts': _data_quality_alerts(availability_status, monitoring_payload, schedule_source),
        },
    }

    if request.include_player_projection:
        selected_game_date = upcoming_game.get('GAME_DATE')
        response['projected_player_performance'] = {
            'home': _merged_player_projection_for_team(
                request.home_team_id,
                home_row,
                game_date=selected_game_date,
                opponent_team_id=request.away_team_id,
            ),
            'away': _merged_player_projection_for_team(
                request.away_team_id,
                away_row,
                game_date=selected_game_date,
                opponent_team_id=request.home_team_id,
            ),
        }

    home_name = response["game"].get("home_team_name", "")
    away_name = response["game"].get("away_team_name", "")
    headlines = _fetch_game_headlines(home_name, away_name, max_items=5) if request.include_headlines else []
    top_rec = _top_stat_recommendation(response.get("projected_player_performance", {}))
    if top_rec and (availability_status.get("is_stale") or availability_status.get("is_empty")):
        prior = float(top_rec.get("confidence_pct", 0.0))
        penalized = max(50.0, prior - 8.0)
        top_rec["confidence_pct"] = round(penalized, 1)
        top_rec["reason"] = (
            f"{top_rec.get('reason', '')} Availability freshness penalty applied due to stale/empty injury snapshot."
        ).strip()
    response["advisory"] = {
        "narrative": _advisory_narrative(response["game"], probability_report, top_rec, headlines),
        "top_recommendation": top_rec,
        "recent_headlines": headlines,
        "confidence_drivers": _confidence_drivers(
            home_prob,
            away_prob,
            availability_status,
            model_spread_std=float(probability_report.get("model_spread_std", 0.0)),
        ),
    }

    _append_prediction_log_row({
        "predicted_at_utc": datetime.now(timezone.utc).isoformat(),
        "game_id": resolved_game_id,
        "game_date": response["game"].get("game_date"),
        "model": model_choice,
        "home_team_id": int(request.home_team_id),
        "away_team_id": int(request.away_team_id),
        "home_win_probability": float(probability_report["home_win_probability"]),
        "away_win_probability": float(probability_report["away_win_probability"]),
        "availability_rows": int(availability_status.get("rows", 0) or 0),
        "availability_age_hours": availability_status.get("age_hours"),
        "availability_is_stale": bool(availability_status.get("is_stale", True)),
        "availability_is_empty": bool(availability_status.get("is_empty", True)),
    })

    return response
