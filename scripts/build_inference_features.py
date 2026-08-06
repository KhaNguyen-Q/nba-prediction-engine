import os
import sys
from datetime import datetime, timezone

import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.team_utils import (
    find_team_location,
    find_team_profile,
    get_team_adult_quality,
    get_team_timezone_offset,
    haversine_distance,
)
from scripts.build_features import load_odds_features


PROCESSED_PATH = "data/processed/games_with_features.csv"
UPCOMING_PATH = "data/raw/upcoming_games.csv"
INJURIES_PATH = "data/raw/injuries_raw.csv"
ODDS_PATH = "data/raw/odds_raw.csv"
OUTPUT_PATH = "data/processed/upcoming_inference_features.csv"

ROLLING_SOURCE_COLS = ['PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'PLUS_MINUS']
ROLLING_WINDOWS = (5, 10)
ODDS_FEATURE_COLS = [
    'HOME_ML_PRICE', 'AWAY_ML_PRICE', 'HOME_ML_PROB', 'AWAY_ML_PROB',
    'HOME_SPREAD_POINT', 'AWAY_SPREAD_POINT', 'TOTAL_POINT',
    'H2H_BOOKMAKERS_COUNT', 'SPREAD_BOOKMAKERS_COUNT', 'TOTALS_BOOKMAKERS_COUNT',
    'TEAM_ML_PRICE', 'TEAM_IMPLIED_PROB', 'OPP_ML_PRICE', 'OPP_IMPLIED_PROB',
    'ODDS_PROB_DIFF', 'PUBLIC_BIAS_INDICATOR', 'TEAM_SPREAD_POINT', 'OPP_SPREAD_POINT',
]


def _utc_naive(series):
    return pd.to_datetime(series, errors='coerce', utc=True).dt.tz_convert(None)


def _current_team_injury_impact(injuries_df, team_id, game_date):
    if injuries_df.empty:
        return 0.0
    team_inj = injuries_df[injuries_df['TEAM_ID'] == team_id].copy()
    if team_inj.empty:
        return 0.0
    active = (
        (team_inj['GAME_DATE'] <= game_date) &
        (team_inj['INJURY_RETURN_DATE'].isna() | (team_inj['INJURY_RETURN_DATE'] >= game_date))
    )
    return float(team_inj.loc[active, 'INJURY_SEVERITY'].sum())


def _rest_days(latest_game_date, target_date):
    if pd.isna(latest_game_date) or pd.isna(target_date):
        return 5.0
    return float(max(0, min((target_date - latest_game_date).days, 14)))


def _team_name(team_id):
    profile = find_team_profile(team_id=int(team_id))
    if profile:
        return profile.get('name')
    return None


def _rolling_features_from_history(history_df):
    """Recompute prior-game rolling features including the latest completed game."""
    out = {}
    if history_df is None or history_df.empty:
        for col in ROLLING_SOURCE_COLS:
            for w in ROLLING_WINDOWS:
                out[f"{col.lower()}_last{w}"] = 0.0
        out['WIN_STREAK'] = 0.0
        return out

    hist = history_df.sort_values('GAME_DATE')
    for col in ROLLING_SOURCE_COLS:
        series = pd.to_numeric(hist[col], errors='coerce').dropna() if col in hist.columns else pd.Series(dtype=float)
        for w in ROLLING_WINDOWS:
            feature = f"{col.lower()}_last{w}"
            out[feature] = float(series.tail(w).mean()) if not series.empty else 0.0

    if 'WIN' in hist.columns:
        wins = pd.to_numeric(hist['WIN'], errors='coerce').dropna()
        out['WIN_STREAK'] = float(wins.tail(5).sum()) if not wins.empty else 0.0
    else:
        out['WIN_STREAK'] = 0.0
    return out


def _last_location(latest_row):
    lat = latest_row.get('CURRENT_LAT') if hasattr(latest_row, 'get') else None
    lon = latest_row.get('CURRENT_LON') if hasattr(latest_row, 'get') else None
    try:
        lat = float(lat) if pd.notna(lat) else None
        lon = float(lon) if pd.notna(lon) else None
    except (TypeError, ValueError):
        lat, lon = None, None

    if lat is not None and lon is not None:
        return lat, lon

    # Fallback: infer last venue from home/away flag on the latest completed game.
    team_id = int(latest_row.get('TEAM_ID')) if pd.notna(latest_row.get('TEAM_ID', np.nan)) else None
    is_away = int(latest_row.get('IS_AWAY', 0) or 0) == 1
    if is_away:
        opponent = latest_row.get('OPPONENT')
        loc = find_team_location(abbreviation=opponent) if isinstance(opponent, str) else None
    else:
        loc = find_team_location(team_id=team_id) if team_id is not None else None
    if not loc:
        return None, None
    return loc.get('lat'), loc.get('lon')


def _last_timezone_offset(latest_row, reference_date):
    offset = latest_row.get('CURRENT_TIMEZONE_OFFSET') if hasattr(latest_row, 'get') else None
    try:
        if pd.notna(offset):
            return float(offset)
    except (TypeError, ValueError):
        pass

    team_id = int(latest_row.get('TEAM_ID')) if pd.notna(latest_row.get('TEAM_ID', np.nan)) else None
    is_away = int(latest_row.get('IS_AWAY', 0) or 0) == 1
    if is_away:
        opponent = latest_row.get('OPPONENT')
        if isinstance(opponent, str):
            return get_team_timezone_offset(abbreviation=opponent, reference_date=reference_date)
    if team_id is not None:
        return get_team_timezone_offset(team_id=team_id, reference_date=reference_date)
    return None


def _upcoming_venue_location(home_team_id):
    loc = find_team_location(team_id=int(home_team_id))
    if not loc:
        return None, None
    return loc.get('lat'), loc.get('lon')


def _compute_travel_and_timezone(latest_row, home_team_id, game_date):
    """Travel/timezone from last completed venue to the upcoming game venue."""
    prev_lat, prev_lon = _last_location(latest_row)
    curr_lat, curr_lon = _upcoming_venue_location(home_team_id)
    travel_km = 0.0
    if prev_lat is not None and prev_lon is not None and curr_lat is not None and curr_lon is not None:
        dist = haversine_distance(prev_lat, prev_lon, curr_lat, curr_lon)
        travel_km = float(dist) if dist is not None else 0.0

    prev_offset = _last_timezone_offset(latest_row, reference_date=latest_row.get('GAME_DATE'))
    curr_offset = get_team_timezone_offset(team_id=int(home_team_id), reference_date=game_date)
    tz_diff = 0.0
    if prev_offset is not None and curr_offset is not None:
        tz_diff = abs(float(curr_offset) - float(prev_offset))

    return travel_km, tz_diff


def _fatigue_index(rest_days, back_to_back, travel_km, tz_diff):
    return (
        0.4 * float(back_to_back) +
        0.3 * (1.0 / (float(rest_days) + 1.0)) +
        0.2 * min(max(float(travel_km) / 3000.0, 0.0), 1.0) +
        0.1 * min(abs(float(tz_diff)), 3.0)
    )


def _adult_entertainment_index(is_away, home_team_id):
    if not is_away:
        return 0.0
    quality = get_team_adult_quality(team_id=int(home_team_id), default=5)
    return float(np.clip(11 - quality, 1, 10))


def _clear_stale_odds(row):
    for col in ODDS_FEATURE_COLS:
        if col in row:
            row[col] = np.nan
    return row


def _apply_odds_for_side(row, odds_row, is_home):
    if odds_row is None or odds_row.empty:
        return _clear_stale_odds(row)

    for col in [
        'HOME_ML_PRICE', 'AWAY_ML_PRICE', 'HOME_ML_PROB', 'AWAY_ML_PROB',
        'HOME_SPREAD_POINT', 'AWAY_SPREAD_POINT', 'TOTAL_POINT',
        'H2H_BOOKMAKERS_COUNT', 'SPREAD_BOOKMAKERS_COUNT', 'TOTALS_BOOKMAKERS_COUNT',
    ]:
        if col in odds_row.index:
            row[col] = odds_row.get(col)

    home_ml_price = odds_row.get('HOME_ML_PRICE')
    away_ml_price = odds_row.get('AWAY_ML_PRICE')
    home_ml_prob = odds_row.get('HOME_ML_PROB')
    away_ml_prob = odds_row.get('AWAY_ML_PROB')
    home_spread = odds_row.get('HOME_SPREAD_POINT')
    away_spread = odds_row.get('AWAY_SPREAD_POINT')

    row['TEAM_ML_PRICE'] = home_ml_price if is_home else away_ml_price
    row['TEAM_IMPLIED_PROB'] = home_ml_prob if is_home else away_ml_prob
    row['OPP_ML_PRICE'] = away_ml_price if is_home else home_ml_price
    row['OPP_IMPLIED_PROB'] = away_ml_prob if is_home else home_ml_prob
    row['TEAM_SPREAD_POINT'] = home_spread if is_home else away_spread
    row['OPP_SPREAD_POINT'] = away_spread if is_home else home_spread

    team_prob = row.get('TEAM_IMPLIED_PROB')
    opp_prob = row.get('OPP_IMPLIED_PROB')
    if pd.notna(team_prob) and pd.notna(opp_prob):
        row['ODDS_PROB_DIFF'] = abs(float(team_prob) - float(opp_prob))
        row['PUBLIC_BIAS_INDICATOR'] = float(team_prob) - float(opp_prob)
    else:
        row['ODDS_PROB_DIFF'] = np.nan
        row['PUBLIC_BIAS_INDICATOR'] = np.nan
    row['TOTAL_POINT'] = odds_row.get('TOTAL_POINT')
    return row


def _lookup_odds_row(odds_by_game, home_id, away_id, game_date):
    if odds_by_game is None or odds_by_game.empty:
        return None
    home_name = _team_name(home_id)
    away_name = _team_name(away_id)
    if not home_name or not away_name:
        return None
    game_day = pd.to_datetime(game_date, errors='coerce')
    if pd.isna(game_day):
        return None
    game_day = game_day.date()
    match = odds_by_game[
        (odds_by_game['HOME_TEAM'] == home_name) &
        (odds_by_game['AWAY_TEAM'] == away_name) &
        (odds_by_game['GAME_DATE'] == game_day)
    ]
    if match.empty:
        return None
    return match.iloc[0]


def build_inference_features(
    processed_path=PROCESSED_PATH,
    upcoming_path=UPCOMING_PATH,
    injuries_path=INJURIES_PATH,
    odds_path=ODDS_PATH,
    output_path=OUTPUT_PATH,
):
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed features not found: {processed_path}")
    if not os.path.exists(upcoming_path):
        raise FileNotFoundError(f"Upcoming games file not found: {upcoming_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed = pd.read_csv(processed_path)
    upcoming = pd.read_csv(upcoming_path)
    if processed.empty or upcoming.empty:
        pd.DataFrame().to_csv(output_path, index=False)
        print(f"Saved empty inference features to {output_path}")
        return pd.DataFrame()

    processed['GAME_DATE'] = pd.to_datetime(processed['GAME_DATE'], errors='coerce')
    processed = processed.dropna(subset=['GAME_DATE', 'TEAM_ID']).copy()
    processed['TEAM_ID'] = pd.to_numeric(processed['TEAM_ID'], errors='coerce')
    processed = processed.dropna(subset=['TEAM_ID'])
    processed['TEAM_ID'] = processed['TEAM_ID'].astype(int)
    processed = processed.sort_values(['TEAM_ID', 'GAME_DATE'])

    upcoming['GAME_DATE'] = pd.to_datetime(upcoming['GAME_DATE'], errors='coerce')
    upcoming['HOME_TEAM_ID'] = pd.to_numeric(upcoming['HOME_TEAM_ID'], errors='coerce')
    upcoming['AWAY_TEAM_ID'] = pd.to_numeric(upcoming['AWAY_TEAM_ID'], errors='coerce')
    upcoming = upcoming.dropna(subset=['GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']).copy()
    upcoming['HOME_TEAM_ID'] = upcoming['HOME_TEAM_ID'].astype(int)
    upcoming['AWAY_TEAM_ID'] = upcoming['AWAY_TEAM_ID'].astype(int)

    injuries = pd.read_csv(injuries_path) if os.path.exists(injuries_path) else pd.DataFrame()
    if not injuries.empty and {'TEAM_ID', 'GAME_DATE'}.issubset(injuries.columns):
        injuries['TEAM_ID'] = pd.to_numeric(injuries['TEAM_ID'], errors='coerce')
        injuries = injuries.dropna(subset=['TEAM_ID']).copy()
        injuries['TEAM_ID'] = injuries['TEAM_ID'].astype(int)
        injuries['GAME_DATE'] = _utc_naive(injuries['GAME_DATE'])
        injuries['INJURY_RETURN_DATE'] = _utc_naive(injuries.get('INJURY_RETURN_DATE'))
        injuries['INJURY_SEVERITY'] = pd.to_numeric(injuries.get('INJURY_SEVERITY'), errors='coerce').fillna(1.0)
    else:
        injuries = pd.DataFrame(columns=['TEAM_ID', 'GAME_DATE', 'INJURY_RETURN_DATE', 'INJURY_SEVERITY'])

    odds_by_game = load_odds_features(odds_path)
    if not odds_by_game.empty and 'GAME_DATE' in odds_by_game.columns:
        odds_by_game['GAME_DATE'] = pd.to_datetime(odds_by_game['GAME_DATE'], errors='coerce').dt.date

    numeric_cols = processed.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude identifiers/flags that we explicitly set for upcoming rows.
    drop_cols = {
        'WIN', 'GAME_ID', 'TEAM_ID', 'HOME', 'IS_AWAY',
        'HOME_TEAM', 'AWAY_TEAM', 'HOME_TEAM_ID', 'AWAY_TEAM_ID'
    }
    numeric_cols = [col for col in numeric_cols if col not in drop_cols]

    latest_by_team = (
        processed.sort_values('GAME_DATE')
        .groupby('TEAM_ID', as_index=False)
        .tail(1)
        .set_index('TEAM_ID')
    )
    last_game_date = processed.groupby('TEAM_ID')['GAME_DATE'].max().to_dict()
    history_by_team = {team_id: group for team_id, group in processed.groupby('TEAM_ID')}

    rows = []
    for _, game in upcoming.iterrows():
        game_id = str(game['GAME_ID'])
        game_date = pd.to_datetime(game['GAME_DATE'], errors='coerce')
        home_id = int(game['HOME_TEAM_ID'])
        away_id = int(game['AWAY_TEAM_ID'])
        odds_row = _lookup_odds_row(odds_by_game, home_id, away_id, game_date)

        for team_id, is_home in [(home_id, True), (away_id, False)]:
            if team_id not in latest_by_team.index:
                continue
            latest = latest_by_team.loc[team_id]
            row = {
                'GAME_ID': game_id,
                'GAME_DATE': game_date,
                'HOME_TEAM_ID': home_id,
                'AWAY_TEAM_ID': away_id,
                'TEAM_ID': team_id,
                'HOME': 1 if is_home else 0,
                'IS_AWAY': 0 if is_home else 1,
            }
            # Keep schema-compatible numeric base from latest row, then overlay recomputed fields.
            for col in numeric_cols:
                row[col] = float(latest.get(col, 0.0)) if pd.notna(latest.get(col, np.nan)) else 0.0

            rolling = _rolling_features_from_history(history_by_team.get(team_id))
            row.update(rolling)

            rest = _rest_days(last_game_date.get(team_id), game_date)
            back_to_back = 1.0 if rest <= 1 else 0.0
            travel_km, tz_diff = _compute_travel_and_timezone(latest, home_id, game_date)

            row['REST_DAYS'] = rest
            row['BACK_TO_BACK'] = back_to_back
            row['TRAVEL_KM'] = travel_km
            row['TZ_DIFF'] = tz_diff
            row['TRAVEL_DISTANCE'] = travel_km
            row['TIMEZONE_SHIFT'] = tz_diff
            row['fatigue_index'] = _fatigue_index(rest, back_to_back, travel_km, tz_diff)
            row['ADULT_ENTERTAINMENT_INDEX'] = _adult_entertainment_index(not is_home, home_id)
            row['INJURY_IMPACT'] = _current_team_injury_impact(injuries, team_id, game_date)
            row = _apply_odds_for_side(row, odds_row, is_home)
            row['HOME_TEAM'] = float(home_id)
            row['AWAY_TEAM'] = float(away_id)
            rows.append(row)

    inference = pd.DataFrame(rows)
    if not inference.empty:
        inference = inference.sort_values(['GAME_DATE', 'GAME_ID', 'TEAM_ID'])
    inference.to_csv(output_path, index=False)
    print(
        f"Saved inference features to {output_path} with {len(inference)} rows "
        "(rolling/travel/fatigue/odds recomputed for upcoming matchups)"
    )
    return inference


def main():
    build_inference_features()


if __name__ == '__main__':
    main()
