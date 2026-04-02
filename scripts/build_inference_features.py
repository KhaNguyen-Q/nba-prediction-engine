import os
import sys
from datetime import datetime, timezone

import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


PROCESSED_PATH = "data/processed/games_with_features.csv"
UPCOMING_PATH = "data/raw/upcoming_games.csv"
INJURIES_PATH = "data/raw/injuries_raw.csv"
OUTPUT_PATH = "data/processed/upcoming_inference_features.csv"


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


def build_inference_features(
    processed_path=PROCESSED_PATH,
    upcoming_path=UPCOMING_PATH,
    injuries_path=INJURIES_PATH,
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

    rows = []
    for _, game in upcoming.iterrows():
        game_id = str(game['GAME_ID'])
        game_date = pd.to_datetime(game['GAME_DATE'], errors='coerce')
        home_id = int(game['HOME_TEAM_ID'])
        away_id = int(game['AWAY_TEAM_ID'])

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
            for col in numeric_cols:
                row[col] = float(latest.get(col, 0.0)) if pd.notna(latest.get(col, np.nan)) else 0.0

            rest = _rest_days(last_game_date.get(team_id), game_date)
            row['REST_DAYS'] = rest
            row['BACK_TO_BACK'] = 1.0 if rest <= 1 else 0.0
            row['TRAVEL_DISTANCE'] = 0.0 if is_home else float(row.get('TRAVEL_DISTANCE', 0.0))
            row['TIMEZONE_SHIFT'] = 0.0 if is_home else float(row.get('TIMEZONE_SHIFT', 0.0))
            row['INJURY_IMPACT'] = _current_team_injury_impact(injuries, team_id, game_date)
            row['HOME_TEAM'] = float(home_id)
            row['AWAY_TEAM'] = float(away_id)
            rows.append(row)

    inference = pd.DataFrame(rows)
    if not inference.empty:
        inference = inference.sort_values(['GAME_DATE', 'GAME_ID', 'TEAM_ID'])
    inference.to_csv(output_path, index=False)
    print(f"Saved inference features to {output_path} with {len(inference)} rows")
    return inference


def main():
    build_inference_features()


if __name__ == '__main__':
    main()
