import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
from nba_api.stats.endpoints import scoreboardv2

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.team_utils import find_team_profile, is_nba_team_by_id


RAW_PATH = "data/raw/upcoming_games.csv"
ODDS_PATH = "data/raw/odds_raw.csv"


def _team_abbr(team_id, line_score_df):
    if not line_score_df.empty and 'TEAM_ID' in line_score_df.columns and 'TEAM_ABBREVIATION' in line_score_df.columns:
        row = line_score_df[line_score_df['TEAM_ID'] == team_id]
        if not row.empty:
            return str(row.iloc[0]['TEAM_ABBREVIATION']).upper()
    profile = find_team_profile(team_id=int(team_id)) if pd.notna(team_id) else None
    if profile:
        return profile.get('abbreviation')
    return None


def _fallback_from_odds(odds_path=ODDS_PATH):
    if not os.path.exists(odds_path):
        return pd.DataFrame()
    try:
        odds = pd.read_csv(odds_path)
    except Exception:
        return pd.DataFrame()
    if odds.empty or not {'COMMENCE_TIME', 'HOME_TEAM', 'AWAY_TEAM'}.issubset(odds.columns):
        return pd.DataFrame()

    odds = odds.copy()
    odds['GAME_DATE'] = pd.to_datetime(odds['COMMENCE_TIME'], errors='coerce', utc=True).dt.tz_convert(None)
    odds = odds.dropna(subset=['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM'])
    if odds.empty:
        return pd.DataFrame()

    rows = []
    for (home_team, away_team, game_date), group in odds.groupby(['HOME_TEAM', 'AWAY_TEAM', 'GAME_DATE']):
        home_profile = find_team_profile(team_name=str(home_team))
        away_profile = find_team_profile(team_name=str(away_team))
        if not home_profile or not away_profile:
            continue
        home_id = int(home_profile['team_id'])
        away_id = int(away_profile['team_id'])
        game_id = f"ODDS_{home_id}_{away_id}_{game_date.strftime('%Y%m%d')}"
        rows.append({
            'GAME_ID': game_id,
            'GAME_DATE': game_date,
            'HOME_TEAM_ID': home_id,
            'AWAY_TEAM_ID': away_id,
            'HOME_TEAM_ABBR': home_profile.get('abbreviation'),
            'AWAY_TEAM_ABBR': away_profile.get('abbreviation'),
        })
    return pd.DataFrame(rows)


def fetch_upcoming_schedule(days_ahead=7, save_path=RAW_PATH):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    start_date = datetime.now(timezone.utc).date()
    rows = []

    for offset in range(days_ahead):
        game_date = start_date + timedelta(days=offset)
        date_str = game_date.strftime("%m/%d/%Y")
        try:
            endpoint = scoreboardv2.ScoreboardV2(
                game_date=date_str,
                day_offset=0,
                league_id='00',
                timeout=30,
            )
            frames = endpoint.get_data_frames()
        except Exception as exc:
            print(f"Warning: failed schedule fetch for {date_str}: {exc}")
            continue

        game_header = pd.DataFrame()
        line_score = pd.DataFrame()
        for frame in frames:
            cols = set(frame.columns)
            if {'GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID'}.issubset(cols):
                game_header = frame.copy()
            elif {'GAME_ID', 'TEAM_ID', 'TEAM_ABBREVIATION'}.issubset(cols):
                line_score = frame.copy()

        if game_header.empty:
            continue

        for _, game in game_header.iterrows():
            try:
                home_team_id = int(game['HOME_TEAM_ID'])
                away_team_id = int(game['VISITOR_TEAM_ID'])
            except Exception:
                continue

            if not is_nba_team_by_id(home_team_id) or not is_nba_team_by_id(away_team_id):
                continue

            rows.append({
                'GAME_ID': str(game['GAME_ID']),
                'GAME_DATE': pd.to_datetime(game.get('GAME_DATE_EST', game_date), errors='coerce'),
                'HOME_TEAM_ID': home_team_id,
                'AWAY_TEAM_ID': away_team_id,
                'HOME_TEAM_ABBR': _team_abbr(home_team_id, line_score),
                'AWAY_TEAM_ABBR': _team_abbr(away_team_id, line_score),
            })

    schedule = pd.DataFrame(rows).drop_duplicates(subset=['GAME_ID'], keep='last')
    if schedule.empty:
        fallback = _fallback_from_odds()
        if not fallback.empty:
            print("Schedule API unavailable; using odds-based fallback schedule rows.")
            schedule = fallback.drop_duplicates(subset=['GAME_ID'], keep='last')
    if not schedule.empty:
        schedule['GAME_DATE'] = pd.to_datetime(schedule['GAME_DATE'], errors='coerce').dt.tz_localize(None)
        schedule = schedule.dropna(subset=['GAME_DATE'])
        schedule = schedule.sort_values(['GAME_DATE', 'GAME_ID'])
    else:
        schedule = pd.DataFrame(columns=[
            'GAME_ID',
            'GAME_DATE',
            'HOME_TEAM_ID',
            'AWAY_TEAM_ID',
            'HOME_TEAM_ABBR',
            'AWAY_TEAM_ABBR',
        ])

    schedule.to_csv(save_path, index=False)
    print(f"Saved upcoming schedule to {save_path} with {len(schedule)} games")
    return schedule


def main():
    fetch_upcoming_schedule(days_ahead=7)


if __name__ == '__main__':
    main()
