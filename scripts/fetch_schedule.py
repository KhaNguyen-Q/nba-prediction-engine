import os
import sys
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from nba_api.stats.endpoints import scoreboardv2

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.team_utils import find_team_profile, is_nba_team_by_id


RAW_PATH = "data/raw/upcoming_games.csv"
ODDS_PATH = "data/raw/odds_raw.csv"
STATUS_PATH = "data/raw/schedule_source_status.json"
SCOREBOARD_TIMEOUT_SECONDS = int(os.environ.get("SCOREBOARD_TIMEOUT_SECONDS", "12"))
ESPN_SCOREBOARD_URL = os.environ.get(
    "ESPN_SCOREBOARD_URL",
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
)
ESPN_TIMEOUT_SECONDS = int(os.environ.get("ESPN_TIMEOUT_SECONDS", "10"))


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


def _fallback_from_espn(days_ahead):
    rows = []
    start_date = datetime.now(timezone.utc).date()
    for offset in range(days_ahead):
        game_date = start_date + timedelta(days=offset)
        date_key = game_date.strftime("%Y%m%d")
        try:
            resp = requests.get(
                ESPN_SCOREBOARD_URL,
                params={"dates": date_key},
                timeout=ESPN_TIMEOUT_SECONDS,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            print(f"Warning: ESPN schedule fallback failed for {date_key}: {exc}")
            continue

        for event in payload.get("events", []) if isinstance(payload, dict) else []:
            competitors = (
                event.get("competitions", [{}])[0].get("competitors", [])
                if isinstance(event, dict)
                else []
            )
            home = None
            away = None
            for comp in competitors:
                team_info = comp.get("team", {}) or {}
                abbr = str(team_info.get("abbreviation", "")).upper()
                profile = find_team_profile(abbreviation=abbr)
                if not profile:
                    continue
                item = {
                    "team_id": int(profile["team_id"]),
                    "abbr": profile.get("abbreviation"),
                }
                if str(comp.get("homeAway", "")).lower() == "home":
                    home = item
                elif str(comp.get("homeAway", "")).lower() == "away":
                    away = item
            if not home or not away:
                continue
            rows.append({
                "GAME_ID": str(event.get("id", f"ESPN_{away['team_id']}_{home['team_id']}_{date_key}")),
                "GAME_DATE": pd.to_datetime(event.get("date"), errors="coerce"),
                "HOME_TEAM_ID": int(home["team_id"]),
                "AWAY_TEAM_ID": int(away["team_id"]),
                "HOME_TEAM_ABBR": home.get("abbr"),
                "AWAY_TEAM_ABBR": away.get("abbr"),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["GAME_ID"], keep="last")


def _write_schedule_status(status_path, payload):
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def fetch_upcoming_schedule(days_ahead=7, save_path=RAW_PATH, status_path=STATUS_PATH):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    existing_non_empty = pd.DataFrame()
    if os.path.exists(save_path):
        try:
            existing = pd.read_csv(save_path)
            if not existing.empty and {"GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID"}.issubset(existing.columns):
                existing["GAME_DATE"] = pd.to_datetime(existing["GAME_DATE"], errors="coerce")
                existing = existing.dropna(subset=["GAME_DATE"]).copy()
                if not existing.empty:
                    existing_non_empty = existing
        except Exception:
            existing_non_empty = pd.DataFrame()

    start_date = datetime.now(timezone.utc).date()
    rows = []
    nba_api_success_days = 0
    nba_api_fail_days = 0

    for offset in range(days_ahead):
        game_date = start_date + timedelta(days=offset)
        date_str = game_date.strftime("%m/%d/%Y")
        try:
            endpoint = scoreboardv2.ScoreboardV2(
                game_date=date_str,
                day_offset=0,
                league_id='00',
                timeout=SCOREBOARD_TIMEOUT_SECONDS,
            )
            frames = endpoint.get_data_frames()
            nba_api_success_days += 1
        except Exception as exc:
            print(f"Warning: failed schedule fetch for {date_str}: {exc}")
            nba_api_fail_days += 1
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
    source_used = "nba_api"
    source_details = {
        "nba_api_success_days": int(nba_api_success_days),
        "nba_api_fail_days": int(nba_api_fail_days),
    }
    if schedule.empty:
        espn_fallback = _fallback_from_espn(days_ahead=days_ahead)
        if not espn_fallback.empty:
            print("NBA schedule API unavailable; using ESPN scoreboard fallback rows.")
            schedule = espn_fallback
            source_used = "espn_fallback"
        fallback = _fallback_from_odds()
        if schedule.empty and not fallback.empty:
            print("Schedule API unavailable; using odds-based fallback schedule rows.")
            schedule = fallback.drop_duplicates(subset=['GAME_ID'], keep='last')
            source_used = "odds_fallback"
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

    if schedule.empty and not existing_non_empty.empty:
        existing_non_empty = existing_non_empty.sort_values(["GAME_DATE", "GAME_ID"])
        existing_non_empty.to_csv(save_path, index=False)
        _write_schedule_status(status_path, {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_used": "retained_previous",
            "rows": int(len(existing_non_empty)),
            "details": source_details,
            "notes": "No source returned rows; retained last non-empty schedule file.",
        })
        print(
            f"Schedule fetch returned 0 rows; retained previous non-empty schedule at {save_path} "
            f"with {len(existing_non_empty)} games"
        )
        return existing_non_empty

    schedule.to_csv(save_path, index=False)
    _write_schedule_status(status_path, {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_used": source_used,
        "rows": int(len(schedule)),
        "details": source_details,
    })
    print(f"Saved upcoming schedule to {save_path} with {len(schedule)} games")
    return schedule


def main():
    fetch_upcoming_schedule(days_ahead=7)


if __name__ == '__main__':
    main()
