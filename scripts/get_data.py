import os
import re
import sys
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from nba_api.stats.endpoints import leaguegamefinder, commonallplayers, playergamelog

# Add project root to sys.path so imports work when scripts are executed directly.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.team_utils import (
    is_nba_team_by_id,
    is_nba_team_by_abbreviation,
    get_nba_abbreviations,
    find_team_profile,
)

RAW_DIR = "data/raw"
ESPN_INJURY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
NBA_STATS_HEALTH_URL = os.environ.get("NBA_STATS_HEALTH_URL", "https://stats.nba.com/stats/scoreboardv3")
NBA_STATS_HEALTH_TIMEOUT = int(os.environ.get("NBA_STATS_HEALTH_TIMEOUT_SECONDS", "6"))
NBA_STATS_REQUEST_TIMEOUT = int(os.environ.get("NBA_STATS_REQUEST_TIMEOUT_SECONDS", "20"))
ODDS_API_SPORT = os.environ.get('ODDS_API_SPORT', 'basketball_nba')
ODDS_API_BASE_URL = os.environ.get('ODDS_API_BASE_URL', 'https://api.the-odds-api.com/v4')
ODDS_API_URL = os.environ.get(
    'ODDS_API_URL',
    f"{ODDS_API_BASE_URL}/sports/{ODDS_API_SPORT}/odds/"
)
ODDS_API_CACHE_PATH = os.environ.get('ODDS_API_CACHE_PATH')
ODDS_USE_CACHE = os.environ.get('ODDS_USE_CACHE', '0').lower() in ('1', 'true', 'yes')
ODDS_SAVE_CACHE = os.environ.get('ODDS_SAVE_CACHE', '1').lower() in ('1', 'true', 'yes')
ODDS_API_KEY_HEADER = os.environ.get('ODDS_API_KEY_HEADER', 'x-api-key')
ODDS_API_HOST = os.environ.get('ODDS_API_HOST')
PLAYER_LOG_SLEEP_SECONDS = float(os.environ.get('PLAYER_LOG_SLEEP_SECONDS', '0.6'))
PLAYER_LOGS_MAX_PLAYERS = int(os.environ.get('PLAYER_LOGS_MAX_PLAYERS', '0'))
PLAYER_LOG_TIMEOUT_SECONDS = int(os.environ.get('PLAYER_LOG_TIMEOUT_SECONDS', '30'))
PLAYER_LOG_MAX_RETRIES = int(os.environ.get('PLAYER_LOG_MAX_RETRIES', '3'))
PLAYER_LOG_RETRY_BACKOFF_SECONDS = float(os.environ.get('PLAYER_LOG_RETRY_BACKOFF_SECONDS', '1.5'))


def ensure_data_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)


def _stats_api_is_reachable(timeout_seconds=NBA_STATS_HEALTH_TIMEOUT):
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nba.com/"}
    try:
        resp = requests.get(NBA_STATS_HEALTH_URL, timeout=timeout_seconds, headers=headers)
        return 200 <= resp.status_code < 500
    except Exception:
        return False


def _fallback_existing_or_empty(save_path, columns, label):
    if os.path.exists(save_path):
        try:
            cached = pd.read_csv(save_path)
            print(f"Using cached {label} from {save_path} with {len(cached)} rows")
            return cached
        except Exception:
            pass
    empty = pd.DataFrame(columns=columns)
    empty.to_csv(save_path, index=False)
    print(f"Saved empty {label} to {save_path}")
    return empty


def _get_current_season(reference_date=None):
    if reference_date is None:
        reference_date = datetime.utcnow()
    if isinstance(reference_date, str):
        reference_date = datetime.fromisoformat(reference_date)
    year = reference_date.year
    if reference_date.month < 10:
        year -= 1
    return f"{year}-{str(year + 1)[-2:]}"


def _parse_athlete_id(athlete):
    if not isinstance(athlete, dict):
        return None
    if athlete.get('id') is not None:
        try:
            return int(athlete['id'])
        except (ValueError, TypeError):
            pass
    for link in athlete.get('links', []) if isinstance(athlete.get('links'), list) else []:
        href = link.get('href')
        if isinstance(href, str):
            match = re.search(r'/id/(\d+)', href)
            if match:
                return int(match.group(1))
    return None


def _map_injury_severity(injury):
    if not isinstance(injury, dict):
        return 0.0
    status = str(injury.get('status', '')).strip().lower()
    fantasy_status = str(injury.get('details', {}).get('fantasyStatus', {}).get('abbreviation', '')).strip().upper()
    if fantasy_status == 'OUT' or status in {'out', 'out for season', 'doubtful', 'suspended', 'inactive'}:
        return 2.0
    if fantasy_status in {'GTD', 'QUESTIONABLE', 'Q'} or status in {'day-to-day', 'questionable'}:
        return 1.2
    if fantasy_status in {'PROBABLE', 'P'} or status in {'probable'}:
        return 0.5
    return 0.0


def fetch_games_data(save_path=os.path.join(RAW_DIR, "games_raw.csv")):
    print("Fetching games from NBA API...")
    if not _stats_api_is_reachable():
        print("Warning: stats.nba.com unreachable; skipping games fetch.")
        return _fallback_existing_or_empty(
            save_path=save_path,
            columns=['TEAM_ID', 'TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE'],
            label='games data',
        )
    try:
        games = leaguegamefinder.LeagueGameFinder(timeout=NBA_STATS_REQUEST_TIMEOUT).get_data_frames()[0]
    except Exception as exc:
        print(f"Warning: failed to fetch games from NBA API: {exc}")
        return _fallback_existing_or_empty(
            save_path=save_path,
            columns=['TEAM_ID', 'TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE'],
            label='games data',
        )

    # Filter to only the 30 NBA franchises (exclude overseas and affiliate teams)
    if 'TEAM_ID' in games.columns:
        games = games[games['TEAM_ID'].apply(is_nba_team_by_id)]
    elif 'TEAM_ABBREVIATION' in games.columns:
        # fallback: keep only known NBA abbreviations when team id is not present
        nba_abbr = get_nba_abbreviations()
        games = games[games['TEAM_ABBREVIATION'].astype(str).str.upper().isin(nba_abbr)]

    games.to_csv(save_path, index=False)
    print(f"Saved raw games to {save_path} (filtered to {len(games)} NBA rows)")
    return games


def fetch_players_data(save_path=os.path.join(RAW_DIR, "players_raw.csv")):
    ensure_data_dirs()
    season = _get_current_season()
    print(f"Fetching current NBA players for season {season}...")

    if not _stats_api_is_reachable():
        print("Warning: stats.nba.com unreachable; skipping player fetch.")
        return _fallback_existing_or_empty(
            save_path=save_path,
            columns=['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'ROSTER_STATUS'],
            label='players data',
        )
    try:
        players_df = commonallplayers.CommonAllPlayers(
            season=season,
            league_id='00',
            is_only_current_season=1,
            timeout=NBA_STATS_REQUEST_TIMEOUT,
        ).get_data_frames()[0]
    except Exception as exc:
        print(f"Warning: failed to fetch player data from NBA API: {exc}")
        return _fallback_existing_or_empty(
            save_path=save_path,
            columns=['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'ROSTER_STATUS'],
            label='players data',
        )

    if not players_df.empty:
        if 'TEAM_ID' in players_df.columns:
            players_df = players_df[players_df['TEAM_ID'].apply(is_nba_team_by_id)]
        elif 'TEAM_ABBREVIATION' in players_df.columns:
            nba_abbr = get_nba_abbreviations()
            players_df = players_df[players_df['TEAM_ABBREVIATION'].astype(str).str.upper().isin(nba_abbr)]

        rename_map = {
            'PERSON_ID': 'PLAYER_ID',
            'DISPLAY_FIRST_LAST': 'PLAYER_NAME',
            'DISPLAY_LAST_COMMA_FIRST': 'PLAYER_NAME',
            'ROSTERSTATUS': 'ROSTER_STATUS',
        }
        columns = [
            'PLAYER_ID',
            'PLAYER_NAME',
            'TEAM_ID',
            'TEAM_ABBREVIATION',
            'ROSTER_STATUS',
        ]
        players_df = players_df.rename(columns=rename_map)
        players_df = players_df[[c for c in columns if c in players_df.columns]]
    else:
        players_df = pd.DataFrame(columns=[
            'PLAYER_ID',
            'PLAYER_NAME',
            'TEAM_ID',
            'TEAM_ABBREVIATION',
            'ROSTER_STATUS',
        ])

    players_df.to_csv(save_path, index=False)
    print(f"Saved players data to {save_path} with {len(players_df)} rows")
    return players_df


def fetch_player_game_logs_data(
    players_path=os.path.join(RAW_DIR, "players_raw.csv"),
    save_path=os.path.join(RAW_DIR, "player_game_logs_raw.csv")
):
    ensure_data_dirs()
    season = _get_current_season()
    logs_frames = []

    if not _stats_api_is_reachable():
        print("Warning: stats.nba.com unreachable; skipping player log fetch.")
        return _fallback_existing_or_empty(
            save_path=save_path,
            columns=[
                'PLAYER_ID', 'TEAM_ID', 'GAME_ID', 'GAME_DATE', 'MATCHUP',
                'MIN', 'PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
                'PLUS_MINUS', 'WL'
            ],
            label='player logs',
        )

    if os.path.exists(players_path):
        players_df = pd.read_csv(players_path)
    else:
        players_df = fetch_players_data(players_path)

    if players_df.empty or 'PLAYER_ID' not in players_df.columns:
        empty = pd.DataFrame(columns=[
            'PLAYER_ID', 'TEAM_ID', 'GAME_ID', 'GAME_DATE', 'MATCHUP',
            'MIN', 'PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
            'PLUS_MINUS', 'WL'
        ])
        empty.to_csv(save_path, index=False)
        print(f"Saved empty player logs to {save_path} (no players available)")
        return empty

    players_df = players_df.dropna(subset=['PLAYER_ID']).copy()
    players_df['PLAYER_ID'] = pd.to_numeric(players_df['PLAYER_ID'], errors='coerce')
    players_df = players_df.dropna(subset=['PLAYER_ID'])
    players_df['PLAYER_ID'] = players_df['PLAYER_ID'].astype(int)
    if PLAYER_LOGS_MAX_PLAYERS > 0:
        players_df = players_df.head(PLAYER_LOGS_MAX_PLAYERS)

    total = len(players_df)
    print(f"Fetching player game logs for {total} players (season {season})...")
    for idx, player in players_df.iterrows():
        player_id = int(player['PLAYER_ID'])
        frame = None
        last_exc = None
        for attempt in range(1, PLAYER_LOG_MAX_RETRIES + 1):
            try:
                frame = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=season,
                    season_type_all_star='Regular Season',
                    timeout=PLAYER_LOG_TIMEOUT_SECONDS,
                ).get_data_frames()[0]
                break
            except Exception as exc:
                last_exc = exc
                if attempt < PLAYER_LOG_MAX_RETRIES:
                    wait = PLAYER_LOG_RETRY_BACKOFF_SECONDS * attempt
                    print(
                        f"Warning: player log fetch retry {attempt}/{PLAYER_LOG_MAX_RETRIES - 1} "
                        f"for PLAYER_ID={player_id} after error: {exc}"
                    )
                    time.sleep(wait)
                else:
                    print(f"Warning: failed player log fetch for PLAYER_ID={player_id}: {last_exc}")

        if frame is not None and not frame.empty:
            if 'Player_ID' in frame.columns and 'PLAYER_ID' not in frame.columns:
                frame = frame.rename(columns={'Player_ID': 'PLAYER_ID'})
            frame['PLAYER_ID'] = player_id
            if 'TEAM_ID' in player:
                frame['TEAM_ID'] = player.get('TEAM_ID')
            logs_frames.append(frame)

        # Keep polite request cadence to reduce rate-limit failures.
        if PLAYER_LOG_SLEEP_SECONDS > 0:
            time.sleep(PLAYER_LOG_SLEEP_SECONDS)
        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"Fetched player logs for {idx + 1}/{total} players")

    if logs_frames:
        logs = pd.concat(logs_frames, ignore_index=True)
        if 'GAME_DATE' in logs.columns:
            logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'], errors='coerce')
            logs = logs.dropna(subset=['GAME_DATE'])
    else:
        logs = pd.DataFrame(columns=[
            'PLAYER_ID', 'TEAM_ID', 'GAME_ID', 'GAME_DATE', 'MATCHUP',
            'MIN', 'PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
            'PLUS_MINUS', 'WL'
        ])

    logs.to_csv(save_path, index=False)
    print(f"Saved player game logs to {save_path} with {len(logs)} rows")
    return logs


def _status_tokens(*values):
    text = " ".join([str(v or "") for v in values]).strip().lower()
    return text


def _is_unavailable_status(status_text):
    text = str(status_text or "").strip().lower()
    out_tokens = [
        "out", "inactive", "suspended", "doubtful", "out for season",
        "ruled out", "will not play", "won't play", "not expected to play",
    ]
    return any(token in text for token in out_tokens)


def fetch_injuries_data(save_path=os.path.join(RAW_DIR, "injuries_raw.csv"), team_ids=None):
    ensure_data_dirs()
    print(f"Fetching injuries data from ESPN: {ESPN_INJURY_URL}")
    rows = []

    try:
        response = requests.get(ESPN_INJURY_URL, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        payload = response.json()
        for team_data in payload.get('injuries', []):
            team_name = team_data.get('displayName')
            for injury in team_data.get('injuries', []):
                athlete = injury.get('athlete', {}) or {}
                athlete_team = athlete.get('team', {}) or {}
                team_abbr = athlete_team.get('abbreviation') or ''
                team_profile = find_team_profile(abbreviation=team_abbr, team_name=team_name)
                team_id = team_profile.get('team_id') if team_profile else None
                player_name = athlete.get('displayName') or athlete.get('shortName') or ''
                player_id = _parse_athlete_id(athlete)
                # Keep report date and return date separate so downstream logic can model active periods.
                injury_date = injury.get('date')
                if injury_date is None:
                    continue
                rows.append({
                    'TEAM_ID': team_id,
                    'TEAM_ABBREVIATION': team_abbr,
                    'TEAM_NAME': team_name,
                    'PLAYER_ID': player_id,
                    'PLAYER_NAME': player_name,
                    'GAME_DATE': injury_date,
                    'INJURY_STATUS': injury.get('status'),
                    'INJURY_TYPE': injury.get('details', {}).get('type'),
                    'INJURY_LOCATION': injury.get('details', {}).get('location'),
                    'INJURY_DETAIL': injury.get('details', {}).get('detail'),
                    'INJURY_RETURN_DATE': injury.get('details', {}).get('returnDate'),
                    'INJURY_SEVERITY': _map_injury_severity(injury),
                    'FANTASY_STATUS': injury.get('details', {}).get('fantasyStatus', {}).get('abbreviation'),
                    'SHORT_COMMENT': injury.get('shortComment'),
                    'LONG_COMMENT': injury.get('longComment'),
                    'DATA_SOURCE': 'ESPN',
                    'FETCHED_AT_UTC': datetime.utcnow().isoformat(),
                })
    except requests.RequestException as err:
        print(f"Warning: failed to fetch injuries data: {err}")
    except ValueError as err:
        print(f"Warning: invalid JSON from injuries source: {err}")

    expected_columns = [
        'TEAM_ID',
        'TEAM_ABBREVIATION',
        'TEAM_NAME',
        'PLAYER_ID',
        'PLAYER_NAME',
        'GAME_DATE',
        'INJURY_STATUS',
        'INJURY_TYPE',
        'INJURY_LOCATION',
        'INJURY_DETAIL',
        'INJURY_RETURN_DATE',
        'INJURY_SEVERITY',
        'FANTASY_STATUS',
        'SHORT_COMMENT',
        'LONG_COMMENT',
        'DATA_SOURCE',
        'FETCHED_AT_UTC',
        'IS_UNAVAILABLE',
        'AVAILABILITY_LABEL',
    ]
    injuries = pd.DataFrame(rows)
    if not injuries.empty:
        injuries['GAME_DATE'] = pd.to_datetime(injuries['GAME_DATE'], errors='coerce')
        injuries['INJURY_RETURN_DATE'] = pd.to_datetime(injuries['INJURY_RETURN_DATE'], errors='coerce')
        injuries = injuries.dropna(subset=['GAME_DATE'])
        injuries['GAME_DATE'] = injuries['GAME_DATE'].dt.normalize()
        injuries['INJURY_RETURN_DATE'] = injuries['INJURY_RETURN_DATE'].dt.normalize()
        if team_ids:
            team_ids = {int(t) for t in team_ids}
            injuries = injuries[injuries['TEAM_ID'].apply(lambda x: pd.notna(x) and int(x) in team_ids)].copy()
        status_text = injuries.apply(
            lambda r: _status_tokens(
                r.get('INJURY_STATUS'),
                r.get('FANTASY_STATUS'),
                r.get('SHORT_COMMENT'),
                r.get('LONG_COMMENT'),
            ),
            axis=1,
        )
        injuries['IS_UNAVAILABLE'] = status_text.apply(_is_unavailable_status)
        injuries['AVAILABILITY_LABEL'] = np.where(
            injuries['IS_UNAVAILABLE'],
            'Out',
            np.where(
                injuries['INJURY_SEVERITY'] >= 1.2,
                'Questionable',
                np.where(injuries['INJURY_SEVERITY'] >= 0.5, 'Probable', 'Available')
            )
        )
    if injuries.empty:
        injuries = pd.DataFrame(columns=expected_columns)
    else:
        for col in expected_columns:
            if col not in injuries.columns:
                injuries[col] = pd.NA
        injuries = injuries[expected_columns]

    injuries.to_csv(save_path, index=False)
    print(f"Saved injuries data to {save_path} with {len(injuries)} rows")
    return injuries


def build_latest_availability_snapshot(
    injuries_df=None,
    raw_path=os.path.join(RAW_DIR, "injuries_raw.csv"),
    save_path=os.path.join(RAW_DIR, "injuries_latest.csv"),
):
    ensure_data_dirs()
    if injuries_df is None:
        if not os.path.exists(raw_path):
            df = pd.DataFrame()
            df.to_csv(save_path, index=False)
            print(f"Saved empty availability snapshot to {save_path}")
            return df
        injuries_df = pd.read_csv(raw_path)

    if injuries_df.empty:
        injuries_df.to_csv(save_path, index=False)
        print(f"Saved empty availability snapshot to {save_path}")
        return injuries_df

    work = injuries_df.copy()
    for col in ['TEAM_ID', 'PLAYER_ID']:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors='coerce')
    if 'GAME_DATE' in work.columns:
        work['GAME_DATE'] = pd.to_datetime(work['GAME_DATE'], errors='coerce')
    if 'FETCHED_AT_UTC' in work.columns:
        work['FETCHED_AT_UTC'] = pd.to_datetime(work['FETCHED_AT_UTC'], errors='coerce')
    sort_cols = [c for c in ['FETCHED_AT_UTC', 'GAME_DATE'] if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols)
    key_col = 'PLAYER_ID' if 'PLAYER_ID' in work.columns else 'PLAYER_NAME'
    if key_col not in work.columns:
        work.to_csv(save_path, index=False)
        print(f"Saved availability snapshot to {save_path} with {len(work)} rows")
        return work
    latest = work.dropna(subset=[key_col]).drop_duplicates(subset=[key_col], keep='last').copy()
    latest.to_csv(save_path, index=False)
    print(f"Saved availability snapshot to {save_path} with {len(latest)} rows")
    return latest


def _normalize_the_odds_api_payload(payload):
    rows = []
    if not isinstance(payload, list):
        return rows
    for event in payload:
        commence = event.get('commence_time')
        home_team = event.get('home_team')
        away_team = event.get('away_team')
        for bookmaker in event.get('bookmakers', []):
            for market in bookmaker.get('markets', []):
                market_key = market.get('key')
                for outcome in market.get('outcomes', []):
                    rows.append({
                        'COMMENCE_TIME': commence,
                        'HOME_TEAM': home_team,
                        'AWAY_TEAM': away_team,
                        'BOOKMAKER': bookmaker.get('title'),
                        'MARKET': market_key,
                        'OUTCOME_NAME': outcome.get('name'),
                        'POINT': outcome.get('point'),
                        'PRICE': outcome.get('price'),
                        'ODDS_SOURCE': bookmaker.get('key'),
                    })
    return rows


def _load_cached_odds_data(cache_path):
    if not cache_path or not os.path.exists(cache_path):
        return None
    print(f"Loading cached odds from {cache_path}")
    try:
        if cache_path.lower().endswith('.json'):
            with open(cache_path, 'r', encoding='utf-8') as fh:
                payload = json.load(fh)
            if isinstance(payload, dict) and 'response' in payload:
                return payload['response']
            return payload
        return pd.read_csv(cache_path)
    except Exception as err:
        print(f"Warning: failed to load cached odds data: {err}")
        return None


def _save_cached_odds_data(payload, cache_path):
    if not cache_path or not ODDS_SAVE_CACHE:
        return
    try:
        if cache_path.lower().endswith('.json'):
            if isinstance(payload, pd.DataFrame):
                payload = payload.to_dict(orient='records')
            with open(cache_path, 'w', encoding='utf-8') as fh:
                json.dump(payload, fh, indent=2)
            print(f"Saved odds cache to {cache_path}")
            return
        if isinstance(payload, pd.DataFrame):
            payload.to_csv(cache_path, index=False)
            print(f"Saved odds cache to {cache_path}")
            return
        if isinstance(payload, list):
            df = pd.DataFrame(_normalize_the_odds_api_payload(payload))
            df.to_csv(cache_path, index=False)
            print(f"Saved odds cache to {cache_path}")
            return
        if isinstance(payload, dict) and 'response' in payload:
            df = pd.json_normalize(payload['response'])
            df.to_csv(cache_path, index=False)
            print(f"Saved odds cache to {cache_path}")
            return
    except Exception as err:
        print(f"Warning: failed to save cached odds data: {err}")


def fetch_odds_data(save_path=os.path.join(RAW_DIR, "odds_raw.csv")):
    ensure_data_dirs()
    odds_api_url = os.environ.get('ODDS_API_URL', ODDS_API_URL)
    api_key = os.environ.get('ODDS_API_KEY')
    api_key_header = os.environ.get('ODDS_API_KEY_HEADER', ODDS_API_KEY_HEADER)
    api_host = os.environ.get('ODDS_API_HOST', ODDS_API_HOST)
    odds_df = pd.DataFrame()

    if ODDS_API_CACHE_PATH and ODDS_USE_CACHE:
        cached = _load_cached_odds_data(ODDS_API_CACHE_PATH)
        if cached is not None:
            print("Using local odds cache instead of remote request.")
            if isinstance(cached, pd.DataFrame):
                odds_df = cached
            else:
                if "the-odds-api.com" in odds_api_url:
                    odds_df = pd.DataFrame(_normalize_the_odds_api_payload(cached))
                else:
                    if isinstance(cached, dict) and 'response' in cached:
                        cached = cached['response']
                    try:
                        odds_df = pd.json_normalize(cached)
                    except Exception as err:
                        print(f"Warning: could not normalize cached odds payload: {err}")
                        odds_df = pd.DataFrame()
            odds_df.to_csv(save_path, index=False)
            print(f"Saved odds data to {save_path} from cache with {len(odds_df)} rows")
            return odds_df

    if api_key:
        print(f"Fetching odds data from {odds_api_url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        params = None

        if "the-odds-api.com" in odds_api_url:
            params = {
                'regions': 'us',
                'markets': 'spreads,h2h,totals',
                'oddsFormat': 'american',
                'dateFormat': 'iso',
                'apiKey': api_key,
            }
        else:
            headers[api_key_header] = api_key
            if api_host:
                headers['x-apisports-host'] = api_host

        try:
            response = requests.get(odds_api_url, params=params, timeout=20, headers=headers)
            response.raise_for_status()
            payload = response.json()
            if "the-odds-api.com" in odds_api_url:
                odds_df = pd.DataFrame(_normalize_the_odds_api_payload(payload))
            else:
                if isinstance(payload, dict) and 'response' in payload:
                    payload = payload['response']
                try:
                    odds_df = pd.json_normalize(payload)
                except Exception as err:
                    print(f"Warning: could not normalize odds provider JSON payload: {err}")
                    odds_df = pd.DataFrame()
            if ODDS_API_CACHE_PATH and ODDS_SAVE_CACHE:
                _save_cached_odds_data(payload, ODDS_API_CACHE_PATH)
        except requests.RequestException as err:
            print(f"Warning: failed to fetch odds from provider: {err}")
        except ValueError as err:
            print(f"Warning: invalid JSON from odds provider: {err}")
    else:
        print("No ODDS_API_KEY found; creating placeholder odds file.")

    if odds_df.empty:
        odds_df = pd.DataFrame(columns=[
            'COMMENCE_TIME',
            'HOME_TEAM',
            'AWAY_TEAM',
            'BOOKMAKER',
            'MARKET',
            'OUTCOME_NAME',
            'POINT',
            'PRICE',
            'ODDS_SOURCE',
        ])

    odds_df.to_csv(save_path, index=False)
    print(f"Saved odds data to {save_path} with {len(odds_df)} rows")
    return odds_df


def main():
    ensure_data_dirs()
    fetch_games_data()
    fetch_players_data()
    fetch_player_game_logs_data()
    injuries = fetch_injuries_data()
    build_latest_availability_snapshot(injuries_df=injuries)
    fetch_odds_data()


if __name__ == '__main__':
    main()
