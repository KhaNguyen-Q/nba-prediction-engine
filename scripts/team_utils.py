import json
import math
import os
from datetime import datetime
from zoneinfo import ZoneInfo

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'nba_teams.json')
LOCATIONS_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'team_locations.json')


def load_nba_teams(config_path=CONFIG_PATH):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            teams = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"NBA teams config not found: {config_path}")

    return teams


def get_nba_team_ids(config_path=CONFIG_PATH):
    teams = load_nba_teams(config_path)
    return {team['team_id'] for team in teams if 'team_id' in team}


def get_nba_abbreviations(config_path=CONFIG_PATH):
    teams = load_nba_teams(config_path)
    return {team['abbreviation'] for team in teams if 'abbreviation' in team}


def is_nba_team_by_id(team_id, config_path=CONFIG_PATH):
    try:
        return int(team_id) in get_nba_team_ids(config_path)
    except (ValueError, TypeError):
        return False


def is_nba_team_by_abbreviation(abbr, config_path=CONFIG_PATH):
    if not isinstance(abbr, str):
        return False
    return abbr.upper() in get_nba_abbreviations(config_path)


def find_team_profile(team_id=None, abbreviation=None, team_name=None, config_path=CONFIG_PATH):
    teams = load_nba_teams(config_path)
    if team_id is not None:
        for team in teams:
            if team.get('team_id') == int(team_id):
                return team
    if abbreviation is not None:
        abbr_upper = abbreviation.upper()
        for team in teams:
            if team.get('abbreviation', '').upper() == abbr_upper:
                return team
    if team_name is not None:
        name_normalized = team_name.strip().lower()
        for team in teams:
            if team.get('name', '').strip().lower() == name_normalized:
                return team
    return None


def get_team_adult_quality(team_id=None, abbreviation=None, config_path=CONFIG_PATH, default=5):
    team = find_team_profile(team_id=team_id, abbreviation=abbreviation, config_path=config_path)
    if not team:
        return default
    return team.get('adult_quality_rating', default)


def load_team_locations(config_path=LOCATIONS_CONFIG_PATH):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Team locations config not found: {config_path}")


def find_team_location(team_id=None, abbreviation=None, config_path=LOCATIONS_CONFIG_PATH):
    teams = load_team_locations(config_path)
    if team_id is not None:
        for team in teams:
            if team.get('team_id') == int(team_id):
                return team
    if abbreviation is not None:
        abbr_upper = abbreviation.upper()
        for team in teams:
            if team.get('abbreviation', '').upper() == abbr_upper:
                return team
    return None


def get_team_timezone(team_id=None, abbreviation=None, config_path=LOCATIONS_CONFIG_PATH):
    location = find_team_location(team_id=team_id, abbreviation=abbreviation, config_path=config_path)
    if location:
        return location.get('timezone')
    return None


def get_timezone_offset(timezone_name, reference_date=None):
    if timezone_name is None:
        return None
    if reference_date is None:
        reference_date = datetime.utcnow()
    if isinstance(reference_date, str):
        reference_date = datetime.fromisoformat(reference_date)
    tz = ZoneInfo(timezone_name)
    offset = tz.utcoffset(reference_date)
    if offset is None:
        return None
    return offset.total_seconds() / 3600


def get_team_timezone_offset(team_id=None, abbreviation=None, config_path=LOCATIONS_CONFIG_PATH, reference_date=None):
    timezone_name = get_team_timezone(team_id=team_id, abbreviation=abbreviation, config_path=config_path)
    return get_timezone_offset(timezone_name, reference_date=reference_date)


def haversine_distance(lat1, lon1, lat2, lon2):
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def distance_between_team_locations(team_a_id=None, team_a_abbreviation=None, team_b_id=None, team_b_abbreviation=None, config_path=LOCATIONS_CONFIG_PATH):
    a = find_team_location(team_id=team_a_id, abbreviation=team_a_abbreviation, config_path=config_path)
    b = find_team_location(team_id=team_b_id, abbreviation=team_b_abbreviation, config_path=config_path)
    if not a or not b:
        return None
    return haversine_distance(a.get('lat'), a.get('lon'), b.get('lat'), b.get('lon'))


def timezone_difference_between_teams(team_a_id=None, team_a_abbreviation=None, team_b_id=None, team_b_abbreviation=None, reference_date=None, config_path=LOCATIONS_CONFIG_PATH):
    offset_a = get_team_timezone_offset(team_id=team_a_id, abbreviation=team_a_abbreviation, reference_date=reference_date, config_path=config_path)
    offset_b = get_team_timezone_offset(team_id=team_b_id, abbreviation=team_b_abbreviation, reference_date=reference_date, config_path=config_path)
    if offset_a is None or offset_b is None:
        return None
    return abs(offset_a - offset_b)
