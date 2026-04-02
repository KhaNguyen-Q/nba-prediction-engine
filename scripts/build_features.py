import os
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path so imports work when scripts are executed directly.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.team_utils import (
    is_nba_team_by_id,
    get_team_adult_quality,
    find_team_location,
    get_team_timezone_offset,
    haversine_distance,
    find_team_profile,
)

RAW_PATH = "data/raw/games_raw.csv"
PROCESSED_PATH = "data/processed/games_with_features.csv"


def load_raw_games(path=RAW_PATH):
    df = pd.read_csv(path)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
    df = df.dropna(subset=['GAME_DATE'])
    if 'TEAM_ID' in df.columns:
        df = df[df['TEAM_ID'].apply(is_nba_team_by_id)]
    return df


def normalize_team_ids(df):
    df['TEAM_ID'] = df['TEAM_ID'].astype(int)
    return df


def add_label(df):
    if 'WL' in df.columns:
        df['WIN'] = df['WL'].map({'W': 1, 'L': 0})
    elif 'PLUS_MINUS' in df.columns:
        df['WIN'] = (df['PLUS_MINUS'] > 0).astype(int)
    else:
        df['WIN'] = np.nan
    return df


def add_matchup_features(df):
    if 'MATCHUP' not in df.columns:
        df['IS_AWAY'] = 0
        df['HOME'] = 1
        df['OPPONENT'] = None
        return df

    matchup = df['MATCHUP'].astype(str)
    df['IS_AWAY'] = matchup.str.contains('@').astype(int)
    df['HOME'] = matchup.str.contains(r'vs\.').astype(int)

    def get_opponent(matchup):
        if not isinstance(matchup, str):
            return None
        parts = matchup.split()
        if len(parts) >= 3 and parts[1] in ['@', 'vs.', 'vs']:
            return parts[-1]
        return None

    df['OPPONENT'] = matchup.apply(get_opponent)
    return df


def add_game_location_features(df):
    df = add_matchup_features(df)

    def get_current_location(abbreviation, is_away, opponent):
        if is_away == 1 and isinstance(opponent, str):
            return find_team_location(abbreviation=opponent)
        if isinstance(abbreviation, str):
            return find_team_location(abbreviation=abbreviation)
        return None

    def get_location_values(row):
        loc = get_current_location(row['TEAM_ABBREVIATION'], row['IS_AWAY'], row['OPPONENT'])
        if not isinstance(loc, dict):
            return pd.Series([None, None, None, None])
        return pd.Series([loc.get('city'), loc.get('state'), loc.get('lat'), loc.get('lon')])

    coords = df.apply(get_location_values, axis=1)
    coords.columns = ['CURRENT_CITY', 'CURRENT_STATE', 'CURRENT_LAT', 'CURRENT_LON']
    df = pd.concat([df, coords], axis=1)
    return df


def _american_odds_to_probability(price):
    try:
        price = float(price)
    except (ValueError, TypeError):
        return None
    if price > 0:
        return 100.0 / (price + 100.0)
    return -price / (-price + 100.0)


def _aggregate_odds_by_game(odds_df):
    if odds_df.empty:
        return pd.DataFrame()

    odds_df['GAME_DATE'] = pd.to_datetime(odds_df['COMMENCE_TIME'], errors='coerce').dt.date
    odds_df = odds_df.dropna(subset=['GAME_DATE'])

    rows = []
    for (home_team, away_team, game_date), group in odds_df.groupby(['HOME_TEAM', 'AWAY_TEAM', 'GAME_DATE']):
        h2h = group[group['MARKET'] == 'h2h']
        home_ml = h2h[h2h['OUTCOME_NAME'] == home_team]['PRICE']
        away_ml = h2h[h2h['OUTCOME_NAME'] == away_team]['PRICE']

        home_ml_price = home_ml.mean() if not home_ml.empty else None
        away_ml_price = away_ml.mean() if not away_ml.empty else None
        home_ml_prob = home_ml.map(_american_odds_to_probability).dropna().mean() if not home_ml.empty else None
        away_ml_prob = away_ml.map(_american_odds_to_probability).dropna().mean() if not away_ml.empty else None

        spreads = group[group['MARKET'] == 'spreads']
        home_spread_point = spreads[spreads['OUTCOME_NAME'] == home_team]['POINT'].mean() if not spreads.empty else None
        away_spread_point = spreads[spreads['OUTCOME_NAME'] == away_team]['POINT'].mean() if not spreads.empty else None

        totals = group[group['MARKET'] == 'totals']
        total_point = totals['POINT'].mean() if not totals.empty else None

        rows.append({
            'HOME_TEAM': home_team,
            'AWAY_TEAM': away_team,
            'GAME_DATE': game_date,
            'HOME_ML_PRICE': home_ml_price,
            'AWAY_ML_PRICE': away_ml_price,
            'HOME_ML_PROB': home_ml_prob,
            'AWAY_ML_PROB': away_ml_prob,
            'HOME_SPREAD_POINT': home_spread_point,
            'AWAY_SPREAD_POINT': away_spread_point,
            'TOTAL_POINT': total_point,
            'H2H_BOOKMAKERS_COUNT': h2h['BOOKMAKER'].nunique(),
            'SPREAD_BOOKMAKERS_COUNT': spreads['BOOKMAKER'].nunique(),
            'TOTALS_BOOKMAKERS_COUNT': totals['BOOKMAKER'].nunique(),
        })

    return pd.DataFrame(rows)


def load_odds_features(path='data/raw/odds_raw.csv'):
    if not os.path.exists(path):
        return pd.DataFrame()
    odds_df = pd.read_csv(path)
    if odds_df.empty:
        return pd.DataFrame()
    return _aggregate_odds_by_game(odds_df)


def _find_full_team_name(abbreviation):
    profile = find_team_profile(abbreviation=abbreviation)
    if profile:
        return profile.get('name')
    return None


def add_odds_features(df, path='data/raw/odds_raw.csv'):
    odds_features = load_odds_features(path)
    print(f"[build_features] loaded {len(odds_features)} aggregated odds rows from {path}")
    if odds_features.empty:
        print("[build_features] no odds features to merge; processed dataset will remain without odds fields.")
        return df

    if 'GAME_DATE' in odds_features.columns:
        odds_date_values = pd.to_datetime(odds_features['GAME_DATE'], errors='coerce').dropna().dt.date
        if not odds_date_values.empty:
            odds_dates = odds_date_values.unique()
            print(f"[build_features] odds date range: {odds_dates.min()} to {odds_dates.max()} ({len(odds_dates)} unique dates)")

    df = add_matchup_features(df)
    if 'GAME_DATE' in df.columns:
        game_date_values = pd.to_datetime(df['GAME_DATE'], errors='coerce').dropna().dt.date
        if not game_date_values.empty:
            print(f"[build_features] processed game date range: {game_date_values.min()} to {game_date_values.max()} ({game_date_values.nunique()} unique dates)")
    df['OPPONENT_NAME'] = df['OPPONENT'].apply(lambda x: _find_full_team_name(x) if isinstance(x, str) else None)
    df['HOME_TEAM_NAME'] = np.where(df['HOME'] == 1, df['TEAM_NAME'], df['OPPONENT_NAME'])
    df['AWAY_TEAM_NAME'] = np.where(df['IS_AWAY'] == 1, df['TEAM_NAME'], df['OPPONENT_NAME'])
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce').dt.date

    df = df.merge(
        odds_features,
        how='left',
        left_on=['HOME_TEAM_NAME', 'AWAY_TEAM_NAME', 'GAME_DATE'],
        right_on=['HOME_TEAM', 'AWAY_TEAM', 'GAME_DATE'],
    )

    df['TEAM_ML_PRICE'] = np.where(df['HOME'] == 1, df['HOME_ML_PRICE'], df['AWAY_ML_PRICE'])
    df['TEAM_IMPLIED_PROB'] = np.where(df['HOME'] == 1, df['HOME_ML_PROB'], df['AWAY_ML_PROB'])
    df['OPP_ML_PRICE'] = np.where(df['HOME'] == 1, df['AWAY_ML_PRICE'], df['HOME_ML_PRICE'])
    df['OPP_IMPLIED_PROB'] = np.where(df['HOME'] == 1, df['AWAY_ML_PROB'], df['HOME_ML_PROB'])
    df['ODDS_PROB_DIFF'] = (df['TEAM_IMPLIED_PROB'] - df['OPP_IMPLIED_PROB']).abs()
    df['PUBLIC_BIAS_INDICATOR'] = df['TEAM_IMPLIED_PROB'] - df['OPP_IMPLIED_PROB']
    df['TEAM_SPREAD_POINT'] = np.where(df['HOME'] == 1, df['HOME_SPREAD_POINT'], df['AWAY_SPREAD_POINT'])
    df['OPP_SPREAD_POINT'] = np.where(df['HOME'] == 1, df['AWAY_SPREAD_POINT'], df['HOME_SPREAD_POINT'])
    df['TOTAL_POINT'] = df['TOTAL_POINT']
    merged_count = df['HOME_ML_PRICE'].notna().sum() if 'HOME_ML_PRICE' in df.columns else 0
    print(f"[build_features] merged odds into {merged_count} game rows")
    if merged_count == 0:
        processed_dates = pd.to_datetime(df['GAME_DATE'], errors='coerce').dropna().dt.date
        odds_dates = pd.to_datetime(odds_features['GAME_DATE'], errors='coerce').dropna().dt.date
        if not processed_dates.empty and not odds_dates.empty and odds_dates.min() > processed_dates.max():
            print("[build_features] odds data is ahead of the current game logs; no merged rows are expected until game logs update.")
        else:
            print("[build_features] no odds rows matched current game logs; verify game date overlaps and team naming.")

        sample_games = (
            df[['HOME_TEAM_NAME', 'AWAY_TEAM_NAME', 'GAME_DATE']]
            .sort_values('GAME_DATE', ascending=False)
            .drop_duplicates()
            .head(5)
        )
        sample_odds = odds_features[['HOME_TEAM', 'AWAY_TEAM', 'GAME_DATE']].drop_duplicates().head(5)
        print("[build_features] sample processed game join keys:")
        print(sample_games.to_string(index=False))
        print("[build_features] sample odds join keys:")
        print(sample_odds.to_string(index=False))
    return df


def add_travel_timezone_features(df):
    df = df.sort_values(['TEAM_ID', 'GAME_DATE'])

    def get_offset(row):
        abbreviation = row['OPPONENT'] if row['IS_AWAY'] == 1 else row['TEAM_ABBREVIATION']
        if not isinstance(abbreviation, str):
            return None
        return get_team_timezone_offset(abbreviation=abbreviation, reference_date=row['GAME_DATE'])

    df['CURRENT_TIMEZONE_OFFSET'] = df.apply(get_offset, axis=1)
    df['PREV_LAT'] = df.groupby('TEAM_ID')['CURRENT_LAT'].shift(1)
    df['PREV_LON'] = df.groupby('TEAM_ID')['CURRENT_LON'].shift(1)
    df['PREV_TIMEZONE_OFFSET'] = df.groupby('TEAM_ID')['CURRENT_TIMEZONE_OFFSET'].shift(1)

    def compute_distance(row):
        if pd.isna(row['PREV_LAT']) or pd.isna(row['PREV_LON']) or pd.isna(row['CURRENT_LAT']) or pd.isna(row['CURRENT_LON']):
            return 0.0
        return haversine_distance(row['PREV_LAT'], row['PREV_LON'], row['CURRENT_LAT'], row['CURRENT_LON'])

    df['TRAVEL_KM'] = df.apply(compute_distance, axis=1)
    df['TRAVEL_KM'] = df['TRAVEL_KM'].fillna(0.0)

    df['TZ_DIFF'] = (df['CURRENT_TIMEZONE_OFFSET'] - df['PREV_TIMEZONE_OFFSET']).abs()
    df['TZ_DIFF'] = df['TZ_DIFF'].fillna(0.0)

    df = df.drop(columns=['PREV_LAT', 'PREV_LON', 'PREV_TIMEZONE_OFFSET'])
    df['TRAVEL_DISTANCE'] = df['TRAVEL_KM']
    df['TIMEZONE_SHIFT'] = df['TZ_DIFF']
    return df


def add_adult_entertainment_feature(df):
    df = add_game_location_features(df)
    df['OPPONENT_ADULT_QUALITY'] = df['OPPONENT'].apply(lambda x: get_team_adult_quality(abbreviation=x) if isinstance(x, str) else 5)
    # players do better in rooms with worse/bad rated adult entertainment, so we invert quality to difficulty index.
    df['ADULT_ENTERTAINMENT_INDEX'] = (11 - df['OPPONENT_ADULT_QUALITY']).clip(lower=1, upper=10)
    # apply only for away games where the team travels
    df['ADULT_ENTERTAINMENT_INDEX'] = np.where(df['IS_AWAY'] == 1, df['ADULT_ENTERTAINMENT_INDEX'], 0)
    return df


def rolling_team_features(df, windows=(5, 10)):
    df = df.sort_values(['TEAM_ID', 'GAME_DATE'])

    numeric_cols = ['PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'PLUS_MINUS']
    for col in numeric_cols:
        if col not in df.columns:
            continue

        for w in windows:
            feature = f"{col.lower()}_last{w}"
            # Use only prior games to avoid leakage from the current game's stats.
            df[feature] = df.groupby('TEAM_ID')[col].transform(
                lambda series: series.shift(1).rolling(w, min_periods=1).mean()
            )

    return df


def add_rest_days(df):
    df = df.sort_values(['TEAM_ID', 'GAME_DATE'])
    df['PREV_GAME_DATE'] = df.groupby('TEAM_ID')['GAME_DATE'].shift(1)
    df['REST_DAYS'] = (df['GAME_DATE'] - df['PREV_GAME_DATE']).dt.days
    df['REST_DAYS'] = df['REST_DAYS'].fillna(999)
    return df


def add_win_streak(df, window=5):
    if 'WIN' not in df.columns:
        return df
    df = df.sort_values(['TEAM_ID', 'GAME_DATE'])
    # Streak should reflect results before the current game only.
    df['WIN_STREAK'] = df.groupby('TEAM_ID')['WIN'].transform(
        lambda series: series.shift(1).rolling(window, min_periods=1).sum()
    ).fillna(0)
    return df


def load_injury_impact(df, path='data/raw/injuries_raw.csv'):
    def _to_utc_naive(series):
        # Normalize mixed tz-aware / tz-naive timestamp inputs into one comparable type.
        return pd.to_datetime(series, errors='coerce', utc=True).dt.tz_convert(None)

    if not os.path.exists(path):
        return pd.Series(0, index=df.index)

    injuries = pd.read_csv(path)
    if injuries.empty:
        return pd.Series(0, index=df.index)

    injuries = injuries.copy()
    injuries['GAME_DATE'] = _to_utc_naive(injuries.get('GAME_DATE'))
    injuries['INJURY_RETURN_DATE'] = _to_utc_naive(injuries.get('INJURY_RETURN_DATE'))
    injuries = injuries.dropna(subset=['GAME_DATE'])
    if injuries.empty:
        return pd.Series(0.0, index=df.index)

    injuries['INJURY_SEVERITY'] = pd.to_numeric(injuries.get('INJURY_SEVERITY'), errors='coerce').fillna(1.0)
    game_dates = _to_utc_naive(df['GAME_DATE'])
    if game_dates.isna().all():
        return pd.Series(0.0, index=df.index)

    if 'TEAM_ID' in df.columns and 'TEAM_ID' in injuries.columns:
        team_col = 'TEAM_ID'
        injuries = injuries.dropna(subset=['TEAM_ID']).copy()
        injuries['TEAM_ID'] = pd.to_numeric(injuries['TEAM_ID'], errors='coerce')
        injuries = injuries.dropna(subset=['TEAM_ID'])
        injuries['TEAM_ID'] = injuries['TEAM_ID'].astype(int)
    elif 'TEAM_ABBREVIATION' in df.columns and 'TEAM_ABBREVIATION' in injuries.columns:
        team_col = 'TEAM_ABBREVIATION'
        injuries[team_col] = injuries[team_col].astype(str).str.upper()
    else:
        return pd.Series(0.0, index=df.index)

    if injuries.empty:
        return pd.Series(0.0, index=df.index)

    # Build active-burden timelines via severity start/end events.
    start_events = injuries[[team_col, 'GAME_DATE', 'INJURY_SEVERITY']].rename(
        columns={'GAME_DATE': 'EVENT_DATE', 'INJURY_SEVERITY': 'DELTA'}
    )
    end_events = injuries.dropna(subset=['INJURY_RETURN_DATE'])[[team_col, 'INJURY_RETURN_DATE', 'INJURY_SEVERITY']].copy()
    if not end_events.empty:
        end_events['EVENT_DATE'] = end_events['INJURY_RETURN_DATE'] + pd.Timedelta(days=1)
        end_events['DELTA'] = -end_events['INJURY_SEVERITY']
        end_events = end_events[[team_col, 'EVENT_DATE', 'DELTA']]

    events = pd.concat([start_events, end_events], ignore_index=True)
    events = events.groupby([team_col, 'EVENT_DATE'], as_index=False)['DELTA'].sum()
    events = events.sort_values([team_col, 'EVENT_DATE'])
    events['ACTIVE_BURDEN'] = events.groupby(team_col)['DELTA'].cumsum().clip(lower=0)

    game_lookup = df[[team_col]].copy()
    game_lookup['GAME_DATE'] = game_dates
    game_lookup['ROW_ID'] = game_lookup.index
    game_lookup = game_lookup.dropna(subset=['GAME_DATE'])
    if game_lookup.empty:
        return pd.Series(0.0, index=df.index)

    burden_frames = []
    for team_value, team_games in game_lookup.groupby(team_col):
        team_events = events[events[team_col] == team_value][['EVENT_DATE', 'ACTIVE_BURDEN']].sort_values('EVENT_DATE')
        team_games = team_games.sort_values('GAME_DATE')
        if team_events.empty:
            team_games['INJURY_IMPACT'] = 0.0
        else:
            merged = pd.merge_asof(
                team_games,
                team_events,
                left_on='GAME_DATE',
                right_on='EVENT_DATE',
                direction='backward'
            )
            merged['INJURY_IMPACT'] = merged['ACTIVE_BURDEN'].fillna(0.0)
            team_games = merged[['ROW_ID', 'INJURY_IMPACT']]
        burden_frames.append(team_games[['ROW_ID', 'INJURY_IMPACT']])

    if not burden_frames:
        return pd.Series(0.0, index=df.index)

    burden_df = pd.concat(burden_frames, ignore_index=True).drop_duplicates(subset=['ROW_ID'])
    result = pd.Series(0.0, index=df.index, dtype=float)
    result.loc[burden_df['ROW_ID'].astype(int).values] = burden_df['INJURY_IMPACT'].astype(float).values
    return result


def build_fatigue_index(df):
    df['BACK_TO_BACK'] = ((df['REST_DAYS'] == 1).astype(int)).fillna(0)

    if 'TRAVEL_KM' not in df.columns:
        df['TRAVEL_KM'] = 0.0
    if 'TZ_DIFF' not in df.columns:
        df['TZ_DIFF'] = 0.0

    df['fatigue_index'] = (
        0.4 * df['BACK_TO_BACK'] +
        0.3 * (1 / (df['REST_DAYS'] + 1)) +
        0.2 * (df['TRAVEL_KM'] / 3000).clip(0, 1) +
        0.1 * df['TZ_DIFF'].abs().clip(0, 3)
    )
    return df


def main():
    df = load_raw_games()
    df = normalize_team_ids(df)
    df = df[df['GAME_DATE'] >= '2022-01-01']
    df = add_label(df)
    df = add_adult_entertainment_feature(df)
    df = rolling_team_features(df, windows=(5, 10))
    df = add_rest_days(df)
    df = add_travel_timezone_features(df)
    df = add_win_streak(df)
    df['INJURY_IMPACT'] = load_injury_impact(df)
    df = add_odds_features(df)
    df = build_fatigue_index(df)

    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved engineered features to {PROCESSED_PATH}.")


if __name__ == '__main__':
    main()
