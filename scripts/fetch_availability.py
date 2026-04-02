import os
import sys

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.get_data import fetch_injuries_data, build_latest_availability_snapshot


UPCOMING_PATH = "data/raw/upcoming_games.csv"
RAW_INJURIES_PATH = "data/raw/injuries_raw.csv"
LATEST_AVAILABILITY_PATH = "data/raw/injuries_latest.csv"


def _upcoming_team_ids(upcoming_path=UPCOMING_PATH):
    if not os.path.exists(upcoming_path):
        return []
    df = pd.read_csv(upcoming_path)
    if df.empty:
        return []
    ids = set()
    for col in ["HOME_TEAM_ID", "AWAY_TEAM_ID"]:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
            ids.update(series.tolist())
    return sorted(ids)


def fetch_availability_for_upcoming():
    team_ids = _upcoming_team_ids()
    injuries = fetch_injuries_data(save_path=RAW_INJURIES_PATH, team_ids=team_ids if team_ids else None)
    latest = build_latest_availability_snapshot(
        injuries_df=injuries,
        save_path=LATEST_AVAILABILITY_PATH,
    )
    print(
        f"Availability refresh complete: teams_scoped={len(team_ids)} "
        f"injuries_rows={len(injuries)} latest_rows={len(latest)}"
    )
    return latest


def main():
    fetch_availability_for_upcoming()


if __name__ == "__main__":
    main()
