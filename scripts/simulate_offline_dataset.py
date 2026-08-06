"""Generate an offline NBA-like team-game dataset for local demos when stats.nba.com is unreachable.

This does NOT invent live NBA results for publication claims. It creates a synthetic
schedule with latent team strength so the real feature/training scripts can run and
produce registry metrics (e.g. for README demos).

Usage:
  python scripts/simulate_offline_dataset.py
  python scripts/build_features.py
  python scripts/build_inference_features.py
  python scripts/train_baseline.py
  python scripts/train_tree_model.py
  python scripts/train_automl_challenger.py
  python scripts/model_promotion.py
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.team_utils import load_nba_teams


def main(seed: int = 42, n_days: int = 200):
    rng = np.random.default_rng(seed)
    teams = load_nba_teams()
    strength = {t["team_id"]: float(rng.normal(0, 1)) for t in teams}
    abbr = {t["team_id"]: t["abbreviation"] for t in teams}
    name = {t["team_id"]: t["name"] for t in teams}
    ids = [t["team_id"] for t in teams]

    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    start = datetime(2022, 10, 18)
    rows = []
    game_counter = 100000

    for day_i in range(n_days):
        game_date = start + timedelta(days=day_i)
        if day_i % 7 == 3:
            continue
        n_games = int(rng.integers(5, 9))
        day_teams = rng.permutation(ids)
        used = set()
        made = 0
        for i in range(0, len(day_teams) - 1, 2):
            if made >= n_games:
                break
            home_id = int(day_teams[i])
            away_id = int(day_teams[i + 1])
            if home_id in used or away_id in used:
                continue
            used.add(home_id)
            used.add(away_id)
            made += 1
            game_counter += 1
            game_id = f"0022{game_counter}"

            edge = 0.35 + 0.9 * (strength[home_id] - strength[away_id])
            p_home = 1 / (1 + np.exp(-edge))
            home_wins = bool(rng.random() < p_home)

            def box(team_id, won, is_home, opp_id):
                base = 110 + 4 * strength[team_id] + (2.5 if is_home else 0)
                pts = float(np.clip(rng.normal(base + (3 if won else -3), 8), 85, 145))
                reb = float(np.clip(rng.normal(44, 5), 28, 60))
                ast = float(np.clip(rng.normal(24 + strength[team_id], 4), 12, 40))
                fg_pct = float(np.clip(rng.normal(0.46 + (0.02 if won else -0.01), 0.04), 0.35, 0.60))
                fg3_pct = float(np.clip(rng.normal(0.36, 0.05), 0.20, 0.50))
                ft_pct = float(np.clip(rng.normal(0.78, 0.05), 0.60, 0.95))
                plus_minus = float(np.clip(rng.normal(6 if won else -6, 8), -35, 35))
                fga = 88
                fgm = int(round(fg_pct * fga))
                fg3a = 35
                fg3m = int(round(fg3_pct * fg3a))
                fta = 22
                ftm = int(round(ft_pct * fta))
                matchup = (
                    f"{abbr[team_id]} vs. {abbr[opp_id]}"
                    if is_home
                    else f"{abbr[team_id]} @ {abbr[opp_id]}"
                )
                return {
                    "SEASON_ID": "22022" if game_date < datetime(2023, 10, 1) else "22023",
                    "TEAM_ID": team_id,
                    "TEAM_ABBREVIATION": abbr[team_id],
                    "TEAM_NAME": name[team_id],
                    "GAME_ID": game_id,
                    "GAME_DATE": game_date.strftime("%Y-%m-%d"),
                    "MATCHUP": matchup,
                    "WL": "W" if won else "L",
                    "MIN": 240,
                    "PTS": pts,
                    "FGM": fgm,
                    "FGA": fga,
                    "FG_PCT": fg_pct,
                    "FG3M": fg3m,
                    "FG3A": fg3a,
                    "FG3_PCT": fg3_pct,
                    "FTM": ftm,
                    "FTA": fta,
                    "FT_PCT": ft_pct,
                    "OREB": float(rng.integers(8, 16)),
                    "DREB": reb - 10,
                    "REB": reb,
                    "AST": ast,
                    "STL": float(rng.integers(5, 12)),
                    "BLK": float(rng.integers(3, 9)),
                    "TOV": float(rng.integers(10, 18)),
                    "PF": float(rng.integers(16, 26)),
                    "PLUS_MINUS": plus_minus,
                }

            rows.append(box(home_id, home_wins, True, away_id))
            rows.append(box(away_id, not home_wins, False, home_id))

    games_df = pd.DataFrame(rows)
    games_df.to_csv("data/raw/games_raw.csv", index=False)
    print(
        f"Wrote data/raw/games_raw.csv rows={len(games_df)} "
        f"games={games_df.GAME_ID.nunique()} "
        f"date_range={games_df.GAME_DATE.min()}..{games_df.GAME_DATE.max()}"
    )

    pd.DataFrame(
        columns=[
            "COMMENCE_TIME",
            "HOME_TEAM",
            "AWAY_TEAM",
            "BOOKMAKER",
            "MARKET",
            "OUTCOME_NAME",
            "POINT",
            "PRICE",
        ]
    ).to_csv("data/raw/odds_raw.csv", index=False)
    pd.DataFrame(
        columns=[
            "PLAYER_ID",
            "PLAYER_NAME",
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "INJURY_STATUS",
            "INJURY_DETAIL",
            "GAME_DATE",
            "INJURY_RETURN_DATE",
            "INJURY_SEVERITY",
            "IS_UNAVAILABLE",
            "AVAILABILITY_LABEL",
        ]
    ).to_csv("data/raw/injuries_raw.csv", index=False)
    pd.DataFrame(
        columns=[
            "PLAYER_ID",
            "PLAYER_NAME",
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "AVAILABILITY_LABEL",
            "INJURY_SEVERITY",
            "IS_UNAVAILABLE",
            "AS_OF",
        ]
    ).to_csv("data/raw/injuries_latest.csv", index=False)
    pd.DataFrame(
        columns=["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "ROSTER_STATUS"]
    ).to_csv("data/raw/players_raw.csv", index=False)
    pd.DataFrame(
        columns=["PLAYER_ID", "TEAM_ID", "GAME_ID", "GAME_DATE", "MATCHUP", "PTS", "REB", "AST", "MIN"]
    ).to_csv("data/raw/player_game_logs_raw.csv", index=False)

    last = pd.to_datetime(games_df.GAME_DATE).max()
    upcoming = []
    for k in range(8):
        a, b = rng.choice(ids, size=2, replace=False)
        upcoming.append(
            {
                "GAME_ID": f"UP{k + 1}",
                "GAME_DATE": (last + timedelta(days=2 + k)).strftime("%Y-%m-%d"),
                "HOME_TEAM_ID": int(a),
                "AWAY_TEAM_ID": int(b),
                "HOME_TEAM_ABBR": abbr[int(a)],
                "AWAY_TEAM_ABBR": abbr[int(b)],
            }
        )
    pd.DataFrame(upcoming).to_csv("data/raw/upcoming_games.csv", index=False)
    print(f"Wrote data/raw/upcoming_games.csv rows={len(upcoming)}")
    print("Offline synthetic dataset ready.")


if __name__ == "__main__":
    main()
