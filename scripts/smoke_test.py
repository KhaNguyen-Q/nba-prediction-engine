import os
import sys
import json
import requests
import joblib
import pandas as pd
from pandas.errors import EmptyDataError


REQUIRED_FILES = [
    "data/raw/games_raw.csv",
    "data/raw/players_raw.csv",
    "data/raw/injuries_raw.csv",
    "data/raw/injuries_latest.csv",
    "data/raw/odds_raw.csv",
    "data/raw/upcoming_games.csv",
    "data/processed/games_with_features.csv",
    "data/processed/upcoming_inference_features.csv",
    "models/logistic_baseline.pkl",
    "models/xgb_tree_model.pkl",
    "models/registry/index.json",
    "reports/monitoring_report.json",
]


def _safe_read_csv(path):
    try:
        return pd.read_csv(path), None
    except EmptyDataError:
        return pd.DataFrame(), "empty_file"
    except Exception as exc:
        return None, str(exc)


def check_files():
    ok = True
    for path in REQUIRED_FILES:
        exists = os.path.exists(path)
        print(f"[file] {path}: {'OK' if exists else 'MISSING'}")
        if not exists:
            ok = False
    return ok


def check_models():
    ok = True
    model_paths = [
        "models/logistic_baseline.pkl",
        "models/xgb_tree_model.pkl",
        "models/player_projection_model.pkl",
    ]
    for path in model_paths:
        if not os.path.exists(path):
            print(f"[model] {path}: MISSING")
            ok = False
            continue
        try:
            _ = joblib.load(path)
            print(f"[model] {path}: OK")
        except Exception as exc:
            print(f"[model] {path}: FAILED ({exc})")
            ok = False
    return ok


def check_data_shapes():
    ok = True
    data_targets = [
        "data/raw/upcoming_games.csv",
        "data/processed/games_with_features.csv",
        "data/processed/upcoming_inference_features.csv",
    ]
    for path in data_targets:
        df, err = _safe_read_csv(path)
        if err is None:
            print(f"[data] {path}: rows={len(df)} cols={len(df.columns)}")
        elif err == "empty_file":
            print(f"[data] {path}: rows=0 cols=0 (empty file)")
        else:
            print(f"[data] {path}: FAILED ({err})")
            ok = False
    return ok


def check_registry():
    index_path = "models/registry/index.json"
    if not os.path.exists(index_path):
        print(f"[registry] {index_path}: MISSING")
        return False
    try:
        with open(index_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        entries = payload.get("entries", []) if isinstance(payload, dict) else []
        print(f"[registry] {index_path}: OK (entries={len(entries)})")
        return True
    except Exception as exc:
        print(f"[registry] {index_path}: FAILED ({exc})")
        return False


def check_team_id_consistency():
    ok = True
    teams_path = "config/nba_teams.json"
    players_path = "data/raw/players_raw.csv"
    upcoming_path = "data/raw/upcoming_games.csv"
    inference_path = "data/processed/upcoming_inference_features.csv"

    if not os.path.exists(teams_path):
        print(f"[consistency] {teams_path}: MISSING")
        return False

    with open(teams_path, "r", encoding="utf-8") as fh:
        teams = json.load(fh)
    id_to_abbr = {
        int(t["team_id"]): str(t["abbreviation"])
        for t in teams
        if "team_id" in t and "abbreviation" in t
    }

    if os.path.exists(players_path):
        players, err = _safe_read_csv(players_path)
        if players is None:
            print(f"[consistency] players read failed: {err}")
            return False
        if {"TEAM_ID", "TEAM_ABBREVIATION"}.issubset(players.columns):
            players["TEAM_ID"] = pd.to_numeric(players["TEAM_ID"], errors="coerce")
            players = players.dropna(subset=["TEAM_ID"]).copy()
            players["TEAM_ID"] = players["TEAM_ID"].astype(int)
            pairs = players[["TEAM_ID", "TEAM_ABBREVIATION"]].drop_duplicates()
            mismatches = []
            for row in pairs.itertuples(index=False):
                expected = id_to_abbr.get(int(row.TEAM_ID))
                actual = str(row.TEAM_ABBREVIATION)
                if expected is not None and actual != expected:
                    mismatches.append((int(row.TEAM_ID), actual, expected))
            if mismatches:
                print(f"[consistency] players TEAM_ID/TEAM_ABBR mismatches: {len(mismatches)}")
                print(f"[consistency] sample: {mismatches[:5]}")
                ok = False
            else:
                print("[consistency] players TEAM_ID/TEAM_ABBR: OK")

    if os.path.exists(upcoming_path):
        upcoming, err = _safe_read_csv(upcoming_path)
        if upcoming is None:
            print(f"[consistency] upcoming read failed: {err}")
            return False
        required = {"GAME_ID", "HOME_TEAM_ID", "AWAY_TEAM_ID"}
        if required.issubset(upcoming.columns):
            upcoming["HOME_TEAM_ID"] = pd.to_numeric(upcoming["HOME_TEAM_ID"], errors="coerce")
            upcoming["AWAY_TEAM_ID"] = pd.to_numeric(upcoming["AWAY_TEAM_ID"], errors="coerce")
            upcoming = upcoming.dropna(subset=["HOME_TEAM_ID", "AWAY_TEAM_ID"]).copy()
            upcoming["HOME_TEAM_ID"] = upcoming["HOME_TEAM_ID"].astype(int)
            upcoming["AWAY_TEAM_ID"] = upcoming["AWAY_TEAM_ID"].astype(int)
            bad_ids = sorted(
                set(upcoming["HOME_TEAM_ID"]).union(set(upcoming["AWAY_TEAM_ID"])) - set(id_to_abbr.keys())
            )
            if bad_ids:
                print(f"[consistency] upcoming unknown team IDs: {bad_ids}")
                ok = False
            else:
                print("[consistency] upcoming team IDs: OK")

            if os.path.exists(inference_path):
                inf, err = _safe_read_csv(inference_path)
                if inf is None:
                    print(f"[consistency] inference read failed: {err}")
                    ok = False
                    return ok
                if err == "empty_file":
                    print("[consistency] inference file empty (likely no upcoming schedule rows); skipping GAME_ID alignment")
                    return ok
                if {"GAME_ID", "TEAM_ID"}.issubset(inf.columns):
                    inf["TEAM_ID"] = pd.to_numeric(inf["TEAM_ID"], errors="coerce")
                    inf = inf.dropna(subset=["TEAM_ID"]).copy()
                    inf["TEAM_ID"] = inf["TEAM_ID"].astype(int)
                    expected_game_ids = set(upcoming["GAME_ID"].astype(str))
                    actual_game_ids = set(inf["GAME_ID"].astype(str))
                    missing = expected_game_ids - actual_game_ids
                    extra = actual_game_ids - expected_game_ids
                    if missing or extra:
                        print(f"[consistency] inference/upcoming GAME_ID mismatch (missing={len(missing)}, extra={len(extra)})")
                        ok = False
                    else:
                        print("[consistency] inference GAME_ID alignment: OK")

    return ok


def check_api(base_url):
    ok = True
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        print(f"[api] /health: {r.status_code} {r.text[:120]}")
        if r.status_code != 200:
            ok = False
    except Exception as exc:
        print(f"[api] /health FAILED ({exc})")
        ok = False
    return ok


def main():
    base_url = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
    checks = [
        ("files", check_files),
        ("models", check_models),
        ("data_shapes", check_data_shapes),
        ("registry", check_registry),
        ("team_consistency", check_team_id_consistency),
        ("api", lambda: check_api(base_url)),
    ]
    results = []
    for name, fn in checks:
        print(f"--- Running check: {name} ---")
        result = fn()
        results.append(result)

    overall_ok = all(results)
    print(f"\nSMOKE TEST RESULT: {'PASS' if overall_ok else 'FAIL'}")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
