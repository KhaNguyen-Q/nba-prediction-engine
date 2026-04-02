import json
import os
import shutil


REGISTRY_INDEX_PATH = "models/registry/index.json"
CHAMPION_MODEL_PATH = "models/champion_team_model.pkl"
CHAMPION_META_PATH = "models/champion_team_model_meta.json"

SUPPORTED = {"logistic_baseline", "xgb_tree_model", "automl_challenger"}


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _model_score(entry_payload):
    metrics = entry_payload.get("metrics", {})
    holdout = metrics.get("holdout") or metrics.get("winner_holdout") or {}
    rolling = metrics.get("rolling_cv") or []
    log_loss = float(holdout.get("log_loss", 9.0))
    brier = float(holdout.get("brier_score", 9.0))
    if rolling:
        try:
            roll_log = sum(float(r.get("log_loss", 0.0)) for r in rolling) / len(rolling)
            roll_brier = sum(float(r.get("brier_score", 0.0)) for r in rolling) / len(rolling)
        except Exception:
            roll_log, roll_brier = log_loss, brier
    else:
        roll_log, roll_brier = log_loss, brier
    # Lower is better. Heavier weight on calibrated probabilistic quality.
    return 0.45 * log_loss + 0.30 * brier + 0.15 * roll_log + 0.10 * roll_brier


def promote_champion_model():
    if not os.path.exists(REGISTRY_INDEX_PATH):
        print(f"Registry index not found: {REGISTRY_INDEX_PATH}")
        return None
    idx = _load_json(REGISTRY_INDEX_PATH)
    entries = idx.get("entries", [])
    candidates = []
    for item in entries:
        model_name = item.get("model_name")
        if model_name not in SUPPORTED:
            continue
        entry_path = item.get("entry_path")
        if not entry_path or not os.path.exists(entry_path):
            continue
        payload = _load_json(entry_path)
        model_path = payload.get("model_path")
        if not model_path or not os.path.exists(model_path):
            continue
        score = _model_score(payload)
        candidates.append({
            "model_name": model_name,
            "model_path": model_path,
            "registry_entry_path": entry_path,
            "score": float(score),
            "trained_at": payload.get("trained_at"),
            "holdout": payload.get("metrics", {}).get("holdout") or payload.get("metrics", {}).get("winner_holdout") or {},
        })

    if not candidates:
        print("No eligible models found for promotion.")
        return None

    candidates = sorted(candidates, key=lambda x: x["score"])
    winner = candidates[0]
    os.makedirs(os.path.dirname(CHAMPION_MODEL_PATH), exist_ok=True)
    shutil.copyfile(winner["model_path"], CHAMPION_MODEL_PATH)

    meta = {
        "champion_model_name": winner["model_name"],
        "champion_model_path": CHAMPION_MODEL_PATH,
        "source_model_path": winner["model_path"],
        "registry_entry_path": winner["registry_entry_path"],
        "score": winner["score"],
        "trained_at": winner.get("trained_at"),
        "holdout": winner.get("holdout", {}),
        "runner_ups": candidates[1:3],
    }
    with open(CHAMPION_META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Promoted champion model: {winner['model_name']} -> {CHAMPION_MODEL_PATH}")
    return meta


def main():
    promote_champion_model()


if __name__ == "__main__":
    main()
