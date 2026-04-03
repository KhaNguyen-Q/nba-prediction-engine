import json
import os
import shutil


REGISTRY_INDEX_PATH = "models/registry/index.json"
CHAMPION_MODEL_PATH = "models/champion_team_model.pkl"
CHAMPION_META_PATH = "models/champion_team_model_meta.json"
PROMOTION_MIN_IMPROVEMENT = float(os.environ.get("PROMOTION_MIN_IMPROVEMENT", "0.005"))

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
    incumbent = None
    if os.path.exists(CHAMPION_META_PATH):
        try:
            incumbent = _load_json(CHAMPION_META_PATH)
        except Exception:
            incumbent = None
    incumbent_score = None
    incumbent_name = None
    incumbent_source_model_path = None
    if isinstance(incumbent, dict):
        incumbent_score = incumbent.get("score")
        incumbent_name = incumbent.get("champion_model_name")
        incumbent_source_model_path = incumbent.get("source_model_path")
        try:
            incumbent_score = float(incumbent_score) if incumbent_score is not None else None
        except Exception:
            incumbent_score = None

    should_promote = True
    promotion_reason = "no incumbent"
    if incumbent_score is not None:
        if winner["score"] <= (incumbent_score - PROMOTION_MIN_IMPROVEMENT):
            should_promote = True
            promotion_reason = (
                f"winner score improved from {incumbent_score:.6f} to {winner['score']:.6f} "
                f"(min improvement={PROMOTION_MIN_IMPROVEMENT:.6f})"
            )
        else:
            should_promote = False
            promotion_reason = (
                f"winner score {winner['score']:.6f} did not beat incumbent {incumbent_score:.6f} "
                f"by required margin {PROMOTION_MIN_IMPROVEMENT:.6f}"
            )

    if should_promote:
        os.makedirs(os.path.dirname(CHAMPION_MODEL_PATH), exist_ok=True)
        shutil.copyfile(winner["model_path"], CHAMPION_MODEL_PATH)
    else:
        if incumbent_source_model_path and os.path.exists(incumbent_source_model_path):
            winner = {
                "model_name": incumbent_name or winner["model_name"],
                "model_path": incumbent_source_model_path,
                "registry_entry_path": incumbent.get("registry_entry_path"),
                "score": float(incumbent_score),
                "trained_at": incumbent.get("trained_at"),
                "holdout": incumbent.get("holdout", {}),
            }
        else:
            should_promote = True
            promotion_reason = "incumbent artifact missing; promoting winner"
            os.makedirs(os.path.dirname(CHAMPION_MODEL_PATH), exist_ok=True)
            shutil.copyfile(candidates[0]["model_path"], CHAMPION_MODEL_PATH)

    meta = {
        "champion_model_name": winner["model_name"],
        "champion_model_path": CHAMPION_MODEL_PATH,
        "source_model_path": winner["model_path"],
        "registry_entry_path": winner["registry_entry_path"],
        "score": winner["score"],
        "trained_at": winner.get("trained_at"),
        "holdout": winner.get("holdout", {}),
        "runner_ups": candidates[1:3],
        "promoted": bool(should_promote),
        "promotion_reason": promotion_reason,
        "promotion_min_improvement": PROMOTION_MIN_IMPROVEMENT,
    }
    with open(CHAMPION_META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    if should_promote:
        print(f"Promoted champion model: {winner['model_name']} -> {CHAMPION_MODEL_PATH}")
    else:
        print(f"Champion unchanged: {promotion_reason}")
    return meta


def main():
    promote_champion_model()


if __name__ == "__main__":
    main()
