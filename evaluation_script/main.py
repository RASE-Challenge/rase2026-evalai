# evaluation_script/main.py
import json
from typing import Dict

# Difficulty weights
W = {"easy": 0.25, "medium": 0.35, "hard": 0.40}

DISPLAY_NAMES = {
    "pesq": "PESQ",
    "estoi": "ESTOI",
    "dnsmos": "DNSMOS",
    "mfcc_cs": "MFCC-CS",
}

def _weighted_metric(d: Dict[str, Dict[str, float]], key: str) -> float:
    """Weighted avg across difficulties for a given metric key."""
    return (
        float(d["easy"][key])   * W["easy"]
        + float(d["medium"][key]) * W["medium"]
        + float(d["hard"][key])   * W["hard"]
    )

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    """
    EvalAI entrypoint. We expect `user_submission_file` to be a JSON file
    containing:
      { "easy": {...}, "medium": {...}, "hard": {...} }
    where each sub-dict has keys: pesq, estoi, dnsmos, mfcc_cs.
    """
    with open(user_submission_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # compute difficulty-weighted metrics
    pesq   = _weighted_metric(metrics, "pesq")
    estoi  = _weighted_metric(metrics, "estoi")
    dnsmos = _weighted_metric(metrics, "dnsmos")
    mfcc   = _weighted_metric(metrics, "mfcc_cs")

    # Total can be any rule you like; here we use the mean of the four
    total  = (pesq + estoi + dnsmos + mfcc) / 4.0

    # Leaderboard split name must match your challenge_config dataset split codename
    split_name = "test_split" if phase_codename == "test" else "train_split"

    split_payload = {
        "PESQ":   round(pesq, 4),
        "ESTOI":  round(estoi, 4),
        "DNSMOS": round(dnsmos, 4),
        "MFCC-CS": round(mfcc, 4),
        "Total":  round(total, 4),
    }

    result_list = [{split_name: split_payload}]

    return {
        "result": result_list,
        # also show in the submission result file
        "submission_result": result_list[0],
    }
