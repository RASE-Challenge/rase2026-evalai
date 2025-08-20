# evaluation_script/main.py
import json
from typing import Dict, Any

# 🔊 Version tag for stdout so we know THIS file is running
print("Evaluator version: v5-constant-output", flush=True)

# Constant, valid metrics that match your leaderboard labels exactly
CONST = {
    "PESQ": 2.5,
    "ESTOI": 0.70,
    "DNSMOS": 3.00,
    "MFCC-CS": 0.85,
    "Total": 0.76,
}

def evaluate(
    test_annotation_file: str,
    phase_codename: str,
    user_annotation_file: str = None,
    user_submission_file: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Minimal, robust evaluator:
    - Ignores the submitted file entirely (so no JSON parse/path issues)
    - Returns both 'old' and 'new' EvalAI formats
    - Uses split codename 'test' (matches your YAML)
    - Uses metric keys exactly as on your leaderboard
    """
    # NEW-STYLE format (split codename -> metrics)
    new_style = [{"test": CONST}]

    # OLD-STYLE format (explicit fields)
    old_style = [{
        "split": "test",
        "show_to_participant": True,
        "accuracies": CONST
    }]

    # Helpful for UI
    submission_result = {"test": CONST}

    print("Returning metrics:", json.dumps(CONST), flush=True)

    return {
        # Provide BOTH styles so whichever the runner expects will be present
        "result": old_style,        # legacy
        "evalai_result": new_style, # newer
        "submission_metadata": {
            "method_name": "Sanity Dummy",
            "score": CONST["Total"]
        },
        "submission_result": submission_result,
    }
