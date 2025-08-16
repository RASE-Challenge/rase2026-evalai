# evaluation_script/main.py
import json
import os
from typing import Dict, List
import numpy as np
import soundfile as sf
import librosa

# PESQ / ESTOI
from pesq import pesq
from pystoi import stoi

# Optional: DNSMOS via onnxruntime (see _dnsmos below)
try:
    import onnxruntime as ort
except Exception:
    ort = None

# -----------------------
# Challenge configuration
# -----------------------
W = {"easy": 0.25, "medium": 0.35, "hard": 0.40}   # difficulty weights
TARGET_SR = 16000                                   # we evaluate at 16k
USE_DNSMOS = False                                  # set True when you provide a model (see _dnsmos())

# -----------------------
# Utils
# -----------------------
def _resample_if_needed(y, sr, target_sr=TARGET_SR):
    if sr == target_sr:
        return y
    return librosa.resample(y=y, orig_sr=sr, target_sr=target_sr)

def _safe_read(path, target_sr=TARGET_SR):
    y, sr = sf.read(path, always_2d=False)
    # Ensure mono float32
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32, copy=False)
    y = _resample_if_needed(y, sr, target_sr)
    return y, target_sr

def _mfcc_cosine(ref, enh, sr=TARGET_SR, n_mfcc=20):
    """Cosine similarity between mean MFCC vectors."""
    ref_mfcc = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=n_mfcc)
    enh_mfcc = librosa.feature.mfcc(y=enh, sr=sr, n_mfcc=n_mfcc)
    ref_vec = np.mean(ref_mfcc, axis=1)
    enh_vec = np.mean(enh_mfcc, axis=1)
    # cosine similarity
    num = np.dot(ref_vec, enh_vec)
    den = np.linalg.norm(ref_vec) * np.linalg.norm(enh_vec) + 1e-12
    return float(num / den)

# ---- Optional DNSMOS (needs a model) ----
# If you want true DNSMOS(P.835), place the ONNX model on disk and point
# DNSMOS_MODEL_PATH to it (or hardcode a path) and set USE_DNSMOS=True above.
def _dnsmos(enh, sr=TARGET_SR, sess=None):
    """
    Placeholder for DNSMOS, returns np.nan unless you wire a model.
    If you have a P.835 onnx model, process 'enh' and return a scalar MOS.
    """
    # Example skeleton (not functional without the actual model preprocessing):
    # if sess is not None:
    #     feats = extract_features_for_dnsmos(enh, sr)   # you implement
    #     out = sess.run(None, {"feats": feats})[0]
    #     return float(out.mean())
    return np.nan

def _weighted_metric(per_diff: Dict[str, List[float]]) -> float:
    """Take per-utterance values grouped by difficulty and return weighted mean."""
    def _mean(x): return float(np.mean(x)) if len(x) else np.nan
    easy   = _mean(per_diff["easy"])
    medium = _mean(per_diff["medium"])
    hard   = _mean(per_diff["hard"])
    return float(
        np.nansum([
            easy   * W["easy"]   if not np.isnan(easy) else 0.0,
            medium * W["medium"] if not np.isnan(medium) else 0.0,
            hard   * W["hard"]   if not np.isnan(hard) else 0.0
        ]) / np.nansum([
            W["easy"]   if not np.isnan(easy) else 0.0,
            W["medium"] if not np.isnan(medium) else 0.0,
            W["hard"]   if not np.isnan(hard) else 0.0
        ] or [np.nan])
    )

def _pred_path_for_utt(out_dir: str, utt_id: str) -> str:
    """Map utt_id -> expected enhanced wav path in submission output directory."""
    return os.path.join(out_dir, f"{utt_id}.wav")

# -----------------------
# Main evaluate()
# -----------------------
def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    """
    EvalAI entrypoint.

    - test_annotation_file: JSON with a list of items:
        [{"utt_id": "...", "ref": ".../clean.wav", "difficulty": "easy|medium|hard"}, ...]
    - user_submission_file: path to submission directory (EvalAI passes the file/path,
      for docker-based phases this is typically the output folder archive or a path
      where the worker has placed outputs; on remote worker we mount /output).
      We treat it as a directory and look for <utt_id>.wav files inside.
    """
    # Resolve the folder that contains participant outputs
    # In remote worker flow, EvalAI supplies a path to the submission. If it is a
    # zip or a file, your worker typically extracts it. We try: if it's a dir, use it;
    # else, use its directory.
    out_dir = user_submission_file if os.path.isdir(user_submission_file) \
        else os.path.dirname(user_submission_file)

    # Optional DNSMOS session
    dnsmos_sess = None
    if USE_DNSMOS and ort is not None:
        model_path = os.environ.get("DNSMOS_MODEL_PATH", "").strip()
        if os.path.isfile(model_path):
            dnsmos_sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Load annotations
    with open(test_annotation_file, "r", encoding="utf-8") as f:
        ann = json.load(f)

    # buckets
    pesq_vals   = {"easy": [], "medium": [], "hard": []}
    estoi_vals  = {"easy": [], "medium": [], "hard": []}
    dnsmos_vals = {"easy": [], "medium": [], "hard": []}
    mfcc_vals   = {"easy": [], "medium": [], "hard": []}

    missing = []

    for item in ann:
        utt_id     = item["utt_id"]
        ref_path   = item["ref"]
        difficulty = item.get("difficulty", "medium")
        if difficulty not in ("easy", "medium", "hard"):
            difficulty = "medium"

        pred_path = _pred_path_for_utt(out_dir, utt_id)

        if not os.path.isfile(pred_path):
            missing.append(utt_id)
            continue

        # Read audio
        ref, sr1 = _safe_read(ref_path, TARGET_SR)
        enh, sr2 = _safe_read(pred_path, TARGET_SR)

        # Match lengths (PESQ/STOI assume aligned)
        L = min(len(ref), len(enh))
        if L < 1:
            continue
        ref = ref[:L]
        enh = enh[:L]

        # PESQ (WB at 16k)
        try:
            pesq_score = pesq(TARGET_SR, ref, enh, "wb")
        except Exception:
            pesq_score = np.nan

        # ESTOI
        try:
            estoi_score = stoi(ref, enh, TARGET_SR, extended=True)
        except Exception:
            estoi_score = np.nan

        # MFCC cosine
        try:
            mfcc_score = _mfcc_cosine(ref, enh, TARGET_SR)
        except Exception:
            mfcc_score = np.nan

        # DNSMOS (optional)
        if USE_DNSMOS and dnsmos_sess is not None:
            try:
                dnsmos_score = _dnsmos(enh, TARGET_SR, dnsmos_sess)
            except Exception:
                dnsmos_score = np.nan
        else:
            dnsmos_score = np.nan

        # accumulate
        pesq_vals[difficulty].append(pesq_score)
        estoi_vals[difficulty].append(estoi_score)
        mfcc_vals[difficulty].append(mfcc_score)
        dnsmos_vals[difficulty].append(dnsmos_score)

    # Weighted metrics across difficulties
    pesq_w   = _weighted_metric(pesq_vals)
    estoi_w  = _weighted_metric(estoi_vals)
    mfcc_w   = _weighted_metric(mfcc_vals)
    dnsmos_w = _weighted_metric(dnsmos_vals)  # will be nan unless you enable it

    # Final Total = mean of the available metrics (ignore NaNs)
    total = float(np.nanmean([pesq_w, estoi_w, dnsmos_w, mfcc_w]))

    # Split name (must match challenge_phase_splits.dataset_split.codename)
    split_name = "test"

    split_payload = {
        "PESQ":    round(float(pesq_w), 4)   if not np.isnan(pesq_w) else None,
        "ESTOI":   round(float(estoi_w), 4)  if not np.isnan(estoi_w) else None,
        "DNSMOS":  round(float(dnsmos_w), 4) if not np.isnan(dnsmos_w) else None,
        "MFCC-CS": round(float(mfcc_w), 4)   if not np.isnan(mfcc_w) else None,
        "Total":   round(float(total), 4)     if not np.isnan(total) else None,
    }

    # Optional debug: penalize missing outputs
    if missing:
        # You can subtract a penalty or set Total=None.
        # Here we leave Total as computed; you can decide policy.
        pass

    result_list = [{split_name: split_payload}]

    return {
        "result": result_list,
        "submission_result": result_list[0],
        "submission_metadata": {
            "missing_outputs": missing,
            "num_items": len(ann)
        }
    }
