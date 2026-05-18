#!/usr/bin/env python3
"""
Re-save gesture pkl checkpoints with joblib compress=3.

joblib compress=3 stores numpy arrays as a compressed npy stream rather than
through Python pickle, so the output files contain no numpy._core module
references and load correctly on any numpy version (1.x or 2.x).

Usage (run once, in a numpy 2.x environment that can already load the files):
    python3 scripts/repack_checkpoints.py [--model-dir checkpoints]

Overwrites in place:
    gesture_landmark_scaler.pkl
    gesture_landmark_model.pkl
"""

import argparse
import os
import re
import pickle
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def check_numpy_core_refs(path):
    with open(path, "rb") as f:
        raw = f.read()
    return sorted(set(re.findall(rb"numpy\._core[.\w]*", raw)))


def repack(model_dir):
    import joblib

    scaler_path = os.path.join(model_dir, "gesture_landmark_scaler.pkl")
    model_path = os.path.join(model_dir, "gesture_landmark_model.pkl")

    for path in (scaler_path, model_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    # ── Load originals (requires numpy 2.x that matches the saved format) ──
    print("Loading originals with pickle …")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(model_path, "rb") as f:
        rf = pickle.load(f)

    orig_scaler_size = os.path.getsize(scaler_path)
    orig_model_size = os.path.getsize(model_path)

    # ── Capture predictions before overwrite for regression check ──
    x_probe = np.zeros((1, scaler.n_features_in_))
    probs_before = rf.predict_proba(scaler.transform(x_probe))

    # ── Re-save with joblib compress=3 ──
    print("Re-saving scaler …")
    joblib.dump(scaler, scaler_path, compress=3)

    print("Re-saving RF model …")
    joblib.dump(rf, model_path, compress=3)

    # ── Verify: no numpy._core refs ──
    for path, label in [(scaler_path, "scaler"), (model_path, "RF model")]:
        refs = check_numpy_core_refs(path)
        status = "CLEAN" if not refs else f"STILL HAS: {refs}"
        new_size = os.path.getsize(path)
        print(f"  {label}: {status}  ({new_size:,} bytes, was {orig_scaler_size if 'scaler' in label else orig_model_size:,})")

    # ── Verify: predictions unchanged ──
    scaler2 = joblib.load(scaler_path)
    rf2 = joblib.load(model_path)
    probs_after = rf2.predict_proba(scaler2.transform(x_probe))
    if np.allclose(probs_before, probs_after):
        print("  Prediction round-trip: OK")
    else:
        raise RuntimeError(f"Prediction mismatch!\n  before: {probs_before}\n  after:  {probs_after}")

    print("\nDone. Load these files with joblib.load() instead of pickle.load().")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="checkpoints")
    args = parser.parse_args()
    repack(args.model_dir)
