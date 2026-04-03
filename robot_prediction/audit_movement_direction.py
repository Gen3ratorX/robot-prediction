#!/usr/bin/env python3
"""
Audit approaching/moving_away samples using signed depth/scale direction.

This script does not relabel files automatically. It reports which sequences:
- agree with their folder label
- contradict their folder label
- are ambiguous

Usage:
    python3 robot_prediction/audit_movement_direction.py \
      --movement_dir data/movement \
      --output_dir audit/movement_direction
"""

import argparse
import csv
import os
from collections import Counter, defaultdict
from glob import glob

import numpy as np

from movement_features import estimate_depth_scale_signature


TARGET_CLASSES = ("approaching", "moving_away")


def classify_direction(signature, threshold):
    approaching_score = float(signature["approaching_score"])
    moving_away_score = float(signature["moving_away_score"])
    margin = approaching_score - moving_away_score

    if margin >= threshold:
        return "approaching", margin
    if margin <= -threshold:
        return "moving_away", margin
    return "ambiguous", margin


def audit_class(class_dir, expected_label, threshold):
    rows = []
    counts = Counter()

    for path in sorted(glob(os.path.join(class_dir, "*.npy"))):
        try:
            sequence = np.load(path)
        except Exception as exc:  # pragma: no cover
            rows.append(
                {
                    "file": path,
                    "expected_label": expected_label,
                    "predicted_label": "load_error",
                    "status": "error",
                    "margin": "",
                    "approaching_score": "",
                    "moving_away_score": "",
                    "signed_depth_delta": "",
                    "signed_scale_delta": "",
                    "error": str(exc),
                }
            )
            counts["error"] += 1
            continue

        signature = estimate_depth_scale_signature(sequence)
        predicted_label, margin = classify_direction(signature, threshold)
        if predicted_label == expected_label:
            status = "agree"
        elif predicted_label == "ambiguous":
            status = "ambiguous"
        else:
            status = "contradict"

        rows.append(
            {
                "file": path,
                "expected_label": expected_label,
                "predicted_label": predicted_label,
                "status": status,
                "margin": f"{margin:.6f}",
                "approaching_score": f"{signature['approaching_score']:.6f}",
                "moving_away_score": f"{signature['moving_away_score']:.6f}",
                "signed_depth_delta": f"{signature['signed_depth_delta']:.6f}",
                "signed_scale_delta": f"{signature['signed_scale_delta']:.6f}",
                "error": "",
            }
        )
        counts[status] += 1

    return rows, counts


def write_csv(path, rows):
    fieldnames = [
        "file",
        "expected_label",
        "predicted_label",
        "status",
        "margin",
        "approaching_score",
        "moving_away_score",
        "signed_depth_delta",
        "signed_scale_delta",
        "error",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--movement_dir", type=str, default="data/movement")
    parser.add_argument("--output_dir", type=str, default="audit/movement_direction")
    parser.add_argument(
        "--direction_threshold",
        type=float,
        default=0.035,
        help="Minimum approaching-vs-away score margin required for a directional decision.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    summary_rows = []
    all_rows = []
    contradiction_rows = defaultdict(list)

    for class_name in TARGET_CLASSES:
        class_dir = os.path.join(args.movement_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Missing class directory: {class_dir}")
            continue

        rows, counts = audit_class(class_dir, class_name, args.direction_threshold)
        all_rows.extend(rows)
        contradiction_rows[class_name] = [
            row for row in rows if row["status"] in {"contradict", "ambiguous"}
        ]

        summary_rows.append(
            {
                "class": class_name,
                "total": len(rows),
                "agree": counts["agree"],
                "ambiguous": counts["ambiguous"],
                "contradict": counts["contradict"],
                "error": counts["error"],
            }
        )

    write_csv(os.path.join(args.output_dir, "direction_audit_all.csv"), all_rows)
    for class_name, rows in contradiction_rows.items():
        write_csv(
            os.path.join(args.output_dir, f"{class_name}_needs_review.csv"),
            rows,
        )

    print("\nDirection audit summary:")
    for row in summary_rows:
        print(
            f"  {row['class']}: total={row['total']} "
            f"agree={row['agree']} ambiguous={row['ambiguous']} "
            f"contradict={row['contradict']} error={row['error']}"
        )

    print(f"\nReports written to: {args.output_dir}")
    print("Files:")
    print("  direction_audit_all.csv")
    for class_name in TARGET_CLASSES:
        print(f"  {class_name}_needs_review.csv")


if __name__ == "__main__":
    main()
