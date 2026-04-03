#!/usr/bin/env python3
"""
Build a cleaned movement dataset from direction-audit results.

Rules:
- copy all non-direction classes unchanged
- for approaching / moving_away, keep only samples with status=agree

Usage:
    python3 robot_prediction/build_movement_clean.py \
      --movement_dir data/movement \
      --audit_csv audit/movement_direction/direction_audit_all.csv \
      --output_dir data/movement_clean
"""

import argparse
import csv
import os
import shutil
from collections import Counter, defaultdict
from glob import glob


DIRECTION_CLASSES = {"approaching", "moving_away"}


def load_agree_files(audit_csv):
    agree_files = set()
    stats = Counter()
    with open(audit_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            expected = row["expected_label"]
            if expected not in DIRECTION_CLASSES:
                continue
            stats[f"{expected}:{row['status']}"] += 1
            if row["status"] == "agree":
                agree_files.add(os.path.normpath(row["file"]))
    return agree_files, stats


def copy_class_files(src_dir, dst_dir, allowed_files=None):
    os.makedirs(dst_dir, exist_ok=True)
    copied = 0
    for path in sorted(glob(os.path.join(src_dir, "*.npy"))):
        norm_path = os.path.normpath(path)
        if allowed_files is not None and norm_path not in allowed_files:
            continue
        shutil.copy2(path, os.path.join(dst_dir, os.path.basename(path)))
        copied += 1
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--movement_dir", type=str, default="data/movement")
    parser.add_argument(
        "--audit_csv",
        type=str,
        default="audit/movement_direction/direction_audit_all.csv",
    )
    parser.add_argument("--output_dir", type=str, default="data/movement_clean")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove and rebuild the output directory if it already exists.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.movement_dir):
        raise SystemExit(f"Missing movement_dir: {args.movement_dir}")
    if not os.path.isfile(args.audit_csv):
        raise SystemExit(f"Missing audit_csv: {args.audit_csv}")

    if os.path.exists(args.output_dir):
        if not args.force:
            raise SystemExit(
                f"Output already exists: {args.output_dir} (use --force to rebuild)"
            )
        shutil.rmtree(args.output_dir)

    agree_files, audit_stats = load_agree_files(args.audit_csv)

    copied_counts = defaultdict(int)
    class_dirs = [
        name
        for name in sorted(os.listdir(args.movement_dir))
        if os.path.isdir(os.path.join(args.movement_dir, name))
    ]

    for class_name in class_dirs:
        src_dir = os.path.join(args.movement_dir, class_name)
        dst_dir = os.path.join(args.output_dir, class_name)
        if class_name in DIRECTION_CLASSES:
            copied_counts[class_name] = copy_class_files(
                src_dir, dst_dir, allowed_files=agree_files
            )
        else:
            copied_counts[class_name] = copy_class_files(src_dir, dst_dir)

    print("\nClean dataset created:")
    for class_name in class_dirs:
        print(f"  {class_name}: {copied_counts[class_name]} files")

    print("\nAudit source stats:")
    for class_name in sorted(DIRECTION_CLASSES):
        print(
            f"  {class_name}: "
            f"agree={audit_stats[f'{class_name}:agree']} "
            f"ambiguous={audit_stats[f'{class_name}:ambiguous']} "
            f"contradict={audit_stats[f'{class_name}:contradict']}"
        )

    print(f"\nOutput written to: {args.output_dir}")


if __name__ == "__main__":
    main()
