#!/usr/bin/env python3
"""Generate summary tables and plots from logged ROS2 experiment CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_events(events_csv: Path) -> list[dict[str, str]]:
    with events_csv.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_metrics(metrics_path: Path, metrics: dict) -> None:
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def write_table(table_path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with table_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_metrics(events: list[dict[str, str]], metadata: dict) -> tuple[dict, list[dict[str, str]]]:
    counts = defaultdict(Counter)
    conf_sums = defaultdict(lambda: defaultdict(float))
    conf_counts = defaultdict(lambda: defaultdict(int))
    cmd_samples = []

    for row in events:
        topic = row["topic"]
        label = row["label"]
        conf = to_float(row["confidence"])
        if label:
            counts[topic][label] += 1
            if conf is not None:
                conf_sums[topic][label] += conf
                conf_counts[topic][label] += 1
        if topic == "cmd_vel":
            lin = to_float(row["linear_x"])
            ang = to_float(row["angular_z"])
            if lin is not None and ang is not None:
                cmd_samples.append((to_float(row["t_rel_sec"]) or 0.0, lin, ang))

    summary_rows = []
    metrics = {
        "run_name": metadata.get("run_name", ""),
        "duration_sec": metadata.get("duration_sec", ""),
        "expected": metadata.get("expected", {}),
        "topics": {},
        "cmd_vel": {},
    }

    for topic, topic_counts in sorted(counts.items()):
        metrics["topics"][topic] = {}
        expected_label = metadata.get("expected", {}).get(topic, "")
        total = sum(topic_counts.values())
        matches = topic_counts.get(expected_label, 0) if expected_label else None
        for label, count in sorted(topic_counts.items()):
            avg_conf = (
                conf_sums[topic][label] / conf_counts[topic][label]
                if conf_counts[topic][label]
                else None
            )
            summary_rows.append(
                {
                    "topic": topic,
                    "label": label,
                    "count": str(count),
                    "percentage": f"{(count / total * 100):.2f}" if total else "",
                    "avg_confidence": "" if avg_conf is None else f"{avg_conf:.4f}",
                    "expected_label": expected_label,
                    "matches_expected": "yes" if expected_label and label == expected_label else "no" if expected_label else "",
                }
            )
            metrics["topics"][topic][label] = {
                "count": count,
                "percentage": (count / total * 100) if total else 0.0,
                "avg_confidence": avg_conf,
            }
        if expected_label:
            metrics["topics"][topic]["expected_match_rate_pct"] = (
                (matches / total * 100) if total else 0.0
            )

    if cmd_samples:
        linear_abs = [abs(v[1]) for v in cmd_samples]
        angular_abs = [abs(v[2]) for v in cmd_samples]
        active = [1 for _, lin, ang in cmd_samples if abs(lin) > 1e-6 or abs(ang) > 1e-6]
        metrics["cmd_vel"] = {
            "samples": len(cmd_samples),
            "mean_abs_linear_x": sum(linear_abs) / len(linear_abs),
            "mean_abs_angular_z": sum(angular_abs) / len(angular_abs),
            "active_command_rate_pct": len(active) / len(cmd_samples) * 100.0,
        }

    return metrics, summary_rows


def make_plots(events: list[dict[str, str]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is not installed. Install it in the environment used for plotting."
        ) from exc

    counts = defaultdict(Counter)
    conf_values = defaultdict(lambda: defaultdict(list))
    cmd_t, cmd_lin, cmd_ang = [], [], []

    for row in events:
        topic = row["topic"]
        label = row["label"]
        conf = to_float(row["confidence"])
        if label:
            counts[topic][label] += 1
            if conf is not None:
                conf_values[topic][label].append(conf)
        if topic == "cmd_vel":
            t = to_float(row["t_rel_sec"])
            lin = to_float(row["linear_x"])
            ang = to_float(row["angular_z"])
            if t is not None and lin is not None and ang is not None:
                cmd_t.append(t)
                cmd_lin.append(lin)
                cmd_ang.append(ang)

    for topic, topic_counts in counts.items():
        labels = list(topic_counts.keys())
        values = [topic_counts[label] for label in labels]
        plt.figure(figsize=(8, 4.5))
        plt.bar(labels, values)
        plt.title(f"{topic.title()} label counts")
        plt.ylabel("Count")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f"{topic}_counts.png", dpi=150)
        plt.close()

        avg_labels = [label for label in labels if conf_values[topic][label]]
        if avg_labels:
            avg_values = [
                sum(conf_values[topic][label]) / len(conf_values[topic][label])
                for label in avg_labels
            ]
            plt.figure(figsize=(8, 4.5))
            plt.bar(avg_labels, avg_values)
            plt.ylim(0, 1.0)
            plt.title(f"{topic.title()} average confidence")
            plt.ylabel("Average confidence")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            plt.savefig(output_dir / f"{topic}_avg_confidence.png", dpi=150)
            plt.close()

    if cmd_t:
        plt.figure(figsize=(9, 4.8))
        plt.plot(cmd_t, cmd_lin, label="linear_x")
        plt.plot(cmd_t, cmd_ang, label="angular_z")
        plt.title("Robot command over time")
        plt.xlabel("Time (s)")
        plt.ylabel("Command value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "cmd_vel_timeseries.png", dpi=150)
        plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot ROS2 experiment results")
    parser.add_argument("--run-dir", required=True, help="Directory containing events.csv and metadata.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    events_csv = run_dir / "events.csv"
    metadata_json = run_dir / "metadata.json"
    if not events_csv.exists():
        raise SystemExit(f"Missing events file: {events_csv}")
    if not metadata_json.exists():
        raise SystemExit(f"Missing metadata file: {metadata_json}")

    events = load_events(events_csv)
    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    metrics, summary_rows = build_metrics(events, metadata)

    write_metrics(run_dir / "metrics.json", metrics)
    write_table(
        run_dir / "summary_table.csv",
        summary_rows,
        ["topic", "label", "count", "percentage", "avg_confidence", "expected_label", "matches_expected"],
    )
    make_plots(events, run_dir)

    print(f"Saved metrics and plots to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
