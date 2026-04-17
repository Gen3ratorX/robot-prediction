#!/usr/bin/env python3
"""Log ROS2 human-intention topics into CSV for analysis."""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from std_msgs.msg import String


def parse_label_message(msg: str) -> tuple[str, float | None]:
    if ":" not in msg:
        return msg.strip(), None
    label, conf = msg.rsplit(":", 1)
    try:
        return label.strip(), float(conf)
    except ValueError:
        return msg.strip(), None


class ExperimentLogger(Node):
    def __init__(self, output_dir: Path, run_name: str, expected: dict[str, str], notes: str):
        super().__init__("experiment_logger")
        self.output_dir = output_dir
        self.run_name = run_name
        self.expected = expected
        self.notes = notes
        self.start_time = time.time()
        self.shutdown_requested = False
        self.event_count = 0
        self.label_counts: dict[str, Counter] = defaultdict(Counter)
        self.confidence_sums: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.confidence_counts: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.csv"
        self.metadata_path = self.output_dir / "metadata.json"
        self.summary_path = self.output_dir / "summary.csv"

        self.events_file = self.events_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.events_file,
            fieldnames=[
                "t_rel_sec",
                "topic",
                "label",
                "confidence",
                "linear_x",
                "angular_z",
            ],
        )
        self.writer.writeheader()

        self.create_subscription(String, "/human_gesture", self._gesture_cb, 10)
        self.create_subscription(String, "/human_movement", self._movement_cb, 10)
        self.create_subscription(String, "/human_action", self._action_cb, 10)
        self.create_subscription(TwistStamped, "/cmd_vel", self._cmd_cb, 10)

    def _t_rel(self) -> float:
        return time.time() - self.start_time

    def _write_event(
        self,
        *,
        topic: str,
        label: str = "",
        confidence: float | None = None,
        linear_x: float | None = None,
        angular_z: float | None = None,
    ) -> None:
        self.writer.writerow(
            {
                "t_rel_sec": f"{self._t_rel():.3f}",
                "topic": topic,
                "label": label,
                "confidence": "" if confidence is None else f"{confidence:.4f}",
                "linear_x": "" if linear_x is None else f"{linear_x:.4f}",
                "angular_z": "" if angular_z is None else f"{angular_z:.4f}",
            }
        )
        self.events_file.flush()
        self.event_count += 1
        if label:
            self.label_counts[topic][label] += 1
            if confidence is not None:
                self.confidence_sums[topic][label] += confidence
                self.confidence_counts[topic][label] += 1

    def _gesture_cb(self, msg: String) -> None:
        label, conf = parse_label_message(msg.data)
        self._write_event(topic="gesture", label=label, confidence=conf)

    def _movement_cb(self, msg: String) -> None:
        label, conf = parse_label_message(msg.data)
        self._write_event(topic="movement", label=label, confidence=conf)

    def _action_cb(self, msg: String) -> None:
        label, conf = parse_label_message(msg.data)
        self._write_event(topic="action", label=label, confidence=conf)

    def _cmd_cb(self, msg: TwistStamped) -> None:
        self._write_event(
            topic="cmd_vel",
            linear_x=float(msg.twist.linear.x),
            angular_z=float(msg.twist.angular.z),
        )

    def finalize(self) -> None:
        if self.shutdown_requested:
            return
        self.shutdown_requested = True

        duration = self._t_rel()
        metadata = {
            "run_name": self.run_name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "duration_sec": round(duration, 3),
            "notes": self.notes,
            "expected": self.expected,
            "event_count": self.event_count,
            "events_csv": str(self.events_path),
            "summary_csv": str(self.summary_path),
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        with self.summary_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "topic",
                    "label",
                    "count",
                    "avg_confidence",
                    "expected_label",
                    "matches_expected",
                ],
            )
            writer.writeheader()
            for topic, counts in sorted(self.label_counts.items()):
                expected_label = self.expected.get(topic, "")
                for label, count in sorted(counts.items()):
                    conf_count = self.confidence_counts[topic][label]
                    avg_conf = (
                        self.confidence_sums[topic][label] / conf_count
                        if conf_count else ""
                    )
                    writer.writerow(
                        {
                            "topic": topic,
                            "label": label,
                            "count": count,
                            "avg_confidence": "" if avg_conf == "" else f"{avg_conf:.4f}",
                            "expected_label": expected_label,
                            "matches_expected": (
                                "yes" if expected_label and label == expected_label else "no"
                                if expected_label else ""
                            ),
                        }
                    )

        self.events_file.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Log ROS2 intention topics to CSV")
    parser.add_argument("--output-dir", default="results", help="Base output directory")
    parser.add_argument("--run-name", default=None, help="Optional run name")
    parser.add_argument("--duration", type=float, default=0.0, help="Auto-stop after N seconds")
    parser.add_argument("--expected-gesture", default="", help="Expected dominant gesture label")
    parser.add_argument("--expected-movement", default="", help="Expected dominant movement label")
    parser.add_argument("--expected-action", default="", help="Expected dominant action label")
    parser.add_argument("--notes", default="", help="Free-text notes for the run")
    args = parser.parse_args(argv)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"run_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    expected = {
        "gesture": args.expected_gesture.strip(),
        "movement": args.expected_movement.strip(),
        "action": args.expected_action.strip(),
    }

    rclpy.init(args=None)
    node = ExperimentLogger(output_dir=output_dir, run_name=run_name, expected=expected, notes=args.notes)

    def _request_shutdown(*_args):
        node.get_logger().info("Stopping logger...")
        node.finalize()
        node.destroy_node()
        rclpy.shutdown()

    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)

    start = time.time()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if args.duration > 0 and (time.time() - start) >= args.duration:
                _request_shutdown()
                break
    finally:
        if not node.shutdown_requested:
            node.finalize()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print(f"Saved run to: {output_dir}")
    print(f"Events: {node.event_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
