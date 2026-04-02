#!/usr/bin/env python3
"""
Standalone Movement Recorder (No ROS Required)
================================================

Records skeleton keypoint sequences using webcam + MediaPipe.
No ROS needed — just run directly with Python.

Usage:
    python3 record_movement.py --class_name approaching --num_samples 50
    python3 record_movement.py --class_name moving_away --num_samples 50
    python3 record_movement.py --class_name moving_left --num_samples 50
    python3 record_movement.py --class_name moving_right --num_samples 50
    python3 record_movement.py --class_name stationary --num_samples 50

Controls:
    SPACE  — Start recording a sequence
    Q      — Quit
    R      — Redo last sample (deletes it and re-records)

Tips:
    - Stand 1-3 meters from camera
    - Make sure full body is visible
    - Good lighting helps MediaPipe detect you
    - Each sequence is 30 frames (~1 second)
    - Move consistently for each class
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2
import mediapipe as mp


def main():
    parser = argparse.ArgumentParser(description='Record movement skeleton data')
    parser.add_argument('--class_name', type=str, required=True,
                       choices=['approaching', 'moving_away', 'moving_left',
                               'moving_right', 'stationary'],
                       help='Movement class to record')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to record (default: 50)')
    parser.add_argument('--seq_length', type=int, default=30,
                       help='Frames per sequence (default: 30)')
    parser.add_argument('--save_dir', type=str, default='data/movement',
                       help='Base save directory')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index (default: 0)')
    args = parser.parse_args()

    # Setup
    class_dir = os.path.join(args.save_dir, args.class_name)
    os.makedirs(class_dir, exist_ok=True)

    # Count existing samples
    existing = sorted([f for f in os.listdir(class_dir) if f.endswith('.npy')])
    sample_count = len(existing)

    # MediaPipe setup
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        print("If you're in a VM, make sure webcam is connected to VM")
        print("Try: ls /dev/video*")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Recording state
    recording = False
    current_sequence = []
    countdown = 0
    countdown_start = 0

    # Movement instructions
    instructions = {
        'approaching': "Walk TOWARD the camera",
        'moving_away': "Walk AWAY from the camera",
        'moving_left': "Walk to YOUR LEFT",
        'moving_right': "Walk to YOUR RIGHT",
        'stationary': "Stand STILL (small movements OK)",
    }

    print("\n" + "=" * 60)
    print(f"RECORDING: {args.class_name}")
    print(f"Instruction: {instructions[args.class_name]}")
    print(f"Target: {args.num_samples} samples of {args.seq_length} frames")
    print(f"Existing samples: {sample_count}")
    print(f"Remaining: {max(0, args.num_samples - sample_count)}")
    print("=" * 60)
    print("\nControls:")
    print("  SPACE  — Start recording (3-second countdown)")
    print("  R      — Redo last sample")
    print("  Q      — Quit")
    print()

    while sample_count < args.num_samples:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Lost camera connection")
            break

        # Mirror frame for more intuitive movement
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        h, w = frame.shape[:2]

        # ---- Handle countdown ----
        if countdown > 0:
            elapsed = time.time() - countdown_start
            remaining = countdown - elapsed
            if remaining <= 0:
                recording = True
                current_sequence = []
                countdown = 0
                print(f"  GO! Recording sample {sample_count}...")
            else:
                # Show countdown
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 0), -1)
                count_text = str(int(remaining) + 1)
                text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX,
                                           5, 10)[0]
                tx = (w - text_size[0]) // 2
                ty = (h + text_size[1]) // 2
                cv2.putText(frame, count_text, (tx, ty),
                           cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)
                cv2.putText(frame, "GET READY!", (w//2 - 120, ty + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # ---- Draw skeleton ----
        elif results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

        # ---- Recording logic ----
        if recording:
            if results.pose_landmarks:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                current_sequence.append(landmarks)

                # Progress bar
                progress = len(current_sequence) / args.seq_length
                bar_y = h - 40
                bar_w = w - 40
                cv2.rectangle(frame, (20, bar_y), (20 + bar_w, bar_y + 20),
                             (50, 50, 50), -1)
                cv2.rectangle(frame, (20, bar_y),
                             (20 + int(bar_w * progress), bar_y + 20),
                             (0, 0, 255), -1)
                cv2.putText(frame, f"Recording: {len(current_sequence)}/{args.seq_length}",
                           (20, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (0, 0, 255), 2)

                # Done recording this sample
                if len(current_sequence) >= args.seq_length:
                    seq_array = np.array(current_sequence, dtype=np.float32)
                    save_path = os.path.join(class_dir, f"seq{sample_count:04d}.npy")
                    np.save(save_path, seq_array)
                    print(f"  Saved: {save_path} — shape {seq_array.shape}")

                    sample_count += 1
                    recording = False
                    current_sequence = []

                    remaining_samples = args.num_samples - sample_count
                    if remaining_samples > 0:
                        print(f"  {remaining_samples} samples remaining. "
                              f"Press SPACE for next.")
            else:
                # No pose detected during recording
                cv2.putText(frame, "NO POSE DETECTED - Move into view!",
                           (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (0, 0, 255), 2)

        # ---- Draw UI overlay ----
        if countdown <= 0:
            # Top bar
            cv2.rectangle(frame, (0, 0), (w, 95), (0, 0, 0), -1)

            status_color = (0, 0, 255) if recording else (0, 255, 0)
            status_text = "RECORDING" if recording else "READY"

            cv2.putText(frame, f"Class: {args.class_name.upper()}",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Status: {status_text}",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(frame, f"Sample: {sample_count}/{args.num_samples}",
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Instruction
            cv2.putText(frame, instructions[args.class_name],
                       (w//2 - 150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 255, 255), 1)

            # Pose detection status
            if not results.pose_landmarks and not recording:
                cv2.putText(frame, "Stand in front of camera (full body visible)",
                           (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (0, 165, 255), 2)

        cv2.imshow('Movement Recorder', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and not recording and countdown <= 0:
            # Start 3-second countdown
            countdown = 3
            countdown_start = time.time()
            print(f"\n  Starting countdown for sample {sample_count}...")

        elif key == ord('r') and not recording:
            # Redo last sample
            if sample_count > 0:
                sample_count -= 1
                redo_path = os.path.join(class_dir, f"seq{sample_count:04d}.npy")
                if os.path.exists(redo_path):
                    os.remove(redo_path)
                    print(f"  Deleted {redo_path} — press SPACE to re-record")

        elif key == ord('q'):
            print("\nQuitting early.")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    final_count = len([f for f in os.listdir(class_dir) if f.endswith('.npy')])
    print(f"\n{'='*60}")
    print(f"DONE — {args.class_name}")
    print(f"Total samples saved: {final_count}")
    print(f"Location: {os.path.abspath(class_dir)}")
    print(f"{'='*60}")

    if final_count < args.num_samples:
        print(f"\nWARNING: Only recorded {final_count}/{args.num_samples} samples.")
        print(f"Run again to continue where you left off.")


if __name__ == '__main__':
    main()
