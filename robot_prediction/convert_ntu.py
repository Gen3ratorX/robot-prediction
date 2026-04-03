#!/usr/bin/env python3
"""
NTU RGB+D to public movement bootstrap dataset converter.

Converts NTU `.skeleton` files into this repo's movement format:
    (30, 33, 3) numpy arrays

Key differences from the original bootstrap script:
- uses safer NTU candidate classes
- preserves sequence-level depth/scale trends
- assigns final labels from measured motion, not source action alone
- writes into a separate output tree by default

Usage:
    python3 convert_ntu.py \
      --skeleton_dir ~/Desktop/ntu-rgbd/nturgb+d_skeletons \
      --output_dir data/movement_public
"""

import argparse
import os
import re
from collections import Counter, defaultdict
from glob import glob

import numpy as np


SEQ_LENGTH = 30
MIN_VALID_FRAMES = 20
X_MOTION_THRESHOLD = 0.08
Z_MOTION_THRESHOLD = 0.12
SCALE_CHANGE_THRESHOLD = 0.10


def parse_skeleton_file(filepath):
    """
    Parse one NTU RGB+D `.skeleton` file.
    Returns a list of frames, each shaped (25, 3) for the first body only.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    idx = 0
    num_frames = int(lines[idx].strip())
    idx += 1

    all_frames = []
    for _ in range(num_frames):
        if idx >= len(lines):
            break

        num_bodies = int(lines[idx].strip())
        idx += 1

        frame_bodies = []
        for _ in range(num_bodies):
            if idx >= len(lines):
                break

            idx += 1  # body info
            num_joints = int(lines[idx].strip())
            idx += 1

            joints = []
            for _ in range(num_joints):
                if idx >= len(lines):
                    break
                parts = lines[idx].strip().split()
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                joints.append([x, y, z])
                idx += 1

            if len(joints) == 25:
                frame_bodies.append(np.array(joints, dtype=np.float32))

        if frame_bodies:
            all_frames.append(frame_bodies[0])

    return all_frames


def kinect25_to_mediapipe33(kinect_joints):
    """Map 25 Kinect joints to 33 MediaPipe-style joints without per-frame normalization."""
    mp = np.zeros((33, 3), dtype=np.float32)

    # Face approximation from head
    mp[0] = kinect_joints[3]
    mp[1] = kinect_joints[3] + [0.02, -0.01, 0.0]
    mp[2] = kinect_joints[3] + [0.03, -0.01, 0.0]
    mp[3] = kinect_joints[3] + [0.04, -0.01, 0.0]
    mp[4] = kinect_joints[3] + [-0.02, -0.01, 0.0]
    mp[5] = kinect_joints[3] + [-0.03, -0.01, 0.0]
    mp[6] = kinect_joints[3] + [-0.04, -0.01, 0.0]
    mp[7] = kinect_joints[3] + [0.05, 0.02, 0.0]
    mp[8] = kinect_joints[3] + [-0.05, 0.02, 0.0]
    mp[9] = kinect_joints[3] + [0.01, 0.03, 0.0]
    mp[10] = kinect_joints[3] + [-0.01, 0.03, 0.0]

    # Body
    mp[11] = kinect_joints[4]
    mp[12] = kinect_joints[8]
    mp[13] = kinect_joints[5]
    mp[14] = kinect_joints[9]
    mp[15] = kinect_joints[6]
    mp[16] = kinect_joints[10]
    mp[17] = kinect_joints[22]
    mp[18] = kinect_joints[24]
    mp[19] = kinect_joints[21]
    mp[20] = kinect_joints[23]
    mp[21] = kinect_joints[22]
    mp[22] = kinect_joints[24]
    mp[23] = kinect_joints[12]
    mp[24] = kinect_joints[16]
    mp[25] = kinect_joints[13]
    mp[26] = kinect_joints[17]
    mp[27] = kinect_joints[14]
    mp[28] = kinect_joints[18]
    mp[29] = kinect_joints[15]
    mp[30] = kinect_joints[19]
    mp[31] = kinect_joints[15] + [0.0, 0.0, 0.05]
    mp[32] = kinect_joints[19] + [0.0, 0.0, 0.05]

    return mp


def normalize_sequence_like_mediapipe(sequence):
    """
    Convert raw camera coordinates into a MediaPipe-like range while preserving
    sequence-wide motion trends. x/y are normalized over the whole sequence,
    z is centered on the first-frame hip midpoint and scaled by sequence std.
    """
    normalized = sequence.copy()

    for axis in (0, 1):
        values = normalized[:, :, axis]
        axis_min = values.min()
        axis_max = values.max()
        if axis_max - axis_min > 1e-6:
            normalized[:, :, axis] = (values - axis_min) / (axis_max - axis_min)
        else:
            normalized[:, :, axis] = 0.5

    hip_center = normalized[:, [23, 24], 2].mean(axis=1)
    z_center = hip_center[0]
    normalized[:, :, 2] = normalized[:, :, 2] - z_center
    z_std = normalized[:, :, 2].std()
    if z_std > 1e-6:
        normalized[:, :, 2] = normalized[:, :, 2] / z_std

    return normalized.astype(np.float32)


def extract_sequence(frames, seq_length=SEQ_LENGTH):
    """Sample a fixed-length sequence from the valid center of a clip."""
    if len(frames) < seq_length:
        indices = np.linspace(0, len(frames) - 1, seq_length)
    else:
        start = max((len(frames) - seq_length) // 2, 0)
        end = start + seq_length
        indices = np.linspace(start, end - 1, seq_length)

    sampled = []
    for idx in indices:
        sampled.append(frames[int(round(idx))].copy())
    return sampled


def compute_motion_signature(frames):
    """
    Summarize raw Kinect motion in camera coordinates.
    x: lateral movement
    z: depth (toward/away from camera)
    """
    seq = np.array(frames, dtype=np.float32)
    hip_center = seq[:, [12, 16], :].mean(axis=1)
    shoulder_width = np.linalg.norm(seq[:, 4, :] - seq[:, 8, :], axis=1)

    dx = float(hip_center[-1, 0] - hip_center[0, 0])
    dz = float(hip_center[-1, 2] - hip_center[0, 2])
    base_width = max(float(np.mean(shoulder_width[:3])), 1e-6)
    scale_change = float((shoulder_width[-1] - shoulder_width[0]) / base_width)

    return {
        'dx': dx,
        'dz': dz,
        'scale_change': scale_change,
    }


def classify_motion(signature):
    """Assign one of the repo's movement labels from measured motion."""
    dx = signature['dx']
    dz = signature['dz']
    scale_change = signature['scale_change']

    if dz <= -Z_MOTION_THRESHOLD or scale_change >= SCALE_CHANGE_THRESHOLD:
        return 'approaching'
    if dz >= Z_MOTION_THRESHOLD or scale_change <= -SCALE_CHANGE_THRESHOLD:
        return 'moving_away'
    if dx <= -X_MOTION_THRESHOLD:
        return 'moving_left'
    if dx >= X_MOTION_THRESHOLD:
        return 'moving_right'
    return 'stationary'


def convert_skeleton_to_sequence(skeleton_file, seq_length=SEQ_LENGTH):
    """Convert one `.skeleton` file into sequence data plus a motion signature."""
    frames = parse_skeleton_file(skeleton_file)
    if len(frames) < MIN_VALID_FRAMES:
        return None, None

    seq_frames = extract_sequence(frames, seq_length=seq_length)
    signature = compute_motion_signature(seq_frames)

    converted = [kinect25_to_mediapipe33(frame) for frame in seq_frames]
    sequence = normalize_sequence_like_mediapipe(np.array(converted, dtype=np.float32))
    return sequence, signature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--skeleton_dir',
        type=str,
        default=os.path.expanduser('~/Desktop/ntu-rgbd/nturgb+d_skeletons')
    )
    parser.add_argument('--output_dir', type=str, default='data/movement_public')
    parser.add_argument('--seq_length', type=int, default=SEQ_LENGTH)
    parser.add_argument('--max_per_class', type=int, default=300)
    args = parser.parse_args()

    # Candidate NTU actions. Final label still comes from measured motion.
    candidate_actions = {
        '028', '029', '034',  # stationary-ish actions
        '058', '059', '060',  # walking interaction actions
    }

    skeleton_files = sorted(glob(os.path.join(args.skeleton_dir, '*.skeleton')))
    print(f"Found {len(skeleton_files)} skeleton files")

    grouped_files = defaultdict(list)
    for path in skeleton_files:
        match = re.search(r'A(\d{3})\.skeleton$', os.path.basename(path))
        if not match:
            continue
        action_num = match.group(1)
        if action_num in candidate_actions:
            grouped_files[action_num].append(path)

    print("\nCandidate NTU actions:")
    for action_num in sorted(grouped_files):
        print(f"  A{action_num}: {len(grouped_files[action_num])} files")

    for class_name in ['approaching', 'moving_away', 'moving_left', 'moving_right', 'stationary']:
        os.makedirs(os.path.join(args.output_dir, class_name), exist_ok=True)

    saved_counts = Counter()
    skipped_counts = Counter()

    files_to_process = []
    for action_num in sorted(grouped_files):
        files_to_process.extend(grouped_files[action_num])

    for skel_file in files_to_process:
        if all(saved_counts[c] >= args.max_per_class for c in saved_counts if saved_counts):
            break

        try:
            sequence, signature = convert_skeleton_to_sequence(
                skel_file, seq_length=args.seq_length
            )
            if sequence is None:
                skipped_counts['too_short'] += 1
                continue

            movement_class = classify_motion(signature)
            if saved_counts[movement_class] >= args.max_per_class:
                skipped_counts[f'{movement_class}_full'] += 1
                continue

            class_dir = os.path.join(args.output_dir, movement_class)
            next_idx = len([f for f in os.listdir(class_dir) if f.endswith('.npy')])
            action_id = re.search(r'A(\d{3})', os.path.basename(skel_file)).group(1)
            save_name = f"ntu_A{action_id}_{next_idx:04d}.npy"
            np.save(os.path.join(class_dir, save_name), sequence)
            saved_counts[movement_class] += 1
        except Exception:
            skipped_counts['errors'] += 1

    print("\nSaved samples:")
    for class_name in ['approaching', 'moving_away', 'moving_left', 'moving_right', 'stationary']:
        total = len([
            f for f in os.listdir(os.path.join(args.output_dir, class_name))
            if f.endswith('.npy')
        ])
        print(f"  {class_name}: +{saved_counts[class_name]} -> {total} total")

    if skipped_counts:
        print("\nSkipped:")
        for reason, count in sorted(skipped_counts.items()):
            print(f"  {reason}: {count}")

    print("\nDone.")
    print("Bootstrap data written to:", args.output_dir)
    print("Suggested next step:")
    print("  1. Train on data/movement_public")
    print("  2. Fine-tune on data/movement")


if __name__ == '__main__':
    main()
