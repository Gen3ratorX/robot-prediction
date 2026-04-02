"""Shared movement feature engineering utilities."""

import numpy as np


HIP_JOINTS = (23, 24)
VELOCITY_JOINTS = (11, 12, 23, 24, 27, 28)
POSITION_FEATURE_SIZE = 33 * 3
VELOCITY_FEATURE_SIZE = len(VELOCITY_JOINTS) * 3
MOVEMENT_FEATURE_SIZE = POSITION_FEATURE_SIZE + VELOCITY_FEATURE_SIZE
LANDMARK_SMOOTHING_ALPHA = 0.6
STATIONARY_MOTION_THRESHOLD = 0.01


def compute_hip_center(skeleton_seq):
    """Use the midpoint between left/right hips as the body center."""
    return skeleton_seq[:, HIP_JOINTS, :].mean(axis=1, keepdims=True)


def smooth_skeleton_sequence(skeleton_seq, alpha=LANDMARK_SMOOTHING_ALPHA):
    """Apply simple EMA smoothing over time to reduce landmark jitter."""
    smoothed = np.array(skeleton_seq, dtype=np.float32, copy=True)
    for t in range(1, smoothed.shape[0]):
        smoothed[t] = alpha * smoothed[t] + (1.0 - alpha) * smoothed[t - 1]
    return smoothed


def normalize_movement_sequence(skeleton_seq):
    """Center a sequence on the hip midpoint and scale by global std."""
    centered = skeleton_seq - compute_hip_center(skeleton_seq)
    std = centered.std()
    if std > 1e-6:
        centered = centered / std
    return centered


def compute_velocity_features(skeleton_seq, joint_indices=VELOCITY_JOINTS):
    """Frame-to-frame displacement for a small set of motion-heavy joints."""
    velocities = np.zeros(
        (skeleton_seq.shape[0], len(joint_indices), skeleton_seq.shape[2]),
        dtype=np.float32
    )
    velocities[1:] = skeleton_seq[1:, joint_indices, :] - skeleton_seq[:-1, joint_indices, :]
    return velocities


def estimate_motion_energy(skeleton_seq, joint_indices=VELOCITY_JOINTS):
    """Estimate average frame-to-frame motion magnitude for selected joints."""
    smoothed = smooth_skeleton_sequence(skeleton_seq)
    velocities = compute_velocity_features(smoothed, joint_indices=joint_indices)
    magnitudes = np.linalg.norm(velocities[1:], axis=2)
    return float(np.mean(magnitudes)) if magnitudes.size else 0.0


def apply_stationary_motion_gate(
    probs,
    idx_to_class,
    motion_energy,
    threshold=STATIONARY_MOTION_THRESHOLD,
):
    """Override to stationary when measured motion is below a small threshold."""
    stationary_idx = next(
        (idx for idx, name in idx_to_class.items() if name == 'stationary'),
        None
    )
    if stationary_idx is None or motion_energy >= threshold:
        pred_idx = int(np.argmax(probs))
        return pred_idx, float(probs[pred_idx]), False

    top_conf = float(np.max(probs))
    stationary_conf = float(probs[stationary_idx])
    return stationary_idx, max(stationary_conf, top_conf), True


def build_movement_features(skeleton_seq):
    """Flatten normalized positions and selected joint velocities per frame."""
    smoothed = smooth_skeleton_sequence(skeleton_seq)
    normalized = normalize_movement_sequence(smoothed)
    positions = normalized.reshape(normalized.shape[0], POSITION_FEATURE_SIZE)
    velocities = compute_velocity_features(normalized).reshape(
        normalized.shape[0], VELOCITY_FEATURE_SIZE
    )
    return np.concatenate([positions, velocities], axis=1)
