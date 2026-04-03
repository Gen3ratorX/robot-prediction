"""Shared movement feature engineering utilities."""

import numpy as np


HIP_JOINTS = (23, 24)
SHOULDER_JOINTS = (11, 12)
TORSO_JOINTS = (11, 12, 23, 24)
VELOCITY_JOINTS = (11, 12, 23, 24, 27, 28)
POSITION_FEATURE_SIZE = 33 * 3
VELOCITY_FEATURE_SIZE = len(VELOCITY_JOINTS) * 3
APPROACH_AWAY_FEATURE_SIZE = 8
MOVEMENT_FEATURE_SIZE = (
    POSITION_FEATURE_SIZE + VELOCITY_FEATURE_SIZE + APPROACH_AWAY_FEATURE_SIZE
)
LANDMARK_SMOOTHING_ALPHA = 0.6
STATIONARY_MOTION_THRESHOLD = 0.01
DEPTH_MOTION_THRESHOLD = 0.008
SCALE_CHANGE_THRESHOLD = 0.03


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


def _pairwise_distance(seq, joint_a, joint_b):
    return np.linalg.norm(seq[:, joint_a, :] - seq[:, joint_b, :], axis=1)


def compute_approach_away_features(skeleton_seq):
    """Per-frame cues that preserve motion toward/away from the camera."""
    hip_center = compute_hip_center(skeleton_seq)[:, 0, :]
    torso_center = skeleton_seq[:, TORSO_JOINTS, :].mean(axis=1)

    hip_depth = hip_center[:, 2]
    torso_depth = torso_center[:, 2]
    shoulder_width = _pairwise_distance(skeleton_seq, 11, 12)
    hip_width = _pairwise_distance(skeleton_seq, 23, 24)
    torso_height = np.linalg.norm(torso_center[:, :2] - hip_center[:, :2], axis=1)
    nose_hip_distance = np.linalg.norm(skeleton_seq[:, 0, :] - hip_center, axis=1)

    features = np.zeros((skeleton_seq.shape[0], APPROACH_AWAY_FEATURE_SIZE), dtype=np.float32)
    features[:, 0] = hip_depth
    features[:, 1] = torso_depth
    features[:, 2] = shoulder_width
    features[:, 3] = hip_width
    features[:, 4] = torso_height
    features[:, 5] = nose_hip_distance
    features[1:, 6] = hip_depth[1:] - hip_depth[:-1]
    features[1:, 7] = shoulder_width[1:] - shoulder_width[:-1]
    return features


def estimate_depth_scale_signature(skeleton_seq):
    """Summarize forward/backward evidence from depth and apparent body scale."""
    smoothed = smooth_skeleton_sequence(skeleton_seq)
    approach_features = compute_approach_away_features(smoothed)
    depth_change = float(np.mean(np.abs(approach_features[1:, 6]))) if approach_features.shape[0] > 1 else 0.0
    signed_depth_delta = float(approach_features[-1, 0] - approach_features[0, 0]) if approach_features.shape[0] > 1 else 0.0

    shoulder_width = approach_features[:, 2]
    baseline_width = max(float(np.mean(shoulder_width[:3])), 1e-6)
    signed_scale_delta = float((shoulder_width[-1] - shoulder_width[0]) / baseline_width)
    scale_change = float(
        np.max(np.abs((shoulder_width - baseline_width) / baseline_width))
    )
    # MediaPipe z tends to become more negative as the subject approaches the camera.
    approaching_score = max(0.0, -signed_depth_delta) + max(0.0, signed_scale_delta)
    moving_away_score = max(0.0, signed_depth_delta) + max(0.0, -signed_scale_delta)
    return {
        'depth_change': depth_change,
        'scale_change': scale_change,
        'signed_depth_delta': signed_depth_delta,
        'signed_scale_delta': signed_scale_delta,
        'approaching_score': approaching_score,
        'moving_away_score': moving_away_score,
    }


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
    depth_change=0.0,
    scale_change=0.0,
    threshold=STATIONARY_MOTION_THRESHOLD,
):
    """Override to stationary when measured motion is below a small threshold."""
    stationary_idx = next(
        (idx for idx, name in idx_to_class.items() if name == 'stationary'),
        None
    )
    depth_or_scale_motion = (
        depth_change >= DEPTH_MOTION_THRESHOLD or
        scale_change >= SCALE_CHANGE_THRESHOLD
    )
    if stationary_idx is None or motion_energy >= threshold or depth_or_scale_motion:
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
    approach_away = compute_approach_away_features(smoothed)
    return np.concatenate([positions, velocities, approach_away], axis=1)
