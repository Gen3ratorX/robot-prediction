import os
import sys
import unittest

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "robot_prediction"))

from movement_features import (  # noqa: E402
    MOVEMENT_FEATURE_SIZE,
    POSITION_FEATURE_SIZE,
    VELOCITY_FEATURE_SIZE,
    VELOCITY_JOINTS,
    build_movement_features,
    compute_hip_center,
    compute_velocity_features,
    normalize_movement_sequence,
)


class MovementFeatureTests(unittest.TestCase):
    def test_hip_center_uses_mediapipe_hips_not_nose(self):
        seq = np.zeros((4, 33, 3), dtype=np.float32)
        seq[:, 0, :] = np.array([100.0, 200.0, -50.0], dtype=np.float32)
        seq[:, 23, :] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        seq[:, 24, :] = np.array([3.0, 4.0, 5.0], dtype=np.float32)

        hip_center = compute_hip_center(seq)

        expected = np.tile(
            np.array([2.0, 3.0, 4.0], dtype=np.float32),
            (seq.shape[0], 1),
        )
        np.testing.assert_allclose(hip_center[:, 0, :], expected)

    def test_normalization_centers_on_average_hip_midpoint(self):
        seq = np.zeros((4, 33, 3), dtype=np.float32)
        seq[:, 0, :] = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        seq[:, 23, :] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        seq[:, 24, :] = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        seq[:, 11, :] = np.array([2.0, 1.0, 3.0], dtype=np.float32)

        normalized = normalize_movement_sequence(seq)
        normalized_hip_center = normalized[:, (23, 24), :].mean(axis=1)

        np.testing.assert_allclose(
            normalized_hip_center,
            np.zeros((4, 3), dtype=np.float32),
            atol=1e-6,
        )

    def test_velocity_features_track_selected_joint_displacement(self):
        seq = np.zeros((5, 33, 3), dtype=np.float32)
        delta = np.array([0.25, -0.5, 0.125], dtype=np.float32)
        for frame_idx in range(seq.shape[0]):
            seq[frame_idx, VELOCITY_JOINTS, :] = frame_idx * delta

        velocities = compute_velocity_features(seq)

        self.assertEqual(velocities.shape, (5, len(VELOCITY_JOINTS), 3))
        np.testing.assert_allclose(velocities[0], 0.0)
        expected = np.tile(delta, (seq.shape[0] - 1, len(VELOCITY_JOINTS), 1))
        np.testing.assert_allclose(velocities[1:], expected)

    def test_build_movement_features_has_expected_position_velocity_layout(self):
        seq = np.zeros((6, 33, 3), dtype=np.float32)
        seq[:, 23, :] = np.array([0.4, 0.5, 0.0], dtype=np.float32)
        seq[:, 24, :] = np.array([0.6, 0.5, 0.0], dtype=np.float32)

        features = build_movement_features(seq)

        self.assertEqual(features.shape, (6, MOVEMENT_FEATURE_SIZE))
        velocity_start = POSITION_FEATURE_SIZE
        velocity_end = velocity_start + VELOCITY_FEATURE_SIZE
        self.assertEqual(
            features[:, velocity_start:velocity_end].shape,
            (6, VELOCITY_FEATURE_SIZE),
        )


if __name__ == "__main__":
    unittest.main()
