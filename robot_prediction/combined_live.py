#!/usr/bin/env python3
"""
Combined Live Test — Gesture Landmarks + Movement LSTM + ST-GCN
=================================================================

Gesture: Landmark-based Random Forest (fast, accurate)
Movement: LSTM on skeleton sequences
Action: ST-GCN on skeleton graph sequences

Usage:
    python3 combined_live.py --model_dir checkpoints
"""

import argparse
import os
import json
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import MovementLSTM, SpatioTemporalGCN
from movement_features import (
    MOVEMENT_FEATURE_SIZE,
    apply_stationary_motion_gate,
    build_movement_features,
    estimate_depth_scale_signature,
    estimate_motion_energy,
)
from gesture_landmarks import extract_advanced_features


def main():
    import cv2
    cv2.namedWindow('Human-Aware Robot Navigation', cv2.WINDOW_NORMAL)

    import mediapipe as mp

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='checkpoints')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--confidence', type=float, default=0.6)
    parser.add_argument('--seq_length', type=int, default=30)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ========================================
    # LOAD GESTURE LANDMARK MODEL
    # ========================================
    gesture_model = None
    gesture_scaler = None
    gesture_classes = None

    model_path = os.path.join(args.model_dir, 'gesture_landmark_model.pkl')
    scaler_path = os.path.join(args.model_dir, 'gesture_landmark_scaler.pkl')
    classes_path = os.path.join(args.model_dir, 'gesture_landmark_classes.json')

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            gesture_model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            gesture_scaler = pickle.load(f)
        with open(classes_path, 'r') as f:
            gesture_classes = json.load(f)
        print(f"Loaded Gesture Landmark Model: {gesture_classes}")
    else:
        print("WARNING: Gesture landmark model not found — skipping")

    # ========================================
    # LOAD MOVEMENT LSTM
    # ========================================
    movement_model = None
    movement_classes = None

    movement_json = os.path.join(args.model_dir, 'movement_classes.json')
    movement_pth = os.path.join(args.model_dir, 'movement_lstm_best.pth')
    if os.path.exists(movement_json) and os.path.exists(movement_pth):
        with open(movement_json) as f:
            classes = json.load(f)
        movement_classes = {v: k for k, v in classes.items()}

        movement_model = MovementLSTM(
            input_size=MOVEMENT_FEATURE_SIZE,
            num_classes=len(classes)
        )
        ckpt = torch.load(movement_pth, map_location=device, weights_only=True)
        movement_model.load_state_dict(ckpt['model_state_dict'])
        movement_model.to(device).eval()
        print(f"Loaded Movement LSTM: {list(classes.keys())}")
    else:
        print("WARNING: Movement model not found — skipping")

    # ========================================
    # LOAD ST-GCN
    # ========================================
    action_model = None
    action_classes = None

    gcn_json = os.path.join(args.model_dir, 'stgcn_classes.json')
    gcn_pth = os.path.join(args.model_dir, 'stgcn_best.pth')
    if os.path.exists(gcn_json) and os.path.exists(gcn_pth):
        with open(gcn_json) as f:
            classes = json.load(f)
        action_classes = {v: k for k, v in classes.items()}

        action_model = SpatioTemporalGCN(num_classes=len(classes), num_joints=33)
        ckpt = torch.load(gcn_pth, map_location=device, weights_only=True)
        action_model.load_state_dict(ckpt['model_state_dict'])
        action_model.to(device).eval()
        print(f"Loaded ST-GCN: {list(classes.keys())}")
    else:
        print("WARNING: ST-GCN model not found — skipping")

    # ========================================
    # MEDIAPIPE
    # ========================================
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # ========================================
    # CAMERA
    # ========================================
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Buffers
    skeleton_buffer = deque(maxlen=args.seq_length)
    gesture_buffer = deque(maxlen=10)
    movement_buffer = deque(maxlen=20)
    action_buffer = deque(maxlen=20)
    fps_buffer = deque(maxlen=30)
    movement_display_name = None
    movement_display_conf = 0.0
    movement_candidate_name = None
    movement_candidate_conf = 0.0
    movement_candidate_streak = 0
    movement_hysteresis_frames = 3

    print("\n" + "=" * 55)
    print("HUMAN-AWARE ROBOT NAVIGATION — COMBINED LIVE TEST")
    print("Gesture: Landmark-based (Random Forest)")
    print("Movement: LSTM (skeleton sequences)")
    print("Action: ST-GCN (skeleton graphs)")
    print("Press Q to quit")
    print("=" * 55 + "\n")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        gesture_text = ""
        gesture_name = ""
        gesture_conf = 0.0
        movement_text = ""
        action_text = ""
        confidence_text = ""
        robot_cmd = "CONTINUE FORWARD"
        hand_found = False
        hand_label = "?"
        move_name = None
        move_conf = 0.0
        move_probs = None
        action_name = None
        action_conf = 0.0
        action_probs = None

        # ============================================
        # GESTURE RECOGNITION (Landmark-based)
        # ============================================
        if gesture_model is not None:
            hand_results = hands.process(frame_rgb)

            if hand_results.multi_hand_landmarks:
                hand_found = True
                hand_landmarks = hand_results.multi_hand_landmarks[0]

                if hand_results.multi_handedness:
                    hand_label = hand_results.multi_handedness[0].classification[0].label[0]

                # Draw hand skeleton
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                )

                # Extract landmarks
                raw = []
                for lm in hand_landmarks.landmark:
                    raw.extend([lm.x, lm.y, lm.z])

                # Get advanced features and predict
                features = extract_advanced_features(raw)
                features_scaled = gesture_scaler.transform([features])
                probs = gesture_model.predict_proba(features_scaled)[0]
                gesture_buffer.append(probs)

                if len(gesture_buffer) >= 3:
                    avg_probs = np.mean(list(gesture_buffer), axis=0)
                    pred_idx = np.argmax(avg_probs)
                    gesture_conf = avg_probs[pred_idx]
                    gesture_name = gesture_classes[pred_idx]

                    if gesture_conf > args.confidence:
                        gesture_text = f"Gesture: {gesture_name} ({gesture_conf:.0%}) [{hand_label}]"
                    else:
                        gesture_text = f"Gesture: uncertain ({gesture_conf:.0%})"
                else:
                    gesture_text = "Gesture: buffering..."
            else:
                gesture_buffer.clear()
                gesture_text = ""

        # ============================================
        # MOVEMENT & ACTION PREDICTION (Pose)
        # ============================================
        pose_results = pose.process(frame_rgb)

        if pose_results.pose_landmarks:
            # Draw pose skeleton
            mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1)
            )

            landmarks = []
            for lm in pose_results.pose_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            skeleton = np.array(landmarks, dtype=np.float32)
            skeleton_buffer.append(skeleton)

            if len(skeleton_buffer) == args.seq_length:
                skeleton_seq = np.array(list(skeleton_buffer))

                # Movement LSTM
                if movement_model is not None:
                    flat = build_movement_features(skeleton_seq)
                    motion_energy = estimate_motion_energy(skeleton_seq)
                    depth_scale = estimate_depth_scale_signature(skeleton_seq)
                    input_t = torch.tensor(flat).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = movement_model(input_t)
                        probs = F.softmax(logits, dim=1)

                    movement_buffer.append(probs.cpu().numpy()[0])

                    if len(movement_buffer) >= 3:
                        avg_probs = np.mean(list(movement_buffer), axis=0)
                        move_probs = avg_probs
                        pred_idx, move_conf, stationary_gated = apply_stationary_motion_gate(
                            avg_probs,
                            movement_classes,
                            motion_energy,
                            depth_change=depth_scale['depth_change'],
                            scale_change=depth_scale['scale_change'],
                        )
                        move_name = movement_classes[pred_idx]
                        if move_conf > args.confidence:
                            if movement_display_name is None:
                                movement_display_name = move_name
                                movement_display_conf = move_conf
                                movement_candidate_name = None
                                movement_candidate_conf = 0.0
                                movement_candidate_streak = 0
                            elif move_name == movement_display_name:
                                movement_display_conf = move_conf
                                movement_candidate_name = None
                                movement_candidate_conf = 0.0
                                movement_candidate_streak = 0
                            else:
                                if move_name == movement_candidate_name:
                                    movement_candidate_streak += 1
                                    movement_candidate_conf = move_conf
                                else:
                                    movement_candidate_name = move_name
                                    movement_candidate_conf = move_conf
                                    movement_candidate_streak = 1

                                if movement_candidate_streak >= movement_hysteresis_frames:
                                    movement_display_name = movement_candidate_name
                                    movement_display_conf = movement_candidate_conf
                                    movement_candidate_name = None
                                    movement_candidate_conf = 0.0
                                    movement_candidate_streak = 0

                            movement_text = (
                                f"Movement: {movement_display_name} "
                                f"({movement_display_conf:.0%})"
                            )
                        elif movement_display_name is not None:
                            movement_text = (
                                f"Movement: {movement_display_name} "
                                f"({movement_display_conf:.0%})"
                            )
                        else:
                            movement_text = f"Movement: uncertain ({move_conf:.0%})"
                        if stationary_gated and movement_display_name == 'stationary':
                            movement_text += " [stable]"

                # ST-GCN
                if action_model is not None:
                    seq_t = skeleton_seq.transpose(2, 0, 1)
                    input_t = torch.tensor(seq_t).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = action_model(input_t)
                        probs = F.softmax(logits, dim=1)

                    action_buffer.append(probs.cpu().numpy()[0])

                    if len(action_buffer) >= 3:
                        avg_probs = np.mean(list(action_buffer), axis=0)
                        action_probs = avg_probs
                        pred_idx = np.argmax(avg_probs)
                        action_conf = avg_probs[pred_idx]
                        action_name = action_classes[pred_idx]
                        if action_conf > args.confidence:
                            action_text = f"Action: {action_name} ({action_conf:.0%})"
                        else:
                            action_text = f"Action: uncertain ({action_conf:.0%})"
        else:
            movement_text = "Movement: No person detected"
            movement_candidate_name = None
            movement_candidate_conf = 0.0
            movement_candidate_streak = 0

        # ============================================
        # ROBOT DECISION (Priority System)
        # ============================================
        # Priority 1: Gesture commands (highest)
        gesture_valid = (hand_found and gesture_name and
                        gesture_conf > args.confidence and
                        "uncertain" not in gesture_text and
                        "buffering" not in gesture_text)

        if gesture_valid and gesture_name == "stop":
            robot_cmd = "STOP (gesture)"
        elif gesture_valid and gesture_name == "forward":
            robot_cmd = "MOVE FORWARD (gesture)"
        elif gesture_valid and gesture_name == "backward":
            robot_cmd = "MOVE BACKWARD (gesture)"
        elif gesture_valid and gesture_name == "left":
            robot_cmd = "TURN LEFT (gesture)"
        elif gesture_valid and gesture_name == "right":
            robot_cmd = "TURN RIGHT (gesture)"
        # Priority 2: Movement avoidance
        else:
            fused_name = None
            fused_conf = 0.0
            if move_probs is not None and action_probs is not None:
                movement_weight = max(move_conf, 1e-6)
                action_weight = max(action_conf, 1e-6)
                fused_scores = {}
                for idx, name in movement_classes.items():
                    fused_scores[name] = fused_scores.get(name, 0.0) + movement_weight * move_probs[idx]
                for idx, name in action_classes.items():
                    fused_scores[name] = fused_scores.get(name, 0.0) + action_weight * action_probs[idx]

                fused_name, fused_score = max(fused_scores.items(), key=lambda item: item[1])
                fused_conf = fused_score / (movement_weight + action_weight)
            elif move_name and move_conf > args.confidence:
                fused_name = move_name
                fused_conf = move_conf
            elif action_name and action_conf > args.confidence:
                fused_name = action_name
                fused_conf = action_conf

            if move_conf > 0.0 or action_conf > 0.0:
                confidence_text = (
                    f"Motion Conf  LSTM:{move_conf:.0%}  ST-GCN:{action_conf:.0%}"
                )
                if fused_name is not None:
                    confidence_text += f"  Fused:{fused_conf:.0%}"

            if fused_name == "approaching" and fused_conf > args.confidence:
                robot_cmd = "STOP + TURN (fused)"
            elif fused_name == "moving_left" and fused_conf > args.confidence:
                robot_cmd = "ADJUST RIGHT (fused)"
            elif fused_name == "moving_right" and fused_conf > args.confidence:
                robot_cmd = "ADJUST LEFT (fused)"
            elif move_name == "approaching" and move_conf > args.confidence:
                robot_cmd = "CAUTION (LSTM: approaching)"

        # ============================================
        # DRAW UI
        # ============================================
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 132), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h - 55), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Predictions
        y = 22
        for text, color in [(gesture_text, (0, 255, 255)),
                            (movement_text, (255, 200, 0)),
                            (action_text, (200, 255, 0))]:
            if text:
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, color, 2)
                y += 22
        if confidence_text:
            cv2.putText(frame, confidence_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (180, 220, 255), 2)

        # Hand indicator
        if hand_found:
            hand_status = f"HAND: {hand_label}"
        else:
            hand_status = "HAND: NO"
        hand_color = (0, 255, 0) if hand_found else (100, 100, 100)
        cv2.putText(frame, hand_status, (w - 120, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, hand_color, 2)

        # Robot command
        if "STOP" in robot_cmd:
            cmd_color = (0, 0, 255)
        elif "TURN" in robot_cmd or "ADJUST" in robot_cmd:
            cmd_color = (0, 200, 255)
        elif "BACKWARD" in robot_cmd:
            cmd_color = (0, 100, 255)
        else:
            cmd_color = (0, 255, 0)
        cv2.putText(frame, f"Robot: {robot_cmd}", (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cmd_color, 2)

        # FPS
        fps_buffer.append(1.0 / max(time.time() - start_time, 0.001))
        fps = np.mean(fps_buffer)
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 80, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Buffers
        cv2.putText(frame, f"Skel: {len(skeleton_buffer)}/{args.seq_length}",
                    (w - 120, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Gest: {len(gesture_buffer)}/10",
                    (w - 120, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow('Human-Aware Robot Navigation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pose.close()
    print("Test complete!")


if __name__ == '__main__':
    main()
