#!/usr/bin/env python3
"""
Live Test — Verify Models with Webcam (No ROS)
================================================

Uses MediaPipe Hands to detect and crop hand region for gesture CNN.
Falls back to full frame if no hand detected.
Uses MediaPipe Pose for skeleton-based movement and action prediction.
Temporal smoothing over 10 frames for stable gesture predictions.
Left hand is flipped to match right hand training images.

Usage:
    python3 test_live.py --model_dir checkpoints

Controls:
    Q — Quit
"""

import argparse
import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import GestureCNN, MovementLSTM, SpatioTemporalGCN


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

    # ---- Load models ----
    models = {}
    class_maps = {}

    # Gesture CNN
    gesture_json = os.path.join(args.model_dir, 'gesture_classes.json')
    gesture_pth = os.path.join(args.model_dir, 'gesture_cnn_best.pth')
    if os.path.exists(gesture_json) and os.path.exists(gesture_pth):
        with open(gesture_json) as f:
            classes = json.load(f)
        class_maps['gesture'] = {v: k for k, v in classes.items()}

        model = GestureCNN(num_classes=len(classes), pretrained=True)
        ckpt = torch.load(gesture_pth, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device).eval()
        models['gesture'] = model
        print(f"Loaded Gesture CNN: {list(classes.keys())}")
    else:
        print("WARNING: Gesture model not found — skipping")

    # Movement LSTM
    movement_json = os.path.join(args.model_dir, 'movement_classes.json')
    movement_pth = os.path.join(args.model_dir, 'movement_lstm_best.pth')
    if os.path.exists(movement_json) and os.path.exists(movement_pth):
        with open(movement_json) as f:
            classes = json.load(f)
        class_maps['movement'] = {v: k for k, v in classes.items()}

        model = MovementLSTM(input_size=33 * 3, num_classes=len(classes))
        ckpt = torch.load(movement_pth, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device).eval()
        models['movement'] = model
        print(f"Loaded Movement LSTM: {list(classes.keys())}")
    else:
        print("WARNING: Movement model not found — skipping")

    # ST-GCN
    gcn_json = os.path.join(args.model_dir, 'stgcn_classes.json')
    gcn_pth = os.path.join(args.model_dir, 'stgcn_best.pth')
    if os.path.exists(gcn_json) and os.path.exists(gcn_pth):
        with open(gcn_json) as f:
            classes = json.load(f)
        class_maps['action'] = {v: k for k, v in classes.items()}

        model = SpatioTemporalGCN(num_classes=len(classes), num_joints=33)
        ckpt = torch.load(gcn_pth, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device).eval()
        models['action'] = model
        print(f"Loaded ST-GCN: {list(classes.keys())}")
    else:
        print("WARNING: ST-GCN model not found — skipping")

    if not models:
        print("ERROR: No models found! Train models first.")
        return

    # ---- MediaPipe ----
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # ---- Camera ----
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    skeleton_buffer = deque(maxlen=args.seq_length)
    gesture_buffer = deque(maxlen=10)
    movement_buffer = deque(maxlen=15)
    action_buffer = deque(maxlen=15)
    fps_buffer = deque(maxlen=30)

    from torchvision import transforms
    gesture_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("\n" + "=" * 50)
    print("LIVE TEST — Press Q to quit")
    print("Hand detection: ON (green box around hand)")
    print("Both hands supported (left hand auto-flipped)")
    print("Temporal smoothing: 10 frames (gesture), 8 frames (movement)")
    print("=" * 50 + "\n")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        gesture_text = ""
        movement_text = ""
        action_text = ""
        robot_cmd = "IDLE"
        hand_found = False
        hand_label = "Unknown"
        box_x1, box_y1, box_x2, box_y2 = 0, 0, 0, 0

        # ============================================
        # GESTURE RECOGNITION (Hand Detection + CNN)
        # ============================================
        if 'gesture' in models:
            hand_results = hands.process(frame_rgb)

            gesture_input = None

            if hand_results.multi_hand_landmarks:
                hand_found = True
                hand_landmarks = hand_results.multi_hand_landmarks[0]

                # Detect which hand (left or right)
                if hand_results.multi_handedness:
                    hand_label = hand_results.multi_handedness[0].classification[0].label

                # Get bounding box of hand
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                # Raw bounding box with generous padding
                pad = 80
                x_min = max(0, int(min(x_coords) * w) - pad)
                y_min = max(0, int(min(y_coords) * h) - pad)
                x_max = min(w, int(max(x_coords) * w) + pad)
                y_max = min(h, int(max(y_coords) * h) + pad)

                # Make square crop (matching 224x224 training images)
                if (x_max - x_min) > 20 and (y_max - y_min) > 20:
                    cx = (x_min + x_max) // 2
                    cy = (y_min + y_max) // 2
                    size = max(x_max - x_min, y_max - y_min)
                    half = size // 2
                    box_x1 = max(0, cx - half)
                    box_y1 = max(0, cy - half)
                    box_x2 = min(w, cx + half)
                    box_y2 = min(h, cy + half)
                    hand_crop = frame_rgb[box_y1:box_y2, box_x1:box_x2]

                    # Flip left hand to look like right hand for CNN
                    if hand_label == "Left":
                        hand_crop = cv2.flip(hand_crop, 1)

                    gesture_input = hand_crop

                    # Draw square bounding box on frame
                    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)

                # Draw hand skeleton
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                )
            else:
                # No hand detected — clear gesture buffer
                gesture_buffer.clear()

            # Fallback: use full frame if no hand detected
            if gesture_input is None:
                gesture_input = frame_rgb

            # Run gesture CNN with temporal smoothing
            try:
                img_tensor = gesture_transform(gesture_input).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = models['gesture'](img_tensor)
                    probs = F.softmax(logits, dim=1)

                # Only add to buffer if hand is detected
                if hand_found:
                    gesture_buffer.append(probs.cpu().numpy()[0])

                if len(gesture_buffer) >= 3:
                    # Average over last N frames for stability
                    avg_probs = np.mean(list(gesture_buffer), axis=0)
                    pred_idx = np.argmax(avg_probs)
                    gesture_conf = avg_probs[pred_idx]
                    gesture_name = class_maps['gesture'][pred_idx]

                    hand_info = f"{hand_label[0]}" if hand_label != "Unknown" else ""
                    if gesture_conf > args.confidence and hand_found:
                        gesture_text = f"Gesture: {gesture_name} ({gesture_conf:.0%}) [{hand_info}]"
                        if hand_found:
                            cv2.putText(frame, f"{gesture_name} {gesture_conf:.0%}",
                                        (box_x1, box_y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        gesture_text = f"Gesture: uncertain ({gesture_conf:.0%}) [{hand_info}]"
                else:
                    gesture_text = "Gesture: buffering..."
            except Exception as e:
                gesture_text = "Gesture: error"

        # ============================================
        # MOVEMENT & ACTION PREDICTION (Pose)
        # ============================================
        pose_results = pose.process(frame_rgb)

        if pose_results.pose_landmarks:
            # Draw pose skeleton (thinner so it doesn't cover hand)
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

                # Movement LSTM with smoothing
                if 'movement' in models:
                    hip = skeleton_seq[:, 0:1, :]
                    normalized = skeleton_seq - hip
                    std = normalized.std()
                    if std > 1e-6:
                        normalized = normalized / std
                    flat = normalized.reshape(args.seq_length, 33 * 3)
                    input_t = torch.tensor(flat).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = models['movement'](input_t)
                        probs = F.softmax(logits, dim=1)

                    movement_buffer.append(probs.cpu().numpy()[0])

                    if len(movement_buffer) >= 2:
                        avg_probs = np.mean(list(movement_buffer), axis=0)
                        pred_idx = np.argmax(avg_probs)
                        move_conf = avg_probs[pred_idx]
                        move_name = class_maps['movement'][pred_idx]
                        if move_conf > args.confidence:
                            movement_text = f"Movement: {move_name} ({move_conf:.0%})"
                        else:
                            movement_text = f"Movement: uncertain ({move_conf:.0%})"

                # ST-GCN with smoothing
                if 'action' in models:
                    seq_t = skeleton_seq.transpose(2, 0, 1)
                    input_t = torch.tensor(seq_t).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = models['action'](input_t)
                        probs = F.softmax(logits, dim=1)

                    action_buffer.append(probs.cpu().numpy()[0])

                    if len(action_buffer) >= 2:
                        avg_probs = np.mean(list(action_buffer), axis=0)
                        pred_idx = np.argmax(avg_probs)
                        action_conf = avg_probs[pred_idx]
                        action_name = class_maps['action'][pred_idx]
                        if action_conf > args.confidence:
                            action_text = f"Action: {action_name} ({action_conf:.0%})"
                        else:
                            action_text = f"Action: uncertain ({action_conf:.0%})"
        else:
            movement_text = "Movement: No person detected"

        # ============================================
        # ROBOT DECISION
        # ============================================
        gesture_lower = gesture_text.lower()
        is_valid = ("uncertain" not in gesture_lower
                    and "error" not in gesture_lower
                    and "buffering" not in gesture_lower
                    and hand_found)

        if is_valid and "stop" in gesture_lower:
            robot_cmd = "STOP (gesture)"
        elif is_valid and "forward" in gesture_lower:
            robot_cmd = "MOVE FORWARD (gesture)"
        elif is_valid and "backward" in gesture_lower:
            robot_cmd = "MOVE BACKWARD (gesture)"
        elif is_valid and "left" in gesture_lower:
            robot_cmd = "TURN LEFT (gesture)"
        elif is_valid and "right" in gesture_lower:
            robot_cmd = "TURN RIGHT (gesture)"
        elif "approaching" in movement_text.lower() and "uncertain" not in movement_text:
            robot_cmd = "STOP + TURN (avoidance)"
        elif "moving_left" in movement_text.lower() and "uncertain" not in movement_text:
            robot_cmd = "ADJUST RIGHT (avoidance)"
        elif "moving_right" in movement_text.lower() and "uncertain" not in movement_text:
            robot_cmd = "ADJUST LEFT (avoidance)"
        else:
            robot_cmd = "CONTINUE FORWARD"

        # ============================================
        # DRAW UI
        # ============================================
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 105), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h - 55), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Predictions text
        y = 22
        for text, color in [(gesture_text, (0, 255, 255)),
                            (movement_text, (255, 200, 0)),
                            (action_text, (200, 255, 0))]:
            if text:
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, color, 2)
                y += 22

        # Hand detection indicator
        if hand_found:
            hand_status = f"HAND: {hand_label[0]}"
        else:
            hand_status = "HAND: NO"
        hand_color = (0, 255, 0) if hand_found else (0, 0, 255)
        cv2.putText(frame, hand_status, (w - 130, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 2)

        # Robot command
        cmd_color = (0, 0, 255) if "STOP" in robot_cmd else (0, 255, 0)
        if "TURN" in robot_cmd or "ADJUST" in robot_cmd:
            cmd_color = (0, 200, 255)
        cv2.putText(frame, f"Robot: {robot_cmd}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, cmd_color, 2)

        # FPS
        fps_buffer.append(1.0 / max(time.time() - start_time, 0.001))
        fps = np.mean(fps_buffer)
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 90, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Buffer status
        cv2.putText(frame, f"Buf: {len(skeleton_buffer)}/{args.seq_length}",
                    (w - 130, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)

        # Gesture buffer status
        cv2.putText(frame, f"GBuf: {len(gesture_buffer)}/10",
                    (w - 130, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)

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
