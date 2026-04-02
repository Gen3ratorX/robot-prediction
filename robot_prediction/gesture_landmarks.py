#!/usr/bin/env python3
"""
Hand Gesture Landmark Recorder & Trainer (v2 — Normalized)
============================================================

Landmarks are normalized relative to the wrist and hand size,
making predictions independent of hand position, distance, and size.

Usage:
    python3 gesture_landmarks.py record --gesture forward --samples 1000
    python3 gesture_landmarks.py train
    python3 gesture_landmarks.py test
"""

import argparse
import os
import sys
import time
import csv
import pickle
import numpy as np


def normalize_landmarks(landmarks_flat):
    """
    Normalize 21 hand landmarks relative to wrist position and hand size.
    
    This makes the features independent of:
    - Hand position in frame (translation invariant)
    - Distance from camera (scale invariant)
    - Slight rotations
    
    Input: list of 63 values (21 landmarks * 3 coords)
    Output: list of 63 normalized values
    """
    coords = np.array(landmarks_flat).reshape(21, 3)
    
    # Center on wrist (landmark 0)
    wrist = coords[0].copy()
    coords = coords - wrist
    
    # Scale by hand size (distance from wrist to middle finger MCP)
    # Landmark 9 = middle finger MCP
    hand_size = np.linalg.norm(coords[9])
    if hand_size > 1e-6:
        coords = coords / hand_size
    
    return coords.flatten().tolist()


def extract_advanced_features(landmarks_flat):
    """
    Extract rich features from hand landmarks:
    - Normalized coordinates (63 features)
    - Finger tip distances from wrist (5 features)
    - Finger tip angles relative to wrist (5 features)
    - Which fingers are extended (5 features)
    - Finger tip to palm distances (5 features)
    
    Total: 83 features
    """
    coords = np.array(landmarks_flat).reshape(21, 3)
    
    # Normalize
    wrist = coords[0].copy()
    centered = coords - wrist
    hand_size = np.linalg.norm(centered[9])
    if hand_size > 1e-6:
        centered = centered / hand_size
    
    features = centered.flatten().tolist()  # 63 features
    
    # Finger tip indices: thumb=4, index=8, middle=12, ring=16, pinky=20
    tips = [4, 8, 12, 16, 20]
    # Finger PIP (second joint) indices
    pips = [3, 6, 10, 14, 18]
    # Finger MCP (knuckle) indices
    mcps = [2, 5, 9, 13, 17]
    
    # Distance from each fingertip to wrist
    for tip in tips:
        dist = np.linalg.norm(centered[tip])
        features.append(dist)
    
    # Angle of each fingertip relative to wrist (in x-y plane)
    for tip in tips:
        angle = np.arctan2(centered[tip][1], centered[tip][0])
        features.append(angle)
    
    # Is each finger extended? (tip further from wrist than PIP joint)
    for tip, pip in zip(tips, pips):
        tip_dist = np.linalg.norm(centered[tip])
        pip_dist = np.linalg.norm(centered[pip])
        extended = 1.0 if tip_dist > pip_dist else 0.0
        features.append(extended)
    
    # Distance from each fingertip to palm center (average of MCPs)
    palm_center = np.mean(centered[mcps], axis=0)
    for tip in tips:
        dist = np.linalg.norm(centered[tip] - palm_center)
        features.append(dist)
    
    return features


def record_gesture(gesture_name, num_samples=1000, save_dir='data/gesture_landmarks', camera=0):
    import cv2
    cv2.namedWindow('Gesture Recorder', cv2.WINDOW_NORMAL)

    import mediapipe as mp

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f'{gesture_name}.csv')

    existing = 0
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            existing = sum(1 for _ in f)
        print(f"Found {existing} existing samples for '{gesture_name}'")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    recording = False
    sample_count = existing
    target = existing + num_samples

    instructions = {
        'forward': "CLOSED FIST (all fingers closed)",
        'backward': "THUMBS DOWN (thumb pointing down)",
        'stop': "OPEN PALM (all 5 fingers spread)",
        'left': "PEACE SIGN (two fingers up)",
        'right': "ROCK/HORNS (index + pinky extended)",
    }

    print("\n" + "=" * 60)
    print(f"RECORDING: {gesture_name}")
    print(f"Instruction: {instructions.get(gesture_name, 'Show your gesture')}")
    print(f"Target: {num_samples} new samples ({target} total)")
    print("=" * 60)
    print("\nTIPS FOR BETTER ACCURACY:")
    print("  - Move your hand around (different positions in frame)")
    print("  - Vary distance from camera (close and far)")
    print("  - Tilt your hand slightly in different directions")
    print("  - Use both left and right hands")
    print("\nControls:")
    print("  SPACE — Start/stop auto-recording")
    print("  Q     — Quit\n")

    csv_file = open(csv_path, 'a', newline='')
    writer = csv.writer(csv_file)

    while sample_count < target:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        results = hands.process(frame_rgb)

        status = "RECORDING" if recording else "PAUSED (press SPACE)"
        color = (0, 0, 255) if recording else (0, 255, 0)

        cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"Gesture: {gesture_name.upper()}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Samples: {sample_count}/{target}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        progress = (sample_count - existing) / num_samples
        bar_w = w - 40
        cv2.rectangle(frame, (20, h - 30), (20 + bar_w, h - 10), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, h - 30), (20 + int(bar_w * progress), h - 10), color, -1)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
            )

            if recording:
                raw = []
                for lm in hand_landmarks.landmark:
                    raw.extend([lm.x, lm.y, lm.z])
                
                # Save advanced normalized features
                features = extract_advanced_features(raw)
                writer.writerow(features)
                sample_count += 1

                if sample_count % 100 == 0:
                    print(f"  Recorded {sample_count}/{target} samples")
        else:
            cv2.putText(frame, "No hand detected — show your gesture!",
                        (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 165, 255), 2)

        cv2.imshow('Gesture Recorder', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            recording = not recording
            if recording:
                print(f"  Recording started...")
            else:
                print(f"  Paused at {sample_count} samples")
        elif key == ord('q'):
            break

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    final_count = 0
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            final_count = sum(1 for _ in f)

    print(f"\nDone! Total samples for '{gesture_name}': {final_count}")
    print(f"Saved to: {os.path.abspath(csv_path)}")


def train_classifier(data_dir='data/gesture_landmarks', save_dir='checkpoints'):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    import json

    print("\n" + "=" * 60)
    print("TRAINING GESTURE LANDMARK CLASSIFIER")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    all_data = []
    all_labels = []
    class_names = []
    expected_features = None

    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith('.csv'):
            gesture_name = fname.replace('.csv', '')
            class_names.append(gesture_name)
            csv_path = os.path.join(data_dir, fname)

            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                count = 0
                for row in reader:
                    if expected_features is None:
                        expected_features = len(row)
                    if len(row) == expected_features:
                        all_data.append([float(x) for x in row])
                        all_labels.append(gesture_name)
                        count += 1
                print(f"  {gesture_name}: {count} samples")

    if not all_data:
        print("ERROR: No data found! Record gestures first.")
        return

    X = np.array(all_data)
    y = np.array(all_labels)

    print(f"\nTotal samples: {len(X)}")
    print(f"Classes: {class_names}")
    print(f"Features per sample: {X.shape[1]}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=30, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test) * 100

    y_pred = rf.predict(X_test)
    print(f"\n{'=' * 60}")
    print(f"TEST RESULTS — Random Forest — Accuracy: {rf_acc:.1f}%")
    print(f"{'=' * 60}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    model_path = os.path.join(save_dir, 'gesture_landmark_model.pkl')
    scaler_path = os.path.join(save_dir, 'gesture_landmark_scaler.pkl')
    classes_path = os.path.join(save_dir, 'gesture_landmark_classes.json')

    with open(model_path, 'wb') as f:
        pickle.dump(rf, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(classes_path, 'w') as f:
        json.dump(class_names, f)

    print(f"\nModel saved to: {model_path}")


def test_live(model_dir='checkpoints', camera=0):
    import cv2
    cv2.namedWindow('Gesture Landmark Test', cv2.WINDOW_NORMAL)

    import mediapipe as mp
    import json
    from collections import deque

    model_path = os.path.join(model_dir, 'gesture_landmark_model.pkl')
    scaler_path = os.path.join(model_dir, 'gesture_landmark_scaler.pkl')
    classes_path = os.path.join(model_dir, 'gesture_landmark_classes.json')

    if not os.path.exists(model_path):
        print("ERROR: Model not found! Train first.")
        return

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(classes_path, 'r') as f:
        class_names = json.load(f)

    print(f"Loaded model with classes: {class_names}")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prediction_buffer = deque(maxlen=10)
    fps_buffer = deque(maxlen=30)

    print("\n" + "=" * 50)
    print("GESTURE LANDMARK TEST — Press Q to quit")
    print("=" * 50 + "\n")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        results = hands.process(frame_rgb)

        gesture_name = ""
        gesture_conf = 0.0
        hand_found = False
        hand_label = "?"

        if results.multi_hand_landmarks:
            hand_found = True
            hand_landmarks = results.multi_hand_landmarks[0]

            if results.multi_handedness:
                hand_label = results.multi_handedness[0].classification[0].label[0]

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
            )

            # Extract same features used during training
            raw = []
            for lm in hand_landmarks.landmark:
                raw.extend([lm.x, lm.y, lm.z])

            features = extract_advanced_features(raw)
            features_scaled = scaler.transform([features])
            probs = model.predict_proba(features_scaled)[0]
            prediction_buffer.append(probs)

            if len(prediction_buffer) >= 3:
                avg_probs = np.mean(list(prediction_buffer), axis=0)
                pred_idx = np.argmax(avg_probs)
                gesture_conf = avg_probs[pred_idx]
                gesture_name = class_names[pred_idx]
        else:
            prediction_buffer.clear()

        # Draw UI
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        if gesture_name and gesture_conf > 0.6:
            cv2.putText(frame, f"Gesture: {gesture_name} ({gesture_conf:.0%}) [{hand_label}]",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, f"Hand: {hand_label}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            robot_cmd = {"forward": "MOVE FORWARD", "backward": "MOVE BACKWARD",
                        "stop": "STOP", "left": "TURN LEFT", "right": "TURN RIGHT"
                        }.get(gesture_name, "UNKNOWN")
            cmd_color = (0, 0, 255) if "STOP" in robot_cmd else (0, 255, 0)
            if "TURN" in robot_cmd:
                cmd_color = (0, 200, 255)
            cv2.putText(frame, f"Robot: {robot_cmd} (gesture)",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, cmd_color, 2)
        elif hand_found:
            cv2.putText(frame, f"Gesture: buffering... [{hand_label}]",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, "Robot: CONTINUE FORWARD",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
            cv2.putText(frame, "Robot: CONTINUE FORWARD",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        fps_buffer.append(1.0 / max(time.time() - start_time, 0.001))
        fps = np.mean(fps_buffer)
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 90, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, f"Buf: {len(prediction_buffer)}/10",
                    (w - 130, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imshow('Gesture Landmark Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Test complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['record', 'train', 'test'])
    parser.add_argument('--gesture', type=str, default='forward')
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='data/gesture_landmarks')
    parser.add_argument('--model_dir', type=str, default='checkpoints')
    parser.add_argument('--camera', type=int, default=0)

    args = parser.parse_args()

    if args.mode == 'record':
        record_gesture(args.gesture, args.samples, args.save_dir, args.camera)
    elif args.mode == 'train':
        train_classifier(args.save_dir, args.model_dir)
    elif args.mode == 'test':
        test_live(args.model_dir, args.camera)
