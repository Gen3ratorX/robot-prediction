# Codex Task: Improve Movement Prediction Accuracy

## Context
This is a human-aware robot navigation system using ROS2. It uses three AI models:
1. **Gesture Recognition** — Random Forest on MediaPipe hand landmarks (100% accuracy, working perfectly)
2. **Movement Prediction (LSTM)** — Bi-directional LSTM on body skeleton sequences (99.8% on test, but flickering in live use)
3. **Action Recognition (ST-GCN)** — Spatio-Temporal Graph CNN on skeleton graphs (99.4% on test)

## Problem
When running live, the movement prediction sometimes flickers — e.g., showing "stationary" then briefly switching to "moving_away" even when the person is sitting still. The models were trained on 50 real recordings per class + 10x data augmentation.

## Files to Modify

### `robot_prediction/models.py`
Contains `MovementLSTM` and `SpatioTemporalGCN` architectures. Potential improvements:
- Add dropout to reduce overfitting on augmented data
- Increase attention mechanism complexity
- Add batch normalization between LSTM layers

### `robot_prediction/datasets.py`
Contains `MovementDataset` and `SkeletonGraphDataset`. Potential improvements:
- Better normalization (currently normalizes relative to hip center joint 0, but MediaPipe joint 0 is nose, not hip — should use joints 23/24 which are the hips)
- Add velocity features (frame-to-frame joint displacement) as additional input
- Add acceleration features

### `robot_prediction/train.py`
Training pipeline. Potential improvements:
- Add mixup or cutmix augmentation during training
- Use cosine annealing instead of ReduceLROnPlateau
- Add label smoothing to CrossEntropyLoss

### `robot_prediction/combined_live.py`
Live inference. Potential improvements:
- Increase smoothing buffer from 15 to 20 frames
- Add confidence threshold — only update prediction if new confidence > 80%
- Add hysteresis — require 3+ consecutive same predictions before changing displayed class

## Data Format

### Movement Data (`data/movement/{class}/seq*.npy`)
- Shape: (30, 33, 3) per file
- 30 frames, 33 MediaPipe Pose landmarks, (x, y, z) coordinates
- x, y are normalized 0-1 (frame position), z is relative depth
- Classes: approaching, moving_away, moving_left, moving_right, stationary
- 50 real recordings + augmented to ~550-770 per class

### Gesture Data (`data/gesture_landmarks/*.csv`)
- 83 features per row (normalized landmarks + finger distances + angles + extension flags)
- 1000 samples per class
- Classes: backward, forward, left, right, stop
- DO NOT modify gesture system — it works perfectly

## Key Fix Priority
1. **Fix hip normalization bug** in `datasets.py` — joint 0 in MediaPipe is NOSE, not hip. Use average of joints 23 and 24 (left hip, right hip) as center.
2. **Add velocity features** — compute frame-to-frame displacement for key joints (hips, shoulders, ankles)
3. **Improve live smoothing** in `combined_live.py` — add hysteresis to prevent flickering

## Constraints
- Must work on CPU (no GPU in VM)
- Must maintain compatibility with ROS2 Jazzy
- Do not modify gesture_landmarks.py or the gesture model
- PyTorch 2.11, MediaPipe 0.10.14, Python 3.12
