# Robot Prediction

Human-aware robot navigation using camera-based hand gesture recognition, body-motion understanding, and ROS2 robot response.

This repository contains the full implementation used to:

- recognize explicit hand gestures such as `stop`, `forward`, `backward`, `left`, and `right`
- classify body motion as:
  - `approaching`
  - `moving_away`
  - `moving_left`
  - `moving_right`
  - `stationary`
- generate robot control behavior from those predictions
- run both in desktop live mode and in a ROS2 environment
- log ROS2 experiment outputs and convert them into plots and tables for thesis results

This README is written as a handoff document for:

- a supervisor reviewing the work
- a teammate continuing the project
- another student or engineer deploying it on a new VM or robot platform

---

## 1. Project Purpose

The main goal of this project is to build a robot perception-and-response pipeline that can infer a human’s intent from natural visual cues rather than relying only on explicit manual commands.

The system combines:

- **gesture recognition** for direct user commands
- **movement recognition** for contextual body motion
- **action recognition** as a secondary skeleton-based signal
- **direction-aware reasoning** to improve front/back human motion interpretation
- **ROS2 integration** for robot response and experiment logging

The practical motivation is safe and intuitive human-robot interaction. The robot should respond differently when a person:

- gives a command with the hand
- walks toward the robot
- walks away from the robot
- moves laterally across the robot’s path
- remains still

---

## 2. Final System Summary

The final implemented system consists of three perception pathways and one decision layer.

### 2.1 Gesture pathway

- input: MediaPipe hand landmarks
- model: landmark-based gesture classifier
- role: explicit command recognition

Output labels:

- `stop`
- `forward`
- `backward`
- `left`
- `right`

### 2.2 Movement pathway

- input: engineered pose-sequence features from MediaPipe body landmarks
- model: LSTM
- role: primary human motion interpretation for navigation behavior

Output labels:

- `approaching`
- `moving_away`
- `moving_left`
- `moving_right`
- `stationary`

### 2.3 Action pathway

- input: pose skeleton sequence
- model: ST-GCN
- role: secondary skeleton-based action/motion signal used for comparison and support

### 2.4 Inference stabilization layer

The live pipeline does not rely only on raw model output. It adds a stabilization layer that:

- smooths predictions over time
- checks model confidence
- suppresses false `stationary` transitions
- uses **signed pose depth change** and **apparent body-scale change** to stabilize `approaching` vs `moving_away`

This direction-aware logic became one of the most important practical contributions in the project because it made live front/back behavior substantially more reliable.

### 2.5 Decision layer

The decision priority is:

1. gesture commands
2. motion/action interpretation
3. conservative fallback behavior

Robot behavior is currently:

- `stop` gesture -> stop
- `forward` gesture -> move forward
- `backward` gesture -> move backward
- `left` gesture -> turn left
- `right` gesture -> turn right
- `approaching` -> stop and turn/avoid
- `moving_left` -> adjust right
- `moving_right` -> adjust left
- `moving_away` -> hold position
- `stationary` -> hold position

This is intentionally conservative for safety.

---

## 3. Main Engineering Problem Solved

The main difficulty was not simply building a classifier. It was making the live system behave correctly.

Early versions of the system showed a common robotics/perception problem:

- offline metrics were very high
- live performance was less reliable

The most important failure case was front/back motion:

- `approaching`
- `moving_away`

The system could:

- flip between these two classes
- incorrectly insert `stationary` during transitions
- respond more slowly than expected because of temporal buffering and smoothing

The final solution combined:

- cleaned forward/backward labels
- retrained movement and ST-GCN models
- signed direction cues from pose geometry
- improved runtime logic

That combined approach is the reason the live system became usable.

---

## 4. Dataset Work and Data Quality

### 4.1 Gesture data

Gesture recognition uses MediaPipe hand landmark samples stored in:

- [data/gesture_landmarks](/Users/st.dominic/robot-prediction/data/gesture_landmarks)

These CSV files provide the training examples for explicit gesture classification.

### 4.2 Movement data

Movement recognition uses pose-sequence data for:

- `approaching`
- `moving_away`
- `moving_left`
- `moving_right`
- `stationary`

The original real dataset is under:

- [data/movement](/Users/st.dominic/robot-prediction/data/movement)

### 4.3 Data audit

The recorded `approaching` and `moving_away` samples were audited using signed pose depth and body-scale trends.

Audit script:

- [audit_movement_direction.py](/Users/st.dominic/robot-prediction/robot_prediction/audit_movement_direction.py)

Observed audit result:

- `approaching`: `441 agree`, `55 ambiguous`, `54 contradict`
- `moving_away`: `404 agree`, `79 ambiguous`, `67 contradict`

This showed that forward/backward labels contained meaningful noise.

### 4.4 Cleaned movement dataset

A cleaned dataset was created using:

- [build_movement_clean.py](/Users/st.dominic/robot-prediction/robot_prediction/build_movement_clean.py)

Cleaned dataset path:

- [data/movement_clean](/Users/st.dominic/robot-prediction/data/movement_clean)

Final cleaned counts:

- `approaching`: `441`
- `moving_away`: `404`
- `moving_left`: `770`
- `moving_right`: `770`
- `stationary`: `550`

This cleaned dataset became the basis for the final movement model retraining.

---

## 5. Models and Training

### 5.1 Gesture model

- type: landmark-based gesture classifier
- input: hand landmark features
- use: direct command recognition

### 5.2 Movement model

- type: LSTM
- input: engineered movement features derived from body pose
- role: primary motion classifier

Observed result after retraining on `movement_clean`:

- test accuracy: `98.6%`

Important note:

- this result was more believable than earlier near-perfect scores from noisier or more weakly controlled splits

### 5.3 ST-GCN

- type: Spatial-Temporal Graph Convolutional Network
- input: pose skeleton graph sequence
- role: secondary motion/action signal

Observed result after retraining:

- test accuracy: `99.5%`

Checkpoint files in [checkpoints](/Users/st.dominic/robot-prediction/checkpoints):

- `movement_lstm_best.pth`
- `movement_classes.json`
- `stgcn_best.pth`
- `stgcn_classes.json`
- `stgcn.onnx`
- gesture model files

### 5.4 Why both movement and action exist

The two model paths are not equal in importance.

- **Movement** is the primary truth for runtime robot behavior
- **Action** is a secondary skeleton-based interpretation used as supporting evidence

For front/back behavior, the strongest runtime truth is:

1. signed direction cue when strong
2. movement classifier
3. action classifier

---

## 6. Runtime Modes

### 6.1 Desktop live mode

Main script:

- [combined_live.py](/Users/st.dominic/robot-prediction/robot_prediction/combined_live.py)

Purpose:

- direct camera testing
- debugging predictions
- checking gesture, movement, and action outputs visually

Useful flags:

- `--model_dir checkpoints`
- `--debug_fast_response`
- `--demo_mode`
- screenshot capture flags

### 6.2 ROS2 runtime

Main ROS2 node:

- [combined_ros2_node.py](/Users/st.dominic/robot-prediction/robot_prediction/combined_ros2_node.py)

Publishes:

- `/cmd_vel`
- `/human_gesture`
- `/human_movement`
- `/human_action`

This node is the runtime integration point for robot behavior.

### 6.3 Experiment logging and plotting

Logging tools:

- [ros2_experiment_logger.py](/Users/st.dominic/robot-prediction/robot_prediction/ros2_experiment_logger.py)
- [plot_experiment_results.py](/Users/st.dominic/robot-prediction/robot_prediction/plot_experiment_results.py)
- [run_ros2_experiment.sh](/Users/st.dominic/robot-prediction/scripts/run_ros2_experiment.sh)

These tools allow ROS2 topic data to be saved and converted into thesis-ready:

- CSV summaries
- JSON metrics
- confidence plots
- count plots
- command time-series plots

This was added specifically to support formal result reporting.

---

## 7. What Has Been Achieved

At the current stage, the project has achieved:

- a working hand-gesture recognition path
- a working body-movement classification path
- a secondary ST-GCN action path
- live front/back improvement using pose-derived direction cues
- ROS2 runtime integration
- Gazebo-compatible command publishing
- experiment logging for thesis result generation

The main milestone achieved is:

- **usable front/back human motion recognition in live operation**

That was the hardest part of the project.

---

## 8. Remaining Limitations

The system is functional, but not complete in every possible dimension.

Current limitations include:

- ST-GCN is still secondary and not the main source of front/back improvement
- slow retreat can still drift toward `stationary`
- left/right refinement has had less focused cleanup than front/back
- gesture recognition works best when the hand is clearly visible and close enough to the camera
- real-world deployment on a physical robot platform would still require hardware-specific integration work

Also:

- offline accuracy should not be treated as the only evidence of quality
- live evaluation remains critical

---

## 9. Result Generation Workflow

This repository supports a thesis-style results workflow:

1. run experiment
2. log ROS2 outputs
3. generate tables and graphs
4. compare to objectives
5. discuss mismatch and design improvements

Typical experiment categories:

- gesture: `stop`
- gesture: `forward`
- gesture: `backward`
- gesture: `left`
- gesture: `right`
- movement: `approaching`
- movement: `moving_away`
- movement: `moving_left`
- movement: `moving_right`
- movement: `stationary`

Recommended minimum:

- at least 3 trials per condition

---

## 10. Repository Structure

Top-level folders and files:

- [robot_prediction](/Users/st.dominic/robot-prediction/robot_prediction)  
  Main Python package. Contains training code, models, live runtime, ROS2 node, data audit tools, and plotting tools.

- [checkpoints](/Users/st.dominic/robot-prediction/checkpoints)  
  Trained model checkpoints and exported model files.

- [data](/Users/st.dominic/robot-prediction/data)  
  Gesture and movement datasets.

- [scripts](/Users/st.dominic/robot-prediction/scripts)  
  Convenience workflow scripts such as one-command experiment logging.

- [SETUP_AND_RUN.md](/Users/st.dominic/robot-prediction/SETUP_AND_RUN.md)  
  Command-focused setup and execution guide.

- [movement_data_export.csv](/Users/st.dominic/robot-prediction/movement_data_export.csv)  
  Exported movement-related data used during development.

Most important code files:

- [train.py](/Users/st.dominic/robot-prediction/robot_prediction/train.py)
- [models.py](/Users/st.dominic/robot-prediction/robot_prediction/models.py)
- [movement_features.py](/Users/st.dominic/robot-prediction/robot_prediction/movement_features.py)
- [gesture_landmarks.py](/Users/st.dominic/robot-prediction/robot_prediction/gesture_landmarks.py)
- [combined_live.py](/Users/st.dominic/robot-prediction/robot_prediction/combined_live.py)
- [combined_ros2_node.py](/Users/st.dominic/robot-prediction/robot_prediction/combined_ros2_node.py)
- [audit_movement_direction.py](/Users/st.dominic/robot-prediction/robot_prediction/audit_movement_direction.py)
- [build_movement_clean.py](/Users/st.dominic/robot-prediction/robot_prediction/build_movement_clean.py)
- [convert_ntu.py](/Users/st.dominic/robot-prediction/robot_prediction/convert_ntu.py)

---

## 11. Recommended First Steps For A New Owner

If someone new takes over the project, the recommended order is:

1. read [SETUP_AND_RUN.md](/Users/st.dominic/robot-prediction/SETUP_AND_RUN.md)
2. set up a fresh VM or ROS2 workspace
3. run desktop live mode first
4. run the ROS2 node
5. run one logged experiment
6. inspect the generated result plots
7. only then change models or runtime logic

This order avoids the confusion that comes from changing runtime logic before verifying the base pipeline.

---

## 12. Practical Conclusion

This project is not just a classifier demo. It is a full perception-to-response pipeline for human-aware robot behavior.

The most important body of knowledge contribution in this repository is that:

- **reliable live human intention recognition required more than model training**
- it required:
  - better labels
  - retraining
  - runtime stabilization
  - and physically interpretable direction reasoning

That is the practical lesson embedded in the final system.
