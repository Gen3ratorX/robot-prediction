# Commands README

This file is the command reference for the `robot_prediction` project.

It is organized in three parts:

1. Yahboom MicroROS-Pi5 quick start (verified working)
2. Fresh VM / desktop install
3. All run commands

---

# 0. Yahboom MicroROS-Pi5 Quick Start

This section covers running the node inside the Yahboom Docker container on a
Pi5. All steps have been verified against the factory `yahboomtechnology/ros-humble:4.1.2`
image (ROS2 Humble, Python 3.10).

> **Safety note — put the robot on a stand before first run.**
> The node publishes to `/cmd_vel` immediately once a confident prediction
> fires. Keep the wheels off the ground until you have verified the topic
> output looks correct with `ros2 topic echo`.

## 0.1 Enter the Yahboom container

```bash
docker exec -it yahboom_ros bash
```

## 0.2 Set ROS_DOMAIN_ID

The factory firmware runs `/YB_Car_Node` on domain 20. This must be set in
every shell that runs the prediction node:

```bash
export ROS_DOMAIN_ID=20
```

Add this to `~/.bashrc` inside the container if you want it persistent.

## 0.3 Install Python dependencies (Pi5 — no torch)

numpy is pinned because mediapipe 0.10.x and cv2 4.8.x both have compiled
extensions that expect the numpy 1.23 ABI. Do not upgrade numpy.

```bash
pip3 install "numpy==1.23.0" --force-reinstall
pip3 install scikit-learn mediapipe
pip3 install onnxruntime  # aarch64 wheel, ~10 MB
```

torch is **not required**. Without torch the LSTM movement branch is
automatically disabled; gesture recognition and ST-GCN action (via
onnxruntime) continue to work.

If you later want torch for the LSTM movement branch, use a Pi5-compatible
aarch64 wheel — the x86 pytorch.org CPU wheels will not install.

## 0.4 Clone and build

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/Gen3ratorX/robot-prediction.git robot_prediction

cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select robot_prediction --symlink-install
source ~/ros2_ws/install/setup.bash
```

## 0.5 Run the node

```bash
export ROS_DOMAIN_ID=20
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 run robot_prediction combined_ros2_node --ros-args \
  -p model_dir:=$HOME/ros2_ws/src/robot_prediction/checkpoints \
  -p use_camera_topic:=false \
  -p camera_device:=0
```

`use_onnx_for_stgcn` defaults to `true` — no extra flag needed on the Pi.

## 0.6 Verify topics

In a second terminal (same container, same `ROS_DOMAIN_ID=20`):

```bash
export ROS_DOMAIN_ID=20
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# Confirm the node is publishing
ros2 topic list | grep -E 'cmd_vel|human'

# Watch predictions
ros2 topic echo /human_gesture
ros2 topic echo /human_movement
ros2 topic echo /human_action

# Watch robot commands (should be zero Twist when nothing detected)
ros2 topic echo /cmd_vel
```

Expected output format on prediction topics: `forward:0.94`, `approaching:0.81`, etc.

## 0.7 Expected startup log

```
[INFO] torch not found — LSTM movement model disabled
[INFO] Gesture model loaded: ['backward', 'forward', 'left', 'right', 'stop']
[INFO] ST-GCN loaded via ONNX (T=30): ['approaching', 'moving_away', ...]
[INFO] Using direct webcam: /dev/video0
[INFO] Human-Aware Navigation Node initialized! ...
```

If ST-GCN shows disabled, verify `stgcn.onnx` is present in `checkpoints/`
and onnxruntime is installed (`python3 -c "import onnxruntime"`).

---

# 1. Fresh Install

## 1.1 Prerequisites

The target machine should have:

- Ubuntu with GUI access
- ROS2 Jazzy
- Python 3.12
- `colcon`
- a webcam
- internet access

## 1.2 Create the ROS2 workspace

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

## 1.3 Clone the project

```bash
git clone https://github.com/Gen3ratorX/robot-prediction.git robot_prediction
```

## 1.4 Create the Python environment

```bash
python3 -m venv ~/robot_env
source ~/robot_env/bin/activate
pip install --upgrade pip
```

## 1.5 Install Python dependencies

Use the pinned NumPy version to avoid MediaPipe / OpenCV compatibility problems:

```bash
pip install "numpy==1.26.4" torch mediapipe scikit-learn matplotlib opencv-python pypdf fpdf
```

## 1.6 Build the ROS2 package

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select robot_prediction --symlink-install
source ~/ros2_ws/install/setup.bash
```

## 1.7 GUI variables

For live preview windows:

```bash
export DISPLAY=:0
export QT_QPA_PLATFORM=xcb
```

## 1.8 Python path for ROS2

When running ROS nodes from the virtual environment:

```bash
export PYTHONPATH=$HOME/robot_env/lib/python3.12/site-packages:$PYTHONPATH
```

## 1.9 Fresh install verification

### Check the repo

```bash
cd ~/ros2_ws/src/robot_prediction
git rev-parse HEAD
git status --short
```

### Check the ROS build

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 pkg list | grep robot_prediction
```

---

# 2. All Commands

## 2.1 Desktop live commands

### Full desktop demo

```bash
source ~/robot_env/bin/activate
cd ~/ros2_ws/src/robot_prediction
export DISPLAY=:0
export QT_QPA_PLATFORM=xcb
python3 robot_prediction/combined_live.py --model_dir checkpoints --demo_mode
```

### Fast debug mode

```bash
source ~/robot_env/bin/activate
cd ~/ros2_ws/src/robot_prediction
export DISPLAY=:0
export QT_QPA_PLATFORM=xcb
python3 robot_prediction/combined_live.py --model_dir checkpoints --debug_fast_response
```

### Gesture model only

```bash
source ~/robot_env/bin/activate
cd ~/ros2_ws/src/robot_prediction
export DISPLAY=:0
export QT_QPA_PLATFORM=xcb
python3 robot_prediction/gesture_landmarks.py test
```

### Movement + action models only

```bash
source ~/robot_env/bin/activate
cd ~/ros2_ws/src/robot_prediction
export DISPLAY=:0
export QT_QPA_PLATFORM=xcb
python3 robot_prediction/test_live.py --model_dir checkpoints
```

## 2.2 ROS2 runtime commands

### Run ROS2 node without preview

```bash
source ~/robot_env/bin/activate
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
export PYTHONPATH=$HOME/robot_env/lib/python3.12/site-packages:$PYTHONPATH

ros2 run robot_prediction combined_ros2_node --ros-args \
  -p model_dir:=$HOME/ros2_ws/src/robot_prediction/checkpoints \
  -p use_camera_topic:=false \
  -p camera_device:=0
```

### Run ROS2 node with preview window

```bash
source ~/robot_env/bin/activate
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
export PYTHONPATH=$HOME/robot_env/lib/python3.12/site-packages:$PYTHONPATH
export DISPLAY=:0
export QT_QPA_PLATFORM=xcb

ros2 run robot_prediction combined_ros2_node --ros-args \
  -p model_dir:=$HOME/ros2_ws/src/robot_prediction/checkpoints \
  -p use_camera_topic:=false \
  -p camera_device:=0 \
  -p show_preview:=true
```

### Lower movement confidence threshold for testing

```bash
source ~/robot_env/bin/activate
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
export PYTHONPATH=$HOME/robot_env/lib/python3.12/site-packages:$PYTHONPATH

ros2 run robot_prediction combined_ros2_node --ros-args \
  -p model_dir:=$HOME/ros2_ws/src/robot_prediction/checkpoints \
  -p use_camera_topic:=false \
  -p camera_device:=0 \
  -p confidence_threshold:=0.4
```

## 2.3 ROS2 inspection commands

### List nodes

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 node list
```

### List topics

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 topic list
```

### Check topic type

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 topic type /human_movement
```

### Echo prediction topics

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 topic echo /human_gesture std_msgs/msg/String
ros2 topic echo /human_movement std_msgs/msg/String
ros2 topic echo /human_action std_msgs/msg/String
```

### Echo robot commands

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 topic echo /cmd_vel geometry_msgs/msg/TwistStamped
```

## 2.4 Gazebo commands

### Launch TurtleBot3 Gazebo

```bash
source /opt/ros/jazzy/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

### Run the ROS2 node while Gazebo is running

```bash
source ~/robot_env/bin/activate
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
export PYTHONPATH=$HOME/robot_env/lib/python3.12/site-packages:$PYTHONPATH

ros2 run robot_prediction combined_ros2_node --ros-args \
  -p model_dir:=$HOME/ros2_ws/src/robot_prediction/checkpoints \
  -p use_camera_topic:=false \
  -p camera_device:=0
```

## 2.5 Logging and results commands

### One-command experiment logging

Keep the ROS2 node running in one terminal. In another terminal:

```bash
cd ~/ros2_ws/src/robot_prediction
./scripts/run_ros2_experiment.sh approaching_run_01 20 \
  --expected-movement approaching \
  --notes "Direct frontal approach, normal lighting"
```

### Gesture run example

```bash
cd ~/ros2_ws/src/robot_prediction
./scripts/run_ros2_experiment.sh gesture_stop_01 15 \
  --expected-gesture stop \
  --notes "Right hand stop gesture"
```

### Movement-away run example

```bash
cd ~/ros2_ws/src/robot_prediction
./scripts/run_ros2_experiment.sh away_run_01 20 \
  --expected-movement moving_away \
  --notes "Subject walking away from camera"
```

### Re-plot an existing run

```bash
cd ~/ros2_ws/src/robot_prediction
python3 -m robot_prediction.plot_experiment_results \
  --run-dir results/approaching_run_01
```

## 2.6 Rebuild commands after pull

```bash
cd ~/ros2_ws/src/robot_prediction
git pull origin main

cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select robot_prediction --symlink-install
source ~/ros2_ws/install/setup.bash
```

## 2.7 Useful maintenance commands

### Check current commit

```bash
cd ~/ros2_ws/src/robot_prediction
git rev-parse HEAD
```

### Check repo cleanliness

```bash
cd ~/ros2_ws/src/robot_prediction
git status --short
```

### Reinstall NumPy if MediaPipe/OpenCV starts misbehaving

```bash
source ~/robot_env/bin/activate
pip install "numpy==1.26.4"
```

---

# Notes

- The checkpoints are expected at:

```bash
~/ros2_ws/src/robot_prediction/checkpoints
```

- If GUI windows do not appear:

```bash
export DISPLAY=:0
export QT_QPA_PLATFORM=xcb
```

- If ROS cannot see the Python packages from the virtual environment:

```bash
export PYTHONPATH=$HOME/robot_env/lib/python3.12/site-packages:$PYTHONPATH
```

- If you see duplicate package errors in `colcon`, make sure only one copy of the project exists in:

```bash
~/ros2_ws/src
```
