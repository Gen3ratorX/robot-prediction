# Commands README

This file is the command reference for the `robot_prediction` project.

It is organized in two parts:

1. Fresh install
2. All run commands

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
