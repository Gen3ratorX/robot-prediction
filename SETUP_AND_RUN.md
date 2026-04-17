# Setup And Run Guide

This document explains how to open, build, run, and log experiments for the `robot_prediction` project on another VM.

## 1. Prerequisites

The target VM should have:

- Ubuntu with GUI access
- ROS2 Jazzy
- Python 3.12
- `colcon`
- a webcam
- internet access for Python package installation

Minimum Python packages needed:

- `torch`
- `mediapipe`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `opencv-python`
- `pypdf`
- `fpdf`

## 2. Clone The Project

Create the ROS workspace if needed:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/Gen3ratorX/robot-prediction.git
```

## 3. Create And Activate The Python Environment

```bash
python3 -m venv ~/robot_env
source ~/robot_env/bin/activate
pip install --upgrade pip
```

Install the required Python packages:

```bash
pip install torch mediapipe numpy scikit-learn matplotlib opencv-python pypdf fpdf
```

## 4. Build The ROS2 Package

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select robot_prediction --symlink-install
source ~/ros2_ws/install/setup.bash
```

## 5. GUI Environment Variables

If the VM uses a desktop session, export:

```bash
export DISPLAY=:0
export QT_QPA_PLATFORM=xcb
```

If the Python environment is used together with ROS:

```bash
export PYTHONPATH=$HOME/robot_env/lib/python3.12/site-packages:$PYTHONPATH
```

## 6. Main Desktop Commands

### Run the full desktop demo

```bash
source ~/robot_env/bin/activate
cd ~/ros2_ws/src/robot_prediction
export DISPLAY=:0
export QT_QPA_PLATFORM=xcb
python3 robot_prediction/combined_live.py --model_dir checkpoints --demo_mode
```

### Run fast debug mode

```bash
source ~/robot_env/bin/activate
cd ~/ros2_ws/src/robot_prediction
export DISPLAY=:0
export QT_QPA_PLATFORM=xcb
python3 robot_prediction/combined_live.py --model_dir checkpoints --debug_fast_response
```

### Test gesture model only

```bash
source ~/robot_env/bin/activate
cd ~/ros2_ws/src/robot_prediction
export DISPLAY=:0
export QT_QPA_PLATFORM=xcb
python3 robot_prediction/gesture_landmarks.py test
```

### Test movement and action models only

```bash
source ~/robot_env/bin/activate
cd ~/ros2_ws/src/robot_prediction
export DISPLAY=:0
export QT_QPA_PLATFORM=xcb
python3 robot_prediction/test_live.py --model_dir checkpoints
```

## 7. ROS2 Runtime Commands

### Run the ROS2 node

```bash
source ~/robot_env/bin/activate
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
export PYTHONPATH=$HOME/robot_env/lib/python3.12/site-packages:$PYTHONPATH

ros2 run robot_prediction combined_ros2_node --ros-args \
  -p model_dir:=$HOME/ros2_ws/src/robot_prediction/checkpoints \
  -p use_camera_topic:=false \
  -p camera_device:=0 \
  -p demo_mode:=true
```

### Inspect nodes and topics

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 node list
ros2 topic list
```

### Monitor outputs

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 topic echo /human_gesture std_msgs/msg/String
ros2 topic echo /human_movement std_msgs/msg/String
ros2 topic echo /human_action std_msgs/msg/String
ros2 topic echo /cmd_vel geometry_msgs/msg/TwistStamped
```

## 8. Gazebo Commands

### Launch TurtleBot3 Gazebo world

```bash
source /opt/ros/jazzy/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

### Run the ROS2 node in another terminal

```bash
source ~/robot_env/bin/activate
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
export PYTHONPATH=$HOME/robot_env/lib/python3.12/site-packages:$PYTHONPATH

ros2 run robot_prediction combined_ros2_node --ros-args \
  -p model_dir:=$HOME/ros2_ws/src/robot_prediction/checkpoints \
  -p use_camera_topic:=false \
  -p camera_device:=0 \
  -p demo_mode:=true
```

## 9. Logging And Results Workflow

The project includes:

- `robot_prediction/ros2_experiment_logger.py`
- `robot_prediction/plot_experiment_results.py`
- `scripts/run_ros2_experiment.sh`

### One-command experiment logging

Keep the ROS2 node running in one terminal.

In another terminal:

```bash
cd ~/ros2_ws/src/robot_prediction
./scripts/run_ros2_experiment.sh approaching_run_01 20 \
  --expected-movement approaching \
  --notes "Direct frontal approach, normal lighting"
```

This creates:

- `results/approaching_run_01/events.csv`
- `results/approaching_run_01/summary.csv`
- `results/approaching_run_01/summary_table.csv`
- `results/approaching_run_01/metrics.json`
- plots as `.png`

### More examples

Gesture experiment:

```bash
./scripts/run_ros2_experiment.sh gesture_stop_01 15 \
  --expected-gesture stop \
  --notes "Right hand stop gesture"
```

Movement experiment:

```bash
./scripts/run_ros2_experiment.sh away_run_01 20 \
  --expected-movement moving_away \
  --notes "Subject walking away from camera"
```

## 10. Recommended Experiment Plan

For thesis-quality results, run at least:

- 3 trials for each gesture:
  - `stop`
  - `forward`
  - `backward`
  - `left`
  - `right`
- 3 trials for each movement:
  - `approaching`
  - `moving_away`
  - `moving_left`
  - `moving_right`
  - `stationary`

Record notes for each run:

- lighting condition
- camera distance
- subject position
- whether the run was clean or had interruptions

## 11. Suggested First-Time Workflow On A Friend’s VM

1. Clone the project into `~/ros2_ws/src`
2. Create `~/robot_env`
3. Install the required Python packages
4. Build the ROS2 package with `colcon`
5. Run the desktop demo first
6. Run the ROS2 node
7. Run one logged experiment
8. Inspect the generated plots and tables

## 12. Notes

- The checkpoints must remain in `~/ros2_ws/src/robot_prediction/checkpoints`
- The ROS2 package must be rebuilt after pulling new code:

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select robot_prediction --symlink-install
source ~/ros2_ws/install/setup.bash
```

- If GUI windows do not appear, ensure:

```bash
export DISPLAY=:0
export QT_QPA_PLATFORM=xcb
```

- If ROS cannot see Python packages from the virtual environment, re-export:

```bash
export PYTHONPATH=$HOME/robot_env/lib/python3.12/site-packages:$PYTHONPATH
```
