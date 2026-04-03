#!/usr/bin/env python3
"""
ROS2 Combined Inference Node — Human-Aware Robot Navigation
==============================================================

Publishes both Twist and TwistStamped for compatibility with
TurtleBot3 Gazebo simulation on ROS2 Jazzy.

Publishes:
    /cmd_vel            (geometry_msgs/Twist + TwistStamped)
    /human_gesture      (std_msgs/String)
    /human_movement     (std_msgs/String)
    /human_action       (std_msgs/String)

Usage:
    ros2 run robot_prediction combined_ros2_node --ros-args \
        -p model_dir:=$HOME/ros2_ws/src/robot_prediction/checkpoints \
        -p use_camera_topic:=false -p camera_device:=0
"""

import rclpy
from rclpy.node import Node
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import json
import os
import pickle
import mediapipe as mp
from collections import deque

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import String
from cv_bridge import CvBridge

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


class HumanAwareNavigationNode(Node):

    def __init__(self):
        super().__init__('human_aware_navigation')

        # Parameters
        self.declare_parameter('model_dir', 'checkpoints')
        self.declare_parameter('seq_length', 30)
        self.declare_parameter('num_joints', 33)
        self.declare_parameter('confidence_threshold', 0.6)
        self.declare_parameter('robot_speed', 0.3)
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('use_camera_topic', True)
        self.declare_parameter('camera_device', 0)
        self.declare_parameter('demo_mode', True)

        self.model_dir = self.get_parameter('model_dir').value
        self.seq_length = self.get_parameter('seq_length').value
        self.num_joints = self.get_parameter('num_joints').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.robot_speed = self.get_parameter('robot_speed').value
        camera_topic = self.get_parameter('camera_topic').value
        self.use_camera_topic = self.get_parameter('use_camera_topic').value
        self.camera_device = self.get_parameter('camera_device').value
        self.demo_mode = self.get_parameter('demo_mode').value

        self.runtime_seq_length = min(self.seq_length, 20) if self.demo_mode else self.seq_length
        self.movement_buffer_size = 5 if self.demo_mode else 8
        self.action_buffer_size = 5 if self.demo_mode else 8
        self.moving_classes = {'approaching', 'moving_away', 'moving_left', 'moving_right'}
        self.motion_continuity_energy_threshold = 0.004
        self.motion_continuity_depth_threshold = 0.003
        self.motion_continuity_scale_threshold = 0.01
        self.motion_continuity_conf_threshold = 0.25
        self.direction_prior_threshold = 0.035
        self.direction_force_threshold = 0.05
        self.direction_prior_boost = 0.18

        self.device = torch.device('cpu')
        self.get_logger().info(f"Device: {self.device}")

        # Load all models
        self._load_gesture_model()
        self._load_movement_model()
        self._load_action_model()

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Buffers
        self.skeleton_buffer = deque(maxlen=self.runtime_seq_length)
        self.gesture_buffer = deque(maxlen=10)
        self.movement_buffer = deque(maxlen=self.movement_buffer_size)
        self.action_buffer = deque(maxlen=self.action_buffer_size)

        # CV Bridge
        self.bridge = CvBridge()


        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.gesture_pub = self.create_publisher(String, '/human_gesture', 10)
        self.movement_pub = self.create_publisher(String, '/human_movement', 10)
        self.action_pub = self.create_publisher(String, '/human_action', 10)

        # Subscribe to camera topic or use direct webcam
        if self.use_camera_topic:
            self.image_sub = self.create_subscription(
                Image, camera_topic, self.image_callback, 1
            )
            self.cap = None
            self.get_logger().info(f"Subscribing to camera topic: {camera_topic}")
        else:
            self.cap = cv2.VideoCapture(self.camera_device)
            if not self.cap.isOpened():
                self.get_logger().error("Cannot open webcam!")
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.timer = self.create_timer(0.033, self.timer_callback)
            self.get_logger().info(f"Using direct webcam: /dev/video{self.camera_device}")

        self.get_logger().info("Human-Aware Navigation Node initialized!")

    def _load_gesture_model(self):
        model_path = os.path.join(self.model_dir, 'gesture_landmark_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'gesture_landmark_scaler.pkl')
        classes_path = os.path.join(self.model_dir, 'gesture_landmark_classes.json')

        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.gesture_model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.gesture_scaler = pickle.load(f)
            with open(classes_path, 'r') as f:
                self.gesture_classes = json.load(f)
            self.get_logger().info(f"Gesture model loaded: {self.gesture_classes}")
        else:
            self.gesture_model = None
            self.get_logger().warn("Gesture model not found")

    def _load_movement_model(self):
        movement_json = os.path.join(self.model_dir, 'movement_classes.json')
        movement_pth = os.path.join(self.model_dir, 'movement_lstm_best.pth')

        if os.path.exists(movement_json) and os.path.exists(movement_pth):
            with open(movement_json) as f:
                classes = json.load(f)
            self.movement_classes = {v: k for k, v in classes.items()}

            self.movement_model = MovementLSTM(
                input_size=MOVEMENT_FEATURE_SIZE, num_classes=len(classes)
            )
            ckpt = torch.load(movement_pth, map_location=self.device, weights_only=True)
            self.movement_model.load_state_dict(ckpt['model_state_dict'])
            self.movement_model.to(self.device).eval()
            self.get_logger().info(f"Movement LSTM loaded: {list(classes.keys())}")
        else:
            self.movement_model = None
            self.get_logger().warn("Movement model not found")

    def _load_action_model(self):
        gcn_json = os.path.join(self.model_dir, 'stgcn_classes.json')
        gcn_pth = os.path.join(self.model_dir, 'stgcn_best.pth')

        if os.path.exists(gcn_json) and os.path.exists(gcn_pth):
            with open(gcn_json) as f:
                classes = json.load(f)
            self.action_classes = {v: k for k, v in classes.items()}

            self.action_model = SpatioTemporalGCN(
                num_classes=len(classes), num_joints=self.num_joints
            )
            ckpt = torch.load(gcn_pth, map_location=self.device, weights_only=True)
            self.action_model.load_state_dict(ckpt['model_state_dict'])
            self.action_model.to(self.device).eval()
            self.get_logger().info(f"ST-GCN loaded: {list(classes.keys())}")
        else:
            self.action_model = None
            self.get_logger().warn("ST-GCN model not found")

    def timer_callback(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            self.process_frame(frame)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.process_frame(frame)
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        gesture_name = ""
        gesture_conf = 0.0
        movement_name = ""
        movement_conf = 0.0
        move_probs = None
        depth_scale = None
        action_name = ""
        action_conf = 0.0
        action_probs = None
        hand_found = False

        # ---- GESTURE RECOGNITION ----
        if self.gesture_model is not None:
            hand_results = self.hands.process(frame_rgb)

            if hand_results.multi_hand_landmarks:
                hand_found = True
                hand_landmarks = hand_results.multi_hand_landmarks[0]

                raw = []
                for lm in hand_landmarks.landmark:
                    raw.extend([lm.x, lm.y, lm.z])

                features = extract_advanced_features(raw)
                features_scaled = self.gesture_scaler.transform([features])
                probs = self.gesture_model.predict_proba(features_scaled)[0]
                self.gesture_buffer.append(probs)

                if len(self.gesture_buffer) >= 3:
                    avg_probs = np.mean(list(self.gesture_buffer), axis=0)
                    pred_idx = np.argmax(avg_probs)
                    gesture_conf = avg_probs[pred_idx]
                    gesture_name = self.gesture_classes[pred_idx]
            else:
                self.gesture_buffer.clear()

        # ---- MOVEMENT & ACTION PREDICTION ----
        pose_results = self.pose.process(frame_rgb)

        if pose_results.pose_landmarks:
            landmarks = []
            for lm in pose_results.pose_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            skeleton = np.array(landmarks, dtype=np.float32)
            self.skeleton_buffer.append(skeleton)

            if len(self.skeleton_buffer) == self.runtime_seq_length:
                skeleton_seq = np.array(list(self.skeleton_buffer))

                # Movement LSTM
                if self.movement_model is not None:
                    flat = build_movement_features(skeleton_seq)
                    motion_energy = estimate_motion_energy(skeleton_seq)
                    depth_scale = estimate_depth_scale_signature(skeleton_seq)
                    input_t = torch.tensor(flat).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        logits = self.movement_model(input_t)
                        probs = F.softmax(logits, dim=1)

                    self.movement_buffer.append(probs.cpu().numpy()[0])

                    if len(self.movement_buffer) >= 3:
                        avg_probs = np.mean(list(self.movement_buffer), axis=0)
                        approach_bias = (
                            depth_scale['approaching_score'] - depth_scale['moving_away_score']
                        )
                        for idx, class_name in self.movement_classes.items():
                            if class_name == 'approaching' and approach_bias >= self.direction_prior_threshold:
                                avg_probs[idx] += self.direction_prior_boost
                            elif class_name == 'moving_away' and approach_bias <= -self.direction_prior_threshold:
                                avg_probs[idx] += self.direction_prior_boost
                        avg_probs = avg_probs / max(float(np.sum(avg_probs)), 1e-6)
                        move_probs = avg_probs
                        pred_idx, movement_conf, _ = apply_stationary_motion_gate(
                            avg_probs,
                            self.movement_classes,
                            motion_energy,
                            depth_change=depth_scale['depth_change'],
                            scale_change=depth_scale['scale_change'],
                        )
                        continuity_motion = (
                            motion_energy >= self.motion_continuity_energy_threshold or
                            depth_scale['depth_change'] >= self.motion_continuity_depth_threshold or
                            depth_scale['scale_change'] >= self.motion_continuity_scale_threshold
                        )
                        if (
                            self.movement_classes[pred_idx] == 'stationary' and
                            continuity_motion
                        ):
                            non_stationary_scores = [
                                (idx, score)
                                for idx, score in enumerate(avg_probs)
                                if self.movement_classes[idx] != 'stationary'
                            ]
                            if non_stationary_scores:
                                alt_idx, alt_conf = max(
                                    non_stationary_scores, key=lambda item: item[1]
                                )
                                if alt_conf >= self.motion_continuity_conf_threshold:
                                    pred_idx = alt_idx
                                    movement_conf = float(alt_conf)
                        if abs(approach_bias) >= self.direction_force_threshold:
                            preferred_name = 'approaching' if approach_bias > 0 else 'moving_away'
                            preferred_idx = next(
                                (
                                    idx for idx, class_name in self.movement_classes.items()
                                    if class_name == preferred_name
                                ),
                                None
                            )
                            if preferred_idx is not None:
                                pred_idx = preferred_idx
                                movement_conf = min(max(float(avg_probs[preferred_idx]), 0.7), 1.0)
                        movement_name = self.movement_classes[pred_idx]

                # ST-GCN
                if self.action_model is not None:
                    seq_t = skeleton_seq.transpose(2, 0, 1)
                    input_t = torch.tensor(seq_t).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        logits = self.action_model(input_t)
                        probs = F.softmax(logits, dim=1)

                    self.action_buffer.append(probs.cpu().numpy()[0])

                    if len(self.action_buffer) >= 3:
                        avg_probs = np.mean(list(self.action_buffer), axis=0)
                        action_probs = avg_probs
                        pred_idx = np.argmax(avg_probs)
                        action_conf = avg_probs[pred_idx]
                        action_name = self.action_classes[pred_idx]
        else:
            self.skeleton_buffer.clear()
            self.movement_buffer.clear()
            self.action_buffer.clear()

        # ---- DECISION & PUBLISH ----
        cmd = Twist()

        # Priority 1: Gesture commands
        gesture_valid = (hand_found and gesture_name and
                        gesture_conf > self.confidence_threshold)

        if gesture_valid:
            if gesture_name == 'stop':
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.get_logger().info(f"GESTURE: STOP ({gesture_conf:.0%})")
            elif gesture_name == 'forward':
                cmd.linear.x = self.robot_speed
                cmd.angular.z = 0.0
                self.get_logger().info(f"GESTURE: FORWARD ({gesture_conf:.0%})")
            elif gesture_name == 'backward':
                cmd.linear.x = -self.robot_speed
                cmd.angular.z = 0.0
                self.get_logger().info(f"GESTURE: BACKWARD ({gesture_conf:.0%})")
            elif gesture_name == 'left':
                cmd.linear.x = 0.0
                cmd.angular.z = 0.5
                self.get_logger().info(f"GESTURE: TURN LEFT ({gesture_conf:.0%})")
            elif gesture_name == 'right':
                cmd.linear.x = 0.0
                cmd.angular.z = -0.5
                self.get_logger().info(f"GESTURE: TURN RIGHT ({gesture_conf:.0%})")

        # Priority 2: Movement avoidance
        else:
            fused_name = None
            fused_conf = 0.0
            if move_probs is not None and action_probs is not None:
                move_scores = {
                    name: max(
                        (
                            move_probs[idx]
                            for idx, class_name in self.movement_classes.items()
                            if class_name == name
                        ),
                        default=0.0
                    )
                    for name in set(self.movement_classes.values())
                }
                action_scores = {
                    name: max(
                        (
                            action_probs[idx]
                            for idx, class_name in self.action_classes.items()
                            if class_name == name
                        ),
                        default=0.0
                    )
                    for name in set(self.action_classes.values())
                }
                direction_bias = (
                    depth_scale['approaching_score'] - depth_scale['moving_away_score']
                    if depth_scale is not None
                    else 0.0
                )
                approaching_move_conf = move_scores.get('approaching', 0.0)
                moving_away_move_conf = move_scores.get('moving_away', 0.0)
                approaching_action_conf = action_scores.get('approaching', 0.0)

                if direction_bias >= self.direction_force_threshold:
                    fused_name = 'approaching'
                    fused_conf = min(max(approaching_move_conf, 0.7), 1.0)
                elif direction_bias <= -self.direction_force_threshold:
                    fused_name = 'moving_away'
                    fused_conf = min(max(moving_away_move_conf, 0.7), 1.0)
                elif (
                    approaching_action_conf > 0.8 and
                    moving_away_move_conf < 0.45 and
                    approaching_move_conf >= 0.2
                ):
                    fused_name = 'approaching'
                    fused_conf = approaching_action_conf
                else:
                    fused_scores = {}
                    all_names = set(move_scores) | set(action_scores)
                    for name in all_names:
                        move_score = move_scores.get(name, 0.0)
                        action_score = action_scores.get(name, 0.0)
                        if name == 'approaching':
                            fused_scores[name] = 0.3 * move_score + 0.7 * action_score
                            if direction_bias >= self.direction_prior_threshold:
                                fused_scores[name] += self.direction_prior_boost
                        elif name == 'moving_away':
                            fused_scores[name] = 0.7 * move_score + 0.3 * action_score
                            if direction_bias <= -self.direction_prior_threshold:
                                fused_scores[name] += self.direction_prior_boost
                        else:
                            fused_scores[name] = 0.7 * move_score + 0.3 * action_score
                    fused_name, fused_conf = max(fused_scores.items(), key=lambda item: item[1])
                    fused_conf = min(float(fused_conf), 1.0)
            elif movement_name and movement_conf > self.confidence_threshold:
                fused_name = movement_name
                fused_conf = movement_conf
            elif action_name and action_conf > self.confidence_threshold:
                fused_name = action_name
                fused_conf = action_conf

            if fused_name == 'approaching' and fused_conf > self.confidence_threshold:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.5
                self.get_logger().warn(f"AVOIDANCE: Approaching ({fused_conf:.0%})")
            elif fused_name == 'moving_left' and fused_conf > self.confidence_threshold:
                cmd.linear.x = self.robot_speed * 0.5
                cmd.angular.z = -0.3
            elif fused_name == 'moving_right' and fused_conf > self.confidence_threshold:
                cmd.linear.x = self.robot_speed * 0.5
                cmd.angular.z = 0.3
            elif fused_name in ('moving_away', 'stationary'):
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0

    

        # Publish TwistStamped for Gazebo TurtleBot3 compatibility
        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = 'base_link'
        stamped.twist = cmd
        self.cmd_vel_pub.publish(stamped)

        # Publish predictions
        if gesture_name and gesture_conf > self.confidence_threshold:
            msg = String()
            msg.data = f"{gesture_name}:{gesture_conf:.2f}"
            self.gesture_pub.publish(msg)

        if movement_name and movement_conf > self.confidence_threshold:
            msg = String()
            msg.data = f"{movement_name}:{movement_conf:.2f}"
            self.movement_pub.publish(msg)

        if action_name and action_conf > self.confidence_threshold:
            msg = String()
            msg.data = f"{action_name}:{action_conf:.2f}"
            self.action_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HumanAwareNavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.cap is not None:
            node.cap.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
