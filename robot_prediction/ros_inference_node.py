#!/usr/bin/env python3
"""
ROS2 Inference Node — Human-Aware Robot Navigation (Jazzy)
============================================================

Pipeline:
    Camera Image
        ├── MediaPipe Pose → skeleton keypoints
        │       ├── LSTM → movement prediction
        │       └── ST-GCN → action recognition
        ├── CNN → gesture recognition
        └── Decision Logic → /cmd_vel (Twist message)

ROS2 Topics:
    Subscribes:
        /camera/image_raw      (sensor_msgs/Image)

    Publishes:
        /cmd_vel               (geometry_msgs/Twist)
        /human_gesture         (std_msgs/String)
        /human_movement        (std_msgs/String)
        /human_action          (std_msgs/String)

Usage:
    ros2 run robot_prediction ros_inference_node
"""

import rclpy
from rclpy.node import Node
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import json
import os
import mediapipe as mp
from collections import deque

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge

# Import models — adjust path if needed
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import GestureCNN, MovementLSTM, SpatioTemporalGCN
from movement_features import (
    MOVEMENT_FEATURE_SIZE,
    apply_stationary_motion_gate,
    build_movement_features,
    estimate_depth_scale_signature,
    estimate_motion_energy,
)


class HumanAwareNavigationNode(Node):
    """Main ROS2 node combining all perception models."""

    def __init__(self):
        super().__init__('human_aware_navigation')

        # Declare parameters
        self.declare_parameter('model_dir', 'checkpoints')
        self.declare_parameter('seq_length', 30)
        self.declare_parameter('num_joints', 33)
        self.declare_parameter('confidence_threshold', 0.6)
        self.declare_parameter('robot_speed', 0.3)
        self.declare_parameter('use_gpu', False)
        self.declare_parameter('camera_topic', '/camera/image_raw')

        # Get parameters
        self.model_dir = self.get_parameter('model_dir').value
        self.seq_length = self.get_parameter('seq_length').value
        self.num_joints = self.get_parameter('num_joints').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.robot_speed = self.get_parameter('robot_speed').value
        self.use_gpu = self.get_parameter('use_gpu').value
        camera_topic = self.get_parameter('camera_topic').value

        # Device
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.get_logger().info(f"Using device: {self.device}")

        # Load models
        self._load_models()

        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Skeleton buffer for temporal models
        self.skeleton_buffer = deque(maxlen=self.seq_length)

        # CV Bridge
        self.bridge = CvBridge()

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.gesture_pub = self.create_publisher(String, '/human_gesture', 10)
        self.movement_pub = self.create_publisher(String, '/human_movement', 10)
        self.action_pub = self.create_publisher(String, '/human_action', 10)

        # Subscriber
        self.image_sub = self.create_subscription(
            Image, camera_topic, self.image_callback, 1
        )

        self.get_logger().info("Human-Aware Navigation Node initialized!")

    def _load_models(self):
        """Load all trained models."""
        # --- Gesture CNN ---
        gesture_classes_path = os.path.join(self.model_dir, 'gesture_classes.json')
        if os.path.exists(gesture_classes_path):
            with open(gesture_classes_path) as f:
                self.gesture_classes = json.load(f)
            self.gesture_idx_to_class = {v: k for k, v in self.gesture_classes.items()}

            self.gesture_model = GestureCNN(
                num_classes=len(self.gesture_classes), pretrained=False
            )
            ckpt = torch.load(
                os.path.join(self.model_dir, 'gesture_cnn_best.pth'),
                map_location=self.device, weights_only=True
            )
            self.gesture_model.load_state_dict(ckpt['model_state_dict'])
            self.gesture_model.to(self.device).eval()
            self.get_logger().info(f"Gesture CNN loaded: {list(self.gesture_classes.keys())}")
        else:
            self.gesture_model = None
            self.get_logger().warn("Gesture model not found — skipping gesture recognition")

        # --- Movement LSTM ---
        movement_classes_path = os.path.join(self.model_dir, 'movement_classes.json')
        if os.path.exists(movement_classes_path):
            with open(movement_classes_path) as f:
                self.movement_classes = json.load(f)
            self.movement_idx_to_class = {v: k for k, v in self.movement_classes.items()}

            self.movement_model = MovementLSTM(
                input_size=MOVEMENT_FEATURE_SIZE,
                num_classes=len(self.movement_classes)
            )
            ckpt = torch.load(
                os.path.join(self.model_dir, 'movement_lstm_best.pth'),
                map_location=self.device, weights_only=True
            )
            self.movement_model.load_state_dict(ckpt['model_state_dict'])
            self.movement_model.to(self.device).eval()
            self.get_logger().info(f"Movement LSTM loaded: {list(self.movement_classes.keys())}")
        else:
            self.movement_model = None
            self.get_logger().warn("Movement model not found — skipping movement prediction")

        # --- ST-GCN ---
        stgcn_classes_path = os.path.join(self.model_dir, 'stgcn_classes.json')
        if os.path.exists(stgcn_classes_path):
            with open(stgcn_classes_path) as f:
                self.action_classes = json.load(f)
            self.action_idx_to_class = {v: k for k, v in self.action_classes.items()}

            self.action_model = SpatioTemporalGCN(
                num_classes=len(self.action_classes),
                num_joints=self.num_joints
            )
            ckpt = torch.load(
                os.path.join(self.model_dir, 'stgcn_best.pth'),
                map_location=self.device, weights_only=True
            )
            self.action_model.load_state_dict(ckpt['model_state_dict'])
            self.action_model.to(self.device).eval()
            self.get_logger().info(f"ST-GCN loaded: {list(self.action_classes.keys())}")
        else:
            self.action_model = None
            self.get_logger().warn("ST-GCN model not found — skipping action recognition")

    def image_callback(self, msg):
        """Process each camera frame."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- 1. Gesture Recognition ---
        gesture, gesture_conf = self._predict_gesture(frame_rgb)

        # --- 2. Pose Estimation ---
        pose_results = self.pose.process(frame_rgb)

        movement = None
        movement_conf = 0.0
        action = None
        action_conf = 0.0

        if pose_results.pose_landmarks:
            landmarks = []
            for lm in pose_results.pose_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            skeleton = np.array(landmarks, dtype=np.float32)

            self.skeleton_buffer.append(skeleton)

            if len(self.skeleton_buffer) == self.seq_length:
                skeleton_seq = np.array(list(self.skeleton_buffer))

                # --- 3. Movement Prediction ---
                movement, movement_conf = self._predict_movement(skeleton_seq)

                # --- 4. Action Recognition ---
                action, action_conf = self._predict_action(skeleton_seq)

        # --- 5. Decision & Command ---
        self._make_decision(gesture, gesture_conf,
                           movement, movement_conf,
                           action, action_conf)

        # Publish predictions
        if gesture:
            msg_out = String()
            msg_out.data = f"{gesture}:{gesture_conf:.2f}"
            self.gesture_pub.publish(msg_out)
        if movement:
            msg_out = String()
            msg_out.data = f"{movement}:{movement_conf:.2f}"
            self.movement_pub.publish(msg_out)
        if action:
            msg_out = String()
            msg_out.data = f"{action}:{action_conf:.2f}"
            self.action_pub.publish(msg_out)

    def _predict_gesture(self, frame_rgb):
        if self.gesture_model is None:
            return None, 0.0

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img_tensor = transform(frame_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.gesture_model(img_tensor)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(1)

        gesture = self.gesture_idx_to_class[pred.item()]
        confidence = conf.item()

        if confidence < self.confidence_threshold:
            return None, confidence

        return gesture, confidence

    def _predict_movement(self, skeleton_seq):
        if self.movement_model is None:
            return None, 0.0

        flat = build_movement_features(skeleton_seq)
        motion_energy = estimate_motion_energy(skeleton_seq)
        depth_scale = estimate_depth_scale_signature(skeleton_seq)

        input_tensor = torch.tensor(flat).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.movement_model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx, confidence, _ = apply_stationary_motion_gate(
            probs,
            self.movement_idx_to_class,
            motion_energy,
            depth_change=depth_scale['depth_change'],
            scale_change=depth_scale['scale_change'],
        )
        movement = self.movement_idx_to_class[pred_idx]

        if confidence < self.confidence_threshold:
            return None, confidence

        return movement, confidence

    def _predict_action(self, skeleton_seq):
        if self.action_model is None:
            return None, 0.0

        seq_transposed = skeleton_seq.transpose(2, 0, 1)
        input_tensor = torch.tensor(seq_transposed).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.action_model(input_tensor)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(1)

        action = self.action_idx_to_class[pred.item()]
        confidence = conf.item()

        if confidence < self.confidence_threshold:
            return None, confidence

        return action, confidence

    def _make_decision(self, gesture, gesture_conf,
                       movement, movement_conf,
                       action, action_conf):
        """
        Priority:
        1. Gesture commands (explicit human instructions)
        2. Movement prediction (collision avoidance)
        3. Default: continue forward
        """
        cmd = Twist()

        # PRIORITY 1: Gesture commands
        if gesture and gesture_conf > self.confidence_threshold:
            if gesture == 'stop':
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.get_logger().info(f"GESTURE: STOP (conf={gesture_conf:.2f})")

            elif gesture == 'forward':
                cmd.linear.x = self.robot_speed
                cmd.angular.z = 0.0
                self.get_logger().info(f"GESTURE: FORWARD (conf={gesture_conf:.2f})")

            elif gesture == 'backward':
                cmd.linear.x = -self.robot_speed
                cmd.angular.z = 0.0
                self.get_logger().info(f"GESTURE: BACKWARD (conf={gesture_conf:.2f})")

            elif gesture == 'left':
                cmd.linear.x = 0.0
                cmd.angular.z = 0.5
                self.get_logger().info(f"GESTURE: TURN LEFT (conf={gesture_conf:.2f})")

            elif gesture == 'right':
                cmd.linear.x = 0.0
                cmd.angular.z = -0.5
                self.get_logger().info(f"GESTURE: TURN RIGHT (conf={gesture_conf:.2f})")

            self.cmd_vel_pub.publish(cmd)
            return

        # PRIORITY 2: Movement prediction (collision avoidance)
        if movement and movement_conf > self.confidence_threshold:
            if movement == 'approaching':
                cmd.linear.x = 0.0
                cmd.angular.z = 0.5
                self.get_logger().warn(f"AVOIDANCE: Human approaching — turning left "
                            f"(conf={movement_conf:.2f})")

            elif movement == 'moving_left':
                cmd.linear.x = self.robot_speed * 0.5
                cmd.angular.z = -0.3
                self.get_logger().info(f"AVOIDANCE: Human moving left — adjusting right")

            elif movement == 'moving_right':
                cmd.linear.x = self.robot_speed * 0.5
                cmd.angular.z = 0.3
                self.get_logger().info(f"AVOIDANCE: Human moving right — adjusting left")

            elif movement in ('moving_away', 'stationary'):
                cmd.linear.x = self.robot_speed
                cmd.angular.z = 0.0

            self.cmd_vel_pub.publish(cmd)
            return

        # PRIORITY 3: Default — move forward slowly
        cmd.linear.x = self.robot_speed * 0.5
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = HumanAwareNavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
