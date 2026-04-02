#!/usr/bin/env python3
"""
NTU RGB+D to Movement Dataset Converter
=========================================

Reads NTU RGB+D .skeleton files and converts them to our
movement dataset format (30 frames, 33 MediaPipe joints, xyz).

NTU has 25 Kinect joints, we need 33 MediaPipe joints.
Missing joints are interpolated from nearby joints.

NTU Action Classes we use:
    A008 = walking towards (approaching)
    A009 = walking apart (moving_away)
    A027 = jump up (stationary - we use standing frames)
    A028 = phone call (stationary - standing still)
    A060 = touch other person's pocket (approaching)
    
    For left/right we mirror walking sequences.

Usage:
    python3 convert_ntu.py --skeleton_dir ~/Desktop/ntu-rgbd/nturgb+d_skeletons --output_dir data/movement
"""

import os
import sys
import argparse
import numpy as np
from glob import glob


def parse_skeleton_file(filepath):
    """
    Parse one NTU RGB+D .skeleton file.
    Returns list of frames, each frame is (num_bodies, 25, 3) array.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    idx = 0
    num_frames = int(lines[idx].strip())
    idx += 1
    
    all_frames = []
    
    for frame_i in range(num_frames):
        if idx >= len(lines):
            break
        
        num_bodies = int(lines[idx].strip())
        idx += 1
        
        frame_bodies = []
        for body_i in range(num_bodies):
            if idx >= len(lines):
                break
            
            # Body info line
            body_info = lines[idx].strip().split()
            idx += 1
            
            num_joints = int(lines[idx].strip())
            idx += 1
            
            joints = []
            for joint_i in range(num_joints):
                if idx >= len(lines):
                    break
                parts = lines[idx].strip().split()
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                joints.append([x, y, z])
                idx += 1
            
            if len(joints) == 25:
                frame_bodies.append(np.array(joints, dtype=np.float32))
        
        if frame_bodies:
            all_frames.append(frame_bodies[0])  # Use first body only
    
    return all_frames


def kinect25_to_mediapipe33(kinect_joints):
    """
    Map 25 Kinect joints to 33 MediaPipe Pose joints.
    
    Kinect 25 joints:
    0=SpineBase, 1=SpineMid, 2=Neck, 3=Head,
    4=ShoulderLeft, 5=ElbowLeft, 6=WristLeft, 7=HandLeft,
    8=ShoulderRight, 9=ElbowRight, 10=WristRight, 11=HandRight,
    12=HipLeft, 13=KneeLeft, 14=AnkleLeft, 15=FootLeft,
    16=HipRight, 17=KneeRight, 18=AnkleRight, 19=FootRight,
    20=SpineShoulder, 21=HandTipLeft, 22=ThumbLeft,
    23=HandTipRight, 24=ThumbRight
    
    MediaPipe 33 joints:
    0=Nose, 1-10=Face, 11=LeftShoulder, 12=RightShoulder,
    13=LeftElbow, 14=RightElbow, 15=LeftWrist, 16=RightWrist,
    17-22=Hands, 23=LeftHip, 24=RightHip,
    25=LeftKnee, 26=RightKnee, 27=LeftAnkle, 28=RightAnkle,
    29-32=Feet
    """
    mp = np.zeros((33, 3), dtype=np.float32)
    
    # Face (approximate from head and neck)
    mp[0] = kinect_joints[3]  # Nose = Head
    mp[1] = kinect_joints[3] + [0.02, -0.01, 0]  # Left eye inner
    mp[2] = kinect_joints[3] + [0.03, -0.01, 0]  # Left eye
    mp[3] = kinect_joints[3] + [0.04, -0.01, 0]  # Left eye outer
    mp[4] = kinect_joints[3] + [-0.02, -0.01, 0]  # Right eye inner
    mp[5] = kinect_joints[3] + [-0.03, -0.01, 0]  # Right eye
    mp[6] = kinect_joints[3] + [-0.04, -0.01, 0]  # Right eye outer
    mp[7] = kinect_joints[3] + [0.05, 0.02, 0]  # Left ear
    mp[8] = kinect_joints[3] + [-0.05, 0.02, 0]  # Right ear
    mp[9] = kinect_joints[3] + [0.01, 0.03, 0]  # Mouth left
    mp[10] = kinect_joints[3] + [-0.01, 0.03, 0]  # Mouth right
    
    # Shoulders
    mp[11] = kinect_joints[4]   # Left shoulder
    mp[12] = kinect_joints[8]   # Right shoulder
    
    # Elbows
    mp[13] = kinect_joints[5]   # Left elbow
    mp[14] = kinect_joints[9]   # Right elbow
    
    # Wrists
    mp[15] = kinect_joints[6]   # Left wrist
    mp[16] = kinect_joints[10]  # Right wrist
    
    # Hands (approximate from hand tips and thumbs)
    mp[17] = kinect_joints[22]  # Left pinky (use thumb as approx)
    mp[18] = kinect_joints[24]  # Right pinky (use thumb as approx)
    mp[19] = kinect_joints[21]  # Left index (use hand tip)
    mp[20] = kinect_joints[23]  # Right index (use hand tip)
    mp[21] = kinect_joints[22]  # Left thumb
    mp[22] = kinect_joints[24]  # Right thumb
    
    # Hips
    mp[23] = kinect_joints[12]  # Left hip
    mp[24] = kinect_joints[16]  # Right hip
    
    # Knees
    mp[25] = kinect_joints[13]  # Left knee
    mp[26] = kinect_joints[17]  # Right knee
    
    # Ankles
    mp[27] = kinect_joints[14]  # Left ankle
    mp[28] = kinect_joints[18]  # Right ankle
    
    # Feet
    mp[29] = kinect_joints[15]  # Left heel (use foot)
    mp[30] = kinect_joints[19]  # Right heel (use foot)
    mp[31] = kinect_joints[15] + [0, 0, 0.05]  # Left foot index
    mp[32] = kinect_joints[19] + [0, 0, 0.05]  # Right foot index
    
    # Normalize to 0-1 range (MediaPipe style)
    for c in range(3):
        col = mp[:, c]
        cmin, cmax = col.min(), col.max()
        if cmax - cmin > 1e-6:
            mp[:, c] = (col - cmin) / (cmax - cmin)
        else:
            mp[:, c] = 0.5
    
    return mp


def extract_sequence(frames, seq_length=30):
    """Extract a fixed-length sequence from frames."""
    if len(frames) < seq_length:
        # Pad by repeating last frame
        while len(frames) < seq_length:
            frames.append(frames[-1])
    
    # Take middle portion
    start = (len(frames) - seq_length) // 2
    return frames[start:start + seq_length]


def convert_skeleton_to_npy(skeleton_file, seq_length=30):
    """Convert one .skeleton file to our format: (30, 33, 3)."""
    frames = parse_skeleton_file(skeleton_file)
    
    if len(frames) < 10:
        return None
    
    # Extract sequence
    seq_frames = extract_sequence(frames, seq_length)
    
    # Convert each frame from Kinect 25 to MediaPipe 33
    converted = []
    for frame in seq_frames:
        mp_joints = kinect25_to_mediapipe33(frame)
        converted.append(mp_joints)
    
    return np.array(converted, dtype=np.float32)  # (30, 33, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skeleton_dir', type=str, 
                        default=os.path.expanduser('~/Desktop/ntu-rgbd/nturgb+d_skeletons'))
    parser.add_argument('--output_dir', type=str, default='data/movement')
    parser.add_argument('--seq_length', type=int, default=30)
    parser.add_argument('--max_per_class', type=int, default=200,
                        help='Max samples per movement class')
    args = parser.parse_args()
    
    # NTU action classes mapped to our movement classes
    action_mapping = {
        'approaching': ['008', '009'],      # Walking towards, touch pocket
        'moving_away': ['010', '009'],       # Walking apart
        'stationary': ['028', '070', '027'], # Phone call, standing, jump up (use start frames)
        'moving_left': ['060'],              # Walking (will use as-is)
        'moving_right': ['060'],             # Walking (will mirror)
    }
    
    # Get all skeleton files
    skeleton_files = sorted(glob(os.path.join(args.skeleton_dir, '*.skeleton')))
    print(f"Found {len(skeleton_files)} skeleton files")
    
    # Group by action class
    action_files = {}
    for f in skeleton_files:
        basename = os.path.basename(f)
        # Extract action number: SsssCcccPpppRrrrAaaa.skeleton
        action_id = basename[-16:-9]  # Gets 'A008' etc
        action_num = basename.split('A')[1].split('.')[0]  # Gets '008'
        if action_num not in action_files:
            action_files[action_num] = []
        action_files[action_num].append(f)
    
    print(f"\nAction classes found:")
    for a in sorted(action_files.keys())[:20]:
        print(f"  A{a}: {len(action_files[a])} files")
    
    # Convert for each movement class
    for movement_class, ntu_actions in action_mapping.items():
        class_dir = os.path.join(args.output_dir, movement_class)
        os.makedirs(class_dir, exist_ok=True)
        
        # Count existing
        existing = len([f for f in os.listdir(class_dir) if f.endswith('.npy')])
        next_idx = existing
        count = 0
        
        for action_num in ntu_actions:
            if action_num not in action_files:
                print(f"  Warning: Action A{action_num} not found in dataset")
                continue
            
            files = action_files[action_num]
            
            for skel_file in files:
                if count >= args.max_per_class:
                    break
                
                try:
                    sequence = convert_skeleton_to_npy(skel_file, args.seq_length)
                    if sequence is not None:
                        save_path = os.path.join(class_dir, f"ntu{next_idx:04d}.npy")
                        np.save(save_path, sequence)
                        next_idx += 1
                        count += 1
                        
                        # For moving_right, also save mirrored version
                        if movement_class == 'moving_right':
                            mirrored = sequence.copy()
                            mirrored[:, :, 0] = 1.0 - mirrored[:, :, 0]
                            save_path = os.path.join(class_dir, f"ntu{next_idx:04d}.npy")
                            np.save(save_path, mirrored)
                            next_idx += 1
                            count += 1
                except Exception as e:
                    continue
            
            if count >= args.max_per_class:
                break
        
        total = len([f for f in os.listdir(class_dir) if f.endswith('.npy')])
        print(f"  {movement_class}: +{count} from NTU → {total} total")
    
    print("\nDone! Now retrain with:")
    print("  python3 robot_prediction/train.py --model movement --movement_dir data/movement --epochs 80")


if __name__ == '__main__':
    main()
