#!/usr/bin/env python3
"""
Movement Data Augmenter
========================

Takes your existing 50 samples per class and generates
hundreds more through realistic transformations:

- Add noise (simulates tracking jitter)
- Scale (simulates different distances)
- Time shift (simulates different timing)
- Mirror (simulates left/right flip)
- Speed variation (simulates faster/slower movement)
- Interpolation (creates new sequences between existing ones)

Usage:
    python3 augment_movement.py --data_dir data/movement --multiplier 10
    
    This turns 50 samples into 500 per class.
"""

import os
import numpy as np
import argparse


def add_noise(sequence, noise_level=0.005):
    """Add small random noise to joint positions."""
    noise = np.random.randn(*sequence.shape).astype(np.float32) * noise_level
    return sequence + noise


def scale_skeleton(sequence, scale_range=(0.8, 1.2)):
    """Scale skeleton size (simulates different distances from camera)."""
    scale = np.random.uniform(scale_range[0], scale_range[1])
    # Center on hip, scale, then move back
    center = sequence[:, 0:1, :].copy()
    centered = sequence - center
    scaled = centered * scale
    return (scaled + center).astype(np.float32)


def time_shift(sequence, max_shift=5):
    """Shift sequence in time (roll frames)."""
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(sequence, shift, axis=0).astype(np.float32)


def mirror_horizontal(sequence):
    """Flip skeleton left-right (mirror x-axis around center)."""
    mirrored = sequence.copy()
    # Flip x coordinates around 0.5 (center of frame)
    mirrored[:, :, 0] = 1.0 - mirrored[:, :, 0]
    
    # Swap left/right joint pairs (MediaPipe Pose)
    left_right_pairs = [
        (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
        (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
    ]
    for left, right in left_right_pairs:
        mirrored[:, [left, right], :] = mirrored[:, [right, left], :]
    
    return mirrored.astype(np.float32)


def speed_variation(sequence, speed_range=(0.7, 1.3)):
    """Change speed by resampling frames."""
    T, J, C = sequence.shape
    speed = np.random.uniform(speed_range[0], speed_range[1])
    new_T = int(T * speed)
    if new_T < 5:
        new_T = 5
    
    # Resample to new length
    old_indices = np.linspace(0, T - 1, new_T)
    new_sequence = np.zeros((new_T, J, C), dtype=np.float32)
    for j in range(J):
        for c in range(C):
            new_sequence[:, j, c] = np.interp(old_indices, np.arange(T), sequence[:, j, c])
    
    # Pad or truncate back to original length
    if new_T < T:
        pad = np.zeros((T - new_T, J, C), dtype=np.float32)
        pad[:] = new_sequence[-1:]  # Repeat last frame
        new_sequence = np.concatenate([new_sequence, pad], axis=0)
    else:
        new_sequence = new_sequence[:T]
    
    return new_sequence


def interpolate_sequences(seq1, seq2, alpha=0.5):
    """Create new sequence by blending two existing sequences."""
    return (seq1 * alpha + seq2 * (1 - alpha)).astype(np.float32)


def random_joint_dropout(sequence, dropout_rate=0.05):
    """Randomly zero out some joints (simulates occlusion)."""
    augmented = sequence.copy()
    T, J, C = augmented.shape
    mask = np.random.random((T, J)) > dropout_rate
    mask = mask[:, :, np.newaxis].repeat(C, axis=2)
    augmented = augmented * mask
    return augmented.astype(np.float32)


def translate_skeleton(sequence, max_shift=0.1):
    """Shift entire skeleton position (simulates different camera position)."""
    shift_x = np.random.uniform(-max_shift, max_shift)
    shift_y = np.random.uniform(-max_shift, max_shift)
    augmented = sequence.copy()
    augmented[:, :, 0] += shift_x
    augmented[:, :, 1] += shift_y
    return augmented.astype(np.float32)


def augment_class(class_dir, multiplier=10):
    """Augment all samples in one class directory."""
    files = sorted([f for f in os.listdir(class_dir) if f.endswith('.npy')])
    original_count = len(files)
    
    if original_count == 0:
        print(f"  No samples found!")
        return 0
    
    # Load all original sequences
    originals = []
    for f in files:
        seq = np.load(os.path.join(class_dir, f))
        originals.append(seq)
    
    new_count = 0
    target = original_count * multiplier
    next_idx = original_count
    
    while new_count < target:
        # Pick a random original
        idx = np.random.randint(len(originals))
        seq = originals[idx]
        
        # Apply random combination of augmentations
        augmented = seq.copy()
        
        # Always add some noise
        augmented = add_noise(augmented, noise_level=np.random.uniform(0.002, 0.01))
        
        # Randomly apply other augmentations
        if np.random.random() < 0.5:
            augmented = scale_skeleton(augmented)
        
        if np.random.random() < 0.3:
            augmented = time_shift(augmented, max_shift=3)
        
        if np.random.random() < 0.3:
            augmented = speed_variation(augmented, speed_range=(0.8, 1.2))
        
        if np.random.random() < 0.2:
            augmented = random_joint_dropout(augmented, dropout_rate=0.03)
        
        if np.random.random() < 0.5:
            augmented = translate_skeleton(augmented, max_shift=0.08)
        
        # Occasionally interpolate between two samples
        if np.random.random() < 0.3 and len(originals) > 1:
            idx2 = np.random.randint(len(originals))
            alpha = np.random.uniform(0.3, 0.7)
            augmented = interpolate_sequences(augmented, originals[idx2], alpha)
        
        # Save
        save_path = os.path.join(class_dir, f"seq{next_idx:04d}.npy")
        np.save(save_path, augmented)
        next_idx += 1
        new_count += 1
    
    return new_count


def augment_mirror_classes(data_dir):
    """
    Special handling: mirror 'moving_left' to create more 'moving_right' and vice versa.
    Also mirror 'approaching' and 'moving_away' for variety.
    """
    mirror_pairs = [
        ('moving_left', 'moving_right'),
        ('moving_right', 'moving_left'),
    ]
    
    for source_cls, target_cls in mirror_pairs:
        source_dir = os.path.join(data_dir, source_cls)
        target_dir = os.path.join(data_dir, target_cls)
        
        if not os.path.isdir(source_dir) or not os.path.isdir(target_dir):
            continue
        
        source_files = [f for f in os.listdir(source_dir) if f.endswith('.npy')]
        existing_target = len([f for f in os.listdir(target_dir) if f.endswith('.npy')])
        next_idx = existing_target
        
        count = 0
        for f in source_files[:20]:  # Mirror up to 20 samples
            seq = np.load(os.path.join(source_dir, f))
            mirrored = mirror_horizontal(seq)
            mirrored = add_noise(mirrored, noise_level=0.005)
            
            save_path = os.path.join(target_dir, f"seq{next_idx:04d}.npy")
            np.save(save_path, mirrored)
            next_idx += 1
            count += 1
        
        if count > 0:
            print(f"  Mirrored {count} samples from {source_cls} → {target_cls}")


def main():
    parser = argparse.ArgumentParser(description='Augment movement skeleton data')
    parser.add_argument('--data_dir', type=str, default='data/movement')
    parser.add_argument('--multiplier', type=int, default=10,
                        help='How many augmented samples per original (default: 10)')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("MOVEMENT DATA AUGMENTATION")
    print("=" * 60)

    # Show original counts
    print("\nOriginal data:")
    for cls in sorted(os.listdir(args.data_dir)):
        cls_dir = os.path.join(args.data_dir, cls)
        if os.path.isdir(cls_dir):
            count = len([f for f in os.listdir(cls_dir) if f.endswith('.npy')])
            print(f"  {cls}: {count} samples")

    # Mirror left/right
    print("\nMirroring left/right classes...")
    augment_mirror_classes(args.data_dir)

    # Augment each class
    print(f"\nAugmenting with {args.multiplier}x multiplier...")
    for cls in sorted(os.listdir(args.data_dir)):
        cls_dir = os.path.join(args.data_dir, cls)
        if os.path.isdir(cls_dir):
            new = augment_class(cls_dir, multiplier=args.multiplier)
            total = len([f for f in os.listdir(cls_dir) if f.endswith('.npy')])
            print(f"  {cls}: +{new} augmented → {total} total")

    # Final counts
    print("\nFinal data:")
    for cls in sorted(os.listdir(args.data_dir)):
        cls_dir = os.path.join(args.data_dir, cls)
        if os.path.isdir(cls_dir):
            count = len([f for f in os.listdir(cls_dir) if f.endswith('.npy')])
            print(f"  {cls}: {count} samples")

    print("\nDone! Now retrain with:")
    print("  python3 robot_prediction/train.py --model movement --movement_dir data/movement --epochs 80")
    print("  python3 robot_prediction/train.py --model gcn --movement_dir data/movement --epochs 80")


if __name__ == '__main__':
    main()
