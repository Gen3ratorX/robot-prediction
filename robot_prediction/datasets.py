"""
Dataset Preparation for Human Movement Prediction & Gesture Recognition
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from movement_features import build_movement_features


class GestureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append((
                        os.path.join(cls_dir, fname),
                        self.class_to_idx[cls]
                    ))
        print(f"[GestureDataset] Found {len(self.samples)} images "
              f"across {len(self.classes)} classes: {self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, label


class MovementDataset(Dataset):
    def __init__(self, root_dir, seq_length=30, normalize=True):
        self.root_dir = root_dir
        self.seq_length = seq_length
        self.normalize = normalize
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.endswith('.npy'):
                    self.samples.append((
                        os.path.join(cls_dir, fname),
                        self.class_to_idx[cls]
                    ))
        print(f"[MovementDataset] Found {len(self.samples)} sequences "
              f"across {len(self.classes)} classes: {self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_path, label = self.samples[idx]
        skeleton_seq = np.load(seq_path).astype(np.float32)
        if len(skeleton_seq) < self.seq_length:
            pad = np.zeros((self.seq_length - len(skeleton_seq),
                          skeleton_seq.shape[1], skeleton_seq.shape[2]),
                          dtype=np.float32)
            skeleton_seq = np.concatenate([skeleton_seq, pad], axis=0)
        else:
            skeleton_seq = skeleton_seq[:self.seq_length]
        if self.normalize:
            skeleton_flat = build_movement_features(skeleton_seq)
        else:
            T, J, C = skeleton_seq.shape
            skeleton_flat = skeleton_seq.reshape(T, J * C)
        return torch.tensor(skeleton_flat), label


class SkeletonGraphDataset(Dataset):
    MEDIAPIPE_EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (11, 23), (12, 24),
        (23, 24), (23, 25), (25, 27), (27, 29), (27, 31), (24, 26), (26, 28),
        (28, 30), (28, 32),
    ]

    def __init__(self, root_dir, seq_length=30, num_joints=33):
        self.root_dir = root_dir
        self.seq_length = seq_length
        self.num_joints = num_joints
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.endswith('.npy'):
                    self.samples.append((
                        os.path.join(cls_dir, fname),
                        self.class_to_idx[cls]
                    ))
        self.adjacency = self._build_adjacency()

    def _build_adjacency(self):
        A = np.zeros((self.num_joints, self.num_joints), dtype=np.float32)
        for i, j in self.MEDIAPIPE_EDGES:
            A[i, j] = 1.0
            A[j, i] = 1.0
        A += np.eye(self.num_joints, dtype=np.float32)
        D = np.sum(A, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        return torch.tensor(A_norm)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_path, label = self.samples[idx]
        skeleton_seq = np.load(seq_path).astype(np.float32)
        if len(skeleton_seq) < self.seq_length:
            pad = np.zeros((self.seq_length - len(skeleton_seq),
                          self.num_joints, 3), dtype=np.float32)
            skeleton_seq = np.concatenate([skeleton_seq, pad], axis=0)
        else:
            skeleton_seq = skeleton_seq[:self.seq_length]
        skeleton_seq = skeleton_seq.transpose(2, 0, 1)
        return torch.tensor(skeleton_seq), label


def create_data_loaders(dataset, batch_size=32, val_split=0.15, test_split=0.15):
    total = len(dataset)
    indices = list(range(total))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_split, random_state=42
    )
    val_ratio = val_split / (1 - test_split)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio, random_state=42
    )
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_idx),
        batch_size=batch_size, shuffle=False, num_workers=2
    )
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val / "
          f"{len(test_idx)} test")
    return train_loader, val_loader, test_loader
