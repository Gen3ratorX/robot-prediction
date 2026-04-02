"""
Model Architectures
====================
1. GestureCNN      — CNN for hand gesture classification
2. MovementLSTM    — LSTM for human movement/trajectory prediction
3. SpatioTemporalGCN — Graph Convolutional Network for skeleton-based action recognition

All models output class logits. Use with CrossEntropyLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from movement_features import MOVEMENT_FEATURE_SIZE


# ============================================================
# MODEL 1: CNN FOR GESTURE RECOGNITION
# ============================================================

class GestureCNN(nn.Module):
    """
    CNN for classifying hand gestures from single RGB images.
    
    Architecture: Lightweight custom CNN (or swap in a pretrained backbone).
    Input:  (B, 3, 224, 224) — RGB image
    Output: (B, num_classes)  — class logits
    
    For better accuracy with small datasets, use pretrained=True 
    to leverage transfer learning from ImageNet.
    """
    
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        self.pretrained = pretrained
        
        if pretrained:
            # Transfer learning with MobileNetV2 (lightweight, good for edge)
            from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
            self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            # Freeze early layers
            for param in list(self.backbone.parameters())[:-20]:
                param.requires_grad = False
            # Replace classifier
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(1280, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
        else:
            # Custom lightweight CNN
            self.features = nn.Sequential(
                # Block 1: 224x224 -> 112x112
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Block 2: 112x112 -> 56x56
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Block 3: 56x56 -> 28x28
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Block 4: 28x28 -> 14x14
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Block 5: 14x14 -> 7x7
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, x):
        if self.pretrained:
            return self.backbone(x)
        else:
            x = self.features(x)
            return self.classifier(x)


# ============================================================
# MODEL 2: LSTM FOR MOVEMENT PREDICTION
# ============================================================

class MovementLSTM(nn.Module):
    """
    Bi-directional LSTM for predicting human movement direction
    from skeleton keypoint sequences.
    
    Input:  (B, T, F)        — T frames of flattened movement features
    Output: (B, num_classes)  — movement direction logits
    
    T = sequence length (e.g., 30 frames = 1 second at 30fps)
    F = normalized joint positions + selected joint velocities
    """
    
    def __init__(self, input_size=MOVEMENT_FEATURE_SIZE, hidden_size=128, num_layers=2,
                 num_classes=5, dropout=0.3):
        """
        Args:
            input_size: movement feature size per frame
            hidden_size: LSTM hidden dimension
            num_layers: number of stacked LSTM layers
            num_classes: movement directions
            dropout: dropout rate between LSTM layers
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bi-directional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism to weight important frames
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, input_size) — skeleton sequences
        Returns:
            logits: (B, num_classes)
        """
        B, T, _ = x.shape
        
        # Project input
        x = self.input_proj(x)  # (B, T, hidden_size)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, T, hidden_size*2)
        
        # Attention
        attn_weights = self.attention(lstm_out)  # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (B, hidden_size*2)
        
        # Classify
        return self.classifier(context)


# ============================================================
# MODEL 3: SPATIO-TEMPORAL GRAPH CONVOLUTIONAL NETWORK (ST-GCN)
# ============================================================

class GraphConvolution(nn.Module):
    """Single graph convolution layer."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x, A):
        """
        Args:
            x: (B, C, T, V) — features per joint per frame
            A: (V, V)       — adjacency matrix
        """
        # Graph convolution: aggregate neighbor features
        # x @ A^T via einsum
        x = torch.einsum('bctv,vw->bctw', x, A)
        x = self.conv(x)
        x = self.bn(x)
        return x


class STGCNBlock(nn.Module):
    """
    One Spatio-Temporal Graph Convolution block.
    
    Combines:
    1. Spatial graph convolution (aggregate neighbor joint info)
    2. Temporal convolution (capture motion over time)
    """
    
    def __init__(self, in_channels, out_channels, stride=1, residual=True):
        super().__init__()
        
        # Spatial graph convolution
        self.gcn = GraphConvolution(in_channels, out_channels)
        
        # Temporal convolution (1D conv along time axis)
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 
                     kernel_size=(9, 1),     # 9-frame temporal kernel
                     padding=(4, 0),         # same padding
                     stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
        )
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        
        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.relu(x)
        x = self.tcn(x)
        x = self.dropout(x)
        x = x + res
        return self.relu(x)


class SpatioTemporalGCN(nn.Module):
    """
    ST-GCN for skeleton-based action/movement recognition.
    
    Input:  (B, 3, T, V)     — skeleton sequences
            3 = xyz coords, T = frames, V = joints
    Output: (B, num_classes)  — action class logits
    
    Architecture follows the original ST-GCN paper 
    (Yan et al., AAAI 2018) with modifications for our use case.
    """
    
    def __init__(self, in_channels=3, num_classes=5, num_joints=33,
                 adjacency_matrix=None):
        super().__init__()
        
        self.num_joints = num_joints
        
        # Register adjacency matrix as buffer (not a parameter)
        if adjacency_matrix is not None:
            self.register_buffer('A', adjacency_matrix)
        else:
            # Default: fully connected with self-loops
            A = torch.ones(num_joints, num_joints) / num_joints
            self.register_buffer('A', A)
        
        # Initial batch norm on input
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        
        # ST-GCN blocks with increasing channels
        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, 64, residual=False),   # 3  -> 64
            STGCNBlock(64, 64),                             # 64 -> 64
            STGCNBlock(64, 64),                             # 64 -> 64
            STGCNBlock(64, 128, stride=2),                  # 64 -> 128, T/2
            STGCNBlock(128, 128),                           # 128 -> 128
            STGCNBlock(128, 128),                           # 128 -> 128
            STGCNBlock(128, 256, stride=2),                 # 128 -> 256, T/4
            STGCNBlock(256, 256),                           # 256 -> 256
            STGCNBlock(256, 256),                           # 256 -> 256
        ])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T, V) where C=3, T=seq_length, V=num_joints
        Returns:
            logits: (B, num_classes)
        """
        B, C, T, V = x.shape
        
        # Batch norm on input
        x_bn = x.permute(0, 3, 1, 2).contiguous().view(B, V * C, T)
        x_bn = self.data_bn(x_bn)
        x = x_bn.view(B, V, C, T).permute(0, 2, 3, 1)  # (B, C, T, V)
        
        # ST-GCN blocks
        for layer in self.layers:
            x = layer(x, self.A)
        
        # Global average pooling over time and joints
        x = x.mean(dim=[2, 3])  # (B, 256)
        
        return self.classifier(x)


# ============================================================
# MODEL SUMMARY HELPER
# ============================================================

def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    print("="*60)
    print("Model Architecture Tests")
    print("="*60)
    
    # Test GestureCNN
    print("\n--- GestureCNN (pretrained=False) ---")
    cnn = GestureCNN(num_classes=3, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = cnn(x)
    total, trainable = count_parameters(cnn)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {total:,} total, {trainable:,} trainable")
    
    # Test MovementLSTM
    print("\n--- MovementLSTM ---")
    lstm = MovementLSTM(input_size=MOVEMENT_FEATURE_SIZE, num_classes=5)
    x = torch.randn(2, 30, MOVEMENT_FEATURE_SIZE)
    out = lstm(x)
    total, trainable = count_parameters(lstm)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {total:,} total, {trainable:,} trainable")
    
    # Test ST-GCN
    print("\n--- SpatioTemporalGCN ---")
    gcn = SpatioTemporalGCN(num_classes=5, num_joints=33)
    x = torch.randn(2, 3, 30, 33)  # 3 coords, 30 frames, 33 joints
    out = gcn(x)
    total, trainable = count_parameters(gcn)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {total:,} total, {trainable:,} trainable")
    
    print("\nAll models built successfully!")
