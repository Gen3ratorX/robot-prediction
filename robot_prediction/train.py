"""
Training Pipeline
==================
Trains all three models:
1. GestureCNN      — from your gesture image dataset
2. MovementLSTM    — from skeleton keypoint sequences  
3. SpatioTemporalGCN — from skeleton graph sequences

Usage:
    # Train gesture CNN
    python train.py --model gesture --data_dir data/gestures --epochs 50
    
    # Train movement LSTM
    python train.py --model movement --data_dir data/movement --epochs 80
    
    # Train ST-GCN
    python train.py --model gcn --data_dir data/movement --epochs 80
    
    # Train all models
    python train.py --model all --gesture_dir data/gestures --movement_dir data/movement
"""

import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets import (
    GestureDataset,
    MovementDataset,
    SkeletonGraphDataset,
    create_data_loaders,
)
from movement_features import MOVEMENT_FEATURE_SIZE
from models import GestureCNN, MovementLSTM, SpatioTemporalGCN, count_parameters


# ============================================================
# TRAINING ENGINE
# ============================================================

class Trainer:
    """Generic trainer for all three model types."""
    
    def __init__(self, model, device='auto', save_dir='checkpoints'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        total, trainable = count_parameters(model)
        print(f"Device: {self.device}")
        print(f"Parameters: {total:,} total, {trainable:,} trainable")

    def _apply_same_class_mixup(self, inputs, labels, alpha=0.2):
        """
        Interpolate samples only with others from the same class.
        Because labels are unchanged, this behaves like a class-preserving
        augmentation without requiring soft targets.
        """
        if inputs.size(0) < 2 or alpha <= 0:
            return inputs

        mixed_inputs = inputs.clone()
        unique_labels = labels.unique()

        for label in unique_labels:
            class_indices = torch.nonzero(labels == label, as_tuple=False).flatten()
            if class_indices.numel() < 2:
                continue

            perm = class_indices[torch.randperm(class_indices.numel(), device=labels.device)]
            lam = np.random.beta(alpha, alpha)
            mixed_inputs[class_indices] = (
                lam * inputs[class_indices] + (1.0 - lam) * inputs[perm]
            )

        return mixed_inputs
    
    def train(self, train_loader, val_loader, epochs=50, lr=1e-3,
              weight_decay=1e-4, patience=10, model_name='model',
              label_smoothing=0.1, same_class_mixup=False, mixup_alpha=0.2,
              class_weights=None):
        """
        Full training loop with:
        - Learning rate scheduling
        - Early stopping
        - Best model checkpointing
        - Training history plots
        """
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
        optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}, LR: {lr}, Patience: {patience}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print()
        
        for epoch in range(epochs):
            # --- Train ---
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                if same_class_mixup:
                    inputs = self._apply_same_class_mixup(
                        inputs, labels, alpha=mixup_alpha
                    )
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping (important for LSTM)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_loss /= train_total
            train_acc = 100. * train_correct / train_total
            
            # --- Validate ---
            val_loss, val_acc = self._evaluate(val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
                  f"LR: {current_lr:.2e}")
            
            # Early stopping / checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_epoch = epoch + 1
                
                # Save best model
                save_path = os.path.join(self.save_dir, f'{model_name}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, save_path)
                print(f"  >> Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1} "
                          f"(best was epoch {best_epoch})")
                    break
        
        # Load best model
        best_path = os.path.join(self.save_dir, f'{model_name}_best.pth')
        checkpoint = torch.load(best_path, map_location=self.device,
                               weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Plot training history
        self._plot_history(model_name)
        
        return self.history
    
    def _evaluate(self, loader, criterion):
        """Evaluate model on a data loader."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / total, 100. * correct / total
    
    def test(self, test_loader, class_names=None):
        """Full evaluation with classification report and confusion matrix."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = (all_preds == all_labels).mean() * 100
        
        print(f"\n{'='*60}")
        print(f"TEST RESULTS — Accuracy: {accuracy:.1f}%")
        print(f"{'='*60}")
        
        if class_names:
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds, 
                                       target_names=class_names))
            
            print("Confusion Matrix:")
            cm = confusion_matrix(all_labels, all_preds)
            print(cm)
        
        return accuracy, all_preds, all_labels
    
    def _plot_history(self, model_name):
        """Save training curves plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history['train_loss'], label='Train')
        ax1.plot(self.history['val_loss'], label='Val')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{model_name} — Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.history['train_acc'], label='Train')
        ax2.plot(self.history['val_acc'], label='Val')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title(f'{model_name} — Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, f'{model_name}_training.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Training plot saved to {plot_path}")
    
    def export_onnx(self, dummy_input, model_name='model'):
        """Export model to ONNX for deployment on robot."""
        self.model.eval()
        onnx_path = os.path.join(self.save_dir, f'{model_name}.onnx')
        torch.onnx.export(
            self.model,
            dummy_input.to(self.device),
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )
        print(f"ONNX model exported to {onnx_path}")


# ============================================================
# TRAINING FUNCTIONS FOR EACH MODEL
# ============================================================

def train_gesture_cnn(data_dir, epochs=50, batch_size=32, lr=1e-3, 
                      pretrained=True, save_dir='checkpoints'):
    """Train the gesture recognition CNN."""
    print("\n" + "="*60)
    print("GESTURE CNN TRAINING")
    print("="*60)
    
    # Load dataset
    dataset = GestureDataset(data_dir)
    num_classes = len(dataset.classes)
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset, batch_size=batch_size
    )
    
    # Create model
    model = GestureCNN(num_classes=num_classes, pretrained=pretrained)
    
    # Train
    trainer = Trainer(model, save_dir=save_dir)
    trainer.train(
        train_loader, val_loader,
        epochs=epochs, lr=lr,
        model_name='gesture_cnn'
    )
    
    # Test
    trainer.test(test_loader, class_names=dataset.classes)
    
    # Export for deployment
    dummy = torch.randn(1, 3, 224, 224)
    trainer.export_onnx(dummy, 'gesture_cnn')
    
    # Save class mapping
    with open(os.path.join(save_dir, 'gesture_classes.json'), 'w') as f:
        json.dump(dataset.class_to_idx, f, indent=2)
    
    return trainer


def train_movement_lstm(data_dir, epochs=80, batch_size=32, lr=1e-3,
                        seq_length=30, num_joints=33, save_dir='checkpoints'):
    """Train the movement prediction LSTM."""
    print("\n" + "="*60)
    print("MOVEMENT LSTM TRAINING")
    print("="*60)
    
    # Load dataset
    dataset = MovementDataset(data_dir, seq_length=seq_length)
    num_classes = len(dataset.classes)
    input_size = MOVEMENT_FEATURE_SIZE
    
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset, batch_size=batch_size
    )
    class_weights = torch.ones(num_classes, dtype=torch.float32)
    approaching_idx = dataset.class_to_idx.get('approaching')
    if approaching_idx is not None:
        class_weights[approaching_idx] = 2.0
    
    # Create model
    model = MovementLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes
    )
    
    # Train
    trainer = Trainer(model, save_dir=save_dir)
    trainer.train(
        train_loader, val_loader,
        epochs=epochs, lr=lr,
        model_name='movement_lstm',
        same_class_mixup=True,
        class_weights=class_weights,
    )
    
    # Test
    trainer.test(test_loader, class_names=dataset.classes)
    
    # Export
    dummy = torch.randn(1, seq_length, input_size)
    trainer.export_onnx(dummy, 'movement_lstm')
    
    with open(os.path.join(save_dir, 'movement_classes.json'), 'w') as f:
        json.dump(dataset.class_to_idx, f, indent=2)
    
    return trainer


def train_stgcn(data_dir, epochs=80, batch_size=16, lr=1e-3,
                seq_length=30, num_joints=33, save_dir='checkpoints'):
    """Train the Spatio-Temporal GCN."""
    print("\n" + "="*60)
    print("ST-GCN TRAINING")
    print("="*60)
    
    # Load dataset
    dataset = SkeletonGraphDataset(
        data_dir, seq_length=seq_length, num_joints=num_joints
    )
    num_classes = len(dataset.classes)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset, batch_size=batch_size
    )
    
    # Create model with the skeleton's adjacency matrix
    model = SpatioTemporalGCN(
        in_channels=3,
        num_classes=num_classes,
        num_joints=num_joints,
        adjacency_matrix=dataset.adjacency
    )
    
    # Train
    trainer = Trainer(model, save_dir=save_dir)
    trainer.train(
        train_loader, val_loader,
        epochs=epochs, lr=lr,
        model_name='stgcn'
    )
    
    # Test
    trainer.test(test_loader, class_names=dataset.classes)
    
    # Export
    dummy = torch.randn(1, 3, seq_length, num_joints)
    trainer.export_onnx(dummy, 'stgcn')
    
    with open(os.path.join(save_dir, 'stgcn_classes.json'), 'w') as f:
        json.dump(dataset.class_to_idx, f, indent=2)
    
    return trainer


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train robot perception models')
    parser.add_argument('--model', type=str, default='all',
                       choices=['gesture', 'movement', 'gcn', 'all'],
                       help='Which model to train')
    parser.add_argument('--gesture_dir', type=str, default='data/gestures',
                       help='Path to gesture image dataset')
    parser.add_argument('--movement_dir', type=str, default='data/movement',
                       help='Path to movement skeleton dataset')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--seq_length', type=int, default=30)
    parser.add_argument('--num_joints', type=int, default=33,
                       help='33 for MediaPipe, 25 for OpenPose')
    
    args = parser.parse_args()
    
    if args.model in ('gesture', 'all'):
        if os.path.exists(args.gesture_dir):
            train_gesture_cnn(
                args.gesture_dir, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr,
                save_dir=args.save_dir
            )
        else:
            print(f"Gesture data dir not found: {args.gesture_dir}")
    
    if args.model in ('movement', 'all'):
        if os.path.exists(args.movement_dir):
            train_movement_lstm(
                args.movement_dir, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr,
                seq_length=args.seq_length, num_joints=args.num_joints,
                save_dir=args.save_dir
            )
        else:
            print(f"Movement data dir not found: {args.movement_dir}")
    
    if args.model in ('gcn', 'all'):
        if os.path.exists(args.movement_dir):
            train_stgcn(
                args.movement_dir, epochs=args.epochs,
                batch_size=min(args.batch_size, 16), lr=args.lr,
                seq_length=args.seq_length, num_joints=args.num_joints,
                save_dir=args.save_dir
            )
        else:
            print(f"Movement data dir not found: {args.movement_dir}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Models and exports saved to: {args.save_dir}/")
    print("\nNext steps:")
    print("  1. Copy .pth or .onnx models to your robot")
    print("  2. Use ros_inference_node.py for real-time inference")
    print("  3. Fine-tune with your own recorded data if accuracy is low")
