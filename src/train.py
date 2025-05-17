import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import random
import time
import argparse
import datetime

# Add src to the Python path if needed
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from src.models.multit_model import MultiModalTransformer
from src.models.dataset import MultiModalDataset
from src.utils.config import ModelConfig

class ExtractedFeaturesDataset(Dataset):
    """Dataset for loading pre-extracted OpenFace and OpenSMILE features from JSON files"""
    def __init__(self, pairs_df, audio_features_dir, video_features_dir, max_seq_len=1000):
        """
        Initialize the dataset.
        
        Args:
            pairs_df (pd.DataFrame): DataFrame with columns ['role', 'video', 'audio']
            audio_features_dir (str): Directory containing audio feature JSON files
            video_features_dir (str): Directory containing video feature JSON files
            max_seq_len (int): Maximum sequence length
        """
        self.pairs_df = pairs_df
        self.audio_features_dir = audio_features_dir
        self.video_features_dir = video_features_dir
        self.max_seq_len = max_seq_len
        
        # Cache for loaded features
        self.audio_features_cache = {}
        self.video_features_cache = {}
        
        # Calculate feature dimensions from first pair
        first_pair = self.pairs_df.iloc[0]
        audio_file = os.path.join(self.audio_features_dir, f"{first_pair['audio']}.json")
        video_file = os.path.join(self.video_features_dir, f"{first_pair['video']}.json")
        
        with open(audio_file, 'r') as f:
            audio_data = json.load(f)
            self.audio_dim = len(audio_data['features'][0])
        
        with open(video_file, 'r') as f:
            video_data = json.load(f)
            # Calculate total video feature dimension
            landmark_dim = len(video_data['features']['landmarks'][0])
            face_desc_dim = len(video_data['features']['face_descriptor'][0])
            eye_dim = len(video_data['features']['eye_aspect_ratios'][0])
            mouth_dim = len(video_data['features']['mouth_aspect_ratio'])  # Single value per frame
            head_pose_dim = len(video_data['features']['head_pose'][0])
            
            # Total video dimension
            self.video_dim = landmark_dim + face_desc_dim + eye_dim + 1 + head_pose_dim  # +1 for mouth_aspect_ratio
            
            # Store dimensions for reference
            self.feature_dims = {
                'audio': self.audio_dim,
                'video_landmarks': landmark_dim,
                'video_face_descriptor': face_desc_dim,
                'video_eye_ratios': eye_dim,
                'video_mouth_ratio': 1,
                'video_head_pose': head_pose_dim,
                'video_total': self.video_dim
            }
    
    def __len__(self):
        return len(self.pairs_df)
    
    def __getitem__(self, idx):
        pair = self.pairs_df.iloc[idx]
        audio_filename = pair['audio']
        video_filename = pair['video']
        
        # Load audio features
        if audio_filename in self.audio_features_cache:
            audio_features = self.audio_features_cache[audio_filename]
        else:
            audio_file = os.path.join(self.audio_features_dir, f"{audio_filename}.json")
            with open(audio_file, 'r') as f:
                audio_data = json.load(f)
                audio_features = np.array(audio_data['features'])
                self.audio_features_cache[audio_filename] = audio_features
        
        # Load video features
        if video_filename in self.video_features_cache:
            video_data = self.video_features_cache[video_filename]
        else:
            video_file = os.path.join(self.video_features_dir, f"{video_filename}.json")
            with open(video_file, 'r') as f:
                video_data = json.load(f)
                self.video_features_cache[video_filename] = video_data
        
        # Extract video feature components
        landmarks = np.array(video_data['features']['landmarks'])
        face_descriptor = np.array(video_data['features']['face_descriptor'])
        eye_ratios = np.array(video_data['features']['eye_aspect_ratios'])
        
        # Convert mouth_aspect_ratio to 2D array if it's 1D
        mouth_ratio = np.array(video_data['features']['mouth_aspect_ratio'])
        if mouth_ratio.ndim == 1:
            mouth_ratio = mouth_ratio.reshape(-1, 1)
        
        head_pose = np.array(video_data['features']['head_pose'])
        
        # Determine minimum length across all features
        min_length = min(
            audio_features.shape[0],
            landmarks.shape[0],
            face_descriptor.shape[0],
            eye_ratios.shape[0],
            mouth_ratio.shape[0],
            head_pose.shape[0]
        )
        
        # Truncate to minimum length
        audio_features = audio_features[:min_length]
        landmarks = landmarks[:min_length]
        face_descriptor = face_descriptor[:min_length]
        eye_ratios = eye_ratios[:min_length]
        mouth_ratio = mouth_ratio[:min_length]
        head_pose = head_pose[:min_length]
        
        # Concatenate video features
        video_features = np.concatenate([
            landmarks,
            face_descriptor,
            eye_ratios,
            mouth_ratio,
            head_pose
        ], axis=1)
        
        # Padding or truncation to max_seq_len
        if audio_features.shape[0] > self.max_seq_len:
            audio_features = audio_features[:self.max_seq_len]
            video_features = video_features[:self.max_seq_len]
        elif audio_features.shape[0] < self.max_seq_len:
            # Padding
            audio_pad = np.zeros((self.max_seq_len - audio_features.shape[0], audio_features.shape[1]))
            video_pad = np.zeros((self.max_seq_len - video_features.shape[0], video_features.shape[1]))
            
            audio_features = np.vstack([audio_features, audio_pad])
            video_features = np.vstack([video_features, video_pad])
        
        # Convert to tensors
        audio_tensor = torch.FloatTensor(audio_features)
        video_tensor = torch.FloatTensor(video_features)
        
        # Generate random labels for now (to be replaced with real annotations later)
        # Using consistent seeds for deterministic behavior
        random.seed(idx)
        valence_label = torch.FloatTensor([random.uniform(0, 1)])
        arousal_label = torch.FloatTensor([random.uniform(0, 1)])
        
        return audio_tensor, video_tensor, valence_label, arousal_label

def create_dataloaders(config):
    """Create training and validation dataloaders from the extracted features"""
    # Create pairing information
    pairs_data = {
        'id': ['1', '2', '3', '4'],
        'role': ['project manager', 'industrial designer', 'user interface designer', 'marketing expert'],
        'video': ['ES2016a.Closeup4', 'ES2016a.Closeup3', 'ES2016a.Closeup2', 'ES2016a.Closeup1'],
        'audio': ['ES2016a.Headset-0', 'ES2016a.Headset-1', 'ES2016a.Headset-2', 'ES2016a.Headset-3']
    }
    pairs_df = pd.DataFrame(pairs_data)
    
    # With only 4 samples, we'll use a simple train-validation split
    # For example, 3 for training, 1 for validation (leave-one-out style)
    train_df = pairs_df.iloc[:3]
    val_df = pairs_df.iloc[3:4]
    
    # Create datasets
    train_dataset = ExtractedFeaturesDataset(
        pairs_df=train_df,
        audio_features_dir=config.audio_features_dir,
        video_features_dir=config.video_features_dir,
        max_seq_len=config.max_seq_len
    )
    
    val_dataset = ExtractedFeaturesDataset(
        pairs_df=val_df,
        audio_features_dir=config.audio_features_dir,
        video_features_dir=config.video_features_dir,
        max_seq_len=config.max_seq_len
    )
    
    # Update config with actual feature dimensions
    config.audio_dim = train_dataset.audio_dim
    config.video_dim = train_dataset.video_dim
    
    print(f"Feature dimensions:")
    for key, value in train_dataset.feature_dims.items():
        print(f"  {key}: {value}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for audio, video, valence_label, arousal_label in tqdm(dataloader, desc="Training"):
        # Move data to device
        audio = audio.to(device)
        video = video.to(device)
        valence_label = valence_label.to(device)
        arousal_label = arousal_label.to(device)
        
        # Forward pass
        valence_pred, arousal_pred = model(audio, video)
        
        # Compute loss
        valence_loss = criterion(valence_pred, valence_label)
        arousal_loss = criterion(arousal_pred, arousal_label)
        loss = valence_loss + arousal_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for audio, video, valence_label, arousal_label in tqdm(dataloader, desc="Validation"):
            # Move data to device
            audio = audio.to(device)
            video = video.to(device)
            valence_label = valence_label.to(device)
            arousal_label = arousal_label.to(device)
            
            # Forward pass
            valence_pred, arousal_pred = model(audio, video)
            
            # Compute loss
            valence_loss = criterion(valence_pred, valence_label)
            arousal_loss = criterion(arousal_pred, arousal_label)
            loss = valence_loss + arousal_loss
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train_model():
    """Main training function"""
    # Load config
    config = ModelConfig()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Create model
    model = MultiModalTransformer(
        audio_dim=config.audio_dim,
        video_dim=config.video_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(device)
    
    # Print model summary
    print(f"MulT Model Architecture:")
    print(f"  Audio dimension: {config.audio_dim}")
    print(f"  Video dimension: {config.video_dim}")
    print(f"  Hidden dimension: {config.hidden_dim}")
    print(f"  Number of heads: {config.num_heads}")
    print(f"  Number of layers: {config.num_layers}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=config.patience // 2
    )
    
    # Create criterion
    criterion = nn.MSELoss()
    
    # Create tensorboard writer
    writer = SummaryWriter(config.log_dir)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print progress
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"New best model saved!")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'audio_dim': config.audio_dim,
                    'video_dim': config.video_dim,
                    'hidden_dim': config.hidden_dim,
                    'num_heads': config.num_heads,
                    'num_layers': config.num_layers,
                    'dropout': config.dropout
                }
            }, config.checkpoint_path)
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    writer.close()
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    return model

if __name__ == "__main__":
    train_model() 