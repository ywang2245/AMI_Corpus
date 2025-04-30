import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from config import ModelConfig
from dataset import MultiModalDataset
from multit_model import MultiModalTransformer
import pandas as pd

def create_dataloaders(config):
    """Create training and validation dataloaders"""
    # Create pairing information
    pairs_data = {
        'id': ['1', '2', '3', '4'],
        'role': ['project manager', 'industrial designer', 'user interface designer', 'marketing expert'],
        'video': ['ES2016a.Closeup4', 'ES2016a.Closeup3', 'ES2016a.Closeup2', 'ES2016a.Closeup1'],
        'audio': ['ES2016a.Headset-0', 'ES2016a.Headset-1', 'ES2016a.Headset-2', 'ES2016a.Headset-3']
    }
    pairs = pd.DataFrame(pairs_data)
    
    # Create dataset
    dataset = MultiModalDataset(
        pairs_df=pairs,
        audio_dir=config.audio_dir,
        video_dir=config.video_dir,
        max_seq_len=config.max_seq_len
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Reduce batch size
        shuffle=True,
        num_workers=0,  # Disable multiprocessing
        pin_memory=False
    )
    
    return dataloader

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for audio, video in tqdm(dataloader, desc="Training"):
        # Move data to device
        audio = audio.to(device)
        video = video.to(device)
        
        # Forward pass
        valence_pred, arousal_pred = model(audio, video)
        
        # Compute loss (using dummy labels for now)
        valence_loss = criterion(valence_pred, torch.zeros_like(valence_pred))
        arousal_loss = criterion(arousal_pred, torch.zeros_like(arousal_pred))
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
        for audio, video in tqdm(dataloader, desc="Validation"):
            # Move data to device
            audio = audio.to(device)
            video = video.to(device)
            
            # Forward pass
            valence_pred, arousal_pred = model(audio, video)
            
            # Compute loss (using dummy labels for now)
            valence_loss = criterion(valence_pred, torch.zeros_like(valence_pred))
            arousal_loss = criterion(arousal_pred, torch.zeros_like(arousal_pred))
            loss = valence_loss + arousal_loss
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train_model():
    """Main training function"""
    # Load config
    config = ModelConfig()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = MultiModalTransformer(
        audio_dim=config.audio_dim,
        video_dim=config.video_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len
    ).to(device)
    
    # Create dataloaders
    train_loader = create_dataloaders(config)
    
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
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate(model, train_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Print progress
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, config.checkpoint_path)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    writer.close()
    return model

if __name__ == "__main__":
    train_model() 