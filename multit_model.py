import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from audio_va_extractor import AudioVAExtractor
from video_va_extractor import EmotionVAExtractor as VideoVAExtractor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiModalTransformer(nn.Module):
    def __init__(self, audio_dim=8, video_dim=8, d_model=64, nhead=8, num_layers=4):
        super(MultiModalTransformer, self).__init__()
        
        # Audio and video feature projections
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.video_proj = nn.Linear(video_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(d_model, nhead)
        
        # Output layers
        self.fc = nn.Linear(d_model * 2, 2)  # Output: valence and arousal
        
    def forward(self, audio_features, video_features):
        # Project features to d_model dimension
        audio_emb = self.audio_proj(audio_features)
        video_emb = self.video_proj(video_features)
        
        # Add positional encoding
        audio_emb = self.pos_encoder(audio_emb)
        video_emb = self.pos_encoder(video_emb)
        
        # Self-attention within each modality
        audio_self = self.transformer_encoder(audio_emb)
        video_self = self.transformer_encoder(video_emb)
        
        # Cross-modal attention
        audio_cross, _ = self.cross_attention(audio_self, video_self, video_self)
        video_cross, _ = self.cross_attention(video_self, audio_self, audio_self)
        
        # Concatenate and pool
        combined = torch.cat([audio_cross.mean(dim=1), video_cross.mean(dim=1)], dim=1)
        
        # Final prediction
        output = self.fc(combined)
        return output

class MultiModalDataset(Dataset):
    def __init__(self, pairs_df, audio_dir, video_dir):
        # Store the pairing information DataFrame
        self.pairs = pairs_df
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.audio_extractor = AudioVAExtractor()
        self.video_extractor = VideoVAExtractor()
        
        # Print pairing information for verification
        print("\n音频-视频配对信息:")
        for idx, row in self.pairs.iterrows():
            print(f"配对 {row['id']}:")
            print(f"  角色: {row['role']}")
            print(f"  音频: {row['audio']}")
            print(f"  视频: {row['video']}")
            print("-" * 50)
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        audio_path = os.path.join(self.audio_dir, f"{row['audio']}.wav")  # 添加.wav后缀
        video_path = os.path.join(self.video_dir, f"{row['video']}.avi")  # 添加.avi后缀
        
        # Verify files exist
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Extract features
        audio_features = self.audio_extractor.extract_features(audio_path)
        video_features = self.video_extractor.extract_features(video_path)
        
        # Convert to tensors
        audio_tensor = torch.FloatTensor(audio_features)
        video_tensor = torch.FloatTensor(video_features)
        
        return audio_tensor, video_tensor

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for audio, video in train_loader:
            optimizer.zero_grad()
            outputs = model(audio, video)
            loss = criterion(outputs, torch.zeros_like(outputs))  # Placeholder for actual labels
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for audio, video in val_loader:
                outputs = model(audio, video)
                loss = criterion(outputs, torch.zeros_like(outputs))
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}')

if __name__ == "__main__":
    # Create dataset to verify pairings
    dataset = MultiModalDataset(
        pair_file='ES2016_video_audio_pair.xlsx',
        audio_dir='ES2016a/Audios',
        video_dir='ES2016a/Videos'
    )
    
    # Initialize model
    model = MultiModalTransformer()
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Train model
    train_model(model, train_loader, val_loader) 