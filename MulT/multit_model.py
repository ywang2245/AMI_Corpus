import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import torchvision
import os
import numpy as np
import cv2
from typing import Tuple
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear transformations
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        return self.out_linear(out)

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class MultiModalTransformer(nn.Module):
    def __init__(self, audio_dim, video_dim, hidden_dim, num_heads, num_layers, dropout=0.1, max_seq_len=1000):
        super().__init__()
        
        # Input projections
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.audio_norm = nn.LayerNorm(hidden_dim)
        self.video_norm = nn.LayerNorm(hidden_dim)
        
        # Positional encoding
        self.audio_pos = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))
        self.video_pos = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(hidden_dim * 2, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output heads
        self.valence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Normalize to [0, 1]
        )
        
        self.arousal_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Normalize to [0, 1]
        )
        
    def forward(self, audio, video):
        batch_size = audio.size(0)
        
        # Project inputs
        audio = self.audio_proj(audio)  # [batch, seq_len, hidden_dim]
        video = self.video_proj(video)  # [batch, seq_len, hidden_dim]
        
        # Apply layer normalization
        audio = self.audio_norm(audio)
        video = self.video_norm(video)
        
        # Add positional encodings
        audio = audio + self.audio_pos[:, :audio.size(1)]
        video = video + self.video_pos[:, :video.size(1)]
        
        # Concatenate modalities along feature dimension
        x = torch.cat([audio, video], dim=-1)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
            
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Predict valence and arousal
        valence = self.valence_head(x)
        arousal = self.arousal_head(x)
        
        return valence, arousal

class MultiModalDataset(Dataset):
    def __init__(
        self,
        pairs_df,
        audio_dir: str,
        video_dir: str,
        audio_duration: float = 3.0,
        audio_sample_rate: int = 16000,
        video_fps: int = 25
    ):
        self.pairs = pairs_df
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.audio_duration = audio_duration
        self.audio_sample_rate = audio_sample_rate
        self.video_fps = video_fps
        
        # Audio processing setup
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=40
        )
        
        print("\nAudio-Video Pairing Information:")
        for idx, row in self.pairs.iterrows():
            print(f"Pair {row['id']}:")
            print(f"  Role: {row['role']}")
            print(f"  Audio: {row['audio']}")
            print(f"  Video: {row['video']}")
            print("-" * 50)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        audio_path = os.path.join(self.audio_dir, f"{row['audio']}.wav")
        video_path = os.path.join(self.video_dir, f"{row['video']}.avi")
        
        # Verify files exist
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Load and process audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.audio_sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.audio_sample_rate)(waveform)
        
        # Ensure correct audio length
        target_length = int(self.audio_duration * self.audio_sample_rate)
        if waveform.size(1) > target_length:
            waveform = waveform[:, :target_length]
        else:
            waveform = F.pad(waveform, (0, target_length - waveform.size(1)))
        
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Load and process video
        cap = cv2.VideoCapture(video_path)
        frames = []
        target_frames = int(self.audio_duration * self.video_fps)
        
        while len(frames) < target_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frame = torch.from_numpy(frame).float() / 255.0
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
        
        cap.release()
        
        # Ensure correct number of video frames
        if len(frames) < target_frames:
            last_frame = frames[-1] if frames else torch.zeros((3, 224, 224))
            while len(frames) < target_frames:
                frames.append(last_frame)
        elif len(frames) > target_frames:
            frames = frames[:target_frames]
        
        video_tensor = torch.stack(frames)
        
        return mel_spec.squeeze(0), video_tensor 