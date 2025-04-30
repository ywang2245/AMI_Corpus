import os
import torch
import numpy as np
import torchaudio
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from video_reader import VideoReader
from audio_feature_extractor import AudioFeatureExtractor

class MultiModalDataset(Dataset):
    def __init__(self, pairs_df, audio_dir, video_dir, max_seq_len=1000, duration_minutes=5):
        """
        Initialize the dataset.
        
        Args:
            pairs_df (pd.DataFrame): DataFrame containing pairing information
            audio_dir (str): Directory containing audio files
            video_dir (str): Directory containing video files
            max_seq_len (int): Maximum sequence length for both audio and video
            duration_minutes (int): Duration to process in minutes
        """
        self.pairs_df = pairs_df
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.max_seq_len = max_seq_len
        self.duration_minutes = duration_minutes
        
        # Initialize video reader
        self.video_reader = VideoReader()
        
        # Initialize audio feature extractor
        self.audio_feature_extractor = AudioFeatureExtractor()
    
    def pad_or_truncate(self, sequence, max_len):
        """Pad or truncate a sequence to max_len"""
        if len(sequence) > max_len:
            return sequence[:max_len]
        elif len(sequence) < max_len:
            pad_len = max_len - len(sequence)
            return np.pad(sequence, ((0, pad_len), (0, 0)), mode='constant')
        return sequence
    
    def __len__(self):
        return len(self.pairs_df)
    
    def __getitem__(self, idx):
        # Get pair information
        pair = self.pairs_df.iloc[idx]
        video_file = pair['video']
        audio_file = pair['audio']
        
        # Load video
        video_path = os.path.join(self.video_dir, f"{video_file}.avi")
        video_frames = self.video_reader.read_video(video_path)
        
        # Load audio
        audio_path = os.path.join(self.audio_dir, f"{audio_file}.wav")
        audio_features = self.audio_feature_extractor.extract_features(audio_path)
        
        # Calculate number of frames/features for 5 minutes
        video_frames_per_5min = self.duration_minutes * 60 * 25  # 25 fps
        audio_features_per_5min = self.duration_minutes * 60 * 100  # 100 Hz
        
        # Truncate to 5 minutes
        video_frames = video_frames[:video_frames_per_5min]
        audio_features = audio_features[:audio_features_per_5min]
        
        # Pad or truncate sequences
        video_frames = self.pad_or_truncate(video_frames, self.max_seq_len)
        audio_features = self.pad_or_truncate(audio_features, self.max_seq_len)
        
        # Convert to tensors
        video_tensor = torch.FloatTensor(video_frames)
        audio_tensor = torch.FloatTensor(audio_features)
        
        return audio_tensor, video_tensor 