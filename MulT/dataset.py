import os
import torch
import numpy as np
import torchaudio
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from openface_extractor import OpenFaceExtractor
from opensmile_extractor import OpenSMILEExtractor

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
        
        # Initialize feature extractors
        self.video_extractor = OpenFaceExtractor()
        self.audio_extractor = OpenSMILEExtractor(feature_set='eGeMAPSv01a')  # Using eGeMAPS for better emotion recognition
    
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
        video_features = self.video_extractor.process_video(video_path, sample_rate=5)  # Process every 5th frame
        
        # Load audio
        audio_path = os.path.join(self.audio_dir, f"{audio_file}.wav")
        audio_features = self.audio_extractor.extract_temporal_features(
            audio_path,
            frame_size=0.025,  # 25ms frames
            hop_size=0.010     # 10ms hop
        )
        
        # Calculate number of frames/features for duration_minutes
        video_frames_per_min = self.duration_minutes * 60 * 5  # 5 fps (due to sample_rate=5)
        audio_frames_per_min = self.duration_minutes * 60 * 100  # 100 Hz (due to hop_size=0.010)
        
        # Truncate to duration_minutes
        video_features['features']['landmarks'] = video_features['features']['landmarks'][:video_frames_per_min]
        video_features['features']['face_descriptor'] = video_features['features']['face_descriptor'][:video_frames_per_min]
        video_features['features']['eye_aspect_ratios'] = video_features['features']['eye_aspect_ratios'][:video_frames_per_min]
        video_features['features']['mouth_aspect_ratio'] = video_features['features']['mouth_aspect_ratio'][:video_frames_per_min]
        video_features['features']['head_pose'] = video_features['features']['head_pose'][:video_frames_per_min]
        
        audio_features['features'] = audio_features['features'][:audio_frames_per_min]
        
        # Pad or truncate sequences
        video_features['features']['landmarks'] = self.pad_or_truncate(video_features['features']['landmarks'], self.max_seq_len)
        video_features['features']['face_descriptor'] = self.pad_or_truncate(video_features['features']['face_descriptor'], self.max_seq_len)
        video_features['features']['eye_aspect_ratios'] = self.pad_or_truncate(video_features['features']['eye_aspect_ratios'], self.max_seq_len)
        video_features['features']['mouth_aspect_ratio'] = self.pad_or_truncate(video_features['features']['mouth_aspect_ratio'], self.max_seq_len)
        video_features['features']['head_pose'] = self.pad_or_truncate(video_features['features']['head_pose'], self.max_seq_len)
        
        audio_features['features'] = self.pad_or_truncate(audio_features['features'], self.max_seq_len)
        
        # Convert to tensors
        video_tensors = {
            'landmarks': torch.FloatTensor(video_features['features']['landmarks']),
            'face_descriptor': torch.FloatTensor(video_features['features']['face_descriptor']),
            'eye_aspect_ratios': torch.FloatTensor(video_features['features']['eye_aspect_ratios']),
            'mouth_aspect_ratio': torch.FloatTensor(video_features['features']['mouth_aspect_ratio']),
            'head_pose': torch.FloatTensor(video_features['features']['head_pose'])
        }
        
        audio_tensor = torch.FloatTensor(audio_features['features'])
        
        return audio_tensor, video_tensors 