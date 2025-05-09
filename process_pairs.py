import pandas as pd
import os
import torch
from multit_model import MultiModalTransformer, MultiModalDataset
from torch.utils.data import DataLoader
import numpy as np

def verify_files(pairs, audio_dir, video_dir):
    """Verify if audio and video files exist"""
    valid_pairs = []
    
    for idx, row in pairs.iterrows():
        audio_path = os.path.join(audio_dir, f"{row['audio']}.wav")  # Add .wav extension
        video_path = os.path.join(video_dir, f"{row['video']}.avi")  # Add .avi extension
        
        is_valid = True
        
        # Check if files exist
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            is_valid = False
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            is_valid = False
            
        if is_valid:
            valid_pairs.append(row)
    
    return pd.DataFrame(valid_pairs)

def main():
    # 1. Create pairing information
    print("Creating pairing information...")
    pairs_data = {
        'id': ['1', '2', '3', '4'],
        'role': ['project manager', 'industrial designer', 'user interface designer', 'marketing expert'],
        'video': ['ES2016a.Closeup4', 'ES2016a.Closeup3', 'ES2016a.Closeup2', 'ES2016a.Closeup1'],
        'audio': ['ES2016a.Headset-0', 'ES2016a.Headset-1', 'ES2016a.Headset-2', 'ES2016a.Headset-3']
    }
    pairs = pd.DataFrame(pairs_data)
    
    # Verify file paths
    print("\nVerifying file paths...")
    pairs = verify_files(pairs, 'ES2016a/Audios', 'ES2016a/Videos')
    
    # Print pairing information
    print("\nAudio-Video Pairing Information:")
    for idx, row in pairs.iterrows():
        print(f"Pair {row['id']}:")
        print(f"  Role: {row['role']}")
        print(f"  Audio: {row['audio']}")
        print(f"  Video: {row['video']}")
        print("-" * 50)
    
    print(f"\nTotal number of pairs: {len(pairs)}")
    
    # Save valid pairs if any found
    if len(pairs) > 0:
        pairs.to_csv('valid_pairs.csv', index=False)
    else:
        print("Error: No valid pairs found!")

if __name__ == "__main__":
    main() 