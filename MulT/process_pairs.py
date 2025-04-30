import pandas as pd
import os
import torch
from multit_model import MultiModalTransformer, MultiModalDataset
from torch.utils.data import DataLoader
import numpy as np

def verify_file_paths(pairs_df, audio_dir, video_dir):
    """Verify if audio and video files exist"""
    valid_pairs = []
    for idx, row in pairs_df.iterrows():
        audio_path = os.path.join(audio_dir, f"{row['audio']}.wav")
        video_path = os.path.join(video_dir, f"{row['video']}.avi")
        
        if os.path.exists(audio_path) and os.path.exists(video_path):
            valid_pairs.append(True)
        else:
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found: {audio_path}")
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found: {video_path}")
            valid_pairs.append(False)
    
    return pairs_df[valid_pairs]

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
    pairs = verify_file_paths(pairs, 'ES2016a/Audios', 'ES2016a/Videos')
    
    # Print pairing information
    print("\nAudio-Video Pairing Information:")
    print("=" * 50)
    for idx, row in pairs.iterrows():
        print(f"Pair {row['id']}:")
        print(f"  Role: {row['role']}")
        print(f"  Audio: {row['audio']}")
        print(f"  Video: {row['video']}")
        print("-" * 50)
    
    print(f"\nTotal pairs: {len(pairs)}")
    print("=" * 50)
    
    if len(pairs) == 0:
        print("Error: No valid pairs found!")
        return
    
    # 2. Create dataset
    print("\nCreating multimodal dataset...")
    dataset = MultiModalDataset(
        pairs_df=pairs,
        audio_dir='ES2016a/Audios',
        video_dir='ES2016a/Videos'
    )
    
    # 3. Initialize model
    print("\nInitializing MulT model...")
    model = MultiModalTransformer(
        audio_dim=8,  # Audio feature dimension
        video_dim=8,  # Video feature dimension
        d_model=64,   # Transformer internal dimension
        nhead=8,      # Number of attention heads
        num_layers=4  # Number of transformer layers
    )
    
    # 4. Create data loader
    print("\nCreating data loader...")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 5. Perform multimodal analysis
    print("\nStarting multimodal analysis...")
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch_idx, (audio_features, video_features) in enumerate(dataloader):
            # Forward pass
            outputs = model(audio_features, video_features)
            
            # Get prediction results
            valence, arousal = outputs[0].numpy()
            
            # Store results
            results.append({
                'pair_id': pairs.iloc[batch_idx]['id'],
                'role': pairs.iloc[batch_idx]['role'],
                'audio_file': pairs.iloc[batch_idx]['audio'],
                'video_file': pairs.iloc[batch_idx]['video'],
                'valence': valence,
                'arousal': arousal
            })
            
            # Print current pair analysis results
            print(f"\nPair {pairs.iloc[batch_idx]['id']} Analysis Results:")
            print(f"Role: {pairs.iloc[batch_idx]['role']}")
            print(f"Audio: {pairs.iloc[batch_idx]['audio']}")
            print(f"Video: {pairs.iloc[batch_idx]['video']}")
            print(f"Valence: {valence:.4f}")
            print(f"Arousal: {arousal:.4f}")
            print("-" * 50)
    
    # 6. Save analysis results
    results_df = pd.DataFrame(results)
    output_path = os.path.join('MulT', 'multimodal_analysis_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nAnalysis results saved to '{output_path}'")

if __name__ == "__main__":
    main() 