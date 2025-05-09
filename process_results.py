import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from multit_model import MultiModalTransformer, MultiModalDataset
from torch.utils.data import DataLoader

def load_model(checkpoint_path):
    """Load the trained model from checkpoint"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = MultiModalTransformer(
        audio_dim=40,
        video_dim=3*224*224,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        max_seq_len=1000
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device

def process_temporal_results():
    """Process and display temporal results for all pairs"""
    # Create pairing information
    pairs_data = {
        'id': ['1', '2', '3', '4'],
        'role': ['project manager', 'industrial designer', 'user interface designer', 'marketing expert'],
        'video': ['ES2016a.Closeup4', 'ES2016a.Closeup3', 'ES2016a.Closeup2', 'ES2016a.Closeup1'],
        'audio': ['ES2016a.Headset-0', 'ES2016a.Headset-1', 'ES2016a.Headset-2', 'ES2016a.Headset-3']
    }
    pairs = pd.DataFrame(pairs_data)
    
    # Load model
    model, device = load_model('checkpoints/model_epoch_1.pt')
    
    # Create dataset
    base_dir = "/Users/yuwang/Documents/GitHub/AMI_Corpus_Updated/AMI_Corpus"
    dataset = MultiModalDataset(
        pairs_df=pairs,
        audio_dir=os.path.join(base_dir, 'ES2016a/Audios'),
        video_dir=os.path.join(base_dir, 'ES2016a/Videos')
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Process each pair
    results = []
    temporal_data = {role: {'valence': [], 'arousal': [], 'timestamps': []} 
                    for role in pairs['role']}
    
    print("\nProcessing temporal results for each participant...")
    with torch.no_grad():
        for batch_idx, (audio, video) in enumerate(tqdm(dataloader)):
            # Get current pair info
            current_pair = pairs.iloc[batch_idx]
            role = current_pair['role']
            
            # Move data to device
            audio = audio.to(device)
            video = video.to(device)
            
            # Get predictions
            valence, arousal = model(audio, video)
            
            # Store results
            temporal_data[role]['valence'] = valence.cpu().numpy().flatten()
            temporal_data[role]['arousal'] = arousal.cpu().numpy().flatten()
            temporal_data[role]['timestamps'] = np.arange(len(valence)) * 0.04  # Assuming 25 fps
            
            # Store summary
            results.append({
                'role': role,
                'mean_valence': float(np.mean(valence.cpu().numpy())),
                'mean_arousal': float(np.mean(arousal.cpu().numpy())),
                'std_valence': float(np.std(valence.cpu().numpy())),
                'std_arousal': float(np.std(arousal.cpu().numpy()))
            })
    
    # Plot temporal results
    plot_temporal_results(temporal_data)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    for result in results:
        print(f"\nRole: {result['role']}")
        print(f"Mean Valence: {result['mean_valence']:.4f} (±{result['std_valence']:.4f})")
        print(f"Mean Arousal: {result['mean_arousal']:.4f} (±{result['std_arousal']:.4f})")
    
    return temporal_data, results

def plot_temporal_results(temporal_data):
    """Plot temporal evolution of valence and arousal for all participants"""
    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    colors = ['b', 'g', 'r', 'purple']
    
    # Plot Valence
    ax1.set_title('Temporal Evolution of Valence', fontsize=14)
    for (role, data), color in zip(temporal_data.items(), colors):
        ax1.plot(data['timestamps'], data['valence'], color=color, label=role, alpha=0.7)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Valence')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Arousal
    ax2.set_title('Temporal Evolution of Arousal', fontsize=14)
    for (role, data), color in zip(temporal_data.items(), colors):
        ax2.plot(data['timestamps'], data['arousal'], color=color, label=role, alpha=0.7)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Arousal')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('temporal_results.pdf')
    plt.close()

if __name__ == "__main__":
    temporal_data, results = process_temporal_results() 