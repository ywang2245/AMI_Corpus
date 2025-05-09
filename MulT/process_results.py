import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from multit_model import MultiModalTransformer
from dataset import MultiModalDataset
from config import ModelConfig
from torch.utils.data import DataLoader

def load_model(checkpoint_path, config):
    """Load the trained model from checkpoint"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model with config
    model = MultiModalTransformer(
        audio_dim=config.audio_dim,
        video_dim=config.video_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len
    ).to(device)
    
    # Load checkpoint or initialize weights
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}, using initialized weights")
        # Initialize model weights
        model.train()  # Ensure BatchNorm is initialized
        dummy_audio = torch.randn(1, 100, config.audio_dim).to(device)
        dummy_video = torch.randn(1, 100, config.video_dim).to(device)
        _ = model(dummy_audio, dummy_video)
    
    model.eval()
    return model, device

def find_high_arousal_positive_emotions(temporal_data, valence_threshold=0.6, arousal_threshold=0.6):
    """Find the first high-arousal positive emotion for each participant"""
    results = {}
    
    for role, data in temporal_data.items():
        # Find points where both valence and arousal are above thresholds
        high_arousal_positive = np.where(
            (data['valence'] > valence_threshold) & 
            (data['arousal'] > arousal_threshold)
        )[0]
        
        if len(high_arousal_positive) > 0:
            first_idx = high_arousal_positive[0]
            results[role] = {
                'timestamp': data['timestamps'][first_idx],
                'valence': data['valence'][first_idx],
                'arousal': data['arousal'][first_idx]
            }
        else:
            results[role] = None
    
    return results

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
    
    # Load config
    config = ModelConfig()
    
    # Load model
    model, device = load_model(config.checkpoint_path, config)
    
    # Create dataset with full duration
    base_dir = "/Users/yuwang/Documents/GitHub/AMI_Corpus_Updated/AMI_Corpus"
    dataset = MultiModalDataset(
        pairs_df=pairs,
        audio_dir=os.path.join(base_dir, 'ES2016a/Audios'),
        video_dir=os.path.join(base_dir, 'ES2016a/Videos'),
        max_seq_len=config.max_seq_len,
        duration_minutes=5  # Only process first 5 minutes
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
            
            # Print debug information
            print(f"\nProcessing {role}:")
            print(f"Audio shape: {audio.shape}")
            print(f"Video shape: {video.shape}")
            
            # Process sequence in chunks
            chunk_size = 100  # Process 100 frames at a time
            num_chunks = audio.shape[1] // chunk_size
            valence_list = []
            arousal_list = []
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size
                
                # Get chunk of data
                audio_chunk = audio[:, start_idx:end_idx, :]
                video_chunk = video[:, start_idx:end_idx, :]
                
                # Move data to device
                audio_chunk = audio_chunk.to(device)
                video_chunk = video_chunk.to(device)
                
                # Get predictions
                output = model(audio_chunk, video_chunk)
                valence = output[0].cpu().numpy().item()
                arousal = output[1].cpu().numpy().item()
                
                valence_list.append(valence)
                arousal_list.append(arousal)
            
            # Convert lists to arrays
            valence_array = np.array(valence_list)
            arousal_array = np.array(arousal_list)
            
            # Calculate timestamps for 5 minutes
            total_duration = 5 * 60  # 5 minutes in seconds
            timestamps = np.linspace(0, total_duration, len(valence_list))
            
            print(f"Number of chunks: {len(valence_list)}")
            print(f"Duration: {timestamps[-1]:.2f} seconds")
            
            # Store results
            temporal_data[role]['valence'] = valence_array
            temporal_data[role]['arousal'] = arousal_array
            temporal_data[role]['timestamps'] = timestamps
            
            # Store summary
            results.append({
                'role': role,
                'mean_valence': float(np.mean(valence_array)),
                'mean_arousal': float(np.mean(arousal_array)),
                'std_valence': float(np.std(valence_array)),
                'std_arousal': float(np.std(arousal_array)),
                'duration': float(timestamps[-1])  # Total duration in seconds
            })
    
    # Find high-arousal positive emotions
    high_arousal_positive = find_high_arousal_positive_emotions(temporal_data)
    
    # Print results
    print("\nFirst High-Arousal Positive Emotions:")
    print("-" * 50)
    for role, result in high_arousal_positive.items():
        if result is not None:
            print(f"\nRole: {role}")
            print(f"Time: {result['timestamp']:.2f} seconds")
            print(f"Valence: {result['valence']:.4f}")
            print(f"Arousal: {result['arousal']:.4f}")
        else:
            print(f"\nRole: {role}")
            print("No high-arousal positive emotions detected")
    
    # Plot temporal results
    plot_temporal_results(temporal_data)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    for result in results:
        print(f"\nRole: {result['role']}")
        print(f"Duration: {result['duration']:.2f} seconds")
        print(f"Mean Valence: {result['mean_valence']:.4f} (±{result['std_valence']:.4f})")
        print(f"Mean Arousal: {result['mean_arousal']:.4f} (±{result['std_arousal']:.4f})")
    
    return temporal_data, results, high_arousal_positive

def plot_temporal_results(temporal_data):
    """Plot temporal evolution of valence and arousal for each participant"""
    plt.rcParams['figure.figsize'] = [15, 10]
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 12
    
    # Create individual plots for each participant
    for role, data in temporal_data.items():
        plt.figure(figsize=(15, 8))
        
        # Ensure data is in numpy array format
        timestamps = np.array(data['timestamps'])
        valence = np.array(data['valence'])
        arousal = np.array(data['arousal'])
        
        # Plot both valence and arousal
        plt.plot(timestamps, valence, 
                color='blue', label='Valence', linewidth=2, alpha=0.8)
        plt.plot(timestamps, arousal, 
                color='red', label='Arousal', linewidth=2, alpha=0.8)
        
        # Add horizontal lines at 0.6 for reference
        plt.axhline(y=0.6, color='gray', linestyle='--', alpha=0.3)
        
        # Customize the plot
        plt.title(f'{role} - Emotion Evolution', fontsize=16, pad=20)
        plt.xlabel('Time (seconds)', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add text annotations for mean values
        mean_valence = np.mean(valence)
        mean_arousal = np.mean(arousal)
        plt.text(0.02, 0.95, f'Mean Valence: {mean_valence:.4f}', 
                transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.02, 0.90, f'Mean Arousal: {mean_arousal:.4f}', 
                transform=plt.gca().transAxes, fontsize=12)
        
        # Set x-axis ticks to show minutes and seconds
        max_time = timestamps[-1]
        if max_time > 60:
            # Create ticks every 30 seconds
            ticks = np.arange(0, max_time + 30, 30)
            tick_labels = [f'{int(t//60)}:{int(t%60):02d}' for t in ticks]
            plt.xticks(ticks, tick_labels)
        else:
            plt.xticks(np.arange(0, max_time + 10, 10))
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f'emotion_evolution_{role.replace(" ", "_")}.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    temporal_data, results, high_arousal_positive = process_temporal_results() 