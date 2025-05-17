import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to the Python path if needed
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# Import MulT model
from src.models.multit_model import MultiModalTransformer

# Set paths
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Go up one level from src
EXCEL_FILE = os.path.join(BASE_DIR, 'ES2016_video_audio_pair.xlsx')
FEATURES_DIR = os.path.join(BASE_DIR, 'extracted_features')
VIDEO_FEATURES_DIR = os.path.join(FEATURES_DIR, 'video_features')
AUDIO_FEATURES_DIR = os.path.join(FEATURES_DIR, 'audio_features')
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'MulT', 'MulT', 'checkpoints', 'mult_model.pt')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis_results')

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_pairing_info():
    """Load video and audio pairing information from Excel file"""
    df = pd.read_excel(EXCEL_FILE)
    # Only keep ES2016a rows
    df = df.iloc[1:5].reset_index(drop=True)
    # Rename columns
    df.columns = ['id', 'role', 'video', 'audio']
    return df

def load_features(video_file, audio_file):
    """Load video and audio features"""
    # Load video features
    video_feature_path = os.path.join(VIDEO_FEATURES_DIR, f"{video_file}.json")
    with open(video_feature_path, 'r') as f:
        video_data = json.load(f)
    
    # Load audio features
    audio_feature_path = os.path.join(AUDIO_FEATURES_DIR, f"{audio_file}.json")
    with open(audio_feature_path, 'r') as f:
        audio_data = json.load(f)
    
    # Extract features
    landmarks = np.array(video_data['features']['landmarks'])
    face_descriptor = np.array(video_data['features']['face_descriptor'])
    eye_ratios = np.array(video_data['features']['eye_aspect_ratios'])
    
    # Process mouth aspect ratio, ensure it's a 2D array
    mouth_ratio = np.array(video_data['features']['mouth_aspect_ratio'])
    if mouth_ratio.ndim == 1:
        mouth_ratio = mouth_ratio.reshape(-1, 1)
    
    head_pose = np.array(video_data['features']['head_pose'])
    
    # Audio features
    audio_features = np.array(audio_data['features'])
    
    # Determine minimum length
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
    
    # Combine video features
    video_features = np.concatenate([
        landmarks,
        face_descriptor,
        eye_ratios,
        mouth_ratio,
        head_pose
    ], axis=1)
    
    # Return features and timestamps
    return {
        'audio_features': audio_features,
        'video_features': video_features,
        'timestamps': audio_data['timestamps'][:min_length],
        'duration': audio_data['duration'],
        'fps': video_data.get('fps', 25)  # Default value is 25fps
    }

def load_model():
    """Load pre-trained MulT model"""
    # Check if pre-trained model exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Pre-trained model not found at {CHECKPOINT_PATH}")
        print("Please run MulT/train.py to train the model first")
        return None
    
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
    
    # Create model
    model_config = checkpoint['config']
    print("Model configuration:", model_config)
    
    try:
        # Create model
        model = MultiModalTransformer(
            audio_dim=model_config['audio_dim'],
            video_dim=model_config['video_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_heads=model_config['num_heads'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout']
        )
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def analyze_features(model, features, window_size=100, stride=50):
    """Analyze features using the MulT model"""
    audio_features = features['audio_features']
    video_features = features['video_features']
    timestamps = features['timestamps']
    
    # Convert to tensors
    audio_tensor = torch.FloatTensor(audio_features)
    video_tensor = torch.FloatTensor(video_features)
    
    # Use sliding window to process long sequences
    seq_length = min(len(audio_tensor), len(video_tensor))
    results = []
    
    for start in range(0, seq_length - window_size, stride):
        end = start + window_size
        
        # Extract features within the window
        audio_window = audio_tensor[start:end]
        video_window = video_tensor[start:end]
        timestamp_window = timestamps[start:end]
        
        # Use model to predict
        with torch.no_grad():
            valence_pred, arousal_pred = model(audio_window.unsqueeze(0), video_window.unsqueeze(0))
        
        # Record results
        results.append({
            'start_time': timestamp_window[0],
            'end_time': timestamp_window[-1],
            'valence': valence_pred.item(),
            'arousal': arousal_pred.item(),
            'window_start': start,
            'window_end': end
        })
    
    return results

def plot_emotions(results, role, output_file):
    """Plot emotion changes"""
    times = [(r['start_time'] + r['end_time']) / 2 for r in results]
    valence = [r['valence'] for r in results]
    arousal = [r['arousal'] for r in results]
    
    plt.figure(figsize=(12, 6))
    
    # Plot emotion changes
    plt.subplot(2, 1, 1)
    plt.plot(times, valence, 'b-', label='Valence')
    plt.plot(times, arousal, 'r-', label='Arousal')
    plt.ylabel('Emotion Value')
    plt.title(f'Emotion Changes for {role}')
    plt.legend()
    plt.grid(True)
    
    # Plot emotion quadrants
    plt.subplot(2, 1, 2)
    plt.scatter(valence, arousal, c=times, cmap='viridis', alpha=0.6)
    plt.colorbar(label='Time (seconds)')
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.title('Emotion Quadrant')
    plt.grid(True)
    
    # Add quadrant labels
    plt.text(0.25, 0.75, 'Negative\nActive', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.75, 0.75, 'Positive\nActive', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.25, 0.25, 'Negative\nPassive', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.75, 0.25, 'Positive\nPassive', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def generate_summary(all_results):
    """Generate analysis summary"""
    summary = []
    
    for result in all_results:
        role = result['role']
        analysis = result['analysis']
        
        # Calculate average emotion values
        avg_valence = np.mean([r['valence'] for r in analysis])
        avg_arousal = np.mean([r['arousal'] for r in analysis])
        
        # Determine main emotion quadrant
        quadrant = ''
        if avg_valence >= 0.5 and avg_arousal >= 0.5:
            quadrant = 'Positive Active'
        elif avg_valence >= 0.5 and avg_arousal < 0.5:
            quadrant = 'Positive Passive'
        elif avg_valence < 0.5 and avg_arousal >= 0.5:
            quadrant = 'Negative Active'
        else:
            quadrant = 'Negative Passive'
        
        # Calculate emotion variability
        valence_std = np.std([r['valence'] for r in analysis])
        arousal_std = np.std([r['arousal'] for r in analysis])
        emotion_variability = (valence_std + arousal_std) / 2
        
        # Add to summary
        summary.append({
            'role': role,
            'avg_valence': avg_valence,
            'avg_arousal': avg_arousal,
            'quadrant': quadrant,
            'emotion_variability': emotion_variability
        })
    
    return summary

def main():
    """Main function"""
    # Load pairing information
    pairs_df = load_pairing_info()
    print("Pairing information:")
    print(pairs_df)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    all_results = []
    
    # Process each video and audio pair
    for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Analyzing data"):
        role = row['role']
        video_file = row['video']
        audio_file = row['audio']
        
        print(f"\nAnalyzing {role}:")
        print(f"Video: {video_file}")
        print(f"Audio: {audio_file}")
        
        # Load features
        features = load_features(video_file, audio_file)
        
        # Analyze features
        results = analyze_features(model, features)
        
        # Save results
        output_file = os.path.join(OUTPUT_DIR, f"{video_file}_{audio_file}_analysis.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot emotions
        plot_file = os.path.join(OUTPUT_DIR, f"{video_file}_{audio_file}_emotions.png")
        plot_emotions(results, role, plot_file)
        
        # Add to all results
        all_results.append({
            'role': role,
            'video': video_file,
            'audio': audio_file,
            'analysis': results
        })
    
    # Generate summary
    summary = generate_summary(all_results)
    
    # Save summary
    summary_file = os.path.join(OUTPUT_DIR, "es2016a_analysis_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate report
    report_file = os.path.join(OUTPUT_DIR, "es2016a_analysis_report.md")
    with open(report_file, 'w') as f:
        f.write("# ES2016a Meeting Analysis Report\n\n")
        f.write("## Participant Emotion Analysis Summary\n\n")
        
        f.write("| Role | Average Valence | Average Arousal | Main Emotion Quadrant | Emotional Variability |\n")
        f.write("|------|----------------|-----------------|----------------------|----------------------|\n")
        
        for s in summary:
            f.write(f"| {s['role']} | {s['avg_valence']:.3f} | {s['avg_arousal']:.3f} | {s['quadrant']} | {s['emotion_variability']:.3f} |\n")
        
        f.write("\n## Detailed Analysis\n\n")
        
        for result in all_results:
            role = result['role']
            video = result['video']
            audio = result['audio']
            
            f.write(f"### {role}\n\n")
            f.write(f"- Video: {video}\n")
            f.write(f"- Audio: {audio}\n\n")
            f.write(f"![Emotion Changes]({video}_{audio}_emotions.png)\n\n")
            
            # Add detailed description
            s = next((s for s in summary if s['role'] == role), None)
            if s:
                f.write(f"**Summary**: This participant's emotions are mainly in the {s['quadrant']} quadrant, ")
                f.write(f"with average valence of {s['avg_valence']:.3f} and average arousal of {s['avg_arousal']:.3f}. ")
                
                if s['emotion_variability'] > 0.15:
                    f.write("There is significant emotional variability, indicating notable mood changes during the meeting.\n\n")
                elif s['emotion_variability'] > 0.1:
                    f.write("There is moderate emotional variability, indicating some mood changes during the meeting.\n\n")
                else:
                    f.write("There is low emotional variability, indicating stable emotions throughout the meeting.\n\n")
    
    print("\nAnalysis complete!")
    print(f"Analysis results saved to: {OUTPUT_DIR}")
    print(f"Analysis report: {report_file}")

if __name__ == "__main__":
    main() 