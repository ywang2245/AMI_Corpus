import os
import sys
import pandas as pd
import json
from tqdm import tqdm

# Add src to the Python path if needed
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from src.extractors.openface_extractor import OpenFaceExtractor
from src.extractors.opensmile_extractor import OpenSMILEExtractor

# Set paths
BASE_DIR = os.getcwd()
VIDEO_DIR = os.path.join(BASE_DIR, 'ES2016a', 'Videos')
AUDIO_DIR = os.path.join(BASE_DIR, 'ES2016a', 'Audios')
EXCEL_FILE = os.path.join(BASE_DIR, 'ES2016_video_audio_pair.xlsx')

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, 'extracted_features')
VIDEO_FEATURES_DIR = os.path.join(OUTPUT_DIR, 'video_features')
AUDIO_FEATURES_DIR = os.path.join(OUTPUT_DIR, 'audio_features')

# Create output directories
os.makedirs(VIDEO_FEATURES_DIR, exist_ok=True)
os.makedirs(AUDIO_FEATURES_DIR, exist_ok=True)

def load_pairing_info():
    """Load video and audio pairing information from Excel file"""
    df = pd.read_excel(EXCEL_FILE)
    # Only keep ES2016a rows
    df = df.iloc[1:5].reset_index(drop=True)
    # Rename columns
    df.columns = ['id', 'role', 'video', 'audio']
    return df

def extract_video_features(video_file):
    """Extract video features using OpenFace"""
    print(f"Processing video: {video_file}")
    video_path = os.path.join(VIDEO_DIR, f"{video_file}.avi")
    
    # Initialize OpenFace feature extractor
    extractor = OpenFaceExtractor()
    
    # Extract features
    features = extractor.process_video(video_path)
    
    return features

def extract_audio_features(audio_file):
    """Extract audio features using OpenSMILE"""
    print(f"Processing audio: {audio_file}")
    audio_path = os.path.join(AUDIO_DIR, f"{audio_file}.wav")
    
    # Initialize OpenSMILE feature extractor
    extractor = OpenSMILEExtractor()
    
    # Extract features (frame level)
    features = extractor.extract_temporal_features(audio_path)
    
    return features

def process_pair(video_file, audio_file):
    """Process a pair of video and audio files"""
    # Extract features
    video_features = extract_video_features(video_file)
    audio_features = extract_audio_features(audio_file)
    
    # Save video features
    video_output_file = os.path.join(VIDEO_FEATURES_DIR, f"{video_file}.json")
    with open(video_output_file, 'w') as f:
        json.dump(video_features, f, indent=2)
    
    # Convert audio features to the correct format
    # OpenSMILE temporal features are in dictionary format, we need to convert to list format
    audio_output = {
        'features': audio_features['features'].tolist(),
        'timestamps': audio_features['timestamps'],
        'sample_rate': audio_features['sample_rate'],
        'num_samples': audio_features['num_samples'],
        'duration': audio_features['duration']
    }
    
    # Save audio features
    audio_output_file = os.path.join(AUDIO_FEATURES_DIR, f"{audio_file}.json")
    with open(audio_output_file, 'w') as f:
        json.dump(audio_output, f, indent=2)
    
    print(f"Features saved: {video_file} and {audio_file}")

def main():
    """Main function"""
    # Load pairing information
    pairs_df = load_pairing_info()
    print("Pairing information:")
    print(pairs_df)
    
    # Process each video and audio pair
    for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df)):
        print(f"\nProcessing pair {idx+1}/{len(pairs_df)}:")
        print(f"Role: {row['role']}")
        print(f"Video: {row['video']}")
        print(f"Audio: {row['audio']}")
        
        process_pair(row['video'], row['audio'])
    
    print("\nFeature extraction complete!")
    print(f"Video features saved to: {VIDEO_FEATURES_DIR}")
    print(f"Audio features saved to: {AUDIO_FEATURES_DIR}")

if __name__ == "__main__":
    main() 