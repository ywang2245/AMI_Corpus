import os
import sys
import pandas as pd
from tqdm import tqdm
import json
import argparse

# Add src to the Python path if needed
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from src.extractors.openface_extractor import OpenFaceExtractor
from src.extractors.opensmile_extractor import OpenSMILEExtractor

def process_pairs(pairs, video_dir, audio_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'video_features'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'audio_features'), exist_ok=True)

    video_extractor = OpenFaceExtractor()
    audio_extractor = OpenSMILEExtractor(feature_set='eGeMAPSv01a')

    for pair in tqdm(pairs, desc="Processing pairs"):
        role = pair['role']
        video_file = pair['video'] + '.avi'
        audio_file = pair['audio'] + '.wav'
        video_path = os.path.join(video_dir, video_file)
        audio_path = os.path.join(audio_dir, audio_file)

        # Process video
        try:
            video_features = video_extractor.process_video(video_path, sample_rate=5)
            video_outfile = os.path.join(output_dir, 'video_features', f"{pair['video']}.json")
            with open(video_outfile, 'w') as f:
                json.dump({
                    'role': role,
                    'features': {
                        'landmarks': video_features['features']['landmarks'].tolist(),
                        'face_descriptor': video_features['features']['face_descriptor'].tolist(),
                        'eye_aspect_ratios': video_features['features']['eye_aspect_ratios'].tolist(),
                        'mouth_aspect_ratio': video_features['features']['mouth_aspect_ratio'].tolist(),
                        'head_pose': video_features['features']['head_pose'].tolist()
                    },
                    'video_properties': video_features['video_properties']
                }, f)
        except Exception as e:
            print(f"Error processing video {video_file}: {str(e)}")

        # Process audio
        try:
            audio_features = audio_extractor.extract_temporal_features(
                audio_path,
                frame_size=0.025,
                hop_size=0.010
            )
            # Remove rows with NaN values
            features_array = np.array(audio_features['features'])
            valid_rows = ~np.isnan(features_array).any(axis=1)
            num_removed = (~valid_rows).sum()
            filtered_features = features_array[valid_rows]
            if num_removed > 0:
                print(f"Filtered out {num_removed} NaN rows from {audio_file}")
            audio_outfile = os.path.join(output_dir, 'audio_features', f"{pair['audio']}.json")
            with open(audio_outfile, 'w') as f:
                json.dump({
                    'role': role,
                    'features': filtered_features.tolist(),
                    'feature_names': audio_features['feature_names'],
                    'audio_properties': audio_features['audio_properties']
                }, f)
        except Exception as e:
            print(f"Error processing audio {audio_file}: {str(e)}")

def main():
    pairs = [
        {'role': 'project manager', 'video': 'ES2016a.Closeup4', 'audio': 'ES2016a.Headset-0'},
        {'role': 'industrial designer', 'video': 'ES2016a.Closeup3', 'audio': 'ES2016a.Headset-1'},
        {'role': 'user interface designer', 'video': 'ES2016a.Closeup2', 'audio': 'ES2016a.Headset-2'},
        {'role': 'marketing expert', 'video': 'ES2016a.Closeup1', 'audio': 'ES2016a.Headset-3'},
    ]
    video_dir = os.path.join('ES2016a', 'Videos')
    audio_dir = os.path.join('ES2016a', 'Audios')
    output_dir = 'extracted_features'
    process_pairs(pairs, video_dir, audio_dir, output_dir)
    print("\nFeature extraction for selected pairs completed!")

if __name__ == '__main__':
    main() 