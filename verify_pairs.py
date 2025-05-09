import pandas as pd
import os

# Read the pairing information from Excel
pairs = pd.read_excel('ES2016_video_audio_pair.xlsx')

# Filter out Overhead and Corner videos
pairs = pairs[~pairs['video_file'].str.contains('Overhead|Corner')]

# Print pairing information
print("\nAudio-Video Pairing Information (Closeup videos only):")
print("=" * 50)
for idx, row in pairs.iterrows():
    print(f"Pair {idx+1}:")
    print(f"  Audio: {row['audio_file']}")
    print(f"  Video: {row['video_file']}")
    print("-" * 50)

# Print summary
print(f"\nTotal pairs: {len(pairs)}")
print("=" * 50) 