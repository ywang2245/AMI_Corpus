import opensmile
import numpy as np
import librosa
from tqdm import tqdm

class OpenSMILEExtractor:
    def __init__(self, feature_set='ComParE_2016'):
        """
        Initialize OpenSMILE feature extractor.
        
        Args:
            feature_set (str): OpenSMILE feature set to use. Options:
                - 'ComParE_2016': 6373 features
                - 'GeMAPSv01a': 62 features
                - 'eGeMAPSv01a': 88 features
        """
        self.feature_set = feature_set
        self.smile = opensmile.Smile(
            feature_set=feature_set,
            feature_level=opensmile.FeatureLevel.Functionals,
            loglevel=2,
            logfile='smile.log'
        )
        
    def extract_features(self, audio_path, sample_rate=16000):
        """
        Extract audio features using OpenSMILE.
        
        Args:
            audio_path (str): Path to the audio file
            sample_rate (int): Target sample rate for audio
            
        Returns:
            dict: Dictionary containing audio features and properties
        """
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sample_rate)
        
        # Extract features using OpenSMILE
        features = self.smile.process_signal(y, sr)
        
        # Get feature names
        feature_names = features.columns.tolist()
        
        # Convert to numpy array
        feature_array = features.values
        
        # Get audio properties
        duration = librosa.get_duration(y=y, sr=sr)
        
        return {
            'features': feature_array,
            'feature_names': feature_names,
            'audio_properties': {
                'sample_rate': sr,
                'duration': duration,
                'num_samples': len(y)
            }
        }
    
    def extract_temporal_features(self, audio_path, frame_size=0.025, hop_size=0.010, sample_rate=16000):
        """
        Extract temporal audio features using OpenSMILE.
        
        Args:
            audio_path (str): Path to the audio file
            frame_size (float): Frame size in seconds
            hop_size (float): Hop size in seconds
            sample_rate (int): Target sample rate for audio
            
        Returns:
            dict: Dictionary containing temporal features and properties
        """
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sample_rate)
        
        # Calculate frame parameters
        frame_length = int(frame_size * sr)
        hop_length = int(hop_size * sr)
        
        # Extract features for each frame
        features_list = []
        timestamps = []
        
        # Process frames with progress bar
        num_frames = int((len(y) - frame_length) / hop_length) + 1
        pbar = tqdm(total=num_frames, desc="Processing audio frames")
        
        for i in range(0, len(y) - frame_length + 1, hop_length):
            # Extract frame
            frame = y[i:i + frame_length]
            
            # Extract features for this frame
            features = self.smile.process_signal(frame, sr)
            features_list.append(features.values[0])
            
            # Calculate timestamp
            timestamp = i / sr
            timestamps.append(timestamp)
            
            pbar.update(1)
        
        pbar.close()
        
        # Stack features
        feature_array = np.stack(features_list)
        
        # Get feature names
        feature_names = features.columns.tolist()
        
        # Get audio properties
        duration = librosa.get_duration(y=y, sr=sr)
        
        return {
            'features': feature_array,
            'feature_names': feature_names,
            'timestamps': np.array(timestamps),
            'audio_properties': {
                'sample_rate': sr,
                'duration': duration,
                'num_samples': len(y),
                'frame_size': frame_size,
                'hop_size': hop_size
            }
        }
    
    def get_feature_set_info(self):
        """
        Get information about the current feature set.
        
        Returns:
            dict: Dictionary containing feature set information
        """
        feature_sets = {
            'ComParE_2016': {
                'num_features': 6373,
                'description': 'Large feature set from the ComParE 2016 challenge',
                'features': [
                    'MFCCs', 'spectral', 'energy', 'voicing', 'F0',
                    'jitter', 'shimmer', 'loudness', 'formants'
                ]
            },
            'GeMAPSv01a': {
                'num_features': 62,
                'description': 'Geneva Minimalistic Acoustic Parameter Set',
                'features': [
                    'F0', 'jitter', 'shimmer', 'HNR', 'formants',
                    'spectral energy', 'MFCCs', 'alpha ratio', 'hammarberg index'
                ]
            },
            'eGeMAPSv01a': {
                'num_features': 88,
                'description': 'Extended Geneva Minimalistic Acoustic Parameter Set',
                'features': [
                    'F0', 'jitter', 'shimmer', 'HNR', 'formants',
                    'spectral energy', 'MFCCs', 'alpha ratio', 'hammarberg index',
                    'spectral slope', 'spectral flux', 'spectral roll-off'
                ]
            }
        }
        
        return feature_sets.get(self.feature_set, {
            'num_features': 'unknown',
            'description': 'Custom feature set',
            'features': []
        }) 