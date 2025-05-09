import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from tqdm import tqdm

class AudioVAExtractor:
    def __init__(self, sample_rate=22050, hop_length=512, segment_duration=3.0):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.segment_duration = segment_duration  # Duration of each segment in seconds
        self.segment_samples = int(segment_duration * sample_rate)
    
    def extract_features(self, audio_path):
        """Extract audio features for V-A prediction"""
        # Load audio file
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # RMS Energy (related to arousal)
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        
        # Spectral Centroid (related to brightness/timbre)
        cent = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate, 
                                               hop_length=self.hop_length)[0]
        
        # Spectral Rolloff (related to timbre)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sample_rate,
                                                 hop_length=self.hop_length)[0]
        
        # Zero Crossing Rate (related to noisiness)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
        
        # Mel-spectrogram features
        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sample_rate,
                                                hop_length=self.hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # MFCC coefficients
        mfcc = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=13,
                                  hop_length=self.hop_length)
        
        # Tempo and beat strength
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sample_rate,
                                               hop_length=self.hop_length)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env,
                                                   sr=self.sample_rate,
                                                   hop_length=self.hop_length)
        
        # Calculate beat strength
        beat_strength = np.zeros_like(onset_env)
        beat_strength[beat_frames] = onset_env[beat_frames]
        
        # Print feature statistics for debugging
        print("\nFeature Statistics:")
        print(f"Spectral Centroid: mean={np.mean(cent):.2f}, max={np.max(cent):.2f}")
        print(f"Zero Crossing Rate: mean={np.mean(zcr):.4f}, max={np.max(zcr):.4f}")
        print(f"MFCC[0]: mean={np.mean(mfcc[0]):.2f}, max={np.max(mfcc[0]):.2f}")
        print(f"Tempo: mean={np.mean(tempo):.2f}")
        
        # Normalize features
        norm_cent = float(np.mean(cent))
        norm_zcr = float(np.mean(zcr))
        norm_mfcc = float(np.mean(mfcc[0]))
        norm_tempo = float(np.mean(tempo)) if isinstance(tempo, np.ndarray) else float(tempo)
        norm_rms = float(np.mean(rms))
        norm_beat = float(np.mean(beat_strength))
        
        # Return features as a dictionary
        features = {
            'rms': norm_rms,
            'spectral_centroid': norm_cent,
            'spectral_rolloff': float(np.mean(rolloff)),
            'zero_crossing_rate': norm_zcr,
            'mel_spectrogram': mel_spec_db,
            'mfcc': mfcc,
            'tempo': norm_tempo,
            'beat_strength': norm_beat
        }
        
        # Convert dictionary values to numpy arrays
        features_array = np.array([
            features['rms'],
            features['spectral_centroid'],
            features['spectral_rolloff'],
            features['zero_crossing_rate'],
            features['tempo'],
            features['beat_strength'],
            np.mean(features['mfcc']),
            np.std(features['mfcc'])
        ])
        
        return features_array
    
    def predict_va(self, features):
        """Predict valence and arousal from audio features"""
        # Normalize features for valence prediction
        norm_centroid = features['spectral_centroid'] / 1500  # Based on mean value
        norm_zcr = features['zero_crossing_rate'] * 20  # Scale up ZCR
        norm_mfcc = float(np.mean(features['mfcc'][0]) + 500) / 400  # Shift and scale MFCC
        norm_tempo = features['tempo'] / 120  # Normalize by typical tempo
        
        # Print normalized values for debugging
        print("\nNormalized Values for Valence:")
        print(f"Normalized Centroid: {norm_centroid:.4f}")
        print(f"Normalized ZCR: {norm_zcr:.4f}")
        print(f"Normalized MFCC: {norm_mfcc:.4f}")
        print(f"Normalized Tempo: {norm_tempo:.4f}")
        
        # Valence prediction with adjusted weights
        valence = (
            0.4 * norm_centroid +  # Brightness
            0.3 * (1 - norm_zcr) +  # Inverse of noisiness
            0.2 * norm_mfcc +  # First MFCC coefficient
            0.1 * norm_tempo  # Tempo influence
        )
        valence = np.clip(valence, 0, 1)  # Clip to [0, 1]
        
        print(f"Calculated Valence: {valence:.4f}")
        
        # Arousal prediction (based on energy, beat strength, and tempo)
        arousal = (
            0.4 * features['rms'] * 10 +  # Scale up RMS
            0.3 * features['beat_strength'] * 5 +  # Scale up beat strength
            0.2 * features['zero_crossing_rate'] * 20 +  # Scale up ZCR
            0.1 * features['tempo'] / 120  # Normalize tempo
        )
        arousal = np.clip(arousal, 0, 1)  # Clip to [0, 1]
        
        return valence, arousal
    
    def process_audio_segments(self, y, sr):
        """Process audio in segments and return temporal V-A values"""
        # Calculate number of segments
        total_samples = len(y)
        num_segments = total_samples // self.segment_samples
        
        # Initialize lists to store temporal data
        timestamps = []
        valence_values = []
        arousal_values = []
        
        # Process each segment
        for i in tqdm(range(num_segments), desc="Processing segments"):
            start_sample = i * self.segment_samples
            end_sample = start_sample + self.segment_samples
            
            # Extract segment
            segment = y[start_sample:end_sample]
            
            # Extract features and predict V-A values
            features = self.extract_features(segment)
            valence, arousal = self.predict_va(features)
            
            # Store results
            timestamps.append(i * self.segment_duration)
            valence_values.append(valence)
            arousal_values.append(arousal)
        
        return timestamps, valence_values, arousal_values
    
    def plot_temporal_va(self, timestamps, valence_values, arousal_values, audio_path, output_dir):
        """Plot temporal evolution of V-A values"""
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot Valence over time
        ax1.plot(timestamps, valence_values, 'b-', label='Valence')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Valence')
        ax1.set_title('Temporal Evolution of Valence')
        ax1.grid(True)
        ax1.set_ylim(0, 1)
        
        # Plot Arousal over time
        ax2.plot(timestamps, arousal_values, 'r-', label='Arousal')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Arousal')
        ax2.set_title('Temporal Evolution of Arousal')
        ax2.grid(True)
        ax2.set_ylim(0, 1)
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = os.path.join(output_dir, 
                                 os.path.basename(audio_path).replace('.wav', '_temporal_va.pdf'))
        plt.savefig(output_path)
        plt.close()
        
        # Create V-A space animation plot
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(valence_values, arousal_values, c=timestamps, cmap='viridis',
                           s=50, alpha=0.6)
        ax.set_xlabel('Valence')
        ax.set_ylabel('Arousal')
        ax.set_title('Emotional Trajectory in V-A Space')
        ax.grid(True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.colorbar(scatter, label='Time (seconds)')
        
        # Save V-A space plot
        output_path = os.path.join(output_dir, 
                                 os.path.basename(audio_path).replace('.wav', '_va_trajectory.pdf'))
        plt.savefig(output_path)
        plt.close()
        
        # Save temporal data to CSV
        df = pd.DataFrame({
            'Time': timestamps,
            'Valence': valence_values,
            'Arousal': arousal_values
        })
        csv_path = os.path.join(output_dir, 
                              os.path.basename(audio_path).replace('.wav', '_temporal_va.csv'))
        df.to_csv(csv_path, index=False)
    
    def process_audio_file(self, audio_path, output_dir='audio_va_plots'):
        """Process a single audio file and generate V-A analysis"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio file
        print(f"Loading audio file: {audio_path}")
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Process audio in segments
        print("Processing audio segments...")
        timestamps, valence_values, arousal_values = self.process_audio_segments(y, sr)
        
        # Generate temporal plots
        print("Generating plots...")
        self.plot_temporal_va(timestamps, valence_values, arousal_values, audio_path, output_dir)
        
        # Calculate overall statistics
        mean_valence = np.mean(valence_values)
        mean_arousal = np.mean(arousal_values)
        std_valence = np.std(valence_values)
        std_arousal = np.std(arousal_values)
        
        return {
            'mean_valence': mean_valence,
            'mean_arousal': mean_arousal,
            'std_valence': std_valence,
            'std_arousal': std_arousal,
            'timestamps': timestamps,
            'valence_values': valence_values,
            'arousal_values': arousal_values
        }

def main():
    # Initialize extractor with 3-second segments
    extractor = AudioVAExtractor(segment_duration=3.0)
    
    # Process audio file
    audio_path = "/Users/yuwang/Desktop/yw/AMI_Corpus/amicorpus/ES2016a/audio/ES2016a.Array1-01.wav"
    
    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return
    
    # Process the audio file
    print("\nStarting audio analysis...")
    results = extractor.process_audio_file(audio_path)
    
    # Print results
    print(f"\nAudio V-A Analysis Results:")
    print(f"Mean Valence: {results['mean_valence']:.3f} (±{results['std_valence']:.3f})")
    print(f"Mean Arousal: {results['mean_arousal']:.3f} (±{results['std_arousal']:.3f})")
    print(f"\nDetailed temporal analysis has been saved to the 'audio_va_plots' directory:")
    print("- Temporal evolution plots (.pdf)")
    print("- V-A space trajectory plot (.pdf)")
    print("- Temporal data (.csv)")

if __name__ == "__main__":
    main() 