import cv2
import numpy as np
import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import warnings
import os
import matplotlib.pyplot as plt
from datetime import timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd
from tqdm import tqdm
import time
warnings.filterwarnings('ignore')

class EmotionVAExtractor:
    def __init__(self):
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize emotion recognition model (using a simpler approach)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.va_mapping = {
            'Angry': (-0.8, 0.8),    # High negative valence, high arousal
            'Disgust': (-0.6, 0.4),  # Negative valence, moderate arousal
            'Fear': (-0.7, 0.6),     # Negative valence, high arousal
            'Happy': (0.8, 0.6),     # High positive valence, high arousal
            'Sad': (-0.6, -0.4),     # Negative valence, low arousal
            'Surprise': (0.3, 0.8),  # Slightly positive valence, high arousal
            'Neutral': (0.0, 0.0)    # Neutral valence and arousal
        }
        
        # Initialize audio model (using Wav2Vec as an example)
        self.audio_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.audio_model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base")
        
        # Supported video formats
        self.supported_formats = ['.avi', '.rm', '.rmvb', '.mp4', '.mov', '.mkv']
        
    def preprocess_face(self, face_img):
        """Preprocess face image for emotion recognition"""
        # Resize to expected input size
        face_img = cv2.resize(face_img, (48, 48))
        # Convert to grayscale if not already
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        return face_img
        
    def get_va_values(self, face_img):
        """Get valence-arousal values using facial features"""
        try:
            # Preprocess face
            gray = self.preprocess_face(face_img)
            
            # Calculate features
            # 1. Brightness features
            brightness = np.mean(gray)
            brightness_norm = (brightness - 128) / 128  # Normalize to [-1, 1]
            
            # 2. Texture features
            contrast = np.std(gray) / 128  # Normalize standard deviation
            
            # 3. Edge features - adjusted thresholds and normalization
            edges = cv2.Canny(gray, 30, 100)  # Lower thresholds to detect more edges
            edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1] * 255)  # Normalize by max possible value
            
            # 4. Region features
            h, w = gray.shape
            upper_face = gray[:h//2, :]
            lower_face = gray[h//2:, :]
            
            upper_brightness = np.mean(upper_face)
            lower_brightness = np.mean(lower_face)
            vertical_contrast = abs(upper_brightness - lower_brightness) / 255
            
            # 5. Local variance as a measure of facial expression intensity
            local_var = cv2.Laplacian(gray, cv2.CV_64F).var() / 10000  # Normalize by typical range
            
            # Calculate valence using brightness and vertical contrast
            valence = np.clip(
                0.6 * brightness_norm + 0.4 * (1 - vertical_contrast),
                -1, 1
            )
            
            # Calculate arousal using multiple features
            # Edge density indicates facial expression intensity
            # Contrast indicates overall facial movement
            # Local variance indicates local changes
            arousal = np.clip(
                0.3 * edge_density +      # Weight for edge information
                0.3 * contrast +          # Weight for overall contrast
                0.4 * local_var,          # Weight for local variations
                -1, 1
            )
            
            return valence, arousal
            
        except Exception as e:
            print(f"Error in V-A prediction: {str(e)}")
            return 0.0, 0.0  # Return neutral values in case of error
        
    def extract_va_from_video(self, video_path, sample_rate=1):
        """
        Extract Valence-Arousal values from video frames with temporal tracking
        Args:
            video_path: Path to the video file
            sample_rate: Process every nth frame
        Returns:
            Dictionary containing temporal V-A data and video properties
        """
        # Check if file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Check file extension
        _, ext = os.path.splitext(video_path)
        if ext.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported video format: {ext}. Supported formats are: {', '.join(self.supported_formats)}")
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            print(f"Video properties:")
            print(f"- FPS: {fps}")
            print(f"- Total frames: {frame_count}")
            print(f"- Duration: {duration:.2f} seconds")
            
            # Initialize data structures for temporal tracking
            temporal_data = {
                'timestamps': [],
                'valence': [],
                'arousal': [],
                'face_count': [],
                'frame_numbers': []
            }
            
            current_frame = 0
            pbar = tqdm(total=frame_count, desc="Processing frames")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if current_frame % sample_rate == 0:
                    # Convert to grayscale for face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    # Calculate timestamp for this frame
                    timestamp = current_frame / fps
                    time_str = str(timedelta(seconds=int(timestamp)))
                    
                    if len(faces) > 0:
                        # For each face in the frame
                        frame_valence = []
                        frame_arousal = []
                        
                        for (x, y, w, h) in faces:
                            face = frame[y:y+h, x:x+w]
                            # Get V-A values using emotion recognition
                            valence, arousal = self.get_va_values(face)
                            
                            frame_valence.append(valence)
                            frame_arousal.append(arousal)
                        
                        # Average V-A values for all faces in this frame
                        avg_valence = np.mean(frame_valence)
                        avg_arousal = np.mean(frame_arousal)
                        
                        # Store temporal data
                        temporal_data['timestamps'].append(time_str)
                        temporal_data['valence'].append(avg_valence)
                        temporal_data['arousal'].append(avg_arousal)
                        temporal_data['face_count'].append(len(faces))
                        temporal_data['frame_numbers'].append(current_frame)
                        
                        print(f"Frame {current_frame} at {time_str}: {len(faces)} faces detected")
                        print(f"Average Valence: {avg_valence:.3f}, Average Arousal: {avg_arousal:.3f}")
                    else:
                        print(f"No face detected in frame {current_frame} at {time_str}")
                
                current_frame += 1
                pbar.update(1)
                
                # Print progress every 100 frames
                if current_frame % 100 == 0:
                    progress = (current_frame / frame_count) * 100
                    print(f"Processing: {progress:.1f}%")
            
            pbar.close()
            cap.release()
            
            if not temporal_data['valence']:
                print("Warning: No faces were detected in the video")
            
            return {
                'temporal_data': temporal_data,
                'video_properties': {
                    'fps': fps,
                    'frame_count': frame_count,
                    'duration': duration
                }
            }
            
        except Exception as e:
            if 'cap' in locals():
                cap.release()
            raise Exception(f"Error processing video: {str(e)}")
    
    def plot_temporal_emotions(self, data, output_path=None):
        """
        Plot temporal evolution of emotions
        Args:
            data: Dictionary containing temporal V-A data
            output_path: Path to save the plot (optional)
        """
        temporal_data = data['temporal_data']
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Convert timestamps to seconds for better x-axis
        timestamps = [int(sum(x * float(t) for x, t in zip([3600, 60, 1], time.split(':'))))
                     for time in temporal_data['timestamps']]
        
        # Create major ticks every 5 minutes (300 seconds)
        major_tick_spacing = 300
        minor_tick_spacing = 60  # Minor ticks every minute
        
        for ax in [ax1, ax2]:
            ax.set_xticks(range(0, max(timestamps) + major_tick_spacing, major_tick_spacing))
            ax.set_xticks(range(0, max(timestamps) + minor_tick_spacing, minor_tick_spacing), minor=True)
            # Convert seconds back to MM:SS format for labels
            ax.set_xticklabels([f"{int(t/60):02d}:{int(t%60):02d}" for t in ax.get_xticks()])
        
        # Plot valence over time
        ax1.plot(timestamps, temporal_data['valence'], 'b-', label='Valence')
        ax1.set_title('Valence over Time')
        ax1.set_xlabel('Time (MM:SS)')
        ax1.set_ylabel('Valence')
        ax1.grid(True, which='major', linestyle='-')
        ax1.grid(True, which='minor', linestyle=':', alpha=0.5)
        ax1.legend()
        ax1.set_ylim(-1, 1)  # Set y-axis limits for valence
        
        # Plot arousal over time
        ax2.plot(timestamps, temporal_data['arousal'], 'r-', label='Arousal')
        ax2.set_title('Arousal over Time')
        ax2.set_xlabel('Time (MM:SS)')
        ax2.set_ylabel('Arousal')
        ax2.grid(True, which='major', linestyle='-')
        ax2.grid(True, which='minor', linestyle=':', alpha=0.5)
        ax2.legend()
        ax2.set_ylim(-1, 1)  # Set y-axis limits for arousal
        
        # Rotate x-axis labels for better readability
        plt.setp(ax1.get_xticklabels(), rotation=45)
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")
        
        plt.show()

def process_single_video(video_path, sample_rate=5):
    """
    Process a single video and analyze emotional changes
    Args:
        video_path: Path to the video file
        sample_rate: Process every nth frame
    """
    extractor = EmotionVAExtractor()
    
    try:
        print(f"Processing video: {video_path}")
        results = extractor.extract_va_from_video(video_path, sample_rate)
        
        # Create output directory for plots
        output_dir = "emotion_plots"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate plot filename from video filename
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        plot_path = os.path.join(output_dir, f"{video_name}_emotions.pdf")
        
        # Plot temporal emotions
        extractor.plot_temporal_emotions(results, plot_path)
        
        # Save data to CSV
        temporal_data = results['temporal_data']
        df = pd.DataFrame({
            'Frame': temporal_data['frame_numbers'],
            'Time': [float(t.split(':')[0]) * 60 + float(t.split(':')[1]) for t in temporal_data['timestamps']],
            'Valence': temporal_data['valence'],
            'Arousal': temporal_data['arousal'],
            'Face_Count': temporal_data['face_count']
        })
        
        csv_path = os.path.join(output_dir, f"{video_name}_emotions.csv")
        df.to_csv(csv_path, index=False)
        print(f"Data saved to CSV: {csv_path}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total frames processed: {len(temporal_data['frame_numbers'])}")
        print(f"Average valence: {np.mean(temporal_data['valence']):.3f}")
        print(f"Average arousal: {np.mean(temporal_data['arousal']):.3f}")
        print(f"Maximum faces detected in a frame: {max(temporal_data['face_count'])}")
        print(f"\nPlot saved as: {plot_path}")
        
        return results
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return None

def main():
    # Process videos from the AMI Corpus
    video_dir = "/Users/yuwang/Desktop/yw/AMI_Corpus/amicorpus/ES2016a/video"
    video_files = [
        "ES2016a.Closeup1.avi",
        "ES2016a.Closeup2.avi",
        "ES2016a.Closeup3.avi",
        "ES2016a.Closeup4.avi"
    ]
    
    print("\nStarting video processing...")
    print("Working directory:", os.getcwd())
    print("Video directory:", video_dir)
    
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        print(f"\nProcessing {video_file}...")
        print(f"Full path: {video_path}")
        print(f"File exists: {os.path.exists(video_path)}")
        
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            continue
            
        try:
            process_single_video(video_path, sample_rate=5)  # Process every 5th frame for faster processing
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()