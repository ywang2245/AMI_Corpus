import torch
import torchaudio
import numpy as np
import cv2
import librosa
import os
import mediapipe as mp
from collections import defaultdict
import matplotlib.pyplot as plt

class PairedVAExtractor:
    """Extract valence-arousal from paired video and audio files"""
    
    def __init__(self):
        print("Initializing paired V-A extractor...")
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize MediaPipe Pose for body tracking
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # V-A model constants
        self.va_smooth_window = 5  # Window size for smoothing V-A values
        self.va_sampling_rate = 1   # How often to sample V-A (in seconds)
    
    def extract_visual_features(self, video_path):
        """Extract visual emotion features from video"""
        print(f"Extracting visual features from: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"  - Video duration: {duration:.2f} seconds, {total_frames} frames at {fps} fps")
        
        # Sample frames at regular intervals
        sample_interval = int(fps * self.va_sampling_rate)
        sample_frames = []
        frame_indices = []
        
        for i in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                sample_frames.append(frame)
                frame_indices.append(i)
        
        cap.release()
        
        # Initialize feature arrays
        num_samples = len(sample_frames)
        face_features = np.zeros((num_samples, 10))
        pose_features = np.zeros((num_samples, 8))
        timestamps = np.array(frame_indices) / fps
        
        # Extract features from each frame
        for i, frame in enumerate(sample_frames):
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract face features
            face_results = self.face_mesh.process(frame_rgb)
            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0].landmark
                
                # Extract facial expression features
                # Eyes (width, height ratio)
                left_eye_width = abs(landmarks[33].x - landmarks[133].x)
                left_eye_height = abs(landmarks[159].y - landmarks[145].y)
                right_eye_width = abs(landmarks[362].x - landmarks[263].x)
                right_eye_height = abs(landmarks[386].y - landmarks[374].y)
                
                # Mouth (width, height, openness)
                mouth_width = abs(landmarks[61].x - landmarks[291].x)
                mouth_height = abs(landmarks[0].y - landmarks[17].y)
                mouth_openness = abs(landmarks[13].y - landmarks[14].y)
                
                # Eyebrow positions (relative to eyes)
                left_brow_height = landmarks[107].y - landmarks[159].y
                right_brow_height = landmarks[336].y - landmarks[386].y
                
                # Face tilt (head orientation)
                face_tilt = abs(landmarks[33].y - landmarks[263].y)
                
                face_features[i] = [
                    left_eye_width/left_eye_height, 
                    right_eye_width/right_eye_height,
                    mouth_width, 
                    mouth_height, 
                    mouth_openness,
                    left_brow_height, 
                    right_brow_height,
                    face_tilt,
                    landmarks[5].x,  # Nose tip x
                    landmarks[5].y   # Nose tip y
                ]
            
            # Extract pose features
            pose_results = self.pose.process(frame_rgb)
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                
                # Body posture features
                shoulder_width = abs(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x - 
                                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
                
                hip_width = abs(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x - 
                               landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x)
                
                torso_height = abs(landmarks[self.mp_pose.PoseLandmark.NOSE].y - 
                                  landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y)
                
                # Arm positions (for gestures)
                left_arm_angle = self.calculate_angle(
                    [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x, 
                     landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y],
                    [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x, 
                     landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y],
                    [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].x, 
                     landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y]
                )
                
                right_arm_angle = self.calculate_angle(
                    [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x, 
                     landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
                    [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x, 
                     landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y],
                    [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].x, 
                     landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y]
                )
                
                # Head position relative to shoulders
                head_forward = landmarks[self.mp_pose.PoseLandmark.NOSE].z
                head_tilt = abs(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].y - 
                              landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR].y)
                
                # Body rotation
                body_rotation = abs(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].z - 
                                   landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].z)
                
                pose_features[i] = [
                    shoulder_width,
                    hip_width,
                    torso_height,
                    left_arm_angle,
                    right_arm_angle,
                    head_forward,
                    head_tilt,
                    body_rotation
                ]
        
        return {'face': face_features, 'pose': pose_features, 'timestamps': timestamps}
    
    def extract_audio_features(self, audio_path, timestamps=None):
        """Extract acoustic emotion features from audio"""
        print(f"Extracting audio features from: {audio_path}")
        
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=audio, sr=sr)
        
        print(f"  - Audio duration: {duration:.2f} seconds, sampling rate: {sr} Hz")
        
        # If no timestamps provided, create evenly spaced samples
        if timestamps is None:
            num_samples = int(duration / self.va_sampling_rate)
            timestamps = np.linspace(0, duration, num_samples)
        
        num_samples = len(timestamps)
        audio_features = np.zeros((num_samples, 6))
        
        # Extract features at each timestamp
        for i, t in enumerate(timestamps):
            # Extract audio segment around timestamp
            start_sample = int((t - 0.5) * sr)
            end_sample = int((t + 0.5) * sr)
            
            if start_sample < 0:
                start_sample = 0
            if end_sample > len(audio):
                end_sample = len(audio)
            
            segment = audio[start_sample:end_sample]
            
            if len(segment) == 0:
                continue
                
            # Extract spectral features
            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            
            # Extract prosodic features
            rms = librosa.feature.rms(y=segment)[0]
            energy = np.mean(rms)
            
            zero_crossings = librosa.feature.zero_crossing_rate(segment)[0]
            zcr = np.mean(zero_crossings)
            
            # Extract pitch (F0) features
            pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
            pitch_mean = 0
            if np.any(magnitudes > 0):
                pitch_mean = np.mean(pitches[magnitudes > 0])
            
            # Extract speech rate (using energy fluctuations as proxy)
            energy_diff = np.diff(rms)
            speech_rate = np.mean(np.abs(energy_diff))
            
            # Extract spectral contrast (for emotional intensity)
            contrast = librosa.feature.spectral_contrast(y=segment, sr=sr)
            contrast_mean = np.mean(contrast)
            
            # Use first 6 features
            audio_features[i] = [
                energy,              # Energy correlates with arousal
                zcr,                 # Zero-crossing rate correlates with arousal
                pitch_mean,          # Pitch correlates with both valence and arousal
                speech_rate,         # Speech rate correlates with arousal
                contrast_mean,       # Spectral contrast correlates with emotional intensity
                mfcc_mean[1]         # First MFCC coefficient correlates with valence
            ]
        
        return {'audio': audio_features, 'timestamps': timestamps}
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def map_features_to_va(self, visual_features, audio_features):
        """Map extracted features to valence-arousal space"""
        print("Mapping features to valence-arousal space...")
        
        # Combine face and pose features
        face = visual_features['face']
        pose = visual_features['pose']
        audio = audio_features['audio']
        
        # Weight different modalities (these would be learned in a real system)
        face_weights_v = [0.1, 0.1, 0.2, 0.2, 0.15, 0.1, 0.1, 0.05, 0, 0]
        face_weights_a = [0.1, 0.1, 0.05, 0.2, 0.3, 0.1, 0.1, 0.05, 0, 0]
        
        pose_weights_v = [0.05, 0.05, 0.1, 0.2, 0.2, 0.15, 0.15, 0.1]
        pose_weights_a = [0.1, 0.05, 0.05, 0.25, 0.25, 0.1, 0.1, 0.1]
        
        audio_weights_v = [0.1, 0.1, 0.3, 0.2, 0.1, 0.2]
        audio_weights_a = [0.3, 0.15, 0.2, 0.2, 0.1, 0.05]
        
        # For demo, use normalized values from features with appropriate mapping
        valence_scores = []
        arousal_scores = []
        
        # Process each time step
        for i in range(len(visual_features['timestamps'])):
            # Get feature vectors for this time step
            face_features = face[i] if i < len(face) else np.zeros(10)
            pose_features = pose[i] if i < len(pose) else np.zeros(8)
            
            # Find closest audio timestamp
            timestamp_diff = np.zeros(len(audio_features['timestamps']))
            for j, ts in enumerate(audio_features['timestamps']):
                timestamp_diff[j] = abs(ts - visual_features['timestamps'][i])
            closest_audio_idx = np.argmin(timestamp_diff)
            
            audio_features_vec = audio[closest_audio_idx] if closest_audio_idx < len(audio) else np.zeros(6)
            
            # Normalize features
            face_std = np.std(face_features)
            if face_std == 0:
                face_std = 1e-6
            face_features_norm = (face_features - np.mean(face_features)) / face_std
            
            pose_std = np.std(pose_features)
            if pose_std == 0:
                pose_std = 1e-6
            pose_features_norm = (pose_features - np.mean(pose_features)) / pose_std
            
            audio_std = np.std(audio_features_vec)
            if audio_std == 0:
                audio_std = 1e-6
            audio_features_norm = (audio_features_vec - np.mean(audio_features_vec)) / audio_std
            
            # Calculate valence from all modalities
            face_valence = np.sum(face_features_norm * face_weights_v)
            pose_valence = np.sum(pose_features_norm * pose_weights_v)
            audio_valence = np.sum(audio_features_norm * audio_weights_v)
            
            # Calculate arousal from all modalities
            face_arousal = np.sum(face_features_norm * face_weights_a)
            pose_arousal = np.sum(pose_features_norm * pose_weights_a)
            audio_arousal = np.sum(audio_features_norm * audio_weights_a)
            
            # Combine modalities with fusion weights
            valence = 0.4 * face_valence + 0.2 * pose_valence + 0.4 * audio_valence
            arousal = 0.3 * face_arousal + 0.3 * pose_arousal + 0.4 * audio_arousal
            
            # Normalize to [-1, 1] range
            valence = np.clip(valence, -1, 1)
            arousal = np.clip(arousal, -1, 1)
            
            valence_scores.append(valence)
            arousal_scores.append(arousal)
        
        # Apply smoothing if enough data points
        if len(valence_scores) >= self.va_smooth_window:
            kernel = np.ones(self.va_smooth_window) / self.va_smooth_window
            valence_scores = np.convolve(valence_scores, kernel, mode='same')
            arousal_scores = np.convolve(arousal_scores, kernel, mode='same')
        
        return {
            'valence': valence_scores,
            'arousal': arousal_scores,
            'timestamps': visual_features['timestamps'],
            'mean_valence': np.mean(valence_scores),
            'mean_arousal': np.mean(arousal_scores),
            'std_valence': np.std(valence_scores),
            'std_arousal': np.std(arousal_scores),
            'max_valence': np.max(valence_scores),
            'max_arousal': np.max(arousal_scores),
            'min_valence': np.min(valence_scores),
            'min_arousal': np.min(arousal_scores)
        }
    
    def extract_va_values(self, video_path, audio_path):
        """Extract V-A values from paired video and audio"""
        print(f"\nExtracting V-A values for participant")
        print(f"Video: {video_path}")
        print(f"Audio: {audio_path}")
        
        # Extract features from each modality
        visual_features = self.extract_visual_features(video_path)
        audio_features = self.extract_audio_features(audio_path, visual_features['timestamps'])
        
        # Map features to V-A space
        va_results = self.map_features_to_va(visual_features, audio_features)
        
        return va_results

def process_paired_data(base_path, meeting_id):
    """Process paired video and audio data for a meeting"""
    print(f"Processing paired data for meeting {meeting_id}")
    
    # Define participant pairings (closeup video paired with headset audio)
    participant_pairs = [
        {
            "id": 1,
            "video": f"{base_path}/{meeting_id}/video/{meeting_id}.Closeup1.avi",
            "audio": f"{base_path}/{meeting_id}/audio/{meeting_id}.Headset-0.wav"
        },
        {
            "id": 2,
            "video": f"{base_path}/{meeting_id}/video/{meeting_id}.Closeup2.avi",
            "audio": f"{base_path}/{meeting_id}/audio/{meeting_id}.Headset-1.wav"
        },
        {
            "id": 3,
            "video": f"{base_path}/{meeting_id}/video/{meeting_id}.Closeup3.avi",
            "audio": f"{base_path}/{meeting_id}/audio/{meeting_id}.Headset-2.wav"
        },
        {
            "id": 4,
            "video": f"{base_path}/{meeting_id}/video/{meeting_id}.Closeup4.avi",
            "audio": f"{base_path}/{meeting_id}/audio/{meeting_id}.Headset-3.wav"
        }
    ]
    
    # Process each participant's data
    extractor = PairedVAExtractor()
    results = {}
    
    for participant in participant_pairs:
        print(f"\nProcessing participant {participant['id']}")
        
        # Check if files exist
        if not os.path.exists(participant['video']):
            print(f"Warning: Video file {participant['video']} not found. Skipping participant {participant['id']}")
            continue
            
        if not os.path.exists(participant['audio']):
            print(f"Warning: Audio file {participant['audio']} not found. Skipping participant {participant['id']}")
            continue
        
        # Extract V-A values
        va_results = extractor.extract_va_values(participant['video'], participant['audio'])
        
        # Store results
        results[f"participant_{participant['id']}"] = va_results
    
    return results

def visualize_results(results, meeting_id):
    """Visualize V-A results"""
    # Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Plot V-A values over time for each participant
    plt.figure(figsize=(15, 10))
    
    for i, (participant, va_values) in enumerate(results.items()):
        plt.subplot(2, 2, i+1)
        
        timestamps = va_values['timestamps']
        valence = va_values['valence']
        arousal = va_values['arousal']
        
        plt.plot(timestamps, valence, 'b-', label='Valence')
        plt.plot(timestamps, arousal, 'r-', label='Arousal')
        
        plt.title(f"{participant.replace('_', ' ').title()}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Emotion Value")
        plt.legend()
        plt.grid(True)
        
        # Add mean values annotation
        plt.annotate(f"Mean V: {va_values['mean_valence']:.2f}, A: {va_values['mean_arousal']:.2f}",
                    xy=(0.05, 0.05), xycoords='axes fraction')
    
    plt.tight_layout()
    plt.savefig(f"results/{meeting_id}_va_time_series.png")
    
    # Plot V-A values in the 2D emotion space
    plt.figure(figsize=(10, 10))
    
    colors = ['b', 'r', 'g', 'orange']
    for i, (participant, va_values) in enumerate(results.items()):
        if i >= len(colors):
            break
            
        valence = va_values['valence']
        arousal = va_values['arousal']
        
        plt.scatter(valence, arousal, c=colors[i], alpha=0.5, label=participant)
        
        # Plot mean V-A point
        plt.scatter(va_values['mean_valence'], va_values['mean_arousal'], 
                   c=colors[i], s=200, marker='*', edgecolors='black')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title(f"Valence-Arousal Space - Meeting {meeting_id}")
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    
    # Add emotion quadrant labels
    plt.text(0.7, 0.7, "HAPPY\nEXCITED", horizontalalignment='center', verticalalignment='center')
    plt.text(-0.7, 0.7, "ANGRY\nFRUSTRATED", horizontalalignment='center', verticalalignment='center')
    plt.text(-0.7, -0.7, "SAD\nBORED", horizontalalignment='center', verticalalignment='center')
    plt.text(0.7, -0.7, "CALM\nRELAXED", horizontalalignment='center', verticalalignment='center')
    
    plt.legend()
    plt.savefig(f"results/{meeting_id}_va_space.png")
    
    # Save numerical results to CSV
    with open(f"results/{meeting_id}_va_summary.csv", 'w') as f:
        f.write("Participant,Mean Valence,Mean Arousal,Min Valence,Max Valence,Min Arousal,Max Arousal,Std Valence,Std Arousal\n")
        
        for participant, va_values in results.items():
            f.write(f"{participant},{va_values['mean_valence']:.4f},{va_values['mean_arousal']:.4f}," +
                   f"{va_values['min_valence']:.4f},{va_values['max_valence']:.4f}," +
                   f"{va_values['min_arousal']:.4f},{va_values['max_arousal']:.4f}," +
                   f"{va_values['std_valence']:.4f},{va_values['std_arousal']:.4f}\n")

if __name__ == "__main__":
    # Process ES2016a meeting with paired data
    base_path = "amicorpus"
    meeting_id = "ES2016a"
    
    print(f"Processing meeting {meeting_id} with paired video-audio data")
    
    # Adjust headset-participant mapping if needed for this specific meeting
    print("Note: For ES2016a, participant mapping:")
    print("  Closeup1 paired with Headset-0")
    print("  Closeup2 paired with Headset-1")
    print("  Closeup3 paired with Headset-2")
    print("  Closeup4 paired with Headset-3")
    
    # Process data
    results = process_paired_data(base_path, meeting_id)
    
    # Visualize results
    if results:
        visualize_results(results, meeting_id)
        print(f"\nResults saved to 'results/{meeting_id}_*' files")
    else:
        print("No results to visualize") 