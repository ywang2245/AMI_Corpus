import cv2
import numpy as np
import dlib
import os
from tqdm import tqdm

class OpenFaceExtractor:
    def __init__(self, model_path=None):
        """
        Initialize OpenFace feature extractor.
        
        Args:
            model_path (str): Path to OpenFace model files. If None, uses default paths.
        """
        # Initialize face detector
        self.detector = dlib.get_frontal_face_detector()
        
        # Initialize facial landmark predictor
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'models')
        predictor_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Initialize face recognition model
        face_rec_model_path = os.path.join(model_path, 'dlib_face_recognition_resnet_model_v1.dat')
        self.face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
        
        # Feature dimensions
        self.landmark_dim = 68 * 2  # 68 landmarks, each with x,y coordinates
        self.face_rec_dim = 128  # Face recognition embedding dimension
        
    def extract_features(self, frame):
        """
        Extract facial features from a single frame.
        
        Args:
            frame: RGB image frame
            
        Returns:
            dict: Dictionary containing facial features
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return None
            
        # Get the first face
        face = faces[0]
        
        # Get facial landmarks
        landmarks = self.predictor(gray, face)
        landmarks_array = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        # Get face recognition features
        face_descriptor = np.array(self.face_rec_model.compute_face_descriptor(frame, landmarks))
        
        # Calculate additional features
        # 1. Eye aspect ratio
        left_eye = landmarks_array[36:42]
        right_eye = landmarks_array[42:48]
        left_ear = self._calculate_eye_aspect_ratio(left_eye)
        right_ear = self._calculate_eye_aspect_ratio(right_eye)
        
        # 2. Mouth aspect ratio
        mouth = landmarks_array[48:60]
        mar = self._calculate_mouth_aspect_ratio(mouth)
        
        # 3. Head pose estimation (simplified)
        nose_bridge = landmarks_array[27:31]
        nose_tip = landmarks_array[30]
        left_eye_center = np.mean(landmarks_array[36:42], axis=0)
        right_eye_center = np.mean(landmarks_array[42:48], axis=0)
        
        # Combine all features
        features = {
            'landmarks': landmarks_array.flatten(),  # 136 dimensions
            'face_descriptor': face_descriptor,      # 128 dimensions
            'eye_aspect_ratios': np.array([left_ear, right_ear]),
            'mouth_aspect_ratio': mar,
            'head_pose': np.concatenate([nose_bridge.flatten(), nose_tip, left_eye_center, right_eye_center])
        }
        
        return features
    
    def _calculate_eye_aspect_ratio(self, eye):
        """Calculate the eye aspect ratio"""
        # Compute the euclidean distances between the vertical eye landmarks
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        # Compute the euclidean distance between the horizontal eye landmarks
        C = np.linalg.norm(eye[0] - eye[3])
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def _calculate_mouth_aspect_ratio(self, mouth):
        """Calculate the mouth aspect ratio"""
        # Compute the euclidean distances between the vertical mouth landmarks
        A = np.linalg.norm(mouth[3] - mouth[9])
        B = np.linalg.norm(mouth[2] - mouth[10])
        C = np.linalg.norm(mouth[4] - mouth[8])
        # Compute the euclidean distance between the horizontal mouth landmarks
        D = np.linalg.norm(mouth[0] - mouth[6])
        # Compute the mouth aspect ratio
        mar = (A + B + C) / (2.0 * D)
        return mar
    
    def process_video(self, video_path, sample_rate=1):
        """
        Process a video file and extract facial features.
        
        Args:
            video_path (str): Path to the video file
            sample_rate (int): Process every nth frame
            
        Returns:
            dict: Dictionary containing temporal features and video properties
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Initialize feature arrays
        features_list = []
        timestamps = []
        
        frame_idx = 0
        pbar = tqdm(total=frame_count, desc="Processing video frames")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % sample_rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Extract features
                features = self.extract_features(frame_rgb)
                
                if features is not None:
                    features_list.append(features)
                    timestamps.append(frame_idx / fps)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        if not features_list:
            raise ValueError("No faces detected in the video")
        
        # Combine features into arrays
        combined_features = {
            'landmarks': np.stack([f['landmarks'] for f in features_list]),
            'face_descriptor': np.stack([f['face_descriptor'] for f in features_list]),
            'eye_aspect_ratios': np.stack([f['eye_aspect_ratios'] for f in features_list]),
            'mouth_aspect_ratio': np.array([f['mouth_aspect_ratio'] for f in features_list]),
            'head_pose': np.stack([f['head_pose'] for f in features_list]),
            'timestamps': np.array(timestamps)
        }
        
        return {
            'features': combined_features,
            'video_properties': {
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration
            }
        } 