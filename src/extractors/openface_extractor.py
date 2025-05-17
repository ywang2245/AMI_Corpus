import os
import cv2
import numpy as np
import dlib
import time
from tqdm import tqdm

class OpenFaceExtractor:
    """
    A feature extractor for facial features using dlib and OpenCV.
    Extracts facial landmarks, face descriptor, eye and mouth measurements, and head pose.
    """
    
    def __init__(self, 
                 face_detector_path=None, 
                 landmark_predictor_path=None, 
                 face_recognition_model_path=None):
        """
        Initialize the OpenFace extractor with model paths.
        
        Args:
            face_detector_path: Path to dlib's face detector model
            landmark_predictor_path: Path to dlib's landmark predictor model
            face_recognition_model_path: Path to dlib's face recognition model
        """
        # Initialize face detector
        self.detector = dlib.get_frontal_face_detector()
        
        # Initialize facial landmark predictor
        landmark_path = landmark_predictor_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'models', 'shape_predictor_68_face_landmarks.dat'
        )
        if not os.path.exists(landmark_path):
            raise FileNotFoundError(f"Landmark predictor model not found at {landmark_path}. Please run 'python src/utils/download_models.py' first.")
        self.predictor = dlib.shape_predictor(landmark_path)
        
        # Initialize face recognition model
        recognition_path = face_recognition_model_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'models', 'dlib_face_recognition_resnet_model_v1.dat'
        )
        if not os.path.exists(recognition_path):
            raise FileNotFoundError(f"Face recognition model not found at {recognition_path}. Please run 'python src/utils/download_models.py' first.")
        self.face_rec_model = dlib.face_recognition_model_v1(recognition_path)
    
    def extract_features(self, frame):
        """
        Extract facial features from a single frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary containing extracted features or None if no face detected
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector(gray)
        if len(faces) == 0:
            return None
        
        # Use the first detected face (assuming single person videos)
        face = faces[0]
        
        # Get facial landmarks
        landmarks = self.predictor(gray, face)
        landmarks_points = []
        
        # Extract landmark coordinates
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            landmarks_points.append((x, y))
        
        landmarks_array = np.array(landmarks_points).flatten()
        
        # Extract face descriptor
        face_descriptor = self.face_rec_model.compute_face_descriptor(frame, landmarks)
        face_descriptor_array = np.array(face_descriptor)
        
        # Calculate eye aspect ratios
        left_eye_points = [landmarks.part(i) for i in range(36, 42)]
        right_eye_points = [landmarks.part(i) for i in range(42, 48)]
        
        left_ear = self._calculate_eye_aspect_ratio(left_eye_points)
        right_ear = self._calculate_eye_aspect_ratio(right_eye_points)
        
        # Calculate mouth aspect ratio
        mouth_points = [landmarks.part(i) for i in range(48, 68)]
        mar = self._calculate_mouth_aspect_ratio(mouth_points)
        
        # Estimate head pose
        # Simplified head pose calculation based on relative positions of key landmarks
        nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
        left_eye_center = ((landmarks.part(36).x + landmarks.part(39).x) // 2, 
                           (landmarks.part(36).y + landmarks.part(39).y) // 2)
        right_eye_center = ((landmarks.part(42).x + landmarks.part(45).x) // 2, 
                            (landmarks.part(42).y + landmarks.part(45).y) // 2)
        
        # Calculate relative angles
        dx_eyes = right_eye_center[0] - left_eye_center[0]
        dy_eyes = right_eye_center[1] - left_eye_center[1]
        
        if dx_eyes == 0:
            dx_eyes = 1  # Avoid division by zero
            
        roll = np.arctan(dy_eyes / dx_eyes) * 180 / np.pi
        
        # Yaw estimation (based on relative position of nose to eye line)
        eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                      (left_eye_center[1] + right_eye_center[1]) // 2)
        dx_nose = nose_tip[0] - eye_center[0]
        yaw = (dx_nose / dx_eyes) * 45  # Approximation
        
        # Pitch estimation (based on nose tip vertical position)
        dy_nose = nose_tip[1] - eye_center[1]
        norm_dy_nose = dy_nose / (frame.shape[0] * 0.1)  # Normalize
        pitch = norm_dy_nose * 45  # Approximation
        
        head_pose = [roll, yaw, pitch]
        
        return {
            'landmarks': landmarks_array,
            'face_descriptor': face_descriptor_array,
            'eye_aspect_ratios': [left_ear, right_ear],
            'mouth_aspect_ratio': mar,
            'head_pose': head_pose
        }
    
    def _calculate_eye_aspect_ratio(self, eye_points):
        """
        Calculate the eye aspect ratio for a given eye.
        
        Args:
            eye_points: List of eye landmark points
            
        Returns:
            Eye aspect ratio (float)
        """
        # Calculate vertical eye distances
        A = np.sqrt((eye_points[1].x - eye_points[5].x)**2 + (eye_points[1].y - eye_points[5].y)**2)
        B = np.sqrt((eye_points[2].x - eye_points[4].x)**2 + (eye_points[2].y - eye_points[4].y)**2)
        
        # Calculate horizontal eye distance
        C = np.sqrt((eye_points[0].x - eye_points[3].x)**2 + (eye_points[0].y - eye_points[3].y)**2)
        
        # Calculate eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def _calculate_mouth_aspect_ratio(self, mouth_points):
        """
        Calculate the mouth aspect ratio.
        
        Args:
            mouth_points: List of mouth landmark points
            
        Returns:
            Mouth aspect ratio (float)
        """
        # Calculate horizontal distances
        horizontal_dist = np.sqrt((mouth_points[6].x - mouth_points[0].x)**2 + 
                                  (mouth_points[6].y - mouth_points[0].y)**2)
        
        # Calculate vertical distances
        A = np.sqrt((mouth_points[2].x - mouth_points[10].x)**2 + 
                    (mouth_points[2].y - mouth_points[10].y)**2)
        B = np.sqrt((mouth_points[4].x - mouth_points[8].x)**2 + 
                    (mouth_points[4].y - mouth_points[8].y)**2)
        
        # Calculate mouth aspect ratio
        mar = (A + B) / (2.0 * horizontal_dist)
        return mar
    
    def process_video(self, video_path, sample_rate=1):
        """
        Process a video file and extract facial features frame by frame.
        
        Args:
            video_path: Path to the video file
            sample_rate: Sample every n-th frame
            
        Returns:
            Dictionary containing processed features and video metadata
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        # Initialize feature containers
        all_landmarks = []
        all_face_descriptors = []
        all_eye_ratios = []
        all_mouth_ratios = []
        all_head_poses = []
        
        # Process frames
        frame_idx = 0
        
        # Create a progress bar for video processing
        with tqdm(total=frame_count, desc=f"Processing {os.path.basename(video_path)}") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame based on sample rate
                if frame_idx % sample_rate == 0:
                    features = self.extract_features(frame)
                    
                    if features is not None:
                        all_landmarks.append(features['landmarks'])
                        all_face_descriptors.append(features['face_descriptor'])
                        all_eye_ratios.append(features['eye_aspect_ratios'])
                        all_mouth_ratios.append(features['mouth_aspect_ratio'])
                        all_head_poses.append(features['head_pose'])
                    else:
                        # If no face detected, use the previous features or zeros
                        if all_landmarks:
                            all_landmarks.append(all_landmarks[-1])
                            all_face_descriptors.append(all_face_descriptors[-1])
                            all_eye_ratios.append(all_eye_ratios[-1])
                            all_mouth_ratios.append(all_mouth_ratios[-1])
                            all_head_poses.append(all_head_poses[-1])
                        else:
                            # Use zeros for the first frame if no face detected
                            all_landmarks.append(np.zeros(68*2))
                            all_face_descriptors.append(np.zeros(128))
                            all_eye_ratios.append([0, 0])
                            all_mouth_ratios.append(0)
                            all_head_poses.append([0, 0, 0])
                
                frame_idx += 1
                pbar.update(1)
        
        # Release the video capture object
        cap.release()
        
        # Package results
        return {
            'features': {
                'landmarks': all_landmarks,
                'face_descriptor': all_face_descriptors,
                'eye_aspect_ratios': all_eye_ratios,
                'mouth_aspect_ratio': all_mouth_ratios,
                'head_pose': all_head_poses
            },
            'video_properties': {
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'sampled_frame_count': len(all_landmarks),
                'sample_rate': sample_rate
            }
        } 