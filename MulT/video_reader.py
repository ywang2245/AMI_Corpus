import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

class VideoReader:
    def __init__(self, target_size=(224, 224)):
        """
        Initialize the video reader.
        
        Args:
            target_size (tuple): Target size for frame resizing (height, width)
        """
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def read_video(self, video_path):
        """
        Read and preprocess video frames.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            np.ndarray: Preprocessed video frames of shape [T, 3*H*W]
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame
            frame = cv2.resize(frame, self.target_size)
            
            # Apply transforms
            frame = self.transform(frame)
            
            # Flatten frame
            frame = frame.view(-1)
            
            frames.append(frame.numpy())
        
        cap.release()
        
        # Stack frames
        frames = np.stack(frames)
        
        return frames 