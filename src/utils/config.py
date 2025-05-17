from dataclasses import dataclass
import os

@dataclass
class ModelConfig:
    # Model architecture parameters
    audio_dim: int = 40  # Number of mel filterbanks
    video_dim: int = 3 * 224 * 224  # RGB image flattened
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    max_seq_len: int = 1000
    
    # Training parameters
    batch_size: int = 4
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 5
    
    # Data processing parameters
    video_fps: int = 25
    audio_sample_rate: int = 16000
    audio_hop_length: int = 160
    audio_n_fft: int = 400
    audio_n_mels: int = 40
    
    # Paths
    base_dir: str = "/Users/yuwang/Documents/GitHub/AMI_Corpus_Updated/AMI_Corpus"
    data_dir: str = os.path.join(base_dir, "ES2016a")
    audio_dir: str = os.path.join(data_dir, "Audios")
    video_dir: str = os.path.join(data_dir, "Videos")
    checkpoint_dir: str = os.path.join(base_dir, "MulT", "checkpoints")
    log_dir: str = os.path.join(base_dir, "MulT", "logs")
    checkpoint_path: str = os.path.join(checkpoint_dir, "best_model.pt")
    
    def __post_init__(self):
        """Create necessary directories and compute derived parameters"""
        # Create directories if they don't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Compute derived parameters
        self.audio_frames_per_second = self.audio_sample_rate // self.audio_hop_length
        self.video_frames_per_second = self.video_fps
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        return cls(**config_dict) 