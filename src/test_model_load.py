import os
import sys
import torch

# Add src to the Python path if needed
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# Import model
from src.models.multit_model import MultiModalTransformer

# Set paths
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Go up one level from src
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'MulT', 'MulT', 'checkpoints', 'mult_model.pt')

def test_load_model():
    """Test loading the model"""
    # Check if pre-trained model exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Pre-trained model not found at {CHECKPOINT_PATH}")
        return None
    
    print(f"Found model file: {CHECKPOINT_PATH}")
    
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
    
    # Print model configuration
    model_config = checkpoint.get('config', {})
    print("Model configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    # Try to create model
    try:
        model = MultiModalTransformer(
            audio_dim=model_config['audio_dim'],
            video_dim=model_config['video_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_heads=model_config['num_heads'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout']
        )
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    model = test_load_model()
    if model:
        print(f"Model type: {type(model)}")
        print(f"Model architecture:\n{model}")
        
        # Test model inference
        print("\nTesting model inference:")
        try:
            batch_size = 1
            seq_len = 100
            audio_dim = 50
            video_dim = 270
            
            # Create random test data
            audio_input = torch.randn(batch_size, seq_len, audio_dim)
            video_input = torch.randn(batch_size, seq_len, video_dim)
            
            # Forward pass
            with torch.no_grad():
                valence, arousal = model(audio_input, video_input)
            
            print(f"Input shapes: Audio {audio_input.shape}, Video {video_input.shape}")
            print(f"Output shapes: Valence {valence.shape}, Arousal {arousal.shape}")
            print(f"Output values: Valence {valence.item():.4f}, Arousal {arousal.item():.4f}")
            print("Model inference successful!")
        except Exception as e:
            print(f"Error during model inference: {e}") 