# Multimodal Transformer (MulT) for Audio-Visual Analysis

This project implements a Multimodal Transformer model for analyzing audio-visual data from the AMI Corpus. The model processes both audio and video inputs to predict valence and arousal values.

## Model Architecture

The MultiModalTransformer consists of:
- Separate audio and video feature projections
- Transformer encoders for each modality
- Cross-modal attention mechanism
- Output layers for valence and arousal prediction

## Dataset

The MultiModalDataset class handles:
- Loading audio-video pairs
- Audio preprocessing using MFCC features
- Video frame extraction and preprocessing
- Batch preparation for training

## Requirements

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

```python
from multit_model import MultiModalTransformer, MultiModalDataset

# Initialize dataset
dataset = MultiModalDataset(
    data_df=your_dataframe,
    audio_dir='path/to/audio',
    video_dir='path/to/video',
    audio_duration=3.0
)

# Initialize model
model = MultiModalTransformer(
    audio_dim=40,  # MFCC features
    video_dim=3,   # RGB channels
    hidden_dim=256,
    num_heads=8,
    num_layers=4,
    dropout=0.1
)

# Training loop implementation follows standard PyTorch practices
``` 