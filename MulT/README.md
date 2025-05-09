# MulT: Multimodal Transformer for Emotion Analysis (AMI Corpus)

## Overview
This project implements a Multimodal Transformer (MulT) model for emotion analysis using the AMI Corpus. The pipeline now uses advanced feature extraction:
- **Video features**: Extracted using OpenFace 2.0 (dlib), including facial landmarks, face descriptors, eye/mouth aspect ratios, and head pose.
- **Audio features**: Extracted using OpenSMILE (eGeMAPSv01a set), providing robust acoustic features for emotion recognition.

## Directory Structure
```
MulT/
├── README.md
├── multit_model.py
├── dataset.py
├── openface_extractor.py
├── opensmile_extractor.py
├── process_features.py
├── download_models.py
├── requirements.txt
├── models/                # OpenFace model files
├── extracted_features/    # Output features (after running extraction)
└── ...
```

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies
- OpenFace model files (downloaded automatically)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Download OpenFace Model Files
Run the following script to download the required dlib models:
```bash
python MulT/download_models.py
```

## Feature Extraction
Extract features from your audio and video files (update paths as needed):
```bash
python MulT/process_features.py
```
- This will process the four main participant pairs (project manager, industrial designer, user interface designer, marketing expert) from the AMI ES2016a session.
- Extracted features are saved in `extracted_features/audio_features/` and `extracted_features/video_features/` as JSON files.

### What is Extracted?
- **Video**: 68 facial landmarks, 128-dim face descriptor, eye/mouth aspect ratios, head pose (per frame)
- **Audio**: eGeMAPSv01a features (per frame, 25ms window, 10ms hop)
- NaN rows in audio features are automatically filtered out.

## Using the MulT Model

### 1. Prepare the Dataset
Modify or use the provided `MultiModalDataset` class to load features from the `extracted_features` directory. The dataset should match pairs of audio and video features for each participant.

### 2. Training/Inference
- You can use leave-one-out or 3-train/1-test cross-validation due to the small number of pairs.
- The MulT model expects tensors for each modality (see `dataset.py`).
- Example (pseudo-code):
```python
from dataset import MultiModalDataset
# Prepare a DataFrame with columns: ['role', 'video', 'audio']
# ...
dataset = MultiModalDataset(pairs_df, audio_dir='extracted_features/audio_features', video_dir='extracted_features/video_features')
# Use with DataLoader, train MulT as usual
```

### 3. Model Output
- The MulT model predicts valence and arousal values for each sample.

## Notes
- The pipeline is now optimized for robust, interpretable features using state-of-the-art open-source toolkits.
- If you wish to process additional pairs, update the `pairs` list in `process_features.py`.
- For troubleshooting, ensure all model files are present in `MulT/models/` and that your environment matches the requirements.

## References
- [OpenFace 2.0](https://github.com/TadasBaltrusaitis/OpenFace)
- [OpenSMILE](https://audeering.github.io/opensmile/)
- [MulT: Multimodal Transformer](https://arxiv.org/abs/1907.05266) 