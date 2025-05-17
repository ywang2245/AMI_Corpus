# MulT Model for AMI Corpus

This directory contains a consolidated version of the Multi-modal Transformer (MulT) model implementation for the AMI Corpus project.

## Directory Structure

```
src/
├── __init__.py                # Makes src a proper Python package
├── analyze_es2016a.py         # Main analysis script for the ES2016a meeting
├── process_es2016a.py         # Script to process ES2016a data
├── process_features.py        # Feature extraction script
├── requirements.txt           # Required packages
├── test_model_load.py         # Test script for model loading
├── train.py                   # MulT model training script
├── extractors/                # Feature extraction implementations
│   ├── __init__.py
│   ├── openface_extractor.py  # Facial feature extraction using OpenFace
│   └── opensmile_extractor.py # Audio feature extraction using OpenSMILE
├── models/                    # Model implementations
│   ├── __init__.py
│   ├── dataset.py             # Dataset loading utilities
│   └── multit_model.py        # MulT model implementation
└── utils/                     # Utility functions
    ├── __init__.py
    ├── config.py              # Configuration settings
    └── download_models.py     # Script to download required models
```

## Usage

To run analysis on the ES2016a meeting:

```bash
python src/analyze_es2016a.py
```

To extract features using OpenFace and OpenSMILE:

```bash
python src/process_es2016a.py
```

To train the MulT model:

```bash
python src/train.py
```

## Dependencies

Install the required packages using:

```bash
pip install -r src/requirements.txt
``` 