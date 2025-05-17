# AMI Corpus MulT Model

This project uses a Multi-modal Transformer (MulT) model to analyze audio and video recordings from the AMI Corpus to detect and evaluate participants' emotional states.

## Project Overview

The AMI Corpus MulT model combines audio and video features, extracts cross-modal features through a transformer architecture, and outputs participants' emotional valence and arousal values. The model uses OpenFace for video feature extraction, OpenSMILE for audio feature extraction, trains a multi-modal transformer model with these features, and ultimately generates an emotion analysis report.

## Project Structure

```
AMI_Corpus/
├── ES2016a/                   # ES2016a meeting data directory
│   ├── Audios/                # Audio recordings
│   └── Videos/                # Video recordings
├── ES2016_video_audio_pair.xlsx # Audio and video pairing information
├── analysis_results/          # Generated analysis results
├── extracted_features/        # Extracted audio and video features
├── run_all.py                 # Script to run all main tools
├── models/                    # Directory for pre-trained model files
└── src/                       # Source code
    ├── extractors/            # Feature extraction implementations
    │   ├── __init__.py        # Package initialization file
    │   ├── openface_extractor.py  # Video feature extraction using OpenFace
    │   └── opensmile_extractor.py # Audio feature extraction using OpenSMILE
    ├── models/                # Model implementations
    │   ├── __init__.py        # Package initialization file
    │   ├── dataset.py         # Dataset loading utilities
    │   └── multit_model.py    # MulT model implementation
    ├── utils/                 # Utility functions
    │   ├── __init__.py        # Package initialization file
    │   ├── config.py          # Configuration settings
    │   └── download_models.py # Script to download required models
    ├── __init__.py            # Package initialization file
    ├── analyze_es2016a.py     # Analysis script
    ├── process_es2016a.py     # Feature extraction script
    ├── process_features.py    # Additional feature processing
    ├── requirements.txt       # Required packages
    ├── test_model_load.py     # Model loading test script
    └── train.py               # Training script
```

## File Descriptions

### Core Scripts

- **run_all.py**: Command-line tool to run all main functions. Supports four main commands: `extract`, `train`, `analyze`, and `test_model`.

### src/extractors/

- **openface_extractor.py**: Implements the `OpenFaceExtractor` class for extracting facial features from videos, including facial landmarks, face descriptors, eye and mouth aspect ratios, and head pose.
  
- **opensmile_extractor.py**: Implements the `OpenSMILEExtractor` class for extracting acoustic features from audio, such as pitch, energy, MFCCs, etc.

### src/models/

- **multit_model.py**: Implements the `MultiModalTransformer` class, the core model of this project. It uses a multi-head attention mechanism to process and fuse audio and video features, and predicts valence and arousal values.

- **dataset.py**: Implements the `MultiModalDataset` class for loading and preprocessing multi-modal data.

### src/utils/

- **config.py**: Contains the `ModelConfig` class for model configuration and training parameters.

- **download_models.py**: Script for downloading required pre-trained model files (such as the dlib facial landmark detector).

### Main Functional Scripts

- **process_es2016a.py**: Processes ES2016a meeting videos and audios, extracts features using OpenFace and OpenSMILE, and saves them to JSON files.

- **train.py**: Script for training the MulT model, including data loading, model initialization, training loop, and validation.

- **analyze_es2016a.py**: Analyzes extracted features, generates emotion analysis reports, creates emotion charts, and summarizes statistics using the trained MulT model.

- **test_model_load.py**: Tests the model loading functionality, ensuring the model can be correctly loaded and used for inference.

## Installation and Setup

1. Clone this repository
```bash
git clone <repository-url>
cd AMI_Corpus
```

2. Create a virtual environment (optional but recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

3. Install dependencies
```bash
pip install -r src/requirements.txt
```

4. Download required models
```bash
python src/utils/download_models.py
```

## Detailed Workflow

### 1. Data Preparation

The project uses ES2016a meeting data from the AMI Corpus, including video and audio recordings. These files should be placed in the appropriate subdirectories under the `ES2016a/` directory. The `ES2016_video_audio_pair.xlsx` file defines the pairing relationship between audio and video files.

### 2. Feature Extraction

The feature extraction process extracts facial features from videos using OpenFace and acoustic features from audio using OpenSMILE.

```bash
python run_all.py extract
```

This command executes the following steps:
1. Loads pairing information from `ES2016_video_audio_pair.xlsx`
2. For each pair of video and audio files:
   - Extracts video features using OpenFace, including facial landmarks, face descriptors, etc.
   - Extracts audio features using OpenSMILE, such as spectral features, pitch, energy, etc.
3. Saves the extracted features to the `extracted_features/` directory

The extracted features are stored in JSON format files for subsequent processing.

### 3. Model Training

Model training uses the extracted features to train the MulT model.

```bash
python run_all.py train
```

This command executes the following steps:
1. Loads the extracted feature data
2. Splits the dataset into training and validation sets
3. Initializes the MulT model
4. Sets up the optimizer, learning rate scheduler, and loss function
5. Trains the model, including:
   - Training and validation for each epoch
   - Recording training and validation losses
   - Saving the best model
   - Implementing early stopping
6. Saves the trained model to the `MulT/MulT/checkpoints/` directory

### 4. Emotion Analysis

Emotion analysis uses the trained model to analyze the emotional states of participants in the ES2016a meeting.

```bash
python run_all.py analyze
```

This command executes the following steps:
1. Loads the trained MulT model
2. For each pair of video and audio features:
   - Uses the model for emotion prediction (generating valence and arousal values)
   - Creates emotion change charts
   - Calculates overall emotion statistics
3. Generates an emotion analysis report, including:
   - A summary table of participants' emotion analysis
   - Detailed analysis for each participant, including charts and text descriptions
4. Saves the results to the `analysis_results/` directory

### 5. Model Testing

Model testing ensures that the model can be correctly loaded and used for inference.

```bash
python run_all.py test_model
```

This command loads the trained model and tests inference using random input data to verify that the model functions correctly.

## Usage Examples

### Complete Workflow

For a complete workflow, execute the following commands in sequence:

```bash
# 1. Download required models
python src/utils/download_models.py

# 2. Extract features
python run_all.py extract

# 3. Train the model
python run_all.py train

# 4. Analyze emotions
python run_all.py analyze

# 5. Test the model (optional)
python run_all.py test_model
```

### Analyzing Existing Data Only

If you already have extracted features and a trained model, and only want to perform analysis:

```bash
python run_all.py analyze
```

## Interpreting Analysis Results

Analysis results are located in the `analysis_results/` directory, including:

1. **JSON files**: Containing raw emotion prediction data
2. **PNG images**: Showing changes in emotional valence and arousal over time, as well as emotion quadrant distribution
3. **Markdown report**: Generated emotion analysis report, including:
   - A summary table of all participants' emotions
   - Detailed emotion analysis for each participant
   - Emotion change charts

Emotions are represented by two dimensions:
- **Valence**: Indicates the positivity/negativity of the emotion (range: 0-1)
- **Arousal**: Indicates the activeness/calmness of the emotion (range: 0-1)

These two dimensions combine to form four emotion quadrants:
- **Positive Active**: High valence, high arousal (e.g., excited, happy)
- **Positive Passive**: High valence, low arousal (e.g., calm, satisfied)
- **Negative Active**: Low valence, high arousal (e.g., angry, anxious)
- **Negative Passive**: Low valence, low arousal (e.g., sad, tired)

## Troubleshooting

1. **Feature Extraction Failures**:
   - Ensure dlib model files have been downloaded to the correct location
   - Check if video/audio file paths are correct
   - Verify that OpenCV and dlib are installed correctly

2. **Model Training Errors**:
   - Ensure features have been extracted correctly
   - Check if GPU memory is sufficient (if applicable)
   - Reduce batch size or model dimensions

3. **Analysis Report Generation Issues**:
   - Ensure the model has been trained and saved
   - Verify that the `analysis_results` directory exists and is writable

4. **Dependency Package Installation Issues**:
   - For dlib installation issues, CMake and a C++ compiler may need to be installed first on some systems
   - For OpenCV issues, try `pip install opencv-python-headless` as an alternative