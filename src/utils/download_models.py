import os
import sys
import requests
import shutil
from tqdm import tqdm

# Add src to the Python path if needed
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)

# Create models directory if it doesn't exist
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

MODELS = {
    'shape_predictor_68_face_landmarks.dat': {
        'url': 'https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2',
        'compressed': True
    },
    'dlib_face_recognition_resnet_model_v1.dat': {
        'url': 'https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2',
        'compressed': True
    }
}

def download_file(url, destination, desc=None):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get('content-length', 0))
    
    desc = desc or os.path.basename(destination)
    progress = tqdm(total=file_size, unit='B', unit_scale=True, desc=desc)
    
    with open(destination, 'wb') as f:
        for data in response.iter_content(chunk_size=1024):
            f.write(data)
            progress.update(len(data))
    progress.close()

def main():
    """Download required models"""
    print(f"Downloading models to {MODELS_DIR}...")
    
    for model_name, model_info in MODELS.items():
        model_path = os.path.join(MODELS_DIR, model_name)
        
        if os.path.exists(model_path):
            print(f"Model {model_name} already exists. Skipping...")
            continue
        
        print(f"Downloading {model_name}...")
        url = model_info['url']
        
        if model_info['compressed']:
            # For compressed files, download to a temporary file and decompress
            temp_path = model_path + '.bz2'
            download_file(url, temp_path, f"Downloading {model_name}")
            
            print(f"Decompressing {model_name}...")
            # Decompress using bzip2
            try:
                import bz2
                with open(model_path, 'wb') as out_file, bz2.BZ2File(temp_path, 'rb') as in_file:
                    shutil.copyfileobj(in_file, out_file)
                os.remove(temp_path)  # Remove the compressed file
            except Exception as e:
                print(f"Error decompressing {model_name}: {e}")
                print("Please manually decompress the file.")
        else:
            # For non-compressed files, download directly
            download_file(url, model_path, f"Downloading {model_name}")
    
    print("All models downloaded successfully!")
    print(f"Models are located in: {MODELS_DIR}")

if __name__ == "__main__":
    main() 