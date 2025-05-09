import os
import urllib.request
import bz2
import shutil

def download_file(url, output_path):
    """Download a file from URL to output_path"""
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded to {output_path}")

def extract_bz2(bz2_path, output_path):
    """Extract a .bz2 file to output_path"""
    print(f"Extracting {bz2_path}...")
    with bz2.open(bz2_path, 'rb') as source, open(output_path, 'wb') as dest:
        shutil.copyfileobj(source, dest)
    print(f"Extracted to {output_path}")
    # Remove the .bz2 file
    os.remove(bz2_path)

def main():
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Download and extract facial landmark predictor
    landmark_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    landmark_bz2 = os.path.join(models_dir, "shape_predictor_68_face_landmarks.dat.bz2")
    landmark_dat = os.path.join(models_dir, "shape_predictor_68_face_landmarks.dat")
    
    if not os.path.exists(landmark_dat):
        download_file(landmark_url, landmark_bz2)
        extract_bz2(landmark_bz2, landmark_dat)
    
    # Download and extract face recognition model
    face_rec_url = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
    face_rec_bz2 = os.path.join(models_dir, "dlib_face_recognition_resnet_model_v1.dat.bz2")
    face_rec_dat = os.path.join(models_dir, "dlib_face_recognition_resnet_model_v1.dat")
    
    if not os.path.exists(face_rec_dat):
        download_file(face_rec_url, face_rec_bz2)
        extract_bz2(face_rec_bz2, face_rec_dat)
    
    print("\nAll model files downloaded and extracted successfully!")

if __name__ == '__main__':
    main() 