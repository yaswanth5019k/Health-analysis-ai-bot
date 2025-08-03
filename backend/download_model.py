import os
import requests
from tensorflow.keras.models import load_model

# You can host your model.h5 file on Google Drive, Dropbox, or any file hosting service
# For now, we'll use a placeholder URL - you need to replace this with your actual model URL
MODEL_URL = 'https://drive.google.com/uc?export=download&id=YOUR_MODEL_FILE_ID'  # Replace with your actual URL
MODEL_PATH = 'model.h5'

def download_model():
    """Download the model file if it doesn't exist"""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model from {MODEL_URL}...")
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Model downloaded successfully.")
            return True
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            print("⚠️  Using mock predictions for demonstration.")
            return False
    else:
        print("✅ Model file already exists.")
        return True

def load_model_safely():
    """Load the model safely with fallback to mock predictions"""
    if download_model():
        try:
            model = load_model(MODEL_PATH)
            print("✅ Model loaded successfully.")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("⚠️  Using mock predictions for demonstration.")
            return None
    else:
        print("⚠️  Using mock predictions for demonstration.")
        return None

if __name__ == "__main__":
    load_model_safely() 