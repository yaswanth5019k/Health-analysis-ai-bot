# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import random


app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Serve static files
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


# Load pre-trained model or use mock model
model = None
try:
    model = load_model("model.h5")
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("⚠️  Model file 'model.h5' not found. Using mock predictions for demonstration.")
except Exception as e:
    print(f"⚠️  Error loading model: {e}. Using mock predictions for demonstration.")

# Define class labels
labels = ["Normal", "Pneumonia", "COVID-19", "Fracture"]  # Example classes

def preprocess_image(img_path):
    """
    Preprocess the uploaded image to match model input requirements.
    """
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust size to your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize if model was trained on normalized images
    return img_array

def mock_predict():
    """
    Generate a mock prediction for demonstration purposes.
    """
    # Simulate model prediction with random results
    prediction = np.random.rand(len(labels))
    prediction = prediction / np.sum(prediction)  # Normalize to probabilities
    class_idx = np.argmax(prediction)
    confidence = float(np.max(prediction))
    return class_idx, confidence

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to receive an image and return AI-generated diagnosis in a conversational format.
    """
    if "image" not in request.files:
        return jsonify({"response": "Please upload a medical image (X-ray, MRI, CT scan) to get a diagnosis."}), 400
    file = request.files["image"]
    img_path = os.path.join("uploads", file.filename)
    file.save(img_path)

    try:
        if model is not None:
            # Use real model prediction
            processed_img = preprocess_image(img_path)
            prediction = model.predict(processed_img)
            class_idx = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))
        else:
            # Use mock prediction
            class_idx, confidence = mock_predict()
        
        diagnosis = labels[class_idx]

        # Conversational response
        response = {
            "response": f"Based on the uploaded image, the AI suggests: {diagnosis} (Confidence: {confidence * 100:.2f}%).",
            "diagnosis": diagnosis,
            "confidence": f"{confidence * 100:.2f}%"
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"response": f"An error occurred: {str(e)}"}), 500


# Placeholder for future chatbot/conversation features
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    # For now, just echo the message. Extend this for real chat features.
    return jsonify({"response": f"You said: {user_message}. Chat features coming soon!"})

if __name__ == "__main__":
    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    # Run the Flask app
    print("🚀 Starting Medical AI Bot...")
    print("📱 Frontend will be available at: http://localhost:5001")
    print("🔧 API endpoints:")
    print("   - POST /predict - Upload medical images for diagnosis")
    print("   - POST /chat - Chat with the AI (coming soon)")
    app.run(debug=True, host='0.0.0.0', port=5001)
