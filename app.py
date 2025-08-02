# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os


app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication


# Load pre-trained model
model = load_model("model.h5")

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
        processed_img = preprocess_image(img_path)
        prediction = model.predict(processed_img)
        class_idx = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
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
    app.run(debug=True)
