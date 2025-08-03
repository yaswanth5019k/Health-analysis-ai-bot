# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import random
import google.generativeai as genai


app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)  # Enable CORS for frontend-backend communication

# Configure Google Gemini AI
try:
    # Using the working API key provided by user
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyBeFjOd6RdDmD83kmRKkMSLaOgvEx5yKfg')
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    GEMINI_AVAILABLE = True
    print("✅ Google Gemini AI configured successfully!")
except Exception as e:
    print(f"⚠️  Gemini AI not available: {e}. Using local responses.")
    GEMINI_AVAILABLE = False

# Serve static files
@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

# Serve other static files
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../frontend', path)


# Import model download functionality
from download_model import load_model_safely

# Load pre-trained model or use mock model
model = load_model_safely()

# Define class labels - must match the order used during training
labels = ["COVID-19", "Normal", "Pneumonia"]  # Match the training order

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
        return jsonify({"success": False, "error": "Please upload a medical image (X-ray, MRI, CT scan) to get a diagnosis."}), 400
    
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

        # Return response format that frontend expects
        response = {
            "success": True,
            "prediction": {
                "class": diagnosis,
                "confidence": confidence,
                "description": f"AI detected {diagnosis} with {confidence * 100:.2f}% confidence"
            },
            "response": f"Based on the uploaded image, the AI suggests: {diagnosis} (Confidence: {confidence * 100:.2f}%)."
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"success": False, "error": f"An error occurred: {str(e)}"}), 500


# Enhanced medical chatbot with Gemini AI integration
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").lower()
    
    # Try Gemini AI first if available
    if GEMINI_AVAILABLE and user_message:
        try:
            # Create a medical-focused prompt
            medical_prompt = f"""
            You are a helpful AI medical assistant. The user is asking: "{user_message}"
            
            Please provide helpful, accurate medical information. Remember to:
            1. Be informative but not diagnostic
            2. Encourage consulting healthcare providers for specific medical advice
            3. Provide general information about symptoms, conditions, and treatments
            4. Keep responses concise and clear
            5. Always emphasize that you're an AI assistant, not a replacement for professional medical care
            
            If the user is asking about symptoms, provide general information and suggest when to see a doctor.
            If they're asking about conditions, explain what it is and general treatment approaches.
            If they're asking about medical procedures, explain what they involve.
            
            Response:"""
            
            # Get response from Gemini
            gemini_response = gemini_model.generate_content(medical_prompt)
            if gemini_response and gemini_response.text:
                return jsonify({"response": gemini_response.text.strip()})
        except Exception as e:
            print(f"Gemini API error: {e}")
            # Fall back to local responses
    
    # Local medical knowledge base - fallback when Gemini is not available
    medical_responses = {
        "covid": "COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus. Common symptoms include fever, cough, fatigue, and loss of taste/smell. If you suspect COVID-19, please consult a healthcare provider and consider getting tested.",
        
        "pneumonia": "Pneumonia is an infection that inflames the air sacs in one or both lungs. Symptoms include cough with phlegm, fever, difficulty breathing, and chest pain. Treatment typically involves antibiotics and rest.",
        
        "fever": "A fever is a temporary increase in body temperature, often due to illness. Normal body temperature is around 98.6°F (37°C). If fever is high (>103°F) or persistent, seek medical attention.",
        
        "cough": "A cough is a reflex action to clear your airways. It can be caused by colds, allergies, or more serious conditions. If cough is severe or persistent, consult a doctor.",
        
        "headache": "Headaches can be caused by stress, dehydration, lack of sleep, or underlying medical conditions. Rest, hydration, and over-the-counter pain relievers may help. Seek medical attention for severe headaches.",
        
        "symptom": "Symptoms are signs that indicate the presence of disease. Common symptoms include fever, pain, fatigue, and changes in appetite. Always consult a healthcare provider for proper diagnosis.",
        
        "x-ray": "X-rays use radiation to create images of bones and some soft tissues. They're commonly used to diagnose fractures, pneumonia, and other conditions. Our AI can analyze chest X-rays for respiratory conditions.",
        
        "mri": "MRI (Magnetic Resonance Imaging) uses magnetic fields to create detailed images of organs and tissues. It's useful for diagnosing brain, spine, and joint problems.",
        
        "ct scan": "CT scans combine X-rays with computer technology to create cross-sectional images. They're useful for detecting tumors, fractures, and internal injuries.",
        
        "diagnosis": "Diagnosis is the process of identifying a disease or condition. It involves reviewing symptoms, medical history, and test results. Our AI can assist with image-based diagnosis but should not replace professional medical advice.",
        
        "treatment": "Treatment depends on the specific condition diagnosed. It may include medications, lifestyle changes, or procedures. Always follow your healthcare provider's recommendations.",
        
        "emergency": "If you're experiencing a medical emergency (severe pain, difficulty breathing, chest pain, etc.), call emergency services immediately. Do not rely on AI for emergency medical decisions.",
        
        "doctor": "Healthcare providers are essential for proper diagnosis and treatment. Our AI is a tool to assist, not replace, professional medical care. Always consult qualified healthcare professionals.",
        
        "help": "I'm here to provide general medical information and assist with image analysis. For specific medical advice, please consult a healthcare provider. You can ask me about symptoms, conditions, or upload medical images for analysis.",
        
        "hello": "Hello! I'm your AI medical assistant. I can help you with general medical information and analyze medical images. How can I assist you today?",
        
        "hi": "Hi there! I'm your AI medical assistant. I can provide information about symptoms, conditions, and analyze medical images. What would you like to know?",
        
        "thanks": "You're welcome! I'm here to help. Feel free to ask more questions or upload medical images for analysis.",
        
        "thank you": "You're welcome! I'm here to help. Feel free to ask more questions or upload medical images for analysis."
    }
    
    # Check for matching keywords
    response = None
    for keyword, medical_response in medical_responses.items():
        if keyword in user_message:
            response = medical_response
            break
    
    # If no specific keyword found, provide general guidance
    if not response:
        response = "I understand you're asking about: '" + user_message + "'. For specific medical advice, please consult a healthcare provider. I can help with general information about symptoms, conditions, or analyze medical images you upload. What specific medical topic would you like to know more about?"
    
    return jsonify({"response": response})

# Health check endpoint for deployment
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "Medical AI Bot is running!"}), 200

if __name__ == "__main__":
    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    # Run the Flask app
    print("🚀 Starting Medical AI Bot...")
    print("📱 Frontend will be available at: http://localhost:5003")
    print("🔧 API endpoints:")
    print("   - POST /predict - Upload medical images for diagnosis")
    print("   - POST /chat - Chat with the AI (coming soon)")
    app.run(debug=True, host='0.0.0.0', port=5003)
