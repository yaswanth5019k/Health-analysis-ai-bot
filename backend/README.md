# 🧠 Medical AI Backend

This directory contains the backend API server for the Medical AI Diagnostic Assistant.

## 📁 Structure

- `app.py` - Main Flask application
- `train_model.py` - Model training script
- `create_small_dataset.py` - Dataset preprocessing utilities
- `model.h5` - Trained ResNet50 model
- `best_model.h5` - Best model during training
- `model_config.json` - Model configuration and results
- `training_history.png` - Training visualization
- `requirements.txt` - Python dependencies
- `README.md` - This file

## 🚀 Features

- **Flask API**: RESTful endpoints for image analysis
- **TensorFlow Model**: Pre-trained ResNet50 for medical image classification
- **CORS Support**: Cross-origin requests enabled
- **File Upload**: Secure image upload and processing
- **Real-time Predictions**: Fast AI inference

## 🛠️ Setup

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Server
```bash
# Start the Flask development server
python app.py

# Server will be available at http://localhost:5001
```

## 🔌 API Endpoints

### POST /predict
Upload and analyze medical images.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `image` file

**Response:**
```json
{
  "response": "Based on the uploaded image, the AI suggests: COVID-19 (Confidence: 85.23%).",
  "diagnosis": "COVID-19",
  "confidence": "85.23%"
}
```

### POST /chat
Chat with the AI assistant.

**Request:**
```json
{
  "message": "Hello, can you help me?"
}
```

**Response:**
```json
{
  "response": "You said: Hello, can you help me?. Chat features coming soon!"
}
```

### GET /
Serves the frontend application.

## 🎯 Model Information

- **Architecture**: ResNet50 with transfer learning
- **Classes**: COVID-19, Normal, Pneumonia
- **Input Size**: 224x224 pixels
- **Accuracy**: 73.33% validation accuracy
- **Dataset**: 300 images (100 per class)

## 🌐 Deployment Options

### Heroku
```bash
# Create Procfile
echo "web: python app.py" > Procfile

# Deploy
git add .
git commit -m "Deploy to Heroku"
heroku create your-app-name
git push heroku main
```

### Railway
```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway login
railway init
railway up
```

### DigitalOcean App Platform
- Connect your GitHub repository
- Select Python runtime
- Set build command: `pip install -r requirements.txt`
- Set run command: `python app.py`

### AWS/GCP/Azure
- Use container deployment
- Create Dockerfile for containerization
- Deploy to cloud services

## 🔧 Environment Variables

- `PORT` - Server port (default: 5001)
- `FLASK_ENV` - Development/production mode
- `MODEL_PATH` - Path to model file

## 📊 Model Training

To retrain the model:
```bash
python train_model.py --dataset tiny --epochs 10
```

## 🔒 Security Notes

- Enable HTTPS in production
- Add authentication if needed
- Validate uploaded files
- Rate limiting recommended 