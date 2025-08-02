# 🩺 Medical AI Bot - Health Analysis Assistant

A Flask-based web application that provides AI-powered medical image analysis and diagnosis using machine learning.

## Features

- **Medical Image Analysis**: Upload X-ray, MRI, or CT scan images for AI diagnosis
- **Real-time Chat Interface**: Interactive web interface for easy image upload and analysis
- **Multiple Diagnosis Categories**: Supports Normal, Pneumonia, COVID-19, and Fracture detection
- **Confidence Scoring**: Provides confidence levels for each diagnosis
- **Demo Mode**: Works without a trained model for demonstration purposes

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd Health-analysis-ai-bot
   ```

2. **Activate the virtual environment**
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies** (if not already installed)
   ```bash
   pip install flask flask-cors tensorflow numpy
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your web browser**
   Navigate to: http://localhost:5001

## Usage

1. **Upload Medical Image**: Click "Choose File" and select a medical image (X-ray, MRI, CT scan)
2. **Get Diagnosis**: Click "Upload and Diagnose" to receive AI analysis
3. **View Results**: The AI will provide a diagnosis with confidence level

## API Endpoints

- `GET /` - Web interface
- `POST /predict` - Upload medical image for diagnosis
- `POST /chat` - Chat with the AI (basic implementation)

## Demo Mode

The application currently runs in demo mode, which means:
- Uses mock predictions instead of a real trained model
- Provides realistic-looking diagnosis results for demonstration
- Shows confidence levels for educational purposes

## Adding a Real Model

To use a real trained model:

1. Place your trained model file as `model.h5` in the project root
2. Ensure the model expects 224x224 pixel images
3. The model should output predictions for the classes: ["Normal", "Pneumonia", "COVID-19", "Fracture"]

## Technical Details

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **ML Framework**: TensorFlow/Keras
- **Image Processing**: PIL/Pillow
- **CORS**: Enabled for cross-origin requests

## Troubleshooting

- **Port 5000 in use**: The app automatically uses port 5001 to avoid conflicts with AirPlay Receiver
- **Model not found**: The app gracefully falls back to demo mode
- **Image upload issues**: Ensure the image is in a supported format (JPEG, PNG, etc.)

## License

This project is licensed under the MIT License - see the LICENSE file for details.