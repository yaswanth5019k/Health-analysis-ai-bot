# 🩺 Medical AI Bot - Health Analysis Assistant

A comprehensive medical image analysis application with separate frontend and backend components, powered by AI for medical diagnosis.

## 📁 Project Structure

```
Health-analysis-ai-bot/
├── frontend/                 # 🎨 Frontend Application
│   ├── index.html           # Main web interface
│   ├── package.json         # Frontend dependencies
│   └── README.md           # Frontend documentation
├── backend/                  # 🧠 Backend API Server
│   ├── app.py              # Flask application
│   ├── train_model.py      # Model training script
│   ├── create_small_dataset.py # Data preprocessing
│   ├── model.h5            # Trained AI model
│   ├── requirements.txt    # Python dependencies
│   └── README.md          # Backend documentation
├── venv/                    # Python virtual environment
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🚀 Quick Start

### Option 1: Run Everything Together (Recommended)
```bash
# Navigate to backend directory
cd backend

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py

# Open http://localhost:5001 in your browser
```

### Option 2: Run Frontend and Backend Separately
```bash
# Terminal 1 - Backend
cd backend
source venv/bin/activate
pip install -r requirements.txt
python app.py

# Terminal 2 - Frontend (optional, for development)
cd frontend
python -m http.server 3000
# Open http://localhost:3000 in your browser
```

## 🎯 Features

### Frontend Features
- **Modern UI**: Beautiful, responsive design
- **Drag & Drop**: Easy image upload
- **Real-time AI**: Live predictions
- **Chat Interface**: Conversational AI
- **Mobile Friendly**: Works on all devices

### Backend Features
- **Flask API**: RESTful endpoints
- **TensorFlow Model**: ResNet50 for image classification
- **CORS Support**: Cross-origin requests
- **File Upload**: Secure image processing
- **Real-time Predictions**: Fast AI inference

## 🔌 API Endpoints

- `GET /` - Serves the frontend application
- `POST /predict` - Upload and analyze medical images
- `POST /chat` - Chat with AI assistant

## 🎯 AI Model

- **Architecture**: ResNet50 with transfer learning
- **Classes**: COVID-19, Normal, Pneumonia
- **Accuracy**: 73.33% validation accuracy
- **Dataset**: 300 medical images (100 per class)

## 🌐 Deployment Options

### Frontend (Static Site)
- **Vercel** (recommended)
- **Netlify**
- **GitHub Pages**
- **Any static hosting**

### Backend (API Server)
- **Heroku**
- **Railway**
- **DigitalOcean App Platform**
- **AWS/GCP/Azure**

## 📊 Model Training

To retrain the model with your own data:
```bash
cd backend
python train_model.py --dataset combined --epochs 20
```

## 🔧 Development

### Frontend Development
```bash
cd frontend
# Edit index.html for UI changes
# Use any static file server for development
python -m http.server 3000
```

### Backend Development
```bash
cd backend
source venv/bin/activate
# Edit app.py for API changes
python app.py
```

## 📝 Documentation

- [Frontend Documentation](./frontend/README.md)
- [Backend Documentation](./backend/README.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## ⚠️ Disclaimer

This application is for educational and demonstration purposes only. It should not be used for actual medical diagnosis without proper validation and certification.