#!/bin/bash

# Medical AI Bot Startup Script

echo "🚀 Starting Medical AI Bot..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Navigate to backend
cd backend

# Check if requirements are installed
if ! python -c "import flask" 2>/dev/null; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
fi

# Start the server
echo "🌐 Starting Flask server..."
echo "📱 Frontend will be available at: http://localhost:5001"
echo "🔧 API endpoints:"
echo "   - POST /predict - Upload medical images for diagnosis"
echo "   - POST /chat - Chat with the AI"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py 