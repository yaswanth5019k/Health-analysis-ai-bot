# 🎨 Medical AI Frontend

This directory contains the frontend application for the Medical AI Diagnostic Assistant.

## 📁 Structure

- `index.html` - Main application interface
- `package.json` - Frontend dependencies and scripts
- `README.md` - This file

## 🚀 Features

- **Modern UI**: Beautiful, responsive design with gradient backgrounds
- **Drag & Drop**: Easy image upload functionality
- **Real-time AI**: Live predictions from the trained model
- **Chat Interface**: Conversational AI responses
- **Mobile Friendly**: Works on all devices

## 🛠️ Development

### Local Development
```bash
# Start a simple HTTP server
python -m http.server 3000

# Or use npm
npm start
```

### Building for Production
```bash
npm run build
```

## 🌐 Deployment

This frontend can be deployed to:
- **Vercel** (recommended for static sites)
- **Netlify**
- **GitHub Pages**
- **Any static hosting service**

## 🔗 API Integration

The frontend communicates with the backend API at:
- `POST /predict` - Image analysis endpoint
- `POST /chat` - Chat functionality endpoint

## 📱 Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge 