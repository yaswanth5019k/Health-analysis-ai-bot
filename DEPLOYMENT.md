# 🚀 Medical AI Bot Deployment Guide

This guide will help you deploy your Medical AI Bot with frontend on Vercel and backend on Render.

## 📋 Prerequisites

- GitHub account with your repository
- Vercel account (free)
- Render account (free)
- Working Gemini API key

## 🎯 Deployment Steps

### **Step 1: Deploy Backend to Render**

1. **Go to Render Dashboard**
   - Visit [render.com](https://render.com)
   - Sign up/Login with your GitHub account

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select your repository: `Health-analysis-ai-bot`

3. **Configure the Service**
   - **Name**: `medical-ai-backend`
   - **Environment**: `Python`
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `cd backend && python app.py`
   - **Plan**: Free

4. **Add Environment Variables**
   - Click "Environment" tab
   - Add these variables:
     ```
     PYTHON_VERSION = 3.11.0
     GEMINI_API_KEY = AIzaSyBeFjOd6RdDmD83kmRKkMSLaOgvEx5yKfg
     ```

5. **Host Your Model File**
   - Follow the instructions in `MODEL_HOSTING.md`
   - Upload your `model.h5` to Google Drive or another hosting service
   - Update the `MODEL_URL` in `backend/download_model.py` with your actual URL

5. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment to complete
   - Copy your Render URL (e.g., `https://medical-ai-backend.onrender.com`)

### **Step 2: Update Frontend API URL**

1. **Update the API URL in frontend**
   - Open `frontend/index.html`
   - Find line with `API_BASE_URL`
   - Replace `'https://your-render-backend-url.onrender.com'` with your actual Render URL

2. **Commit and push changes**
   ```bash
   git add frontend/index.html
   git commit -m "Update API URL for deployment"
   git push origin main
   ```

### **Step 3: Deploy Frontend to Vercel**

1. **Go to Vercel Dashboard**
   - Visit [vercel.com](https://vercel.com)
   - Sign up/Login with your GitHub account

2. **Import Project**
   - Click "New Project"
   - Import your GitHub repository
   - Select `Health-analysis-ai-bot`

3. **Configure Project**
   - **Framework Preset**: Other
   - **Root Directory**: `./` (leave default)
   - **Build Command**: Leave empty
   - **Output Directory**: Leave empty

4. **Add Environment Variables**
   - Click "Environment Variables"
   - Add:
     ```
     REACT_APP_API_URL = https://your-render-backend-url.onrender.com
     ```

5. **Deploy**
   - Click "Deploy"
   - Wait for deployment to complete
   - Your frontend will be available at `https://your-project.vercel.app`

## 🔧 Configuration Files

### **vercel.json** (Frontend)
```json
{
  "version": 2,
  "builds": [
    {
      "src": "frontend/**/*",
      "use": "@vercel/static"
    }
  ],
  "routes": [
    {
      "src": "/",
      "dest": "/frontend/index.html"
    },
    {
      "src": "/styles.css",
      "dest": "/frontend/styles.css"
    },
    {
      "src": "/(.*)",
      "dest": "/frontend/$1"
    }
  ]
}
```

### **render.yaml** (Backend)
```yaml
services:
  - type: web
    name: medical-ai-backend
    env: python
    plan: free
    buildCommand: pip install -r backend/requirements.txt
    startCommand: cd backend && python app.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: GEMINI_API_KEY
        value: AIzaSyBeFjOd6RdDmD83kmRKkMSLaOgvEx5yKfg
    healthCheckPath: /health
    autoDeploy: true
```

## 🧪 Testing Your Deployment

1. **Test Backend Health**
   - Visit: `https://your-render-backend-url.onrender.com/health`
   - Should return: `{"status": "healthy", "message": "Medical AI Bot is running!"}`

2. **Test Frontend**
   - Visit your Vercel URL
   - Try uploading an image
   - Try the chat functionality

## 🔍 Troubleshooting

### **Backend Issues**
- Check Render logs for errors
- Verify environment variables are set
- Ensure `requirements.txt` is in the backend folder

### **Frontend Issues**
- Check browser console for errors
- Verify API URL is correct
- Test API endpoints directly

### **CORS Issues**
- Backend has CORS enabled for all origins
- If issues persist, check Render logs

## 📞 Support

If you encounter issues:
1. Check the logs in Render/Vercel dashboards
2. Verify all environment variables are set
3. Test API endpoints individually
4. Check browser console for frontend errors

## 🎉 Success!

Once deployed, your Medical AI Bot will be available at:
- **Frontend**: `https://your-project.vercel.app`
- **Backend**: `https://your-render-backend-url.onrender.com`

Both image diagnosis and AI chat will work seamlessly! 🚀 