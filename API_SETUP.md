# 🚀 Free API Integration Guide for Medical AI Bot

## 📋 **Available Free APIs**

### **1. Google Gemini AI (Recommended)**
- **Free Tier**: 15 requests/minute, 1500 requests/day
- **Cost**: Completely free for basic usage
- **Best for**: Medical conversations, symptom analysis

### **2. OpenAI GPT-3.5**
- **Free Tier**: 3 requests/minute, 200 requests/day
- **Cost**: $0.002 per 1K tokens after free tier
- **Best for**: Advanced medical conversations

### **3. Hugging Face Inference API**
- **Free Tier**: 30,000 requests/month
- **Cost**: Free for basic models
- **Best for**: Medical text analysis

## 🔧 **Setup Instructions**

### **Option 1: Google Gemini (Recommended)**

1. **Get Free API Key**:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Sign in with your Google account
   - Click "Create API Key"
   - Copy the API key

2. **Install Dependencies**:
   ```bash
   cd backend
   pip install google-generativeai==0.3.2
   ```

3. **Set Environment Variable**:
   ```bash
   # On macOS/Linux
   export GEMINI_API_KEY="your-api-key-here"
   
   # Or create a .env file in backend directory
   echo "GEMINI_API_KEY=your-api-key-here" > .env
   ```

4. **Test the Integration**:
   - Start your server: `python app.py`
   - Go to `http://localhost:5003`
   - Type medical questions in the chat box

### **Option 2: OpenAI GPT-3.5**

1. **Get Free API Key**:
   - Go to [OpenAI Platform](https://platform.openai.com/api-keys)
   - Sign up for free account
   - Create API key

2. **Install Dependencies**:
   ```bash
   pip install openai==1.3.0
   ```

3. **Set Environment Variable**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

### **Option 3: Hugging Face**

1. **Get Free API Key**:
   - Go to [Hugging Face](https://huggingface.co/settings/tokens)
   - Create account
   - Generate access token

2. **Install Dependencies**:
   ```bash
   pip install requests
   ```

## 🎯 **How It Works**

### **With API Integration**:
1. User types medical question in chat box
2. Question sent to AI API (Gemini/OpenAI)
3. AI provides intelligent medical response
4. Response displayed in results section

### **Without API (Fallback)**:
1. User types medical question
2. Local keyword matching finds relevant response
3. Pre-written medical information displayed

## 💡 **Example Questions to Test**

- "What are the symptoms of diabetes?"
- "How do I know if I have a fever?"
- "What is pneumonia?"
- "When should I see a doctor for a headache?"
- "What is an MRI scan?"
- "How to treat a common cold?"

## 🔒 **Security & Privacy**

- **No medical data stored**: All conversations are processed in real-time
- **No personal information**: API calls don't include personal details
- **Medical disclaimer**: Always emphasizes consulting healthcare providers
- **Local fallback**: Works even without internet/API access

## 🚨 **Important Notes**

1. **Not for Emergency Use**: Always call emergency services for urgent medical situations
2. **Not a Replacement**: AI responses are informational, not diagnostic
3. **Consult Professionals**: Always consult healthcare providers for medical advice
4. **Free Tier Limits**: Be aware of API rate limits

## 🛠️ **Troubleshooting**

### **API Key Issues**:
- Check if API key is correctly set
- Verify API key has proper permissions
- Check API service status

### **Rate Limiting**:
- Free tiers have request limits
- System automatically falls back to local responses
- Consider upgrading for higher limits

### **Network Issues**:
- System works offline with local responses
- Check internet connection for API features
- Restart server if needed

## 📞 **Support**

- **Google Gemini**: [AI Studio Help](https://ai.google.dev/docs)
- **OpenAI**: [Platform Documentation](https://platform.openai.com/docs)
- **Hugging Face**: [API Documentation](https://huggingface.co/docs/api-inference)

---

**🎉 Your Medical AI Bot now has intelligent conversation capabilities!** 