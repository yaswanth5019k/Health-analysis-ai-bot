# 📁 Model File Hosting Guide

Your `model.h5` file needs to be hosted online for deployment. Here are several options:

## 🎯 **Option 1: Google Drive (Recommended)**

1. **Upload your model.h5 to Google Drive**
2. **Get the sharing link**
3. **Convert to direct download link**
   - Replace: `https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing`
   - With: `https://drive.google.com/uc?export=download&id=YOUR_FILE_ID`

4. **Update the URL in `backend/download_model.py`**
   ```python
   MODEL_URL = 'https://drive.google.com/uc?export=download&id=YOUR_ACTUAL_FILE_ID'
   ```

## 🎯 **Option 2: GitHub Releases**

1. **Create a GitHub release**
2. **Upload model.h5 as an asset**
3. **Get the direct download URL**
4. **Update the URL in `backend/download_model.py`**

## 🎯 **Option 3: Dropbox**

1. **Upload model.h5 to Dropbox**
2. **Get the sharing link**
3. **Convert to direct download link**
   - Replace: `https://www.dropbox.com/s/...`
   - With: `https://www.dropbox.com/s/...?dl=1`

## 🎯 **Option 4: AWS S3 (Advanced)**

1. **Upload to S3 bucket**
2. **Make it publicly accessible**
3. **Use the S3 URL**

## 🔧 **Quick Setup for Google Drive:**

1. **Upload your model.h5 to Google Drive**
2. **Right-click → Share → Copy link**
3. **Extract the file ID from the URL**
4. **Update `backend/download_model.py`:**

```python
MODEL_URL = 'https://drive.google.com/uc?export=download&id=YOUR_ACTUAL_FILE_ID'
```

## ✅ **Test the Download:**

```bash
cd backend
python download_model.py
```

## 🚀 **For Deployment:**

Once you have the correct URL, the model will be automatically downloaded during deployment on Render.

## 📝 **Example:**

If your Google Drive link is:
`https://drive.google.com/file/d/1ABC123DEF456/view?usp=sharing`

Your download URL should be:
`https://drive.google.com/uc?export=download&id=1ABC123DEF456`

Update `backend/download_model.py` with this URL and commit the changes. 