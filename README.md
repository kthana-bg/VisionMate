# VisionMate - AI Eye Strain Monitor & Ergonomic Coach

## 🚀 Setup Instructions

### Prerequisites
- Python 3.9+
- Webcam access
- Internet connection (for STUN servers)

### Installation Steps

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Run the Application**
```bash
streamlit run app.py
```

3. **Access the App**
- Local: `http://localhost:8501`
- The app will automatically download MediaPipe models on first run

## 🎯 Features

### 1. **Real-Time Monitoring**
- Live webcam feed with AI analysis
- Uses trained CNN and LSTM models (not just EAR rules)
- Real-time eye strain and posture detection

### 2. **6 AI Models**
#### Eye Strain Detection (3 models):
- **Custom CNN** (Default) - 98.99% accuracy
- **MobileNetV2** - 96% accuracy  
- **EfficientNetB0** - 95% accuracy

#### Posture Detection (3 models):
- **Custom LSTM** (Default) - Temporal sequence analysis
- **YOLOv8-Pose / MoveNet DNN** - Advanced landmark detection
- **MediaPipe Pose** - Rule-based geometry

### 3. **Model Comparison**
- Switch between different models in real-time
- Compare performance metrics
- View latency and accuracy statistics

### 4. **Analytics Dashboard**
- Track eye strain and posture over time
- Export data as CSV
- Visualize trends with Plotly charts

## 📱 Deployment to Streamlit Cloud

### Step 1: Prepare Repository
Ensure these files are in your GitHub repository:
- `app.py`
- `requirements.txt`
- `models/` folder with `.h5` files
- All utility modules

### Step 2: Streamlit Cloud Configuration
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Set main file: `app.py`
4. Python version: 3.9

### Step 3: Optional TURN Server (for better connectivity)
Add these secrets in Streamlit Cloud dashboard if you have a TURN server:
```toml
TURN_URL = "turn:your-turn-server.com:3478"
TURN_USER = "username"
TURN_CREDENTIAL = "password"
```

**Note**: The app works fine without TURN server using multiple STUN servers. TURN is only needed for restrictive corporate networks.

## 🔧 How It Works

### Model Integration
The system now uses **actual trained models** instead of simple EAR rules:

1. **Eye Strain Detection**:
   - Extracts 64x32px eye ROI from face landmarks
   - Runs through Custom CNN model
   - Outputs: NORMAL or STRAINED with confidence score

2. **Posture Detection**:
   - Extracts 10-point landmark feature vector
   - Runs through Custom LSTM model  
   - Outputs: GOOD or SLOUCHING with probability

3. **Live Feed**:
   - Uses WebRTC with multiple STUN servers
   - Processes every 5th frame for performance
   - Real-time inference on server-side

### Architecture
```
User's Browser (WebRTC) 
    ↓ 
Streamlit Server
    ↓
MediaPipe (Landmark Detection)
    ↓
Custom Models (CNN/LSTM)
    ↓
Results Display + Database Logging
```

## 🎮 Usage Guide

### 1. Registration
- Click camera to capture your face
- Enter your name
- Face embedding is stored for login

### 2. Login  
- Click camera to capture face
- System matches against stored embeddings
- Automatic login on match

### 3. Live Monitoring
- Click "Start" in webcam stream
- Allow camera permissions
- Watch real-time AI analysis
- System uses your selected models (default: Custom CNN + Custom LSTM)

### 4. Change Models
- Go to "Comparative Analysis" tab
- Select different eye/posture models
- Click "Apply Selection"
- Changes apply immediately to live monitoring

### 5. View Analytics
- Go to "Analytics" tab
- Select time range (1-72 hours)
- View strain patterns and trends
- Export data as CSV

## 🐛 Troubleshooting

### Issue: "Camera not working"
**Solution**: 
- Check browser permissions
- Use HTTPS (required for WebRTC)
- Try different browser (Chrome recommended)

### Issue: "Models not loading"
**Solution**:
- Ensure all `.h5` files are in `/models/` directory
- Check file names match exactly:
  - `models/eye_strain/custom_cnn.h5`
  - `models/eye_strain/mobilenetv2.h5`
  - `models/eye_strain/efficientnetb0.h5`
  - `models/posture/custom_lstm.h5`
  - `models/posture/yolo_movenet_dnn.h5`

### Issue: "Streamlit Cloud deployment fails"
**Solution**:
- Check `requirements.txt` has all packages
- Ensure Python version is 3.9
- Models must be committed to Git (< 100MB each)
- Use Git LFS for large model files if needed

### Issue: "Live feed is laggy"
**Solution**:
- Frame skipping is already enabled (every 5th frame)
- Reduce video resolution in `monitoring_tab.py`
- Close other heavy applications
- Check internet connection

## 📊 Model Performance

| Model | Task | Accuracy | Latency |
|-------|------|----------|---------|
| Custom CNN | Eye Strain | 98.99% | 83ms |
| MobileNetV2 | Eye Strain | 96% | 45ms |
| EfficientNetB0 | Eye Strain | 95% | 120ms |
| Custom LSTM | Posture | 95% | 50ms |
| YOLOv8-MoveNet | Posture | 94% | 85ms |
| MediaPipe | Posture | 92% | 5ms |

## 🔒 Privacy & Security

- All processing happens on server-side (Streamlit Cloud)
- Face embeddings stored in local SQLite database
- No data sent to third parties
- Webcam feed not recorded or stored

## 📝 Technical Stack

- **Frontend**: Streamlit
- **ML Framework**: TensorFlow 2.15 + Keras
- **Computer Vision**: MediaPipe, OpenCV
- **Models**: Custom CNN, LSTM, Transfer Learning
- **Real-time Streaming**: WebRTC (streamlit-webrtc)
- **Database**: SQLite
- **Deployment**: Streamlit Cloud

## 🎓 Academic Context

This project is part of Final Year Project (FYP) at Faculty of Artificial Intelligence and Cyber Security (FAIX), Universiti Teknikal Malaysia Melaka.

**Supervisor**: [Your Supervisor Name]
**Student**: [Your Name]
**Session**: 2024/2025

## 📄 License

Academic project - All rights reserved.

## 🙏 Acknowledgments

- MediaPipe team for landmark detection models
- Streamlit for the excellent web framework
- TensorFlow/Keras community

## 📧 Support

For issues or questions:
- Check the Troubleshooting section above
- Review project proposal document
- Contact project supervisor

---

**Last Updated**: January 2025
**Version**: 2.0 (With Trained Models Integration)
