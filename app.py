import streamlit as st
import cv2
import numpy as np
from detector import EyeStrainDetector

st.set_page_config(page_title="VisionMate AI", layout="wide")

# Custom CSS for that violet aesthetic
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #E0B0FF; }
    .metric-card {
        background: rgba(187, 134, 252, 0.1);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #BB86FC;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("💜 VisionMate Live AI")

# Initialize Detector
@st.cache_resource
def get_detector():
    return EyeStrainDetector()

detector = get_detector()

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    run = st.checkbox("Start Camera", value=True)
    threshold = st.slider("EAR Threshold", 0.10, 0.30, 0.20)

# Layout
col1, col2 = st.columns([2, 1])
with col1:
    view = st.image([])
with col2:
    st.markdown("### Session Stats")
    ear_text = st.empty()
    blink_text = st.empty()

# Camera Loop
if run:
    # Use index 0 for local testing; Streamlit Cloud handles the permissions
    cap = cv2.VideoCapture(0)
    blink_count = 0
    blink_active = False

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        frame = cv2.flip(frame, 1)
        ear, _ = detector.process_frame(frame)

        # Logic
        if ear > 0:
            if ear < threshold:
                blink_active = True
            elif blink_active:
                blink_count += 1
                blink_active = False

        # Update UI
        ear_text.markdown(f"<div class='metric-card'>EAR: {ear:.3f}</div>", unsafe_allow_html=True)
        blink_text.markdown(f"<div class='metric-card'>Blinks: {blink_count}</div>", unsafe_allow_html=True)
        
        # Display Video
        view.image(frame, channels="BGR")

    cap.release()