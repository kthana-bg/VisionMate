import streamlit as st
import cv2
import numpy as np
from detector import EyeStrainDetector
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import threading
import time
import os

st.set_page_config(
    page_title="VisionMate",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "ear_history" not in st.session_state:
    st.session_state.ear_history = [0.25] * 40
if "blink_count" not in st.session_state:
    st.session_state.blink_count = 0

st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        height: 100vh;
        overflow: hidden;
    }
    .stApp {
        background: linear-gradient(rgba(26, 26, 46, 0.9), rgba(26, 26, 46, 0.9)), 
                    url("https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
    }
    [data-testid="column"] > div {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(25px);
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding: 20px !important;
        height: 82vh;
    }
    .metric-value { 
        font-size: 40px; 
        color: #BB86FC; 
        text-shadow: 0 0 10px rgba(187, 134, 252, 0.5);
        text-align: center;
        font-weight: bold;
    }
    .metric-label {
        font-size: 11px;
        opacity: 0.7;
        text-align: center;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    button[kind="secondary"], footer, [data-testid="stSidebar"] { display: none !important; }
    </style>
""", unsafe_allow_html=True)

GLOBAL_LOCK = threading.Lock()
AI_DATA = {"ear": 0.0, "blinks": 0, "status": "Initializing"}

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.detector = EyeStrainDetector()
        self.threshold = 0.20

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        ear, _, _ = self.detector.process_frame(img)
        
        with GLOBAL_LOCK:
            if ear > 0:
                blinks, _ = self.detector.update_blink_state(ear, self.threshold)
                AI_DATA["ear"] = ear
                AI_DATA["blinks"] = blinks
                AI_DATA["status"] = "HIGH STRAIN" if ear < self.threshold else "OPTIMAL"
                color = (75, 75, 255) if ear < self.threshold else (50, 255, 150)
            else:
                AI_DATA["ear"] = 0.0
                AI_DATA["status"] = "NO FACE"
                color = (128, 128, 128)
            cv2.rectangle(img, (0, 0), (w, h), color, 4)
        return img

st.markdown("<h2 style='text-align: center; color: #E0B0FF; margin:0;'>VISIONMATE</h2>", unsafe_allow_html=True)

col1, col2 = st.columns([1.7, 1])

with col1:
    webrtc_streamer(
        key="visionmate-v4",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
        desired_playing_state=True,
        video_html_attrs={"style": {"width": "100%", "border-radius": "15px"}, "controls": False, "autoPlay": True}
    )

with col2:
    st.markdown('<p class="metric-label">Blink Count</p>', unsafe_allow_html=True)
    blink_place = st.empty()
    st.markdown('<p class="metric-label">Current EAR</p>', unsafe_allow_html=True)
    ear_place = st.empty()
    status_place = st.empty()
    chart_place = st.empty()
    coach_place = st.empty()

while True:
    with GLOBAL_LOCK:
        c_ear = AI_DATA["ear"]
        c_blinks = AI_DATA["blinks"]
        c_status = AI_DATA["status"]

    st.session_state.ear_history = st.session_state.ear_history[1:] + [c_ear]
    
    blink_place.markdown(f'<div class="metric-value">{c_blinks}</div>', unsafe_allow_html=True)
    ear_place.markdown(f'<div class="metric-value" style="font-size:30px;">{c_ear:.3f}</div>', unsafe_allow_html=True)
    
    if c_status == "HIGH STRAIN":
        status_place.error("HIGH STRAIN")
        coach_place.warning("Take a break!")
    elif c_status == "OPTIMAL":
        status_place.success("OPTIMAL")
        coach_place.empty()
    else:
        status_place.info("Searching...")
        
    chart_place.line_chart(st.session_state.ear_history, height=150)
    time.sleep(0.1)
