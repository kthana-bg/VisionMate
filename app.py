import streamlit as st
import cv2
import numpy as np
from detector import EyeStrainDetector
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import threading
import time
import os

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

st.set_page_config(page_title="VisionMate", layout="wide", initial_sidebar_state="collapsed")

def init_session_state():
    if "ear_history" not in st.session_state:
        st.session_state.ear_history = [0.25] * 20
    if "metrics" not in st.session_state:
        st.session_state.metrics = {"ear": 0.0, "blinks": 0, "status": "Initializing"}

init_session_state()

st.markdown("""
<style>
html, body, .stApp {
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}
.main .block-container {
    max-width: 100% !important;
    padding: 0.5rem 1rem !important;
    height: 100vh !important;
    overflow: hidden !important;
}
[data-testid="stVerticalBlock"] {
    gap: 0.5rem !important;
}
h1 {
    font-size: 1.5rem !important;
    margin: 0 !important;
    padding: 0 !important;
    color: #E0B0FF !important;
    text-align: center !important;
}
h2, h3 {
    font-size: 1rem !important;
    margin: 0.2rem 0 !important;
    color: #E0B0FF !important;
}
p {
    margin: 0 !important;
    font-size: 0.8rem !important;
    color: #B0B0B0 !important;
    text-align: center !important;
}
[data-testid="column"] > div {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding: 0.8rem !important;
    height: calc(100vh - 2rem) !important;
    overflow: hidden !important;
}
.metric-value { 
    font-size: 2rem !important;
    color: #BB86FC !important;
    text-align: center !important;
    font-weight: bold !important;
    line-height: 1 !important;
}
.metric-label {
    font-size: 0.6rem !important;
    opacity: 0.7 !important;
    text-align: center !important;
    text-transform: uppercase !important;
    margin-bottom: 0.2rem !important;
}
.status-optimal { color: #00E676 !important; }
.status-warning { color: #FFD600 !important; }
.status-danger { color: #FF1744 !important; }
.stElementContainer {
    margin: 0 !important;
    padding: 0 !important;
}
video {
    width: 100% !important;
    height: auto !important;
    max-height: 50vh !important;
    border-radius: 8px !important;
}
.stSlider {
    padding: 0 !important;
}
.stSlider > div {
    padding: 0 !important;
}
.css-1cpxqw2, .css-1x8cf1d, .css-1q1n0qu {
    padding: 0 !important;
}
footer, .stDeployButton, .viewerBadge_container__1QSob, .stSpinner {
    display: none !important;
}
button[kind="secondary"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

def get_ice_servers():
    account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
    auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
    if account_sid and auth_token and TWILIO_AVAILABLE:
        try:
            client = Client(account_sid, auth_token)
            token = client.tokens.create()
            return token.ice_servers
        except:
            pass
    return [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["turn:openrelay.metered.ca:80"], "username": "openrelayproject", "credential": "openrelayproject"},
        {"urls": ["turn:openrelay.metered.ca:443"], "username": "openrelayproject", "credential": "openrelayproject"}
    ]

col1, col2 = st.columns([2, 1])

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.detector = EyeStrainDetector()
        self.threshold = 0.20
        self.blink_count = 0
        self.current_ear = 0.0
        self.status = "Initializing"
        self.last_update = time.time()
        
    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            
            ear, landmarks, _ = self.detector.process_frame(img)
            
            if ear > 0:
                current_blinks, _ = self.detector.update_blink_state(ear, self.threshold)
                self.blink_count = current_blinks
                self.current_ear = ear
                if ear < self.threshold:
                    self.status = "HIGH STRAIN"
                else:
                    self.status = "OPTIMAL"
            else:
                self.status = "NO FACE"
                self.current_ear = 0.0
            
            if time.time() - self.last_update > 0.5:
                st.session_state.metrics = {
                    "ear": self.current_ear,
                    "blinks": self.blink_count,
                    "status": self.status
                }
                if self.current_ear > 0:
                    st.session_state.ear_history = st.session_state.ear_history[1:] + [self.current_ear]
                self.last_update = time.time()
            
            if self.status == "HIGH STRAIN":
                color = (255, 50, 50)
            elif self.status == "OPTIMAL":
                color = (50, 255, 150)
            else:
                color = (128, 128, 128)
            
            cv2.rectangle(img, (0, 0), (w-1, h-1), color, 2)
            
            return img
            
        except Exception as e:
            return frame.to_ndarray(format="bgr24")

with col2:
    st.markdown('<p class="metric-label">Current EAR</p>', unsafe_allow_html=True)
    ear_placeholder = st.empty()
    st.markdown('<p class="metric-label">Total Blinks</p>', unsafe_allow_html=True)
    blink_placeholder = st.empty()
    st.markdown('<p class="metric-label">Status</p>', unsafe_allow_html=True)
    status_placeholder = st.empty()
    st.markdown("---")
    st.markdown('<p class="metric-label">EAR History</p>', unsafe_allow_html=True)
    chart_placeholder = st.empty()
    st.markdown("---")
    st.subheader("Coach")
    coach_placeholder = st.empty()
    
    threshold = st.slider("EAR Threshold", 0.15, 0.30, 0.20, 0.01)
    if st.button("Reset"):
        st.session_state.ear_history = [0.25] * 20
        st.session_state.metrics = {"ear": 0.0, "blinks": 0, "status": "Initializing"}

with col1:
    st.subheader("Live Feed")
    
    ctx = webrtc_streamer(
        key="visionmate",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": get_ice_servers(), "iceTransportPolicy": "all"},
        media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": 30}, "audio": False},
        async_processing=True,
        desired_playing_state=True,
        video_html_attrs={"style": {"width": "100%", "height": "auto"}, "controls": False, "autoPlay": True}
    )
    
    metrics = st.session_state.metrics
    
    if metrics["status"] == "OPTIMAL":
        status_class = "status-optimal"
    elif metrics["status"] == "HIGH STRAIN":
        status_class = "status-danger"
    else:
        status_class = "status-warning"
    
    ear_placeholder.markdown(f'<div class="metric-value {status_class}">{metrics["ear"]:.3f}</div>', unsafe_allow_html=True)
    blink_placeholder.markdown(f'<div class="metric-value" style="color: #BB86FC;">{metrics["blinks"]}</div>', unsafe_allow_html=True)
    status_placeholder.markdown(f'<div class="metric-value {status_class}" style="font-size: 1.2rem;">{metrics["status"]}</div>', unsafe_allow_html=True)
    
    chart_placeholder.line_chart({"EAR": st.session_state.ear_history}, height=100, use_container_width=True)
    
    if metrics["status"] == "NO FACE":
        coach_placeholder.warning("Position your face in front of camera")
    elif metrics["status"] == "HIGH STRAIN":
        coach_placeholder.error("Eye strain detected. Take a break.")
    elif metrics["blinks"] < 5:
        coach_placeholder.success("System Active. Monitoring...")
    else:
        coach_placeholder.info("Remember 20-20-20 rule")
