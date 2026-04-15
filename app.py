import streamlit as st
import cv2
import numpy as np
from detector import EyeStrainDetector
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import threading
import time
import os
from collections import deque

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

st.set_page_config(page_title="VisionMate", layout="wide", initial_sidebar_state="expanded")

if "shared" not in st.session_state:
    st.session_state.shared = {
        "ear": 0.0,
        "blinks": 0,
        "status": "Initializing",
        "lock": threading.Lock()
    }
if "history" not in st.session_state:
    st.session_state.history = deque([0.25] * 40, maxlen=40)
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.20

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { height: 100vh; overflow: hidden; }
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
    height: 85vh;
    overflow: hidden;
}
.metric-value { 
    font-size: 36px; 
    color: #BB86FC; 
    text-shadow: 0 0 10px rgba(187, 134, 252, 0.5);
    text-align: center;
    font-weight: bold;
}
.metric-label {
    font-size: 10px;
    opacity: 0.7;
    text-align: center;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.status-optimal { color: #00E676 !important; }
.status-warning { color: #FFD600 !important; }
.status-danger { color: #FF1744 !important; }
button[kind="secondary"], footer { display: none !important; }
video { width: 100% !important; height: auto !important; max-height: 50vh !important; border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

def get_ice_servers():
    return [{"urls": ["stun:stun.l.google.com:19302"]}, {"urls": ["stun:stun1.l.google.com:19302"]}]

with st.sidebar:
    st.markdown("## VisionMate Control")
    run_monitor = st.checkbox("Enable Monitor", value=True)
    new_threshold = st.slider("EAR Threshold", 0.15, 0.30, st.session_state.threshold, 0.01)
    if new_threshold != st.session_state.threshold:
        st.session_state.threshold = new_threshold
    if st.button("Reset Stats", width="stretch"):
        st.session_state.history = deque([0.25] * 40, maxlen=40)
        with st.session_state.shared["lock"]:
            st.session_state.shared["ear"] = 0.0
            st.session_state.shared["blinks"] = 0
        st.rerun()
    st.info("VisionMate monitors eye strain using AI.")

st.markdown("<h1 style='text-align: center; font-size: 1.5rem;'>VISIONMATE</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #B0B0B0; font-size: 0.8rem;'>AI Eye-Strain Monitor</p>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = EyeStrainDetector()
        self.frame_count = 0
        self.blink_count = 0
        self.last_ear = 0.0
        self.last_status = "Initializing"
        self.last_update = 0
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        self.frame_count += 1
        
        if self.frame_count % 3 == 0:
            small = cv2.resize(img, (320, 240))
            ear, _, _ = self.detector.process_frame(small)
            
            if ear > 0:
                blinks, _ = self.detector.update_blink_state(ear, st.session_state.threshold)
                self.blink_count = blinks
                self.last_ear = ear
                self.last_status = "OPTIMAL" if ear >= st.session_state.threshold else "HIGH STRAIN"
            else:
                self.last_status = "NO FACE"
                self.last_ear = 0.0
            
            current_time = time.time()
            if current_time - self.last_update > 0.3:
                with st.session_state.shared["lock"]:
                    st.session_state.shared["ear"] = self.last_ear
                    st.session_state.shared["blinks"] = self.blink_count
                    st.session_state.shared["status"] = self.last_status
                if self.last_ear > 0:
                    st.session_state.history.append(self.last_ear)
                self.last_update = current_time
        
        color = (50, 255, 150) if self.last_status == "OPTIMAL" else (255, 50, 50) if self.last_status == "HIGH STRAIN" else (128, 128, 128)
        cv2.rectangle(img, (0, 0), (w-1, h-1), color, 3)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

with col2:
    st.subheader("Analytics")
    st.markdown('<p class="metric-label">Current EAR</p>', unsafe_allow_html=True)
    ear_display = st.empty()
    st.markdown('<p class="metric-label">Total Blinks</p>', unsafe_allow_html=True)
    blink_display = st.empty()
    st.markdown('<p class="metric-label">Status</p>', unsafe_allow_html=True)
    status_display = st.empty()
    st.divider()
    st.markdown('<p class="metric-label">History</p>', unsafe_allow_html=True)
    chart_display = st.empty()
    st.divider()
    st.subheader("Coach")
    coach_display = st.empty()

with col1:
    st.subheader("Live Feed")
    if run_monitor:
        ctx = webrtc_streamer(
            key="visionmate-v3",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            rtc_configuration={"iceServers": get_ice_servers()},
            media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": 30}, "audio": False},
            async_processing=True,
            desired_playing_state=True,
            video_html_attrs={"style": {"width": "100%", "height": "auto"}, "controls": False, "autoPlay": True, "playsInline": True, "muted": True}
        )
        
        with st.session_state.shared["lock"]:
            ear = st.session_state.shared["ear"]
            blinks = st.session_state.shared["blinks"]
            status = st.session_state.shared["status"]
        
        status_class = "status-optimal" if status == "OPTIMAL" else "status-danger" if status == "HIGH STRAIN" else "status-warning"
        
        ear_display.markdown(f'<div class="metric-value {status_class}">{ear:.3f}</div>', unsafe_allow_html=True)
        blink_display.markdown(f'<div class="metric-value" style="color: #BB86FC;">{blinks}</div>', unsafe_allow_html=True)
        status_display.markdown(f'<div class="metric-value {status_class}" style="font-size: 18px;">{status}</div>', unsafe_allow_html=True)
        
        chart_display.line_chart(list(st.session_state.history), height=100, width="stretch")
        
        if status == "HIGH STRAIN":
            coach_display.error("Eye strain detected. Take a break.")
        elif status == "NO FACE":
            coach_display.warning("Position your face in camera.")
        else:
            coach_display.success("Monitoring active.")
        
        time.sleep(0.2)
        st.rerun()
    else:
        st.info("Enable monitor in sidebar.")

st.markdown("<p style='text-align: center; color: #666; font-size: 10px;'>VisionMate FYP | BAXU 3973 | UTeM</p>", unsafe_allow_html=True)
