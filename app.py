import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import cv2
import numpy as np
from detector import EyeStrainDetector
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import threading
import time
import os
from queue import Queue

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

st.set_page_config(page_title="VisionMate", layout="wide", initial_sidebar_state="expanded")

def init_session_state():
    if "ear_history" not in st.session_state:
        st.session_state.ear_history = [0.25] * 40
    if "current_ear" not in st.session_state:
        st.session_state.current_ear = 0.0
    if "blink_count" not in st.session_state:
        st.session_state.blink_count = 0
    if "status" not in st.session_state:
        st.session_state.status = "Initializing"
    if "run_monitor" not in st.session_state:
        st.session_state.run_monitor = True

init_session_state()

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

metrics_queue = Queue()

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
    
    # Use only STUN servers (no TURN to avoid 403 errors)
    # This works if direct peer connection is possible
    return [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]}
    ]

with st.sidebar:
    st.markdown("## VisionMate Control")
    run_monitor = st.checkbox("Enable Live AI Monitor", value=st.session_state.run_monitor, key="monitor_toggle")
    st.session_state.run_monitor = run_monitor
    threshold = st.slider("Blink Sensitivity (EAR Threshold)", min_value=0.15, max_value=0.30, value=0.20, step=0.01)
    if st.button("Reset Session Stats", width="stretch"):
        st.session_state.ear_history = [0.25] * 40
        st.session_state.current_ear = 0.0
        st.session_state.blink_count = 0
        st.rerun()
    st.divider()
    st.markdown("### About")
    st.info("VisionMate monitors your eye strain using AI-powered eye tracking.")

st.markdown("<h1 style='text-align: center;'>VISIONMATE</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #B0B0B0;'>AI Eye-Strain Monitor and Ergonomic Coach</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1.8, 1])

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = EyeStrainDetector()
        self.threshold = 0.20
        self.blink_count = 0
        self.current_ear = 0.0
        self.status = "Initializing"
        self.frame_counter = 0
        self.last_metrics_update = time.time()
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        self.frame_counter += 1
        
        if self.frame_counter % 2 == 0:
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
            
            if time.time() - self.last_metrics_update > 0.5:
                try:
                    metrics_queue.put_nowait({
                        "ear": self.current_ear,
                        "blinks": self.blink_count,
                        "status": self.status
                    })
                except:
                    pass
                self.last_metrics_update = time.time()
        
        if self.status == "HIGH STRAIN":
            color = (255, 50, 50)
        elif self.status == "OPTIMAL":
            color = (50, 255, 150)
        else:
            color = (128, 128, 128)
        cv2.rectangle(img, (0, 0), (w-1, h-1), color, 3)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

with col2:
    st.subheader("Session Analytics")
    st.markdown('<p class="metric-label">Current EAR</p>', unsafe_allow_html=True)
    ear_placeholder = st.empty()
    st.markdown('<p class="metric-label">Total Blinks</p>', unsafe_allow_html=True)
    blink_placeholder = st.empty()
    st.markdown('<p class="metric-label">System Status</p>', unsafe_allow_html=True)
    status_placeholder = st.empty()
    st.divider()
    st.markdown('<p class="metric-label">EAR History</p>', unsafe_allow_html=True)
    chart_placeholder = st.empty()
    st.divider()
    st.subheader("Real-time Coach")
    coach_placeholder = st.empty()

with col1:
    st.subheader("Live AI Vision Feed")
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    
    if st.session_state.run_monitor:
        ice_servers = get_ice_servers()
        rtc_configuration = {"iceServers": ice_servers}
        media_constraints = {"video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 30}}, "audio": False}
        
        try:
            ctx = webrtc_streamer(
                key="visionmate-stun-v1",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=VideoProcessor,
                rtc_configuration=rtc_configuration,
                media_stream_constraints=media_constraints,
                async_processing=True,
                desired_playing_state=True,
                video_html_attrs={"style": {"width": "100%", "height": "auto", "max-height": "480px"}, "controls": False, "autoPlay": True, "playsInline": True, "muted": True}
            )
            
            while not metrics_queue.empty():
                try:
                    data = metrics_queue.get_nowait()
                    st.session_state.current_ear = data["ear"]
                    st.session_state.blink_count = data["blinks"]
                    st.session_state.status = data["status"]
                    if data["ear"] > 0:
                        st.session_state.ear_history = st.session_state.ear_history[1:] + [data["ear"]]
                except:
                    break
            
            current_ear = st.session_state.current_ear
            blink_count = st.session_state.blink_count
            status = st.session_state.status
            
            if status == "OPTIMAL":
                status_class = "status-optimal"
            elif status == "HIGH STRAIN":
                status_class = "status-danger"
            else:
                status_class = "status-warning"
            
            ear_display = f"{current_ear:.3f}" if current_ear > 0 else "0.000"
            ear_placeholder.markdown(f'<div class="metric-value {status_class}">{ear_display}</div>', unsafe_allow_html=True)
            blink_placeholder.markdown(f'<div class="metric-value" style="color: #BB86FC;">{blink_count}</div>', unsafe_allow_html=True)
            status_placeholder.markdown(f'<div class="metric-value {status_class}" style="font-size: 20px;">{status}</div>', unsafe_allow_html=True)
            chart_placeholder.line_chart({"EAR": st.session_state.ear_history}, height=120, width="stretch")
            
            if status == "NO FACE":
                coach_placeholder.warning("Please position your face in front of the camera")
            elif status == "HIGH STRAIN":
                coach_placeholder.error("Eye strain detected. Consider taking a break.")
            elif blink_count < 5:
                coach_placeholder.success("System Active. Monitoring your eye health.")
            else:
                coach_placeholder.info("Remember the 20-20-20 rule")
            
            time.sleep(0.3)
            st.rerun()
            
        except Exception as e:
            st.error(f"WebRTC Error: {str(e)}")
            st.info("Please check camera permissions and refresh the page.")
    else:
        st.info("System standby. Enable the monitor in the sidebar to begin.")
    
    st.markdown('</div>', unsafe_allow_html=True)
