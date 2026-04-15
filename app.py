import streamlit as st
import cv2
import numpy as np
from detector import EyeStrainDetector
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import threading
import time
import os

# Try to import Twilio, fallback if not available
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

# --- PAGE CONFIGURATION (MUST BE FIRST) ---
st.set_page_config(
    page_title="VisionMate",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SHARED DATA CLASS (Thread-safe communication) ---
class SharedMetrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.ear = 0.0
        self.blinks = 0
        self.status = "Initializing"
        self.history = [0.25] * 40
    
    def update(self, ear, blinks, status):
        with self.lock:
            self.ear = ear
            self.blinks = blinks
            self.status = status
            if ear > 0:
                self.history = self.history[1:] + [ear]
    
    def get(self):
        with self.lock:
            return self.ear, self.blinks, self.status, self.history

# Initialize the shared metrics object in session state
if "shared" not in st.session_state:
    st.session_state.shared = SharedMetrics()

# --- ORIGINAL CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(26, 26, 46, 0.9), rgba(26, 26, 46, 0.9)), 
                    url("https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-attachment: fixed;
    }
    section[data-testid="stSidebar"] {
        background: rgba(40, 20, 80, 0.6) !important;
        backdrop-filter: blur(20px) !important;
    }
    [data-testid="column"] > div {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(25px);
        border-radius: 24px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding: 30px !important;
    }
    h1, h2, h3 { color: #E0B0FF !important; font-weight: 300 !important; }
    .metric-value { 
        font-size: 48px; 
        color: #BB86FC; 
        text-shadow: 0 0 10px rgba(187, 134, 252, 0.5);
        text-align: center;
        font-weight: bold;
    }
    .metric-label {
        font-size: 12px;
        opacity: 0.7;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .status-optimal { color: #00E676 !important; }
    .status-warning { color: #FFD600 !important; }
    .status-danger { color: #FF1744 !important; }
    
    .video-container {
        width: 100% !important;
        border-radius: 16px !important;
        overflow: hidden !important;
    }
    
    /* Hide default webrtc buttons for clean UI */
    button[kind="secondary"] { display: none !important; }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- ICE SERVERS LOGIC ---
def get_ice_servers():
    account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
    auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
    if account_sid and auth_token and TWILIO_AVAILABLE:
        try:
            client = Client(account_sid, auth_token)
            token = client.tokens.create()
            return token.ice_servers
        except: pass
    return [{"urls": ["stun:stun.l.google.com:19302"]}]

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("## VisionMate Control")
    run_monitor = st.checkbox("Enable Live AI Monitor", value=True)
    threshold = st.slider("Blink Sensitivity", 0.15, 0.30, 0.20, 0.01)
    
    if st.button("Reset Session Stats", width="stretch"):
        st.session_state.shared = SharedMetrics()
        st.rerun()
    
    st.divider()
    st.info("VisionMate monitors eye strain using AI-powered eye tracking.")

# --- MAIN LAYOUT ---
st.markdown("<h1 style='text-align: center;'>VISIONMATE</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #B0B0B0;'>AI Eye-Strain Monitor and Ergonomic Coach</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1.8, 1])

# --- VIDEO PROCESSOR (Fixed for 2025) ---
class VideoProcessor:
    def __init__(self):
        self.detector = EyeStrainDetector()
        self.frame_counter = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        self.frame_counter += 1
        # Process every 2nd frame for stability
        if self.frame_counter % 2 == 0:
            ear, _, _ = self.detector.process_frame(img)
            if ear > 0:
                blinks, _ = self.detector.update_blink_state(ear, threshold)
                status = "OPTIMAL" if ear >= threshold else "HIGH STRAIN"
                st.session_state.shared.update(ear, blinks, status)
            else:
                st.session_state.shared.update(0.0, st.session_state.shared.blinks, "NO FACE")

        # Draw status border
        current_ear, _, status, _ = st.session_state.shared.get()
        color = (50, 255, 150) if status == "OPTIMAL" else (50, 50, 255) if status == "HIGH STRAIN" else (128, 128, 128)
        cv2.rectangle(img, (0, 0), (w-1, h-1), color, 6)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- ANALYTICS COLUMN ---
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

# --- VIDEO FEED COLUMN ---
with col1:
    st.subheader("Live AI Vision Feed")
    if run_monitor:
        ctx = webrtc_streamer(
            key="visionmate-stable",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            rtc_configuration={"iceServers": get_ice_servers()},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            desired_playing_state=True,
            video_html_attrs={
                "style": {"width": "100%", "height": "auto", "border-radius": "16px"},
                "controls": False, "autoPlay": True, "playsInline": True, "muted": True
            }
        )

        # Update UI Loop
        if ctx.state.playing:
            while ctx.state.playing:
                ear, blinks, status, history = st.session_state.shared.get()
                
                # Update Labels
                status_class = "status-optimal" if status == "OPTIMAL" else "status-danger" if status == "HIGH STRAIN" else "status-warning"
                
                ear_placeholder.markdown(f'<div class="metric-value {status_class}">{ear:.3f}</div>', unsafe_allow_html=True)
                blink_placeholder.markdown(f'<div class="metric-value" style="color: #BB86FC;">{blinks}</div>', unsafe_allow_html=True)
                status_placeholder.markdown(f'<div class="metric-value {status_class}" style="font-size: 20px;">{status}</div>', unsafe_allow_html=True)
                
                chart_placeholder.line_chart(history, height=120, width="stretch")
                
                if status == "HIGH STRAIN":
                    coach_placeholder.error("Eye strain detected. Consider taking a break.")
                elif status == "NO FACE":
                    coach_placeholder.warning("Face not detected.")
                else:
                    coach_placeholder.success("System Active. Remember to blink!")
                
                time.sleep(0.1)
    else:
        st.info("System standby. Enable the monitor in the sidebar.")
