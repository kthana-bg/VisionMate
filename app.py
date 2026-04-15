import streamlit as st
import cv2
import numpy as np
from detector import EyeStrainDetector
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
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

# MUST BE FIRST: Page configuration
st.set_page_config(
    page_title="VisionMate",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state with shared values for cross-thread communication
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

# Custom CSS - Clean UI without icons, auto-start video
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
    
    /* Video container sizing */
    .video-container {
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
        border-radius: 16px !important;
        overflow: hidden !important;
    }
    
    /* CRITICAL: Hide the start/stop button */
    button[kind="secondary"] {
        display: none !important;
    }
    
    /* Hide all streamlit buttons in webrtc */
    .stButton > button {
        display: none !important;
    }
    
    /* Alternative: hide specific webrtc controls */
    .webrtc-container button {
        display: none !important;
    }
    
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Function to get ICE servers
def get_ice_servers():
    account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
    auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
    
    if account_sid and auth_token and TWILIO_AVAILABLE:
        try:
            client = Client(account_sid, auth_token)
            token = client.tokens.create()
            return token.ice_servers
        except Exception as e:
            pass
    
    # Fallback Open Relay servers
    return [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        },
        {
            "urls": ["turn:openrelay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
    ]

# Sidebar - No icons
with st.sidebar:
    st.markdown("## VisionMate Control")
    
    run_monitor = st.checkbox("Enable Live AI Monitor", 
                              value=st.session_state.run_monitor,
                              key="monitor_toggle")
    st.session_state.run_monitor = run_monitor
    
    threshold = st.slider("Blink Sensitivity (EAR Threshold)", 
                         min_value=0.15, max_value=0.30, 
                         value=0.20, step=0.01)
    
         if st.button("Reset Session Stats", width="stretch"): 
        st.session_state.ear_history = [0.25] * 40
        st.session_state.current_ear = 0.0
        st.session_state.blink_count = 0
        st.rerun()
    
    st.divider()
    st.markdown("### About")
    st.info("VisionMate monitors your eye strain using AI-powered eye tracking.")

# Main Layout - No icons in title
st.markdown("<h1 style='text-align: center;'>VISIONMATE</h1>", 
            unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #B0B0B0;'>AI Eye-Strain Monitor and Ergonomic Coach</p>", 
            unsafe_allow_html=True)

col1, col2 = st.columns([1.8, 1])

# Thread-safe value storage
class SharedMetrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.ear = 0.0
        self.blinks = 0
        self.status = "Initializing"
    
    def update(self, ear, blinks, status):
        with self.lock:
            self.ear = ear
            self.blinks = blinks
            self.status = status
    
    def get(self):
        with self.lock:
            return self.ear, self.blinks, self.status

# Global shared metrics instance
shared_metrics = SharedMetrics()

# Video Processor - Clean video, updates shared metrics
class VideoProcessor: # No longer needs to inherit from VideoTransformerBase
    def __init__(self):
        self.detector = EyeStrainDetector()
        self.threshold = 0.20
        self.blink_count = 0
        self.current_ear = 0.0
        self.status = "Initializing"
        self.frame_counter = 0
        
    def recv(self, frame): # Changed from transform to recv
        try:
            # Get image from the frame
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
                    self.status = "HIGH STRAIN" if ear < self.threshold else "OPTIMAL"
                else:
                    self.status = "NO FACE"
                    self.current_ear = 0.0
                
                shared_metrics.update(self.current_ear, self.blink_count, self.status)
            
            # Draw border
            color = (255, 50, 50) if self.status == "HIGH STRAIN" else (50, 255, 150)
            cv2.rectangle(img, (0, 0), (w-1, h-1), color, 3)
            
            # RETURN must be an av.VideoFrame
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            return frame

# Analytics Column
with col2:
    st.subheader("Session Analytics")
    
    # EAR Metric
    st.markdown('<p class="metric-label">Current EAR</p>', unsafe_allow_html=True)
    ear_placeholder = st.empty()
    
    # Blink Count Metric
    st.markdown('<p class="metric-label">Total Blinks</p>', unsafe_allow_html=True)
    blink_placeholder = st.empty()
    
    # Status Indicator
    st.markdown('<p class="metric-label">System Status</p>', unsafe_allow_html=True)
    status_placeholder = st.empty()
    
    st.divider()
    
    # EAR History Chart
    st.markdown('<p class="metric-label">EAR History</p>', unsafe_allow_html=True)
    chart_placeholder = st.empty()
    
    st.divider()
    
    # Real-time Coach
    st.subheader("Real-time Coach")
    coach_placeholder = st.empty()

# Video Feed Column - Auto starts, no buttons
with col1:
    st.subheader("Live AI Vision Feed")
    
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    
    if st.session_state.run_monitor:
        # Get ICE servers
        ice_servers = get_ice_servers()
        
        rtc_configuration = {
            "iceServers": ice_servers,
            "iceTransportPolicy": "all"
        }
        
        media_constraints = {
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 30}
            },
            "audio": False
        }
        
        try:
            ctx = webrtc_streamer(
                key="visionmate-v2",
                mode=WebRtcMode.SENDRECV,
                video_frame_callback=None, # If using processor_factory
                video_processor_factory=VideoProcessor,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}] # Simplified to avoid 403
                },
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
                desired_playing_state=True,
            )
            
            # ... update chart using new width param
            chart_placeholder.line_chart(
                {"EAR": st.session_state.ear_history}, 
                height=120, 
                width="stretch" # Changed from use_container_width
            )
            # Read from shared metrics (works across threads)
            current_ear, blink_count, status = shared_metrics.get()
            
            # Update session state for chart history
            if current_ear > 0:
                st.session_state.current_ear = current_ear
                st.session_state.blink_count = blink_count
                st.session_state.status = status
                st.session_state.ear_history = st.session_state.ear_history[1:] + [current_ear]
            
            # Determine display colors
            if status == "OPTIMAL":
                ear_color = "#00E676"
                status_class = "status-optimal"
            elif status == "HIGH STRAIN":
                ear_color = "#FF1744"
                status_class = "status-danger"
            else:
                ear_color = "#FFD600"
                status_class = "status-warning"
            
            # Update EAR display
            ear_display = f"{current_ear:.3f}" if current_ear > 0 else "0.000"
            ear_placeholder.markdown(
                f'<div class="metric-value {status_class}">{ear_display}</div>', 
                unsafe_allow_html=True
            )
            
            # Update Blink count
            blink_placeholder.markdown(
                f'<div class="metric-value" style="color: #BB86FC;">{blink_count}</div>', 
                unsafe_allow_html=True
            )
            
            # Update Status - No icons, just text
            status_placeholder.markdown(
                f'<div class="metric-value {status_class}" style="font-size: 20px;">{status}</div>', 
                unsafe_allow_html=True
            )
            
            # Update Chart
            chart_placeholder.line_chart(
                {"EAR": st.session_state.ear_history}, 
                height=120, 
                use_container_width=True
            )
            
            # Coach message - No icons
            if status == "NO FACE":
                coach_placeholder.warning("Please position your face in front of the camera")
            elif status == "HIGH STRAIN":
                coach_placeholder.error("Eye strain detected. Consider taking a break.")
            elif blink_count < 5:
                coach_placeholder.success("System Active. Monitoring your eye health.")
            else:
                coach_placeholder.info("Remember the 20-20-20 rule")
                
        except Exception as e:
            st.error(f"WebRTC Error: {str(e)}")
            st.info("Please check camera permissions and refresh the page.")
    else:
        st.info("System standby. Enable the monitor in the sidebar to begin.")
    
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.markdown("<p style='text-align: center; color: #666; font-size: 12px;'>VisionMate FYP | BAXU 3973 | UTeM</p>", 
           unsafe_allow_html=True)
