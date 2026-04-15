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
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    if "ear_history" not in st.session_state:
        st.session_state.ear_history = [0.25] * 40
    if "run_monitor" not in st.session_state:
        st.session_state.run_monitor = True

init_session_state()

# Custom CSS - Glassmorphism UI with video sizing
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
    
    /* Video container sizing - CRITICAL FIX */
    .video-container {
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
        border-radius: 16px !important;
        overflow: hidden !important;
    }
    .video-container video {
        width: 100% !important;
        height: auto !important;
        max-height: 480px !important;
        object-fit: contain !important;
        border-radius: 16px !important;
    }
    
    /* Remove default streamlit iframe borders */
    iframe {
        border: none !important;
        border-radius: 16px !important;
    }
    
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Function to get ICE servers (Twilio or fallback)
def get_ice_servers():
    account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
    auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
    
    if account_sid and auth_token and TWILIO_AVAILABLE:
        try:
            client = Client(account_sid, auth_token)
            token = client.tokens.create()
            return token.ice_servers
        except Exception as e:
            st.warning(f"⚠️ Twilio failed ({e}), using Open Relay fallback")
    else:
        if not TWILIO_AVAILABLE:
            st.info("ℹ️ Twilio not installed, using Open Relay")
        elif not account_sid:
            st.info("ℹ️ Twilio credentials not set, using Open Relay")
    
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
        },
        {
            "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
    ]

# Sidebar
with st.sidebar:
    st.markdown("## VisionMate Control")
    
    run_monitor = st.checkbox("Enable Live AI Monitor", 
                              value=st.session_state.run_monitor,
                              key="monitor_toggle")
    st.session_state.run_monitor = run_monitor
    
    threshold = st.slider("Blink Sensitivity (EAR Threshold)", 
                         min_value=0.15, max_value=0.30, 
                         value=0.20, step=0.01)
    
    if st.button("Reset Session Stats", use_container_width=True):
        st.session_state.ear_history = [0.25] * 40
        st.rerun()
    
    st.divider()
    st.markdown("### About")
    st.info("VisionMate monitors your eye strain using AI-powered eye tracking.")

# Main Layout
st.markdown("<h1 style='text-align: center;'>VISIONMATE</h1>", 
            unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #B0B0B0;'>AI Eye-Strain Monitor & Ergonomic Coach</p>", 
            unsafe_allow_html=True)

col1, col2 = st.columns([1.8, 1])

# Video Processor - Clean video without overlays
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.detector = EyeStrainDetector()
        self.threshold = 0.20
        self.blink_count = 0
        self.current_ear = 0.25
        self.status = "Initializing..."
        
    def transform(self, frame):
        try:
            # Get original frame without resizing to avoid blur
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)  # Mirror for natural feel
            h, w, _ = img.shape
            
            # Process on original resolution to maintain quality
            # Only scale down for MediaPipe if needed, then scale back
            process_img = img.copy()
            
            # Detect eyes
            ear, landmarks, _ = self.detector.process_frame(process_img)
            
            # Update metrics
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
                self.current_ear = 0
            
            # Return clean frame - NO TEXT OVERLAYS
            # Only draw subtle border color based on status
            if self.status == "HIGH STRAIN":
                color = (255, 50, 50)  # Red
            elif self.status == "OPTIMAL":
                color = (50, 255, 150)  # Green
            else:
                color = (128, 128, 128)  # Gray
            
            # Thin border only - no text, no overlays
            cv2.rectangle(img, (0, 0), (w-1, h-1), color, 3)
            
            return img
            
        except Exception as e:
            img = frame.to_ndarray(format="bgr24")
            cv2.putText(img, f"Error: {str(e)[:50]}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return img

# Analytics Column - All metrics displayed here
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

# Video Feed Column
with col1:
    st.subheader("Live AI Vision Feed")
    
    # Video container with CSS class
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    
    if st.session_state.run_monitor:
        # Get ICE servers
        ice_servers = get_ice_servers()
        
        rtc_configuration = {
            "iceServers": ice_servers,
            "iceTransportPolicy": "all"
        }
        
        # Video constraints - maintain aspect ratio
        media_constraints = {
            "video": {
                "width": {"ideal": 640, "min": 320},
                "height": {"ideal": 480, "min": 240},
                "aspectRatio": {"ideal": 1.333},  # 4:3 aspect ratio
                "frameRate": {"ideal": 30, "max": 30}
            },
            "audio": False
        }
        
        try:
            ctx = webrtc_streamer(
                key="visionmate-clean-v1",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=VideoProcessor,
                rtc_configuration=rtc_configuration,
                media_stream_constraints=media_constraints,
                async_processing=True,
                # Video HTML attributes for proper sizing
                video_html_attrs={
                    "style": {
                        "width": "100%",
                        "height": "auto",
                        "max-height": "480px",
                        "border-radius": "16px"
                    },
                    "controls": False,
                    "autoPlay": True,
                    "playsInline": True
                }
            )
            
            # Update dashboard metrics from processor
            if ctx and ctx.video_processor:
                processor = ctx.video_processor
                ear_value = processor.current_ear if processor.current_ear > 0 else 0.0
                blink_count = processor.blink_count
                status = processor.status
                
                # Determine colors based on status
                if status == "OPTIMAL":
                    ear_color = "#00E676"  # Green
                    status_color = "status-optimal"
                    status_icon = "✅"
                elif status == "HIGH STRAIN":
                    ear_color = "#FF1744"  # Red
                    status_color = "status-danger"
                    status_icon = "⚠️"
                else:
                    ear_color = "#FFD600"  # Yellow
                    status_color = "status-warning"
                    status_icon = "👤"
                
                # Update EAR display
                ear_display = f"{ear_value:.3f}" if ear_value > 0 else "--.---"
                ear_placeholder.markdown(
                    f'<div class="metric-value {status_color}">{ear_display}</div>', 
                    unsafe_allow_html=True
                )
                
                # Update Blink count
                blink_placeholder.markdown(
                    f'<div class="metric-value" style="color: #BB86FC;">{blink_count}</div>', 
                    unsafe_allow_html=True
                )
                
                # Update Status
                status_placeholder.markdown(
                    f'<div class="metric-value {status_color}" style="font-size: 24px;">{status_icon} {status}</div>', 
                    unsafe_allow_html=True
                )
                
                # Update Chart
                if ear_value > 0:
                    st.session_state.ear_history = st.session_state.ear_history[1:] + [ear_value]
                chart_placeholder.line_chart(
                    {"EAR": st.session_state.ear_history}, 
                    height=120, 
                    use_container_width=True
                )
                
                # Coach message based on status
                if status == "NO FACE":
                    coach_placeholder.warning("👤 Please position your face in front of the camera")
                elif status == "HIGH STRAIN":
                    coach_placeholder.error("⚠️ Eye strain detected! Consider taking a 20-20-20 break.")
                elif blink_count < 5:
                    coach_placeholder.success("✅ System Active: Monitoring your eye health...")
                else:
                    coach_placeholder.info("💡 Tip: Every 20 mins, look 20 feet away for 20 seconds")
            else:
                # Waiting state
                ear_placeholder.markdown(
                    '<div class="metric-value" style="color: #666;">--.---</div>', 
                    unsafe_allow_html=True
                )
                blink_placeholder.markdown(
                    '<div class="metric-value" style="color: #666;">0</div>', 
                    unsafe_allow_html=True
                )
                status_placeholder.markdown(
                    '<div class="metric-value status-warning" style="font-size: 20px;">⏳ Connecting...</div>', 
                    unsafe_allow_html=True
                )
                coach_placeholder.info("⏳ Waiting for camera connection...")
                
        except Exception as e:
            st.error(f"WebRTC Error: {str(e)}")
            st.info("Try refreshing the page or check browser camera permissions.")
    else:
        st.info("📹 System standby. Enable the monitor in the sidebar to begin.")
    
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.markdown("<p style='text-align: center; color: #666; font-size: 12px;'>VisionMate FYP | BAXU 3973 | UTeM</p>", 
           unsafe_allow_html=True)
