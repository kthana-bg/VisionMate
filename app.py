import streamlit as st
import cv2
import numpy as np
from detector import EyeStrainDetector
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import threading
import time

# MUST BE FIRST: Page configuration
st.set_page_config(
    page_title="VisionMate",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CRITICAL: Initialize session state BEFORE any other Streamlit code
def init_session_state():
    if "ear_history" not in st.session_state:
        st.session_state.ear_history = [0.25] * 40
    if "run_monitor" not in st.session_state:
        st.session_state.run_monitor = True

init_session_state()

# Custom CSS
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
    }
    .stAlert { border-radius: 12px !important; }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

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

# Video Processor - Creates its own detector instance (thread-safe)
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        # Create new detector instance for this thread
        self.detector = EyeStrainDetector()
        self.threshold = 0.20
        self.blink_count = 0
        self.current_ear = 0.25
        self.status = "Initializing..."
        self.last_update = time.time()
        
    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)  # Mirror
            h, w, _ = img.shape
            
            # Process at lower resolution for performance
            small_img = cv2.resize(img, (320, 240))
            
            # Detect
            ear, landmarks, _ = self.detector.process_frame(small_img)
            
            # Update blink detection
            if ear > 0:
                current_blinks, _ = self.detector.update_blink_state(ear, self.threshold)
                self.blink_count = current_blinks
                self.current_ear = ear
                
                # Determine status
                if ear < self.threshold:
                    color = (255, 50, 50)  # Red
                    status = "HIGH STRAIN"
                else:
                    color = (50, 255, 150)  # Green
                    status = "OPTIMAL"
            else:
                color = (128, 128, 128)
                status = "NO FACE"
                self.current_ear = 0
            
            self.status = status
            
            # Draw UI on original frame
            cv2.rectangle(img, (0, 0), (w-1, h-1), color, 4)
            
            # Info panel
            overlay = img.copy()
            cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            
            # Text
            cv2.putText(img, f"EAR: {self.current_ear:.3f}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(img, f"Blinks: {self.blink_count}", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(img, status, (20, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            return img
            
        except Exception as e:
            # Return error frame
            img = frame.to_ndarray(format="bgr24")
            cv2.putText(img, f"Error: {str(e)[:50]}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return img

# Analytics Column
with col2:
    st.subheader("Session Analytics")
    
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.markdown("<p style='font-size:12px; opacity:0.7; text-align:center;'>CURRENT EAR</p>", 
                   unsafe_allow_html=True)
        ear_placeholder = st.empty()
        
    with m_col2:
        st.markdown("<p style='font-size:12px; opacity:0.7; text-align:center;'>TOTAL BLINKS</p>", 
                   unsafe_allow_html=True)
        blink_placeholder = st.empty()
    
    st.divider()
    st.markdown("<p style='font-size:12px; opacity:0.7;'>EAR HISTORY</p>", 
               unsafe_allow_html=True)
    chart_placeholder = st.empty()
    
    st.divider()
    st.subheader("Real-time Coach")
    coach_placeholder = st.empty()

# Video Feed Column
with col1:
    st.subheader("Live AI Vision Feed")
    
    if st.session_state.run_monitor:
        # CRITICAL FIX: Use TURN server for Streamlit Cloud
        # Open Relay provides free TURN servers
        rtc_configuration = {
            "iceServers": [
                # Multiple STUN servers for redundancy
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                # FREE TURN server from Open Relay (required for Streamlit Cloud)
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
            ],
            "iceTransportPolicy": "all"  # Allow both STUN and TURN
        }
        
        media_constraints = {
            "video": {
                "width": {"ideal": 640, "max": 1280},
                "height": {"ideal": 480, "max": 720},
                "frameRate": {"ideal": 15, "max": 30}
            },
            "audio": False
        }
        
        try:
            # Use unique key to avoid conflicts
            ctx = webrtc_streamer(
                key="visionmate-turn-v1",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=VideoProcessor,
                rtc_configuration=rtc_configuration,
                media_stream_constraints=media_constraints,
                async_processing=True,
                video_html_attrs={
                    "style": {"width": "100%", "height": "auto", "max-height": "480px"},
                    "controls": False,
                    "autoPlay": True
                }
            )
            
            # Get processor instance to read values
            if ctx and ctx.video_processor:
                processor = ctx.video_processor
                ear_value = processor.current_ear if processor.current_ear > 0 else 0.25
                blink_count = processor.blink_count
                status = processor.status
                
                # Update UI
                ear_color = "#BB86FC" if status == "OPTIMAL" else "#FF1744" if status == "HIGH STRAIN" else "#FFD600"
                
                ear_placeholder.markdown(
                    f"<div class='metric-value' style='color: {ear_color};'>{ear_value:.3f}</div>", 
                    unsafe_allow_html=True
                )
                
                blink_placeholder.markdown(
                    f"<div class='metric-value'>{blink_count}</div>", 
                    unsafe_allow_html=True
                )
                
                # Update chart
                st.session_state.ear_history = st.session_state.ear_history[1:] + [ear_value]
                chart_placeholder.line_chart(
                    {"EAR": st.session_state.ear_history}, 
                    height=150, 
                    use_container_width=True
                )
                
                # Coach message
                if status == "NO FACE":
                    coach_placeholder.warning("👤 Please position your face in front of the camera")
                elif status == "HIGH STRAIN":
                    coach_placeholder.error("⚠️ Eye strain detected! Take a break.")
                elif blink_count < 5:
                    coach_placeholder.success("✅ System Active: Monitoring...")
                else:
                    coach_placeholder.info("💡 Remember the 20-20-20 rule!")
            else:
                # Waiting for connection
                ear_placeholder.markdown("<div class='metric-value'>--.---</div>", unsafe_allow_html=True)
                blink_placeholder.markdown("<div class='metric-value'>0</div>", unsafe_allow_html=True)
                coach_placeholder.info("⏳ Waiting for camera connection...")
                
        except Exception as e:
            st.error(f"WebRTC Error: {str(e)}")
            st.info("Try refreshing the page or check browser camera permissions.")
            
    else:
        st.info("📹 System standby. Enable the monitor in the sidebar to begin.")

st.divider()
st.markdown("<p style='text-align: center; color: #666; font-size: 12px;'>VisionMate FYP | BAXU 3973 | UTeM</p>", 
           unsafe_allow_html=True)
