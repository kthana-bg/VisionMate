import streamlit as st
import cv2
import numpy as np
from detector import EyeStrainDetector
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import threading
import time

# Page configuration
st.set_page_config(
    page_title="VisionMate",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism UI
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
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    }
    h1, h2, h3 { color: #E0B0FF !important; font-weight: 300 !important; }
    .metric-value { 
        font-size: 48px; 
        color: #BB86FC; 
        text-shadow: 0 0 10px rgba(187, 134, 252, 0.5);
        text-align: center;
    }
    .status-optimal { color: #00E676; }
    .status-warning { color: #FFD600; }
    .status-danger { color: #FF1744; }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if "detector" not in st.session_state:
    st.session_state.detector = EyeStrainDetector()
if "ear_history" not in st.session_state:
    st.session_state.ear_history = [0.25] * 40
if "total_blinks" not in st.session_state:
    st.session_state.total_blinks = 0
if "run_monitor" not in st.session_state:
    st.session_state.run_monitor = True

# Sidebar
with st.sidebar:
    st.markdown("## VisionMate Control")
    
    run_monitor = st.checkbox("Enable Live AI Monitor", 
                              value=st.session_state.run_monitor,
                              key="monitor_toggle")
    st.session_state.run_monitor = run_monitor
    
    # EAR Threshold slider
    threshold = st.slider("Blink Sensitivity (EAR Threshold)", 
                         min_value=0.15, max_value=0.30, 
                         value=0.20, step=0.01)
    
    if st.button("Reset Session Stats", use_container_width=True):
        st.session_state.detector.reset_blink_count()
        st.session_state.total_blinks = 0
        st.session_state.ear_history = [0.25] * 40
        st.rerun()
    
    st.divider()
    st.markdown("### About")
    st.info("VisionMate monitors your eye strain using AI-powered eye tracking.")

# Main Layout
st.markdown("<h1 style='text-align: center; text-shadow: 0 0 20px rgba(187, 134, 252, 0.5);'>VISIONMATE</h1>", 
            unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #B0B0B0;'>AI Eye-Strain Monitor & Ergonomic Coach</p>", 
            unsafe_allow_html=True)

col1, col2 = st.columns([1.8, 1])

# Shared state for communication between transformer and main thread
class SharedState:
    def __init__(self):
        self.blink_count = 0
        self.current_ear = 0.25
        self.status = "Initializing..."
        self.lock = threading.Lock()
        
    def update(self, blink_count, ear, status):
        with self.lock:
            self.blink_count = blink_count
            self.current_ear = ear
            self.status = status
            
    def get(self):
        with self.lock:
            return self.blink_count, self.current_ear, self.status

shared_state = SharedState()

# Video Processor Class
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.detector = st.session_state.detector
        self.threshold = 0.20
        self.frame_count = 0
        self.last_update = time.time()
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Mirror flip for natural feel
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        # Resize for performance (optional)
        scale = 0.5
        img_small = cv2.resize(img, (int(w*scale), int(h*scale)))
        
        # Process frame
        ear, landmarks, processed_img = self.detector.process_frame(img_small)
        
        # Scale back up if needed
        if scale != 1.0:
            processed_img = cv2.resize(processed_img, (w, h))
        
        # Update blink state
        blink_count, is_blinking = self.detector.update_blink_state(ear, self.threshold)
        
        # Determine status
        if ear == 0:
            color = (128, 128, 128)
            status = "NO FACE DETECTED"
            status_class = "status-warning"
        elif ear < self.threshold:
            color = (255, 50, 50)  # Red - strain
            status = "HIGH STRAIN"
            status_class = "status-danger"
        else:
            color = (50, 255, 150)  # Green - optimal
            status = "OPTIMAL"
            status_class = "status-optimal"
        
        # Draw UI overlay
        # Border
        cv2.rectangle(processed_img, (0, 0), (w-1, h-1), color, 4)
        
        # Info panel background
        overlay = processed_img.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, processed_img, 0.4, 0, processed_img)
        
        # Text overlays
        cv2.putText(processed_img, f"EAR: {ear:.3f}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(processed_img, f"Blinks: {blink_count}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(processed_img, status, (20, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Update shared state (throttled)
        current_time = time.time()
        if current_time - self.last_update > 0.5:  # Update every 500ms
            shared_state.update(blink_count, ear, status)
            self.last_update = current_time
        
        return processed_img

# Analytics Column
with col2:
    st.subheader("Session Analytics")
    
    # Metrics
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.markdown("<p style='font-size:12px; opacity:0.7; text-align:center;'>CURRENT EAR</p>", 
                   unsafe_allow_html=True)
        ear_placeholder = st.empty()
        
    with m_col2:
        st.markdown("<p style='font-size:12px; opacity:0.7; text-align:center;'>TOTAL BLINKS</p>", 
                   unsafe_allow_html=True)
        blink_placeholder = st.empty()
    
    # EAR Chart
    st.divider()
    st.markdown("<p style='font-size:12px; opacity:0.7;'>EAR HISTORY</p>", 
               unsafe_allow_html=True)
    chart_placeholder = st.empty()
    
    # Coach message
    st.divider()
    st.subheader("Real-time Coach")
    coach_placeholder = st.empty()

# Video Feed Column
with col1:
    st.subheader("Live AI Vision Feed")
    
    if st.session_state.run_monitor:
        # WebRTC configuration for stability
        rtc_config = {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]}
            ]
        }
        
        media_constraints = {
            "video": {
                "width": {"ideal": 640, "max": 1280},
                "height": {"ideal": 480, "max": 720},
                "frameRate": {"ideal": 15, "max": 30}
            },
            "audio": False
        }
        
        ctx = webrtc_streamer(
            key="visionmate-v2",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            rtc_configuration=rtc_config,
            media_stream_constraints=media_constraints,
            async_processing=True,
        )
        
        # Update dashboard from shared state
        blink_count, current_ear, status = shared_state.get()
        
        # Update metrics
        ear_value = current_ear if current_ear > 0 else 0.25
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
        chart_data = {"EAR": st.session_state.ear_history}
        chart_placeholder.line_chart(chart_data, height=150, use_container_width=True)
        
        # Coach feedback
        if status == "NO FACE DETECTED":
            coach_placeholder.warning("Please position your face in front of the camera")
        elif status == "HIGH STRAIN":
            coach_placeholder.error("Eye strain detected! Consider taking a break.")
        elif blink_count < 5:
            coach_placeholder.success("System Active: Monitoring your eye health...")
        else:
            coach_placeholder.info("Tip: Follow the 20-20-20 rule - Every 20 mins, look at something 20 feet away for 20 seconds")
            
    else:
        st.info("System standby. Enable the monitor in the sidebar to begin.")
        st.image("https://img.icons8.com/color/480/null/webcam.png", width=200)

# Footer
st.divider()
st.markdown("<p style='text-align: center; color: #666; font-size: 12px;'>VisionMate FYP | BAXU 3973 | Faculty of AI & Cyber Security - UTeM</p>", 
           unsafe_allow_html=True)
