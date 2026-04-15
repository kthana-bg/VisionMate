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
    layout="wide",
    initial_sidebar_state="expanded"
)

# CRITICAL: Initialize session state BEFORE any other Streamlit code
def init_session_state():
    """Initialize all session state variables"""
    if "detector" not in st.session_state:
        try:
            st.session_state.detector = EyeStrainDetector()
            print("Detector initialized successfully")  # Debug
        except Exception as e:
            st.error(f"Failed to initialize detector: {e}")
            st.session_state.detector = None
    
    if "ear_history" not in st.session_state:
        st.session_state.ear_history = [0.25] * 40
    if "total_blinks" not in st.session_state:
        st.session_state.total_blinks = 0
    if "run_monitor" not in st.session_state:
        st.session_state.run_monitor = True
    if "shared_blink_count" not in st.session_state:
        st.session_state.shared_blink_count = 0
    if "shared_ear" not in st.session_state:
        st.session_state.shared_ear = 0.25
    if "shared_status" not in st.session_state:
        st.session_state.shared_status = "Initializing..."

# Initialize immediately
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
        if st.session_state.detector:
            st.session_state.detector.reset_blink_count()
        st.session_state.shared_blink_count = 0
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

# Thread-safe shared state using Streamlit's session state
class StreamlitSharedState:
    """Uses Streamlit's session state for cross-thread communication"""
    
    @staticmethod
    def update(blink_count, ear, status):
        st.session_state.shared_blink_count = blink_count
        st.session_state.shared_ear = ear
        st.session_state.shared_status = status
        
    @staticmethod
    def get():
        return (
            st.session_state.get('shared_blink_count', 0),
            st.session_state.get('shared_ear', 0.25),
            st.session_state.get('shared_status', "Initializing...")
        )

# Video Processor - FIXED: Creates its own detector instance
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        # CRITICAL FIX: Create new detector instance instead of using session state
        # WebRTC runs in a separate thread where session_state may not be available
        try:
            self.detector = EyeStrainDetector()
            print("VideoProcessor: Detector created successfully")
        except Exception as e:
            print(f"VideoProcessor: Failed to create detector: {e}")
            self.detector = None
            
        self.threshold = 0.20
        self.frame_count = 0
        self.last_update = time.time()
        self.blink_count = 0
        
    def transform(self, frame):
        # Safety check
        if self.detector is None:
            img = frame.to_ndarray(format="bgr24")
            cv2.putText(img, "ERROR: Detector not loaded", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return img
        
        # Process frame
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Mirror
        h, w, _ = img.shape
        
        # Resize for performance (optional - remove if quality is priority)
        process_frame = cv2.resize(img, (320, 240))
        
        # Detect
        ear, landmarks, _ = self.detector.process_frame(process_frame)
        
        # Scale EAR coordinates back if needed (for visualization)
        # For now, we process on small frame but display on original
        
        # Update blink detection
        if ear > 0:  # Valid detection
            current_blinks, _ = self.detector.update_blink_state(ear, self.threshold)
            self.blink_count = current_blinks
        
        # Determine status and color
        if ear == 0:
            color = (128, 128, 128)
            status = "NO FACE"
        elif ear < self.threshold:
            color = (255, 50, 50)  # Red
            status = "HIGH STRAIN"
        else:
            color = (50, 255, 150)  # Green
            status = "OPTIMAL"
        
        # Draw UI on original frame
        # Border
        cv2.rectangle(img, (0, 0), (w-1, h-1), color, 4)
        
        # Info panel
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # Text
        cv2.putText(img, f"EAR: {ear:.3f}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(img, f"Blinks: {self.blink_count}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(img, status, (20, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Update shared state (throttled)
        current_time = time.time()
        if current_time - self.last_update > 0.3:  # Update every 300ms
            try:
                StreamlitSharedState.update(self.blink_count, ear, status)
            except Exception as e:
                pass  # Ignore update errors
            self.last_update = current_time
        
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
        # Check detector is ready
        if st.session_state.detector is None:
            st.error("⚠️ EyeStrainDetector failed to initialize. Check logs.")
        else:
            # WebRTC with robust configuration
            try:
                ctx = webrtc_streamer(
                    key="visionmate-stable-v3",
                    mode=WebRtcMode.SENDRECV,
                    video_processor_factory=VideoProcessor,
                    rtc_configuration={
                        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                    },
                    media_stream_constraints={
                        "video": {
                            "width": {"ideal": 640},
                            "height": {"ideal": 480},
                            "frameRate": {"ideal": 15}
                        },
                        "audio": False
                    },
                    async_processing=True,
                )
                
                # Get values from shared state
                blink_count, current_ear, status = StreamlitSharedState.get()
                
                # Update UI
                ear_color = "#BB86FC" if status == "OPTIMAL" else "#FF1744" if status == "HIGH STRAIN" else "#FFD600"
                ear_value = current_ear if current_ear > 0 else 0.25
                
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
                    coach_placeholder.warning("Please position your face in front of the camera")
                elif status == "HIGH STRAIN":
                    coach_placeholder.error("Eye strain detected! Take a break.")
                elif blink_count < 5:
                    coach_placeholder.success("System Active: Monitoring...")
                else:
                    coach_placeholder.info("Remember the 20-20-20 rule!")
                    
            except Exception as e:
                st.error(f"WebRTC Error: {str(e)}")
                st.info("Try refreshing the page or check browser camera permissions.")
    else:
        st.info("📹 System standby. Enable the monitor in the sidebar to begin.")
