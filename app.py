import streamlit as st
import cv2
import numpy as np
from detector import EyeStrainDetector
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

st.set_page_config(
    page_title="VisionMate",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism UI
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(26, 26, 46, 0.8), rgba(26, 26, 46, 0.8)), 
                    url("https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
    }
    section[data-testid="stSidebar"] {
        background: rgba(40, 20, 80, 0.5) !important;
        backdrop-filter: blur(20px) !important;
    }
    [data-testid="column"] > div {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(25px);
        border-radius: 24px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        padding: 30px !important;
    }
    h1, h2, h3 { color: #E0B0FF !important; font-weight: 300 !important; }
    .metric-value { font-size: 48px; color: #BB86FC; text-shadow: 0 0 10px rgba(187, 134, 252, 0.5); }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialize Session State properly
if "blink_total" not in st.session_state:
    st.session_state.blink_total = 0
if "ear_history" not in st.session_state:
    st.session_state.ear_history = [0.25] * 40

# Sidebar
with st.sidebar:
    st.markdown("## VisionMate Control")
    run_monitor = st.checkbox("Enable Live AI Monitor", value=True)
    if st.button("Reset Session Stats"):
        st.session_state.blink_total = 0
        st.rerun()
    st.divider()

st.markdown("<h1 style='text-align: center;'>VISIONMATE</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1.6, 1])

with col1:
    st.subheader("Live AI Vision Feed")
    # WebRTC container will appear here

with col2:
    st.subheader("Session Analytics")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("<p style='font-size:12px; opacity:0.7;'>CURRENT EAR</p>", unsafe_allow_html=True)
        ear_text = st.empty()
    with m_col2:
        st.markdown("<p style='font-size:12px; opacity:0.7;'>TOTAL BLINKS</p>", unsafe_allow_html=True)
        blink_text = st.empty()

    st.divider()
    st.subheader("Real-time Coach")
    coach_msg = st.empty()

# Detector Logic
@st.cache_resource
def load_detector():
    return EyeStrainDetector()

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.detector = load_detector()
        self.threshold = 0.20
        self.blink_active = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        ear, _ = self.detector.process_frame(img)

        # AI Feedback Logic
        color = (6, 214, 160) # Default Green (Optimal)
        status_label = "STATE: OPTIMAL"

        if ear > 0:
            # Blink Detection logic
            if ear < self.threshold:
                self.blink_active = True
                color = (75, 75, 255) # Red (Strain)
                status_label = "STATE: HIGH STRAIN"
            else:
                if self.blink_active:
                    st.session_state.blink_total += 1
                    self.blink_active = False

            # Draw AI Visuals on the Frame
            cv2.putText(img, f"EAR: {ear:.3f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(img, status_label, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(img, (0, 0), (w, h), color, 4)
        else:
            cv2.putText(img, "SEARCHING FOR FACE...", (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return img

# Main App Execution
if run_monitor:
    with col1:
        ctx = webrtc_streamer(
            key="visionmate-stream",
            # This makes it start automatically
            mode=st.WebRtcMode.SENDRECV, 
            video_processor_factory=VideoProcessor,
            # Enhanced network configuration
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]},
                    {"urls": ["stun:stun.services.mozilla.com"]}
                ]
            },
            media_stream_constraints={
                "video": True, 
                "audio": False
            },
            # This helps the UI stay responsive while the video connects
            async_processing=True, 
        )
    
    # Analytics Update
    blink_text.markdown(f"<div class='metric-value'>{st.session_state.blink_total}</div>", unsafe_allow_html=True)
    
    if st.session_state.blink_total < 5:
        coach_msg.success("You are maintaining great focus!")
    else:
        coach_msg.warning("Consider taking a short break soon.")

else:
    st.info("System standby. Enable the monitor in the sidebar to begin.")
