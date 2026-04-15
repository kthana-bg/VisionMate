import streamlit as st
import cv2
import numpy as np
import av
import threading
import time
from detector import EyeStrainDetector
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# --- PAGE SETUP ---
st.set_page_config(
    page_title="VisionMate | AI Eye-Strain Monitor",
    page_icon="👁️",
    layout="wide"
)

# --- HUMANIZED CSS (Making it look modern and dark) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    /* Style the containers for a 'Glass' look */
    [data-testid="column"] > div {
        background: rgba(255, 255, 255, 0.05);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    /* Big Metric Styles */
    .metric-box {
        text-align: center;
        padding: 10px;
    }
    .metric-val {
        font-size: 2.5rem;
        font-weight: bold;
        color: #bb86fc;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        opacity: 0.7;
    }
    /* Hide the ugly default Streamlit WebRTC buttons */
    button[kind="secondary"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- SHARED STATE MANAGEMENT ---
# Since the camera runs in a background thread, we need a safe way to move 
# data from the camera to the dashboard UI.
class SessionData:
    def __init__(self):
        self.lock = threading.Lock()
        self.ear = 0.25
        self.blinks = 0
        self.status = "Initializing"
        self.history = [0.25] * 50

    def update(self, ear, blinks, status):
        with self.lock:
            self.ear = ear
            self.blinks = blinks
            self.status = status
            self.history = self.history[1:] + [ear]

# Initialize session data once
if "data" not in st.session_state:
    st.session_state.data = SessionData()

# Cache the detector so we don't reload the AI model every time the screen flickers
@st.cache_resource
def load_detector():
    return EyeStrainDetector()

# --- THE VIDEO ENGINE ---
class VideoProcessor:
    def __init__(self):
        self.detector = load_detector()
        self.threshold = 0.20 # Sensitivity

    def recv(self, frame):
        # 1. Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror effect
        
        # 2. Run the AI Detection
        ear, landmarks, annotated_img = self.detector.process_frame(img)
        
        # 3. Handle Blink Logic
        if ear > 0:
            blinks, _ = self.detector.update_blink_state(ear, self.threshold)
            status = "OPTIMAL" if ear >= self.threshold else "HIGH STRAIN"
            # Send data to the UI thread
            st.session_state.data.update(ear, blinks, status)
        else:
            st.session_state.data.update(0.0, st.session_state.data.blinks, "NO FACE")

        # 4. Return the processed frame to the browser
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# --- UI LAYOUT ---
st.title("VISIONMATE")
st.markdown("AI-Powered Eye Health Assistant")

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Live Feed")
    # This starts the camera automatically
    ctx = webrtc_streamer(
        key="visionmate-v1",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        desired_playing_state=True, # Auto-starts camera
    )

with col2:
    st.subheader("Real-time Stats")
    
    # Create empty slots so we can update values without refreshing the whole page
    ear_slot = st.empty()
    blink_slot = st.empty()
    status_slot = st.empty()
    chart_slot = st.empty()
    coach_slot = st.empty()

    # Loop to pull data from the camera thread and show it on the UI
    if ctx.state.playing:
        while ctx.state.playing:
            # Grab the latest numbers
            ear = st.session_state.data.ear
            blinks = st.session_state.data.blinks
            status = st.session_state.data.status
            history = st.session_state.data.history

            # Update UI components
            ear_slot.markdown(f'<div class="metric-box"><div class="metric-label">Eye Aperture (EAR)</div><div class="metric-val">{ear:.3f}</div></div>', unsafe_allow_html=True)
            blink_slot.markdown(f'<div class="metric-box"><div class="metric-label">Total Blinks</div><div class="metric-val">{blinks}</div></div>', unsafe_allow_html=True)
            
            # Status styling
            color = "#00E676" if status == "OPTIMAL" else "#FF1744"
            status_slot.markdown(f"<h3 style='text-align:center; color:{color};'>{status}</h3>", unsafe_allow_html=True)
            
            # Chart - using 'width="stretch"' for 2025 compatibility
            chart_slot.line_chart(history, height=150, width="stretch")

            # Coaching Logic
            if status == "HIGH STRAIN":
                coach_slot.error("⚠️ You aren't blinking enough! Look away for 20 seconds.")
            elif blinks < 5:
                coach_slot.info("System monitoring... Keep your head centered.")
            else:
                coach_slot.success("Doing great! Remember the 20-20-20 rule.")
            
            time.sleep(0.1) # Prevents the CPU from exploding
    else:
        st.warning("Camera is currently stopped. Please allow access to start monitoring.")

st.divider()
st.caption("VisionMate FYP | Powered by MediaPipe & Streamlit")
