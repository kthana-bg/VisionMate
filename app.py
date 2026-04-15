import streamlit as st
import cv2
import numpy as np
from detector import EyeStrainDetector
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

st.set_page_config(
    page_title="VisionMate", 
    layout="wide", 
    initial_sidebar_state="expanded" 
)

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(26, 26, 46, 0.8), rgba(26, 26, 46, 0.8)), 
                    url("https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
    }

    /* Visibality of the sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(40, 20, 80, 0.5) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        z-index: 100;
    }

    /* Sidebar text */
    section[data-testid="stSidebar"] .stText, section[data-testid="stSidebar"] label {
        color: #E0B0FF !important;
    }

    /* Containers */
    [data-testid="column"] > div {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(25px) saturate(170%) !important;
        -webkit-backdrop-filter: blur(25px) saturate(170%) !important;
        border-radius: 24px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        padding: 30px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5) !important;
        //margin-bottom: 1rem;
    }

    /* Toggle button of the sidebar */
    header[data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
        visibility: visible !important;
    }
    
    button[kind="headerNoContext"] {
        background-color: rgba(187, 134, 252, 0.2) !important;
        border-radius: 50% !important;
        color: white !important;
    }

    h1, h2, h3 {
        color: #E0B0FF !important;
        font-family: 'Inter', sans-serif;
        font-weight: 300 !important;
    }

    .metric-value {
        font-size: 52px;
        font-weight: 200;
        color: #BB86FC;
        text-shadow: 0 0 20px rgba(187, 134, 252, 0.6);
    }

    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.detector = EyeStrainDetector()
        self.threshold = 0.20
        self.blink_total = 0
        self.blink_active = False
        self.ear = 0.0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        ear, _ = self.detector.process_frame(img)
        self.ear = ear

        if ear > 0:
            if ear < self.threshold:
                self.blink_active = True
            else:
                if self.blink_active:
                    self.blink_total += 1
                    self.blink_active = False

        # Draw on video
        cv2.putText(img, f"EAR: {ear:.3f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.putText(img, f"Blinks: {self.blink_total}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        return img

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## VisionMate Control")
    run_monitor = st.checkbox("Enable Live AI Monitor", value=True)

# ---------------- MAIN UI ----------------
st.markdown("<h1 style='text-align:center;'>VISIONMATE</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1.6, 1])

with col1:
    st.subheader("Live Vision Stream")

with col2:
    st.subheader("Session Analytics")
    ear_display = st.empty()
    blink_display = st.empty()
    status_display = st.empty()
    suggestion_display = st.empty()

# ---------------- WEBRTC ----------------
if run_monitor:
    ctx = webrtc_streamer(
        key="visionmate",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    # ---------------- LIVE DATA DISPLAY ----------------
    if ctx.video_processor:
        processor = ctx.video_processor

        ear_display.markdown(
            f"<div class='metric-value'>EAR: {processor.ear:.3f}</div>",
            unsafe_allow_html=True
        )

        blink_display.markdown(
            f"<div class='metric-value'>Blinks: {processor.blink_total}</div>",
            unsafe_allow_html=True
        )

        # Status Logic
        if processor.ear == 0:
            status_display.warning("Searching for face...")
        elif processor.ear < 0.20:
            status_display.error("HIGH STRAIN / EYES CLOSED")
            suggestion_display.warning("Look 20 feet away for 20 seconds 👀")
        else:
            status_display.success("Optimal Condition ✅")
            suggestion_display.success("Good blinking rate 👍")

else:
    st.info("Monitoring Paused")
