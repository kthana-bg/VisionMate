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

with st.sidebar:
    st.markdown("## VisionMate Control")
    run_monitor = st.checkbox("Enable Live AI Monitor", value=True)
    st.divider()


# ================= TITLE =================
st.markdown("<h1 style='text-align: center;'>VISIONMATE</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1.6, 1])

with col1:
    st.subheader("Live Vision Stream")
    status_placeholder = st.empty()

with col2:
    st.subheader("Session Analytics")

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown("<p style='font-size:12px;'>EYE ASPECT RATIO</p>", unsafe_allow_html=True)
        ear_placeholder = st.empty()

    with m_col2:
        st.markdown("<p style='font-size:12px;'>TOTAL BLINKS</p>", unsafe_allow_html=True)
        blink_placeholder = st.empty()

    st.divider()
    st.subheader("Coach Recommendations")
    suggestions = st.empty()


# ================= SESSION STATE =================
if "blink_total" not in st.session_state:
    st.session_state.blink_total = 0

if "ear_history" not in st.session_state:
    st.session_state.ear_history = [0.25] * 40


# ================= DETECTOR =================
@st.cache_resource
def load_detector():
    return EyeStrainDetector()

detector = load_detector()
threshold = 0.20


# ================= VIDEO PROCESSOR =================
class VideoProcessor(VideoTransformerBase):

    def __init__(self):
        self.detector = load_detector()
        self.threshold = 0.20
        self.blink_active = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        ear, landmarks = self.detector.process_frame(img)

        # update history
        if ear > 0:
            st.session_state.ear_history.append(ear)
            st.session_state.ear_history = st.session_state.ear_history[-40:]

            # blink detection
            if ear < self.threshold:
                self.blink_active = True
            else:
                if self.blink_active:
                    st.session_state.blink_total += 1
                    self.blink_active = False

        # update UI
        ear_placeholder.markdown(
            f"<div class='metric-value'>{ear:.3f}</div>",
            unsafe_allow_html=True
        )

        blink_placeholder.markdown(
            f"<div class='metric-value'>{st.session_state.blink_total}</div>",
            unsafe_allow_html=True
        )

        # coach logic
        if ear > 0 and ear < self.threshold:
            status_placeholder.markdown(
                "<span style='color:#ff4b4b;font-weight:bold;'>STATE: HIGH STRAIN / EYE CLOSED</span>",
                unsafe_allow_html=True
            )
            suggestions.warning(
                "Coach: High eye strain detected. Look 20 feet away for 20 seconds."
            )

        elif ear == 0:
            status_placeholder.markdown(
                "<span style='color:#ffd166;'>STATE: SEARCHING FOR USER...</span>",
                unsafe_allow_html=True
            )

        else:
            status_placeholder.markdown(
                "<span style='color:#06d6a0;font-weight:bold;'>STATE: OPTIMAL FLOW</span>",
                unsafe_allow_html=True
            )
            suggestions.success(
                "Coach: Good blink rhythm. Keep it up!"
            )

        return img


# ================= MAIN APP =================
if run_monitor:

    webrtc_streamer(
        key="visionmate",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

else:
    status_placeholder.info("Monitoring Paused. Enable via Sidebar.")
