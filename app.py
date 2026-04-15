import streamlit as st
import cv2
import numpy as np
from detector import EyeStrainDetector
import time

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
    st.markdown("##VisionMate Control")
    run_monitor = st.checkbox("Enable Live AI Monitor", value=True)
    st.divider()

st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>VISIONMATE</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1.6, 1])

with col1:
    st.subheader("Live Vision Stream")
    FRAME_WINDOW = st.image([])
    status_placeholder = st.empty()

with col2:
    st.subheader("Session Analytics")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("<p style='font-size: 12px; opacity: 0.6; letter-spacing: 1px;'>EYE ASPECT RATIO</p>", unsafe_allow_html=True)
        ear_placeholder = st.empty()
    with m_col2:
        st.markdown("<p style='font-size: 12px; opacity: 0.6; letter-spacing: 1px;'>TOTAL BLINKS</p>", unsafe_allow_html=True)
        blink_placeholder = st.empty()

    st.divider()
    st.subheader("Coach Recommendations")
    suggestions = st.empty()

@st.cache_resource
def load_detector():
    return EyeStrainDetector()

detector = load_detector()
threshold = 0.20
if 'ear_history' not in st.session_state:
    st.session_state.ear_history = [0.25] * 40
if 'blink_total' not in st.session_state:
    st.session_state.blink_total = 0

if run_monitor:
    cap = cv2.VideoCapture(0)
    blink_active = False 

    while run_monitor:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) 
        ear, landmarks = detector.process_frame(frame)
        
        #data processing
        if ear > 0:
            st.session_state.ear_history.append(ear)
            st.session_state.ear_history = st.session_state.ear_history[-40:]
            
            #blink logic 
            if ear < threshold:
                blink_active = True #eye is currently closed
            else:
                if blink_active:
                    st.session_state.blink_total += 1 #count 1 full blink
                    blink_active = False #reset the state

        ear_placeholder.markdown(f"<div class='metric-value'>{ear:.3f}</div>", unsafe_allow_html=True)
        blink_placeholder.markdown(f"<div class='metric-value'>{st.session_state.blink_total}</div>", unsafe_allow_html=True)
        
        # coach logic
        if ear > 0 and ear < threshold:
            status_placeholder.markdown("<span style='color: #ff4b4b; font-weight: bold;'>STATE: HIGH STRAIN / EYE CLOSED</span>", unsafe_allow_html=True)
            suggestions.warning("Coach: High eye strain detected. Please focus on an object 20 feet away for 20 seconds.")
        elif ear == 0:
            status_placeholder.markdown("<span style='color: #ffd166;'>STATE: SEARCHING FOR USER...</span>", unsafe_allow_html=True)
        else:
            status_placeholder.markdown("<span style='color: #06d6a0; font-weight: bold;'>STATE: OPTIMAL FLOW</span>", unsafe_allow_html=True)
            suggestions.success("Coach: You are doing great! Maintain this blink frequency to avoid CVS.")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb, use_container_width=True)
        
        time.sleep(0.01)

    cap.release()
else:
    status_placeholder.info("Monitoring Paused. Enable via Sidebar.")
