import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import time
from datetime import datetime

from database_manager import DatabaseManager
from model_comparator import ModelComparator
from utils.face_auth import FaceAuthenticator

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="VisionMate",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}
.main-header {
    font-size: 2.2rem; font-weight: 700; text-align: center;
    background: linear-gradient(135deg, #fff, #a0a0ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.status-card {
    background: rgba(255,255,255,0.1); backdrop-filter: blur(20px);
    border-radius: 20px; padding: 1.5rem;
    border: 1px solid rgba(255,255,255,0.2); text-align: center;
}
.status-normal { color: #4caf50; font-size: 2rem; font-weight: bold; }
.status-danger {
    color: #f44336; font-size: 2rem; font-weight: bold;
    animation: pulse 0.8s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# Session state initialisation
# ------------------------------------------------------------------ #

@st.cache_resource(show_spinner="Loading AI models...")
def _load_comparator():
    """Load .keras models once per server session."""
    try:
        comparator = ModelComparator()
        return comparator
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        # Return a minimal comparator that won't crash the app
        class DummyComparator:
            def __init__(self):
                self.face_landmarker = None
                self.pose_landmarker = None
                self.eye_models = {}
                self.posture_models = {}
                self.active_eye_model = "C1"
                self.active_posture_model = "C2"
            def get_model_status(self):
                return {"active_eye": "C1", "active_posture": "C2",
                        "C1_loaded": False, "C2_loaded": False,
                        "B1_loaded": False, "A1_loaded": False,
                        "B2_loaded": False, "A2_loaded": False}
            def set_active_models(self, eye, posture):
                pass
            def process_frame(self, frame):
                return {"eye": {"C1": {"classification": "UNKNOWN", "fatigue_score": 0, "latency_ms": 0, "ear": 0}},
                        "posture": {"C2": {"status": "UNKNOWN", "slouching_prob": 0, "latency_ms": 0, "angle_y": 0}},
                        "eye_consensus": "UNKNOWN", "posture_consensus": "UNKNOWN", "health_score": 50}
        return DummyComparator()


def _init_state():
    defaults = {
        "logged_in":      False,
        "user_id":        None,
        "user_name":      None,
        "session_id":     None,
        "db":             None,  # Initialize lazily
        "auth":           None,  # Initialize lazily
        # live monitoring state
        "live_eye":       "NORMAL",
        "live_posture":   "GOOD",
        "live_health":    50,
        "alert_eye":      False,
        "alert_posture":  False,
        "last_result":    None,
        "session_start":  None,
        # model selection (defaults: C1 eye, C2 posture)
        "sel_eye_model":     "C1",
        "sel_posture_model": "C2",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    
    # Lazy initialize heavy objects
    if st.session_state.db is None:
        try:
            st.session_state.db = DatabaseManager()
        except Exception as e:
            st.error(f"Database initialization failed: {e}")
    
    if st.session_state.auth is None:
        try:
            st.session_state.auth = FaceAuthenticator()
        except Exception as e:
            st.error(f"Face authenticator initialization failed: {e}")


_init_state()


# ------------------------------------------------------------------ #
# Login / Register page
# ------------------------------------------------------------------ #

def _show_auth_page():
    # Modern dark theme styling
    st.markdown("""
    <style>
    /* Dark theme base */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
    }
    
    /* Header styling */
    .auth-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .auth-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00ff87 0%, #60efff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .auth-subtitle {
        color: #8b92b0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
        border-bottom: 2px solid #2a3042;
        padding: 0 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background-color: transparent;
        border: none;
        color: #8b92b0;
        font-size: 1.1rem;
        font-weight: 500;
        padding: 0 2rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #00ff87;
        border-bottom: 3px solid #00ff87;
    }
    
    /* Card styling */
    .auth-card {
        background: rgba(26, 31, 58, 0.6);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    /* Input styling */
    .stTextInput input {
        background-color: #1e2430 !important;
        border: 1px solid #2a3042 !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        padding: 0.8rem 1rem !important;
        font-size: 1rem !important;
    }
    .stTextInput input:focus {
        border-color: #00ff87 !important;
        box-shadow: 0 0 0 1px #00ff87 !important;
    }
    
    /* Button styling */
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #00ff87 0%, #60efff 100%);
        color: #0a0e27;
        border: none;
        border-radius: 8px;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        margin-top: 1rem;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 255, 135, 0.3);
    }
    
    /* Camera section */
    .camera-section {
        background: #1e2430;
        border: 2px solid #2a3042;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    /* Status messages */
    .status-success {
        background: rgba(0, 255, 135, 0.1);
        border-left: 4px solid #00ff87;
        padding: 1rem;
        border-radius: 8px;
        color: #00ff87;
        margin: 1rem 0;
    }
    .status-error {
        background: rgba(255, 82, 82, 0.1);
        border-left: 4px solid #ff5252;
        padding: 1rem;
        border-radius: 8px;
        color: #ff5252;
        margin: 1rem 0;
    }
    .status-warning {
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        color: #ffc107;
        margin: 1rem 0;
    }
    
    /* Section labels */
    .section-label {
        color: #8b92b0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="auth-header">', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">VisionMate</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-subtitle">AI-Powered Eye Strain & Posture Monitor</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tabs for Login and Register
    tab1, tab2 = st.tabs(["🔐 Login", "✨ Create Account"])
    
    # ==================== LOGIN TAB ====================
    with tab1:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Face Authentication</p>', unsafe_allow_html=True)
        st.markdown('<div class="camera-section">', unsafe_allow_html=True)
        
        login_img = st.camera_input(
            "Position your face in the camera", 
            key="login_cam",
            label_visibility="collapsed"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🔓 Login with Face", key="login_btn", type="primary"):
            if login_img is None:
                st.markdown(
                    '<div class="status-warning">⚠️ Please capture your face first</div>',
                    unsafe_allow_html=True
                )
            else:
                with st.spinner("🔍 Recognizing your face..."):
                    frame = _decode_camera_image(login_img)
                    auth  = st.session_state.auth
                    
                    if auth is None:
                        st.error("Authentication system not initialized")
                        return
                    
                    emb   = auth.extract_embedding(frame)
                    
                    if emb is None:
                        st.markdown(
                            '<div class="status-error">❌ No face detected. Please ensure:<br>'
                            '• Good lighting<br>• Face clearly visible<br>• Look directly at camera</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        reduced = auth.reduce_dimension(emb)
                        users   = st.session_state.db.get_all_users()
                        best_match, best_score = None, 0
                        
                        for u in users:
                            stored = np.array(u["face_embedding"])
                            score  = auth.compare_embeddings(reduced, stored)
                            if score > best_score and score > 0.75:
                                best_score = score
                                best_match = u
                        
                        if best_match:
                            st.markdown(
                                f'<div class="status-success">✅ Welcome back, {best_match["user_name"]}!</div>',
                                unsafe_allow_html=True
                            )
                            time.sleep(1)
                            _do_login(best_match)
                        else:
                            st.markdown(
                                '<div class="status-error">❌ Face not recognized<br>'
                                'Please register first if you\'re a new user</div>',
                                unsafe_allow_html=True
                            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Info section
        st.markdown("""
        <div style="margin-top: 2rem; padding: 1.5rem; background: rgba(0, 255, 135, 0.05); 
                    border-radius: 12px; border: 1px solid rgba(0, 255, 135, 0.2);">
            <h4 style="color: #00ff87; margin-bottom: 1rem;">ℹ️ How Face Login Works</h4>
            <ul style="color: #8b92b0; line-height: 1.8;">
                <li>Secure face recognition using AI</li>
                <li>No passwords needed</li>
                <li>Instant authentication</li>
                <li>Your face data is encrypted and stored locally</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== REGISTER TAB ====================
    with tab2:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Personal Information</p>', unsafe_allow_html=True)
        
        name = st.text_input(
            "Full Name",
            placeholder="Enter your full name",
            key="reg_name"
        )
        
        st.markdown('<p class="section-label" style="margin-top: 1.5rem;">Face Registration</p>', unsafe_allow_html=True)
        st.markdown('<div class="camera-section">', unsafe_allow_html=True)
        
        reg_img = st.camera_input(
            "Capture your face for registration",
            key="reg_cam",
            label_visibility="collapsed"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Registration tips
        st.markdown("""
        <div style="background: rgba(96, 239, 255, 0.1); border-radius: 8px; padding: 1rem; margin: 1rem 0;">
            <p style="color: #60efff; font-weight: 600; margin-bottom: 0.5rem;">📸 Tips for best results:</p>
            <ul style="color: #8b92b0; font-size: 0.9rem; margin: 0;">
                <li>Ensure good lighting</li>
                <li>Remove glasses if possible</li>
                <li>Look directly at the camera</li>
                <li>Keep a neutral expression</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("✨ Create Account", key="reg_btn", type="primary"):
            if not name or len(name.strip()) < 2:
                st.markdown(
                    '<div class="status-warning">⚠️ Please enter your full name (at least 2 characters)</div>',
                    unsafe_allow_html=True
                )
            elif reg_img is None:
                st.markdown(
                    '<div class="status-warning">⚠️ Please capture your face photo</div>',
                    unsafe_allow_html=True
                )
            else:
                with st.spinner("🤖 Processing your face data..."):
                    frame = _decode_camera_image(reg_img)
                    auth  = st.session_state.auth
                    
                    if auth is None:
                        st.error("Authentication system not initialized")
                        return
                    
                    emb   = auth.extract_embedding(frame)
                    
                    if emb is None:
                        st.markdown(
                            '<div class="status-error">❌ No face detected. Please ensure:<br>'
                            '• Good lighting<br>• Face clearly visible<br>• Look directly at camera</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        # Check for duplicate face registration
                        reduced = auth.reduce_dimension(emb)
                        users   = st.session_state.db.get_all_users()
                        
                        duplicate_found = False
                        for u in users:
                            stored = np.array(u["face_embedding"])
                            score  = auth.compare_embeddings(reduced, stored)
                            if score > 0.75:
                                duplicate_found = True
                                st.markdown(
                                    f'<div class="status-error">❌ Face Already Registered!<br><br>'
                                    f'This face is already registered as <strong>{u["user_name"]}</strong><br><br>'
                                    f'Each person can only register once. Please use the Login tab to sign in.</div>',
                                    unsafe_allow_html=True
                                )
                                break
                        
                        if not duplicate_found:
                            # No duplicate found, proceed with registration
                            try:
                                st.session_state.db.create_user(name.strip(), reduced.tolist())
                                st.markdown(
                                    f'<div class="status-success">✅ Account Created Successfully!<br><br>'
                                    f'Welcome, <strong>{name.strip()}</strong>!<br><br>'
                                    f'You can now login using the Login tab.</div>',
                                    unsafe_allow_html=True
                                )
                                st.balloons()
                                time.sleep(2)
                                st.rerun()
                            except Exception as e:
                                st.markdown(
                                    f'<div class="status-error">❌ Registration failed: {str(e)}</div>',
                                    unsafe_allow_html=True
                                )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Security info
        st.markdown("""
        <div style="margin-top: 2rem; padding: 1.5rem; background: rgba(96, 239, 255, 0.05); 
                    border-radius: 12px; border: 1px solid rgba(96, 239, 255, 0.2);">
            <h4 style="color: #60efff; margin-bottom: 1rem;">🔒 Privacy & Security</h4>
            <ul style="color: #8b92b0; line-height: 1.8;">
                <li><strong>One Face, One Account:</strong> Each face can only be registered once</li>
                <li><strong>Encrypted Storage:</strong> Your face data is securely encrypted</li>
                <li><strong>Local Processing:</strong> All data stays on the server</li>
                <li><strong>No Third Parties:</strong> Your data is never shared</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


def _do_login(user: dict):
    db = st.session_state.db
    comparator = _load_comparator()
    # Apply current model selection to the comparator
    comparator.set_active_models(
        st.session_state.sel_eye_model,
        st.session_state.sel_posture_model,
    )
    st.session_state.logged_in    = True
    st.session_state.user_id      = user["user_id"]
    st.session_state.user_name    = user["user_name"]
    st.session_state.session_id   = db.start_session(user["user_id"])
    st.session_state.session_start = time.time()
    st.session_state.comparator   = comparator
    st.rerun()


def _decode_camera_image(cam_file) -> np.ndarray:
    """Convert st.camera_input file to a BGR numpy array."""
    data = np.frombuffer(cam_file.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ------------------------------------------------------------------ #
# Live Monitor tab
# ------------------------------------------------------------------ #

def _live_monitor_tab():
    st.markdown("### Live Ergonomic Monitoring")

    comparator = st.session_state.get("comparator")
    if comparator is None:
        st.error("Model not loaded. Please log out and back in.")
        return

    # Show current model selection
    status = comparator.get_model_status()
    c1_ok  = status["C1_loaded"]
    c2_ok  = status["C2_loaded"]
    st.caption(
        f"Active models: Eye = **{comparator.EYE_MODELS[status['active_eye']]}** "
        f"{'(loaded)' if c1_ok else '(file missing - using EAR rule)'}  |  "
        f"Posture = **{comparator.POSTURE_MODELS[status['active_posture']]}** "
        f"{'(loaded)' if c2_ok else '(file missing - using angle rule)'}"
    )

    # ---- Status cards ----
    col_eye, col_posture, col_health = st.columns(3)
    eye_ok     = st.session_state.live_eye     == "NORMAL"
    posture_ok = st.session_state.live_posture == "GOOD"
    health     = st.session_state.live_health

    with col_eye:
        if not eye_ok:
            st.markdown("""
            <div class="status-card">
                <div class="status-danger">STRAINED</div>
                <div style="color:#f44336;">Eye Strain Detected</div>
                <div style="color:rgba(255,255,255,0.6);font-size:0.8rem;">
                    Take a 20-second break</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-card">
                <div class="status-normal">NORMAL</div>
                <div style="color:#4caf50;">Eye Status Healthy</div>
                <div style="color:rgba(255,255,255,0.6);font-size:0.8rem;">
                    Continue good habits</div>
            </div>""", unsafe_allow_html=True)

    with col_posture:
        if not posture_ok:
            st.markdown("""
            <div class="status-card">
                <div class="status-danger">SLOUCHING</div>
                <div style="color:#f44336;">Poor Posture Detected</div>
                <div style="color:rgba(255,255,255,0.6);font-size:0.8rem;">
                    Sit up straight</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-card">
                <div class="status-normal">GOOD</div>
                <div style="color:#4caf50;">Posture Correct</div>
                <div style="color:rgba(255,255,255,0.6);font-size:0.8rem;">
                    Maintain this position</div>
            </div>""", unsafe_allow_html=True)

    with col_health:
        hc = "#4caf50" if health >= 70 else "#ff9800" if health >= 40 else "#f44336"
        hs = "Good" if health >= 70 else "Moderate" if health >= 40 else "Critical"
        st.markdown(f"""
        <div class="status-card">
            <div style="font-size:2rem;font-weight:bold;color:{hc};">{health}</div>
            <div style="color:{hc};">Health Score</div>
            <div style="color:rgba(255,255,255,0.6);font-size:0.8rem;">
                Status: {hs}</div>
        </div>""", unsafe_allow_html=True)

    if not eye_ok:
        st.warning("Eye Strain Detected. Look away from screen for 20 seconds.")
    if not posture_ok:
        st.warning("Poor Posture Detected. Sit up straight and support your back.")

    st.divider()

    # ---- Camera capture + processing ----
    st.markdown("#### Capture Frame for Analysis")
    st.info(
        "Click the camera button below to capture a frame. "
        "Your custom AI models will analyse your eye strain and posture instantly."
    )

    cam_img = st.camera_input(
        "Click the shutter button to analyse",
        key="monitor_cam",
    )

    if cam_img is not None:
        frame = _decode_camera_image(cam_img)

        with st.spinner("Running AI analysis..."):
            result = comparator.process_frame(frame)

        # Update session state for status cards
        st.session_state.live_eye     = result["eye"]["C1"]["classification"]
        st.session_state.live_posture = result["posture"]["C2"]["status"]
        st.session_state.live_health  = result["health_score"]
        st.session_state.last_result  = result

        # Show the annotated frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            st.image(frame_rgb, channels="RGB", use_container_width=True,
                     caption="AI Analysis Result")
        except TypeError:
            st.image(frame_rgb, channels="RGB", use_column_width=True,
                     caption="AI Analysis Result")

        # Log to database
        db  = st.session_state.db
        sid = st.session_state.session_id
        uid = st.session_state.user_id
        if sid and uid:
            try:
                db.log_model_comparison(sid, uid, result)
            except Exception as e:
                print(f"DB log error: {e}")

        # Alert cooldown logic
        eye_strained   = result["eye"]["C1"]["classification"] == "STRAINED"
        posture_poor   = result["posture"]["C2"]["status"]     == "SLOUCHING"
        now            = time.time()
        last_alert     = st.session_state.get("last_alert_time", 0)
        if now - last_alert > 10:
            st.session_state.alert_eye     = eye_strained
            st.session_state.alert_posture = posture_poor
            if eye_strained or posture_poor:
                st.session_state.last_alert_time = now

        st.rerun()

    # ---- Model result detail cards ----
    if st.session_state.get("last_result"):
        st.divider()
        st.markdown("#### Model Output Details")
        r      = st.session_state.last_result
        c1     = r["eye"]["C1"]
        c2     = r["posture"]["C2"]
        dc1, dc2 = st.columns(2)

        ec = "#f44336" if c1["classification"] == "STRAINED" else "#4caf50"
        with dc1:
            st.markdown(f"""
            <div class="status-card" style="text-align:left;">
                <b>C1 - Custom Eye CNN</b><br>
                Fatigue score : {c1['fatigue_score']:.3f}<br>
                Status : <span style="color:{ec};">{c1['classification']}</span><br>
                Latency : {c1['latency_ms']:.1f} ms<br>
                EAR : {c1.get('ear', 0):.3f}
            </div>""", unsafe_allow_html=True)

        pc = "#f44336" if c2["status"] == "SLOUCHING" else "#4caf50"
        with dc2:
            st.markdown(f"""
            <div class="status-card" style="text-align:left;">
                <b>C2 - Custom LSTM Posture</b><br>
                Slouch prob : {c2['slouching_prob']:.3f}<br>
                Status : <span style="color:{pc};">{c2['status']}</span><br>
                Latency : {c2['latency_ms']:.1f} ms<br>
                Neck angle : {c2.get('angle_y', 0):.1f} deg
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Quick Tips")
    t1, t2, t3 = st.columns(3)
    t1.markdown("**20-20-20 Rule**")
    t1.caption("Every 20 min, look 20 ft away for 20 seconds.")
    t2.markdown("**Ergonomic Setup**")
    t2.caption("Top of screen at eye level, back supported.")
    t3.markdown("**Take Breaks**")
    t3.caption("Stand and stretch every 30-60 minutes.")


# ------------------------------------------------------------------ #
# Comparative Analysis tab
# ------------------------------------------------------------------ #

def _comparison_tab():
    st.markdown("### Comparative Analysis")
    st.caption(
        "Select which trained model to use for live monitoring. "
        "Currently only your custom models (C1 and C2) are available."
    )

    comparator = st.session_state.get("comparator")
    if comparator is None:
        st.warning("Please log in first.")
        return

    status = comparator.get_model_status()

    st.markdown("#### Active Model Selection")
    st.markdown(
        "The selected models are applied immediately to the Live Monitoring tab."
    )

    col_eye, col_posture = st.columns(2)

    with col_eye:
        st.markdown("**Eye Strain Model**")
        eye_choice = st.radio(
            "Eye model",
            options=list(comparator.EYE_MODELS.keys()),
            format_func=lambda k: comparator.EYE_MODELS[k],
            index=list(comparator.EYE_MODELS.keys()).index(
                st.session_state.sel_eye_model
            ),
            key="radio_eye",
            label_visibility="collapsed",
        )
        loaded = status.get("C1_loaded") if eye_choice == "C1" else False
        if loaded:
            st.success(f"{comparator.EYE_MODELS[eye_choice]} — model loaded and ready.")
        else:
            st.warning(
                f"{comparator.EYE_MODELS[eye_choice]} — model file not found. "
                "EAR rule-based fallback will be used."
            )

    with col_posture:
        st.markdown("**Posture Model**")
        posture_choice = st.radio(
            "Posture model",
            options=list(comparator.POSTURE_MODELS.keys()),
            format_func=lambda k: comparator.POSTURE_MODELS[k],
            index=list(comparator.POSTURE_MODELS.keys()).index(
                st.session_state.sel_posture_model
            ),
            key="radio_posture",
            label_visibility="collapsed",
        )
        loaded = status.get("C2_loaded") if posture_choice == "C2" else False
        if loaded:
            st.success(f"{comparator.POSTURE_MODELS[posture_choice]} — model loaded and ready.")
        else:
            st.warning(
                f"{comparator.POSTURE_MODELS[posture_choice]} — model file not found. "
                "Angle rule-based fallback will be used."
            )

    if st.button("Apply Selection", type="primary", use_container_width=True):
        st.session_state.sel_eye_model     = eye_choice
        st.session_state.sel_posture_model = posture_choice
        comparator.set_active_models(eye_choice, posture_choice)
        st.success(
            f"Now using: {comparator.EYE_MODELS[eye_choice]} (eye)  +  "
            f"{comparator.POSTURE_MODELS[posture_choice]} (posture)"
        )

    # ---- Session performance stats ----
    st.divider()
    st.markdown("#### Session Performance")
    sid = st.session_state.get("session_id")
    if sid:
        try:
            latencies = st.session_state.db.get_average_latencies(sid)
            if any(v > 0 for v in latencies.values()):
                lat_data = {
                    "Model": ["C1 Eye CNN", "C2 Posture LSTM"],
                    "Avg Latency (ms)": [
                        round(latencies.get("C1", 0), 2),
                        round(latencies.get("C2", 0), 2),
                    ],
                }
                df_lat = pd.DataFrame(lat_data)
                fig = px.bar(
                    df_lat, x="Model", y="Avg Latency (ms)",
                    title="Average Inference Latency This Session",
                    color="Model",
                    color_discrete_map={
                        "C1 Eye CNN":       "#3498db",
                        "C2 Posture LSTM":  "#e74c3c",
                    },
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Capture frames in the Live Monitor tab to see latency data.")
        except Exception as e:
            st.info(f"No session performance data yet. ({e})")
    else:
        st.info("Start a session to see performance metrics.")


# ------------------------------------------------------------------ #
# Analytics tab
# ------------------------------------------------------------------ #

def _analytics_tab():
    st.markdown("### Analytics Dashboard")

    hours = st.selectbox(
        "Time range",
        [1, 6, 12, 24, 48, 72],
        index=3,
        format_func=lambda x: f"Last {x} hours",
    )

    try:
        uid   = st.session_state.user_id
        stats = st.session_state.db.get_strain_statistics(uid, hours=hours)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Eye Strain Events",  stats["eye_strain_count"])
        c2.metric("Poor Posture Events",stats["posture_poor_count"])
        c3.metric("Eye Strain Rate",    f"{stats['eye_strain_percentage']:.1f}%")
        c4.metric("Poor Posture Rate",  f"{stats['posture_poor_percentage']:.1f}%")

        st.divider()

        df = st.session_state.db.get_user_analytics(uid, hours=hours)
        if not df.empty:
            fig1 = px.line(
                df, x="timestamp", y="eye_score",
                title="Eye Strain Score Over Time",
                labels={"eye_score": "Fatigue Score", "timestamp": "Time"},
            )
            fig1.add_hline(y=0.5, line_dash="dash", line_color="red",
                           annotation_text="Strain threshold")
            fig1.update_layout(
                height=380,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
            )
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.line(
                df, x="timestamp", y="posture_score",
                title="Posture Slouch Score Over Time",
                labels={"posture_score": "Slouch Score", "timestamp": "Time"},
            )
            fig2.add_hline(y=0.5, line_dash="dash", line_color="red",
                           annotation_text="Slouch threshold")
            fig2.update_layout(
                height=380,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
            )
            st.plotly_chart(fig2, use_container_width=True)

            if st.button("Export CSV", type="primary"):
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV", csv,
                    f"visionmate_{st.session_state.user_name}_{hours}h.csv",
                    "text/csv",
                )
        else:
            st.info("No data yet. Capture frames in Live Monitor to generate analytics.")

    except Exception as e:
        st.error(f"Analytics error: {e}")


# ------------------------------------------------------------------ #
# Main dashboard
# ------------------------------------------------------------------ #

def _show_dashboard():
    col_h, col_btn = st.columns([4, 1])
    with col_h:
        st.markdown('<div class="main-header">VisionMate Dashboard</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align:center;color:rgba(255,255,255,0.7);">'
            'Real-time Health Monitoring</div>',
            unsafe_allow_html=True,
        )
    with col_btn:
        if st.button("Logout", use_container_width=True):
            sid = st.session_state.session_id
            if sid:
                st.session_state.db.end_session(sid)
            for k in ["logged_in","user_id","user_name","session_id",
                      "comparator","last_result","session_start"]:
                st.session_state[k] = None if k != "logged_in" else False
            st.session_state.live_eye     = "NORMAL"
            st.session_state.live_posture = "GOOD"
            st.session_state.live_health  = 50
            st.session_state.alert_eye    = False
            st.session_state.alert_posture= False
            st.rerun()

    st.divider()

    if st.session_state.session_id:
        try:
            info = st.session_state.db.get_current_session(
                st.session_state.session_id
            )
            mins = info.get("duration_minutes", 0) or 0
            if st.session_state.session_start:
                mins = int((time.time() - st.session_state.session_start) / 60)
            st.caption(
                f"Session active: {mins} min  |  "
                f"User: {st.session_state.user_name}"
            )
        except Exception:
            pass

    tab_live, tab_compare, tab_analytics = st.tabs([
        "Live Monitor",
        "Comparative Analysis",
        "Analytics",
    ])

    with tab_live:
        _live_monitor_tab()

    with tab_compare:
        _comparison_tab()

    with tab_analytics:
        _analytics_tab()


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main():
    try:
        if st.session_state.get("logged_in") and st.session_state.get("comparator"):
            _show_dashboard()
        else:
            _show_auth_page()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please refresh the page.")


if __name__ == "__main__":
    main()
