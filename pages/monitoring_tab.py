import os
import sys
import time

import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.frame_processor import (
    FrameResult,
    FrameProcessor,
    VisionMateTransformer,
    WEBRTC_AVAILABLE,
    load_mediapipe_landmarkers,
)
from utils.voice_guidance import voice_guidance
from database.db_manager import save_health_metric


# ------------------------------------------------------------------ #
# UI helpers
# ------------------------------------------------------------------ #

def _status_color(status: str, good_value: str = "Normal") -> str:
    return "#2ecc71" if status == good_value else "#e74c3c"


def _health_color(score: float) -> str:
    if score >= 75:
        return "#2ecc71"
    if score >= 50:
        return "#f39c12"
    return "#e74c3c"


def _metric_card(label: str, value: str, color: str, sub: str = ""):
    sub_html = (
        f"<p style='font-size:12px;color:#aaa;margin:2px 0 0 0;'>{sub}</p>"
        if sub else ""
    )
    st.markdown(
        f"""
        <div style="
            background:#1e2130;
            border-left:4px solid {color};
            border-radius:8px;
            padding:14px 16px;
            margin-bottom:10px;">
          <p style="font-size:11px;color:#aaa;margin:0 0 4px 0;
                    text-transform:uppercase;letter-spacing:1px;">{label}</p>
          <p style="font-size:24px;font-weight:bold;color:{color};margin:0;">{value}</p>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metrics(result: FrameResult, eye_name: str, posture_name: str):
    """Right-hand metrics panel."""
    _metric_card(
        "Health Score",
        f"{result.health_score:.0f} / 100",
        _health_color(result.health_score),
        "Combined eye and posture",
    )
    _metric_card(
        "Eye Status",
        result.eye_status,
        _status_color(result.eye_status, "Normal"),
        f"EAR: {result.ear_value:.3f}",
    )
    _metric_card(
        "Posture Status",
        result.posture_status,
        _status_color(result.posture_status, "Good"),
        f"Neck angle: {result.posture_angle:.1f} deg",
    )
    st.markdown(
        f"""
        <div style="font-size:11px;color:#aaa;margin-top:10px;
                    background:#1e2130;border-radius:6px;padding:10px;">
          <b>Eye model</b>: {eye_name}<br>
          <b>Posture model</b>: {posture_name}<br>
          Eye latency: {result.eye_latency_ms:.1f} ms<br>
          Posture latency: {result.posture_latency_ms:.1f} ms
        </div>
        """,
        unsafe_allow_html=True,
    )
    fc = "#2ecc71" if result.face_detected else "#e74c3c"
    ft = "Face Detected" if result.face_detected else "No Face"
    st.markdown(
        f"""
        <div style="margin-top:8px;padding:6px 10px;
                    background:{fc}22;border-radius:6px;
                    border:1px solid {fc};color:{fc};
                    font-size:12px;font-weight:600;">
          {ft}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ #
# Lazy-load MediaPipe once per server session
# ------------------------------------------------------------------ #

@st.cache_resource(show_spinner="Loading MediaPipe landmarkers...")
def _get_landmarkers():
    """
    Load MediaPipe FaceLandmarker and PoseLandmarker once.
    Cached at the server level so they survive page reruns.
    """
    return load_mediapipe_landmarkers()


# ------------------------------------------------------------------ #
# WebRTC monitoring  (primary path on Streamlit Cloud)
# ------------------------------------------------------------------ #

def _render_webrtc_monitoring(
    eye_model,
    eye_model_name: str,
    posture_model,
    posture_model_name: str,
    user_id: int,
):
    """
    Streams webcam through WebRTC to VisionMateTransformer.
    Your .h5 models run server-side on every received frame.
    """
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

    # TURN server config — required for cloud deployment behind NAT.
    # Free TURN credentials from Twilio or Metered.ca work here.
    # Without TURN the ICE negotiation sometimes fails in restricted networks.
    # Set TURN_URL / TURN_USER / TURN_CREDENTIAL as Streamlit Secrets or env vars.
    turn_url  = os.environ.get("TURN_URL",  "")
    turn_user = os.environ.get("TURN_USER", "")
    turn_pass = os.environ.get("TURN_CREDENTIAL", "")

    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    if turn_url:
        ice_servers.append({
            "urls":       [turn_url],
            "username":   turn_user,
            "credential": turn_pass,
        })

    rtc_config = RTCConfiguration(
        {"iceServers": ice_servers}
    )

    # Load MediaPipe once
    face_lm, pose_lm = _get_landmarkers()

    # Build transformer factory so WebRTC can create a fresh instance
    # but we can inject models before streaming begins.
    def transformer_factory():
        t = VisionMateTransformer()
        t.face_landmarker    = face_lm
        t.pose_landmarker    = pose_lm
        t.eye_model          = eye_model
        t.eye_model_name     = eye_model_name
        t.posture_model      = posture_model
        t.posture_model_name = posture_model_name
        return t

    # Session timer start
    if not st.session_state.get("session_start"):
        st.session_state["session_start"] = time.time()

    video_col, metrics_col = st.columns([2, 1])

    with video_col:
        # webrtc_streamer returns a context object
        ctx = webrtc_streamer(
            key="visionmate-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_transformer_factory=transformer_factory,
            media_stream_constraints={
                "video": {
                    "width":     {"ideal": 640},
                    "height":    {"ideal": 480},
                    "frameRate": {"ideal": 30},
                },
                "audio": False,
            },
            async_processing=True,   # process frames in a thread, never block UI
        )

        # Session timer
        if "session_start" in st.session_state:
            start_time = st.session_state.get("session_start")
            if start_time is None:
                start_time = time.time()
                st.session_state["session_start"] = start_time
            
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            hrs, mins  = divmod(mins, 60)
            timer_str  = (
                f"{hrs:02d}:{mins:02d}:{secs:02d}" if hrs > 0
                else f"{mins:02d}:{secs:02d}"
            )
            st.caption(f"Session duration: {timer_str}")

    with metrics_col:
        # Pull the latest inference result from the transformer
        if ctx.video_transformer is not None:
            result = ctx.video_transformer.get_result()
        else:
            result = FrameResult()   # blank defaults before stream starts

        _render_metrics(result, eye_model_name, posture_model_name)

        # Voice guidance
        voice_guidance.update_condition(
            "eye_strain", result.eye_status == "Strained"
        )
        voice_guidance.update_condition(
            "slouching",  result.posture_status == "Slouching"
        )
        if "session_start" in st.session_state:
            mins_elapsed = (time.time() - st.session_state["session_start"]) / 60.0
            voice_guidance.update_condition("break_reminder", mins_elapsed > 20)

        # Save metric to DB every 5 s
        last_save = st.session_state.get("last_metric_save", 0)
        if ctx.state.playing and time.time() - last_save >= 5:
            save_health_metric(
                user_id              = user_id,
                eye_status           = result.eye_status,
                ear_value            = result.ear_value,
                posture_status       = result.posture_status,
                posture_angle        = result.posture_angle,
                health_score         = result.health_score,
                active_eye_model     = eye_model_name,
                active_posture_model = posture_model_name,
            )
            st.session_state["last_metric_save"] = time.time()

        # Model update at runtime — user switched model in comparison tab
        if ctx.video_transformer is not None:
            ctx.video_transformer.eye_model          = eye_model
            ctx.video_transformer.eye_model_name     = eye_model_name
            ctx.video_transformer.posture_model      = posture_model
            ctx.video_transformer.posture_model_name = posture_model_name

    # Hint when stream is not yet connected
    if not (ctx.state.playing if ctx else False):
        st.info(
            "Click the camera icon above to start the webcam stream. "
            "Allow camera access when your browser asks."
        )


# ------------------------------------------------------------------ #
# Local cv2 fallback  (development only, not used on Streamlit Cloud)
# ------------------------------------------------------------------ #

def _render_local_monitoring(
    processor: FrameProcessor,
    eye_model_name: str,
    posture_model_name: str,
    user_id: int,
):
    """cv2.VideoCapture fallback for local development."""
    col_start, col_stop, col_voice = st.columns([1, 1, 2])

    with col_start:
        if st.button("Start Session", use_container_width=True, key="start_local"):
            if not st.session_state.get("monitoring_active", False):
                processor.start(camera_index=0)
                st.session_state["monitoring_active"] = True
                st.session_state["session_start"]     = time.time()
                voice_guidance.reset_all()

    with col_stop:
        if st.button("Stop Session", use_container_width=True, key="stop_local"):
            if st.session_state.get("monitoring_active", False):
                processor.stop()
                st.session_state["monitoring_active"] = False
                voice_guidance.reset_all()

    with col_voice:
        if st.button("Test Voice Alert", use_container_width=True, key="test_voice"):
            voice_guidance.speak_now("break_reminder")

    st.divider()

    if not st.session_state.get("monitoring_active", False):
        st.info("Click 'Start Session' to begin monitoring.")
        return

    import cv2

    video_col, metrics_col = st.columns([2, 1])
    result = processor.get_latest_result()

    with video_col:
        if result.frame_bgr is not None:
            frame_rgb = cv2.cvtColor(result.frame_bgr, cv2.COLOR_BGR2RGB)
            try:
                st.image(frame_rgb, channels="RGB", use_container_width=True)
            except TypeError:
                st.image(frame_rgb, channels="RGB", use_column_width=True)
        else:
            st.markdown(
                """<div style="background:#1e2130;border-radius:8px;
                    height:240px;display:flex;align-items:center;
                    justify-content:center;color:#555;font-size:14px;">
                    Waiting for webcam...</div>""",
                unsafe_allow_html=True,
            )
        if "session_start" in st.session_state:
            elapsed    = int(time.time() - st.session_state["session_start"])
            mins, secs = divmod(elapsed, 60)
            st.caption(f"Session duration: {mins:02d}:{secs:02d}")

    with metrics_col:
        _render_metrics(result, eye_model_name, posture_model_name)

        last_save = st.session_state.get("last_metric_save", 0)
        if time.time() - last_save >= 5:
            save_health_metric(
                user_id=user_id,
                eye_status=result.eye_status,
                ear_value=result.ear_value,
                posture_status=result.posture_status,
                posture_angle=result.posture_angle,
                health_score=result.health_score,
                active_eye_model=eye_model_name,
                active_posture_model=posture_model_name,
            )
            st.session_state["last_metric_save"] = time.time()

    time.sleep(0.5)
    st.rerun()


# ------------------------------------------------------------------ #
# Public entry point called from app.py
# ------------------------------------------------------------------ #

def render_monitoring_tab(
    processor: FrameProcessor,
    eye_model_name: str,
    posture_model_name: str,
    user_id: int,
):
    st.header("Live Monitoring")

    # Retrieve the actual model objects from app.py's cache
    eye_models     = st.session_state.get("_eye_models",     {}) or {}
    posture_models = st.session_state.get("_posture_models", {}) or {}
    eye_model      = eye_models.get(eye_model_name)
    posture_model  = posture_models.get(posture_model_name)

    if WEBRTC_AVAILABLE:
        # ── Streamlit Cloud path: WebRTC + your .h5 models ──
        _render_webrtc_monitoring(
            eye_model        = eye_model,
            eye_model_name   = eye_model_name,
            posture_model    = posture_model,
            posture_model_name = posture_model_name,
            user_id          = user_id,
        )
    else:
        # ── Local dev fallback: cv2.VideoCapture ──
        st.warning(
            "streamlit-webrtc not installed. "
            "Running in local cv2 fallback mode."
        )
        _render_local_monitoring(
            processor          = processor,
            eye_model_name     = eye_model_name,
            posture_model_name = posture_model_name,
            user_id            = user_id,
        )
