"""
Minimal Streamlit app for testing deployment
If this works, the issue is with model loading
"""
import streamlit as st

st.set_page_config(page_title="VisionMate Test", layout="wide")

st.title("🎯 VisionMate - Deployment Test")

st.success("✅ Streamlit is working!")

st.info("""
This is a minimal version to test deployment.
If you see this page, Streamlit Cloud is working correctly.

Next steps:
1. Check if models can load
2. Test imports
3. Enable full app
""")

# Test imports
st.subheader("Testing Imports...")

status = {}

try:
    import cv2
    status["OpenCV"] = f"✓ {cv2.__version__}"
except Exception as e:
    status["OpenCV"] = f"✗ {str(e)[:50]}"

try:
    import mediapipe
    status["MediaPipe"] = f"✓ {mediapipe.__version__}"
except Exception as e:
    status["MediaPipe"] = f"✗ {str(e)[:50]}"

try:
    import tensorflow as tf
    status["TensorFlow"] = f"✓ {tf.__version__}"
except Exception as e:
    status["TensorFlow"] = f"✗ {str(e)[:50]}"

try:
    from database_manager import DatabaseManager
    status["DatabaseManager"] = "✓ OK"
except Exception as e:
    status["DatabaseManager"] = f"✗ {str(e)[:50]}"

for lib, stat in status.items():
    if "✓" in stat:
        st.success(f"{lib}: {stat}")
    else:
        st.error(f"{lib}: {stat}")

st.divider()
st.caption("Once all imports work, we can enable the full app")
