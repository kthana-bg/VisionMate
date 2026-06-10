#!/usr/bin/env python3
"""
Minimal health check script for Streamlit Cloud debugging
"""
import sys
import os

print("="*60)
print("VisionMate Health Check")
print("="*60)

# Check 1: Python version
print(f"\n1. Python version: {sys.version}")

# Check 2: Working directory
print(f"\n2. Working directory: {os.getcwd()}")
print(f"   Files: {os.listdir('.')[:10]}")

# Check 3: Import streamlit
try:
    import streamlit as st
    print(f"\n3. ✓ Streamlit imported: {st.__version__}")
except Exception as e:
    print(f"\n3. ✗ Streamlit import failed: {e}")
    sys.exit(1)

# Check 4: Import core dependencies
try:
    import cv2
    print(f"4. ✓ OpenCV imported: {cv2.__version__}")
except Exception as e:
    print(f"4. ✗ OpenCV import failed: {e}")

try:
    import mediapipe
    print(f"5. ✓ MediaPipe imported: {mediapipe.__version__}")
except Exception as e:
    print(f"5. ✗ MediaPipe import failed: {e}")

try:
    import tensorflow as tf
    print(f"6. ✓ TensorFlow imported: {tf.__version__}")
except Exception as e:
    print(f"6. ✗ TensorFlow import failed: {e}")

# Check 5: Import local modules
print("\n7. Importing local modules...")
try:
    from database_manager import DatabaseManager
    print("   ✓ DatabaseManager")
except Exception as e:
    print(f"   ✗ DatabaseManager: {e}")

try:
    from model_comparator import ModelComparator
    print("   ✓ ModelComparator")
except Exception as e:
    print(f"   ✗ ModelComparator: {e}")

try:
    from utils.face_auth import FaceAuthenticator
    print("   ✓ FaceAuthenticator")
except Exception as e:
    print(f"   ✗ FaceAuthenticator: {e}")

# Check 6: Model files
print("\n8. Checking model files...")
model_paths = [
    "models/eye_strain/custom_cnn.h5",
    "models/posture/custom_lstm.h5"
]
for path in model_paths:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"   ✓ {path} ({size_mb:.1f} MB)")
    else:
        print(f"   ✗ {path} NOT FOUND")

print("\n" + "="*60)
print("Health check complete!")
print("="*60)
