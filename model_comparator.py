"""
Model Comparator - Core system for running and comparing all 6 AI models
Handles eye strain detection (3 models) and posture detection (3 models)
"""
import os
import sys
import time
import numpy as np
import cv2

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.model_loader import (
    load_all_eye_models,
    load_all_posture_models,
    load_results,
)
from utils.eye_detection import (
    extract_eye_landmarks,
    calculate_ear,
    get_eye_roi,
    preprocess_eye_image,
    EAR_THRESHOLD,
)
from utils.posture_detection import (
    extract_pose_landmarks,
    calculate_neck_tilt_angle,
    extract_landmark_feature_vector,
    SLOUCH_ANGLE_THRESHOLD,
)
from utils.frame_processor import load_mediapipe_landmarkers


class ModelComparator:
    """
    Manages all 6 models (3 eye + 3 posture) and runs comparative inference.
    """
    
    # Model name mappings
    EYE_MODELS = {
        "C1": "Custom CNN",
        "B1": "MobileNetV2", 
        "A1": "EfficientNetB0",
    }
    
    POSTURE_MODELS = {
        "C2": "Custom LSTM/DNN",
        "B2": "YOLOv8-Pose / MoveNet DNN",
        "A2": "MediaPipe Pose (Rule-Based)",
    }
    
    def __init__(self):
        """Load all models and MediaPipe landmarkers at initialization."""
        print("Loading ModelComparator...")
        
        # Load MediaPipe for landmark detection
        self.face_landmarker, self.pose_landmarker = load_mediapipe_landmarkers()
        
        # Load all trained models
        self.eye_models = load_all_eye_models()
        self.posture_models = load_all_posture_models()
        
        # Active model selection (default: C1 for eye, C2 for posture)
        self.active_eye_model = "C1"
        self.active_posture_model = "C2"
        
        # Frame counter for EAR consecutive frames logic
        self.ear_consec_counter = 0
        
        print(f"Eye models loaded: {list(self.eye_models.keys())}")
        print(f"Posture models loaded: {list(self.posture_models.keys())}")
    
    def set_active_models(self, eye_model_key: str, posture_model_key: str):
        """Set which models to use for live monitoring."""
        if eye_model_key in self.EYE_MODELS:
            self.active_eye_model = eye_model_key
        if posture_model_key in self.POSTURE_MODELS:
            self.active_posture_model = posture_model_key
    
    def get_model_status(self) -> dict:
        """Return which models are loaded and active."""
        return {
            "active_eye": self.active_eye_model,
            "active_posture": self.active_posture_model,
            "C1_loaded": self.eye_models.get("Custom CNN") is not None,
            "B1_loaded": self.eye_models.get("MobileNetV2") is not None,
            "A1_loaded": self.eye_models.get("EfficientNetB0") is not None,
            "C2_loaded": self.posture_models.get("Custom LSTM/DNN") is not None,
            "B2_loaded": self.posture_models.get("YOLOv8-Pose / MoveNet DNN") is not None,
            "A2_loaded": True,  # MediaPipe is always available (rule-based)
        }
    
    def process_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        Process a single frame through the active models.
        Returns a comprehensive result dictionary.
        """
        import mediapipe as mp
        
        h, w = frame_bgr.shape[:2]
        
        # Convert to MediaPipe format
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
        )
        
        # ==================== EYE STRAIN DETECTION ====================
        eye_result = self._process_eye_detection(mp_image, frame_bgr, w, h)
        
        # ==================== POSTURE DETECTION ====================
        posture_result = self._process_posture_detection(mp_image, w, h)
        
        # ==================== COMPUTE HEALTH SCORE ====================
        eye_status = eye_result[self.active_eye_model]["classification"]
        posture_status = posture_result[self.active_posture_model]["status"]
        
        eye_score = 50.0
        if eye_status == "STRAINED":
            eye_score = 20.0
        elif eye_status == "NORMAL":
            eye_score = 50.0
        
        posture_score = 50.0
        if posture_status == "SLOUCHING":
            posture_score = 20.0
        elif posture_status == "GOOD":
            posture_score = 50.0
        
        health_score = eye_score + posture_score
        
        return {
            "eye": eye_result,
            "posture": posture_result,
            "eye_consensus": eye_status,
            "posture_consensus": posture_status,
            "health_score": int(health_score),
        }
    
    def _process_eye_detection(self, mp_image, frame_bgr, w, h) -> dict:
        """Run eye strain detection using active model."""
        result = {
            "C1": {"classification": "NORMAL", "fatigue_score": 0.0, "latency_ms": 0.0, "ear": 0.0},
            "B1": {"classification": "NORMAL", "fatigue_score": 0.0, "latency_ms": 0.0, "ear": 0.0},
            "A1": {"classification": "NORMAL", "fatigue_score": 0.0, "latency_ms": 0.0, "ear": 0.0},
        }
        
        if self.face_landmarker is None:
            return result
        
        try:
            face_result = self.face_landmarker.detect(mp_image)
            
            if not face_result.face_landmarks:
                return result
            
            landmarks_478 = face_result.face_landmarks[0]
            
            # Create proxy object for landmark access
            class _LMProxy:
                def __init__(self, lms):
                    self.landmark = lms
            
            proxy = _LMProxy(landmarks_478)
            
            # Extract eye landmarks and calculate EAR
            left_eye, right_eye = extract_eye_landmarks(proxy, w, h)
            ear_left = calculate_ear(left_eye)
            ear_right = calculate_ear(right_eye)
            ear_avg = (ear_left + ear_right) / 2.0
            
            # Update consecutive counter for rule-based detection
            if ear_avg < EAR_THRESHOLD:
                self.ear_consec_counter += 1
            else:
                self.ear_consec_counter = 0
            
            # Run inference on active model only (for performance)
            active_key = self.active_eye_model
            active_model_name = self.EYE_MODELS[active_key]
            active_model = self.eye_models.get(active_model_name)
            
            if active_model is not None:
                # Get eye ROI for CNN inference
                roi = get_eye_roi(frame_bgr, left_eye, padding=10)
                if roi is not None:
                    start_time = time.perf_counter()
                    
                    # Preprocess and run inference
                    input_tensor = preprocess_eye_image(roi)
                    prediction = active_model.predict(input_tensor, verbose=0)[0]
                    
                    # Interpret prediction
                    if len(prediction) == 1:
                        # Binary output: single neuron sigmoid
                        fatigue_score = float(prediction[0])
                        classification = "STRAINED" if fatigue_score > 0.5 else "NORMAL"
                    else:
                        # Two-class softmax output
                        normal_prob = float(prediction[0])
                        strained_prob = float(prediction[1])
                        fatigue_score = strained_prob
                        classification = "STRAINED" if strained_prob > 0.5 else "NORMAL"
                    
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    
                    result[active_key] = {
                        "classification": classification,
                        "fatigue_score": fatigue_score,
                        "latency_ms": latency_ms,
                        "ear": ear_avg,
                    }
            else:
                # Fallback to EAR rule-based
                if self.ear_consec_counter >= 30:
                    classification = "STRAINED"
                elif ear_avg < EAR_THRESHOLD:
                    classification = "BLINKING"
                else:
                    classification = "NORMAL"
                
                result[active_key] = {
                    "classification": classification,
                    "fatigue_score": 0.0 if classification == "NORMAL" else 1.0,
                    "latency_ms": 0.0,
                    "ear": ear_avg,
                }
        
        except Exception as e:
            print(f"Eye detection error: {e}")
        
        return result
    
    def _process_posture_detection(self, mp_image, w, h) -> dict:
        """Run posture detection using active model."""
        result = {
            "C2": {"status": "GOOD", "slouching_prob": 0.0, "latency_ms": 0.0, "angle_y": 0.0},
            "B2": {"status": "GOOD", "slouching_prob": 0.0, "latency_ms": 0.0, "angle_y": 0.0},
            "A2": {"status": "GOOD", "slouching_prob": 0.0, "latency_ms": 0.0, "angle_y": 0.0},
        }
        
        if self.pose_landmarker is None:
            return result
        
        try:
            pose_result = self.pose_landmarker.detect(mp_image)
            
            if not pose_result.pose_landmarks:
                return result
            
            lms = pose_result.pose_landmarks[0]
            
            # Extract key landmarks
            def to_px(idx):
                return (int(lms[idx].x * w), int(lms[idx].y * h))
            
            landmarks = {
                "nose": to_px(0),
                "left_ear": to_px(7),
                "right_ear": to_px(8),
                "left_shoulder": to_px(11),
                "right_shoulder": to_px(12),
            }
            
            # Calculate neck tilt angle
            ear_mid = (
                (landmarks["left_ear"][0] + landmarks["right_ear"][0]) // 2,
                (landmarks["left_ear"][1] + landmarks["right_ear"][1]) // 2,
            )
            shoulder_mid = (
                (landmarks["left_shoulder"][0] + landmarks["right_shoulder"][0]) // 2,
                (landmarks["left_shoulder"][1] + landmarks["right_shoulder"][1]) // 2,
            )
            
            angle = calculate_neck_tilt_angle(ear_mid, shoulder_mid)
            
            # Run inference on active model only
            active_key = self.active_posture_model
            active_model_name = self.POSTURE_MODELS[active_key]
            active_model = self.posture_models.get(active_model_name)
            
            if active_key == "A2":
                # MediaPipe rule-based (no model file)
                status = "SLOUCHING" if angle > SLOUCH_ANGLE_THRESHOLD else "GOOD"
                result[active_key] = {
                    "status": status,
                    "slouching_prob": 1.0 if status == "SLOUCHING" else 0.0,
                    "latency_ms": 0.0,
                    "angle_y": angle,
                }
            elif active_model is not None:
                # LSTM or DNN model
                start_time = time.perf_counter()
                
                # Extract feature vector
                feature_vector = extract_landmark_feature_vector(landmarks)
                
                # Prepare input based on model architecture
                if "LSTM" in active_model_name:
                    # LSTM expects (batch, timesteps, features)
                    input_tensor = feature_vector.reshape(1, 1, -1)
                else:
                    # DNN expects (batch, features)
                    input_tensor = feature_vector.reshape(1, -1)
                
                prediction = active_model.predict(input_tensor, verbose=0)[0]
                
                # Interpret prediction
                if len(prediction) == 1:
                    slouching_prob = float(prediction[0])
                else:
                    good_prob = float(prediction[0])
                    slouching_prob = float(prediction[1])
                
                status = "SLOUCHING" if slouching_prob > 0.5 else "GOOD"
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                
                result[active_key] = {
                    "status": status,
                    "slouching_prob": slouching_prob,
                    "latency_ms": latency_ms,
                    "angle_y": angle,
                }
            else:
                # Fallback to angle-based rule
                status = "SLOUCHING" if angle > SLOUCH_ANGLE_THRESHOLD else "GOOD"
                result[active_key] = {
                    "status": status,
                    "slouching_prob": 1.0 if status == "SLOUCHING" else 0.0,
                    "latency_ms": 0.0,
                    "angle_y": angle,
                }
        
        except Exception as e:
            print(f"Posture detection error: {e}")
        
        return result
