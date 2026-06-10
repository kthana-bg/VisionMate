import os, sys, json
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

MODELS_DIR  = os.path.join(_ROOT, "models")
RESULTS_DIR = os.path.join(_ROOT, "results")

EYE_MODEL_PATHS = {
    "Custom CNN":     os.path.join(MODELS_DIR, "eye_strain", "custom_cnn.h5"),
    "MobileNetV2":    os.path.join(MODELS_DIR, "eye_strain", "mobilenetv2.h5"),
    "EfficientNetB0": os.path.join(MODELS_DIR, "eye_strain", "efficientnetb0.h5"),
}

POSTURE_MODEL_PATHS = {
    "Custom LSTM/DNN":           os.path.join(MODELS_DIR, "posture", "custom_lstm.h5"),
    "MediaPipe Pose (Rule-Based)": None,   # no model file — pure geometry
    "YOLOv8-Pose / MoveNet DNN": os.path.join(MODELS_DIR, "posture", "yolo_movenet_dnn.h5"),
}

RESULTS_PATHS = {
    "Custom CNN":                  os.path.join(RESULTS_DIR, "custom_cnn_results.json"),
    "MobileNetV2":                 os.path.join(RESULTS_DIR, "mobilenetv2_results.json"),
    "EfficientNetB0":              os.path.join(RESULTS_DIR, "efficientnetb0_results.json"),
    "Custom LSTM/DNN":             os.path.join(RESULTS_DIR, "custom_lstm_results.json"),
    "MediaPipe Pose (Rule-Based)": os.path.join(RESULTS_DIR, "mediapipe_results.json"),
    "YOLOv8-Pose / MoveNet DNN":  os.path.join(RESULTS_DIR, "yolo_movenet_results.json"),
}

# Placeholder values shown when real results JSON is missing
_DEMO_RESULTS = {
    "Custom CNN":                  {"accuracy": 0.87, "f1_score": 0.86, "latency_ms": 12.3},
    "MobileNetV2":                 {"accuracy": 0.91, "f1_score": 0.90, "latency_ms": 8.7},
    "EfficientNetB0":              {"accuracy": 0.94, "f1_score": 0.93, "latency_ms": 15.2},
    "Custom LSTM/DNN":             {"accuracy": 0.85, "f1_score": 0.84, "latency_ms": 5.1},
    "MediaPipe Pose (Rule-Based)": {"accuracy": 0.82, "f1_score": 0.81, "latency_ms": 2.4},
    "YOLOv8-Pose / MoveNet DNN":  {"accuracy": 0.92, "f1_score": 0.91, "latency_ms": 18.6},
}


def load_keras_model(model_path: str):
    """
    Load Keras model with advanced compatibility handling.
    Handles Keras 3.x models trained on Kaggle in TF 2.15 environment.
    """
    if not model_path or not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None

    import tensorflow as tf
    from tensorflow import keras
    
    # Register custom objects for compatibility
    custom_objects = {
        'TrueDivide': tf.math.truediv,
        'DepthwiseConv2D': keras.layers.DepthwiseConv2D,
    }

    # Strategy 1: Load with safe mode (ignore unrecognized arguments)
    try:
        # Temporarily modify deserialize to be more lenient
        import tensorflow.python.keras.saving.legacy.serialization as serialization_legacy
        
        _original_deserialize = serialization_legacy.deserialize_keras_object
        
        def lenient_deserialize(identifier, module_objects=None, custom_objects=None, printable_module_name='object'):
            try:
                return _original_deserialize(identifier, module_objects, custom_objects, printable_module_name)
            except TypeError as e:
                # If it's a keyword argument error, try stripping problematic keys
                if 'Keyword argument not understood' in str(e) or 'Unrecognized keyword' in str(e):
                    if isinstance(identifier, dict) and 'config' in identifier:
                        config = identifier['config'].copy()
                        # Remove known problematic keys
                        config.pop('quantization_config', None)
                        config.pop('optional', None)
                        config.pop('batch_shape', None)
                        # Replace dtype if it's a dict
                        if isinstance(config.get('dtype'), dict):
                            config['dtype'] = 'float32'
                        identifier = identifier.copy()
                        identifier['config'] = config
                        return _original_deserialize(identifier, module_objects, custom_objects, printable_module_name)
                raise
        
        # Apply monkey patch
        serialization_legacy.deserialize_keras_object = lenient_deserialize
        
        try:
            model = keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects=custom_objects
            )
            print(f"✓ Loaded: {os.path.basename(model_path)}")
            return model
        finally:
            # Restore original
            serialization_legacy.deserialize_keras_object = _original_deserialize
            
    except Exception as e1:
        print(f"Strategy 1 failed for {os.path.basename(model_path)}: {str(e1)[:100]}")

    # Strategy 2: Try with h5py direct weight loading
    try:
        import h5py
        
        # This is a simplified approach - rebuild model from architecture
        # For now, return None as we'd need the exact architecture
        pass
    except Exception as e2:
        pass

    print(f"✗ Could not load: {os.path.basename(model_path)}")
    return None


def load_all_eye_models() -> dict:
    models = {}
    for name, path in EYE_MODEL_PATHS.items():
        models[name] = load_keras_model(path)
    return models


def load_all_posture_models() -> dict:
    models = {}
    for name, path in POSTURE_MODEL_PATHS.items():
        models[name] = load_keras_model(path) if path else None
    return models


def load_results(model_name: str) -> dict:
    path = RESULTS_PATHS.get(model_name)
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return _DEMO_RESULTS.get(model_name, {"accuracy": 0.80, "f1_score": 0.79, "latency_ms": 10.0})


def load_all_results() -> dict:
    return {name: load_results(name) for name in RESULTS_PATHS}
