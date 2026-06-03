import base64
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from utils.frame_processor import VisionMateTransformer
from utils.model_loader import load_all_eye_models, load_all_posture_models

app = FastAPI()

# allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# LOAD YOUR MODELS ONCE
# ----------------------------
eye_models, posture_models = load_all_eye_models(), load_all_posture_models()

transformer = VisionMateTransformer()
transformer.face_landmarker, transformer.pose_landmarker = None, None  # optional if you load separately

# assign default models (change names if needed)
transformer.eye_model = eye_models["Custom CNN"]
transformer.posture_model = posture_models["MediaPipe Pose (Rule-Based)"]


# ----------------------------
# FRAME DECODER
# ----------------------------
def decode_frame(base64_str: str):
    img_data = base64.b64decode(base64_str.split(",")[-1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


# ----------------------------
# WEBSOCKET ENDPOINT
# ----------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        try:
            # 1. receive frame
            data = await websocket.receive_text()

            # 2. decode
            frame = decode_frame(data)

            # 3. run your model (THIS IS YOUR EXISTING LOGIC)
            result = transformer.process_frame(frame)

            # 4. send result back
            await websocket.send_json({
                "health_score": float(result.health_score),
                "eye_status": result.eye_status,
                "posture_status": result.posture_status,
                "ear_value": float(result.ear_value),
                "posture_angle": float(result.posture_angle),
                "face_detected": bool(result.face_detected)
            })

        except Exception as e:
            await websocket.send_json({"error": str(e)})
