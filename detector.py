import cv2
import numpy as np
import mediapipe as mp
import threading

# --- ROBUST IMPORT FIX ---
# Instead of mp.solutions.face_mesh, we import the specific class directly
from mediapipe.python.solutions.face_mesh import FaceMesh
from mediapipe.python.solutions import drawing_utils as mp_drawing
# -------------------------

class EyeStrainDetector:
    def __init__(self):
        # Initialize FaceMesh directly using the imported class
        self.face_mesh = FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.blink_count = 0
        self.blink_active = False
        self.lock = threading.Lock()

    def calculate_ear(self, landmarks, eye_indices):
        try:
            p1 = np.array(landmarks[eye_indices[1]])
            p5 = np.array(landmarks[eye_indices[5]])
            p2 = np.array(landmarks[eye_indices[2]])
            p4 = np.array(landmarks[eye_indices[4]])
            p0 = np.array(landmarks[eye_indices[0]])
            p3 = np.array(landmarks[eye_indices[3]])

            v1 = np.linalg.norm(p1 - p5)
            v2 = np.linalg.norm(p2 - p4)
            h = np.linalg.norm(p0 - p3)
            return (v1 + v2) / (2.0 * h) if h > 0 else 0.0
        except: return 0.0

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        ear = 0.0
        if res.multi_face_landmarks:
            # Get landmark coordinates
            m = [(lm.x, lm.y) for lm in res.multi_face_landmarks[0].landmark]
            ear = (self.calculate_ear(m, self.LEFT_EYE) + self.calculate_ear(m, self.RIGHT_EYE)) / 2.0
        return ear, res.multi_face_landmarks, frame

    def update_blink_state(self, ear, threshold):
        with self.lock:
            if ear < threshold and ear > 0 and not self.blink_active:
                self.blink_active = True
            elif ear >= threshold and self.blink_active:
                self.blink_count += 1
                self.blink_active = False
            return self.blink_count, self.blink_active
