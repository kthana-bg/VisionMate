import cv2
import numpy as np
import mediapipe as mp

# We use this specific path to avoid the AttributeError on Linux servers
from mediapipe.python.solutions import face_mesh as mp_face_mesh

class EyeStrainDetector:
    def __init__(self):
        # Initialize directly from the imported module
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # These landmark indices are standard for EAR calculation
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
    def calculate_ear(self, landmarks, eye_indices):
        try:
            # Euclidean distance math
            p1, p2, p3, p4, p5, p6 = [np.array(landmarks[i]) for i in eye_indices]
            v1 = np.linalg.norm(p2 - p6)
            v2 = np.linalg.norm(p3 - p5)
            h = np.linalg.norm(p1 - p4)
            return (v1 + v2) / (2.0 * h)
        except Exception:
            return 0.0

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        ear_avg = 0.0
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            # Convert to list of [x, y]
            mesh_coords = [[lm.x, lm.y] for lm in landmarks]
            
            left_ear = self.calculate_ear(mesh_coords, self.LEFT_EYE)
            right_ear = self.calculate_ear(mesh_coords, self.RIGHT_EYE)
            ear_avg = (left_ear + right_ear) / 2.0
            
        return ear_avg, results.multi_face_landmarks
