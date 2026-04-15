import cv2
import numpy as np
from mediapipe.python.solutions import face_mesh as mp_face_mesh

class EyeStrainDetector:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    def calculate_ear(self, landmarks, eye_indices):
        try:
            #calculate the distance between top and bottom eyelids 
            v1 = np.linalg.norm(np.array(landmarks[eye_indices[1]]) - np.array(landmarks[eye_indices[5]]))
            v2 = np.linalg.norm(np.array(landmarks[eye_indices[2]]) - np.array(landmarks[eye_indices[4]]))
            #calculates the distance from left to right corner of the eye
            h = np.linalg.norm(np.array(landmarks[eye_indices[0]]) - np.array(landmarks[eye_indices[3]]))
            return (v1 + v2) / (2.0 * h)
        except Exception:
            return 0.0

    #takes raw image from webcam and convert into mathematical data
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        ear_avg = 0.0
        
        if results.multi_face_landmarks:
            #convert landmarks to simple coordinate list
            mesh_coords = [[lm.x, lm.y] for lm in results.multi_face_landmarks[0].landmark]
            
            left_ear = self.calculate_ear(mesh_coords, self.LEFT_EYE)
            right_ear = self.calculate_ear(mesh_coords, self.RIGHT_EYE)
            ear_avg = (left_ear + right_ear) / 2.0
            
        return ear_avg, results.multi_face_landmarks
