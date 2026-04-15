import cv2
import numpy as np
import mediapipe as mp
import threading

try:
    mp_face_mesh = mp.solutions.face_mesh
except AttributeError:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh


class EyeStrainDetector:
    def __init__(self):
        try:
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            print(f"MediaPipe initialization error: {e}")
            raise

        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

        self._blink_count = 0
        self._blink_active = False
        self._lock = threading.Lock()
        self._last_ear = 0.25

    def calculate_ear(self, landmarks, eye_indices):
        try:
            p1 = np.array(landmarks[eye_indices[1]])
            p2 = np.array(landmarks[eye_indices[2]])
            p3 = np.array(landmarks[eye_indices[4]])
            p4 = np.array(landmarks[eye_indices[5]])
            p0 = np.array(landmarks[eye_indices[0]])
            p3_h = np.array(landmarks[eye_indices[3]])

            v1 = np.linalg.norm(p1 - p4)
            v2 = np.linalg.norm(p2 - p3)
            h = np.linalg.norm(p0 - p3_h)

            if h == 0:
                return 0.0

            return (v1 + v2) / (2.0 * h)
        except:
            return 0.0

    def process_frame(self, frame):
        if frame is None or frame.size == 0:
            return 0.0, None, frame

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        ear_avg = 0.0
        annotated_frame = frame.copy()

        if results.multi_face_landmarks:
            mesh_coords = [
                [lm.x, lm.y]
                for lm in results.multi_face_landmarks[0].landmark
            ]

            left_ear = self.calculate_ear(mesh_coords, self.LEFT_EYE)
            right_ear = self.calculate_ear(mesh_coords, self.RIGHT_EYE)
            ear_avg = (left_ear + right_ear) / 2.0

            # Draw eye landmarks
            h, w, _ = frame.shape
            for idx in self.LEFT_EYE + self.RIGHT_EYE:
                x = int(mesh_coords[idx][0] * w)
                y = int(mesh_coords[idx][1] * h)
                cv2.circle(annotated_frame, (x, y), 2, (0, 255, 0), -1)

        return ear_avg, results.multi_face_landmarks, annotated_frame

    def update_blink_state(self, ear, threshold=0.20):
        with self._lock:
            self._last_ear = ear

            if ear < threshold and ear > 0 and not self._blink_active:
                self._blink_active = True
            elif ear >= threshold and self._blink_active:
                self._blink_count += 1
                self._blink_active = False

            return self._blink_count, self._blink_active

    def get_blink_count(self):
        with self._lock:
            return self._blink_count

    def reset_blink_count(self):
        with self._lock:
            self._blink_count = 0
            self._blink_active = False
