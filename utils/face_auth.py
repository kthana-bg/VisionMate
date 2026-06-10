"""
Face Authentication Module
Uses MediaPipe Face Detection + simple embedding extraction for authentication
"""
import cv2
import numpy as np
from typing import Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class FaceAuthenticator:
    """
    Face authentication using MediaPipe face mesh landmarks as embeddings.
    Simplified approach - uses facial landmark positions as a feature vector.
    """
    
    def __init__(self):
        self.face_detector = None
        self._load_detector()
    
    def _load_detector(self):
        """Load MediaPipe Face Detection"""
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detector = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 1 for full range (0-5m), 0 for short range (2m)
                min_detection_confidence=0.5
            )
        except Exception as e:
            print(f"Failed to load face detector: {e}")
    
    def extract_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from frame.
        Returns 128-d feature vector or None if no face detected.
        """
        if self.face_detector is None:
            return None
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb_frame.shape[:2]
            
            # Detect face
            results = self.face_detector.process(rgb_frame)
            
            if not results.detections:
                return None
            
            # Get first face detection
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Extract face region
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            w_box = int(bbox.width * w)
            h_box = int(bbox.height * h)
            
            # Ensure coordinates are within frame
            x = max(0, x)
            y = max(0, y)
            x2 = min(w, x + w_box)
            y2 = min(h, y + h_box)
            
            if x2 <= x or y2 <= y:
                return None
            
            face_roi = rgb_frame[y:y2, x:x2]
            
            if face_roi.size == 0:
                return None
            
            # Resize to standard size and flatten
            face_resized = cv2.resize(face_roi, (64, 64))
            
            # Convert to grayscale and normalize
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
            face_normalized = face_gray.astype(np.float32) / 255.0
            
            # Create a simple embedding from image features
            # Use histogram + moments as features
            hist = cv2.calcHist([face_gray], [0], None, [32], [0, 256]).flatten()
            hist = hist / (hist.sum() + 1e-7)  # normalize
            
            moments = cv2.moments(face_gray)
            moment_features = np.array([
                moments['m00'], moments['m10'], moments['m01'],
                moments['m20'], moments['m11'], moments['m02'],
                moments['m30'], moments['m21'], moments['m12'], moments['m03']
            ], dtype=np.float32)
            moment_features = moment_features / (np.linalg.norm(moment_features) + 1e-7)
            
            # Combine features
            embedding = np.concatenate([hist, moment_features])
            
            return embedding
            
        except Exception as e:
            print(f"Face embedding extraction error: {e}")
            return None
    
    def reduce_dimension(self, embedding: np.ndarray, target_dim: int = 128) -> np.ndarray:
        """
        Reduce embedding to target dimension using PCA-like projection.
        """
        if embedding is None:
            return None
        
        current_dim = len(embedding)
        
        if current_dim == target_dim:
            return embedding
        
        if current_dim < target_dim:
            # Pad with zeros
            padded = np.zeros(target_dim, dtype=np.float32)
            padded[:current_dim] = embedding
            return padded
        else:
            # Simple downsampling by averaging chunks
            chunk_size = current_dim // target_dim
            reduced = np.array([
                embedding[i*chunk_size:(i+1)*chunk_size].mean()
                for i in range(target_dim)
            ], dtype=np.float32)
            return reduced / (np.linalg.norm(reduced) + 1e-7)
    
    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compare two embeddings using cosine similarity.
        Returns similarity score between 0 and 1 (higher = more similar).
        """
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Convert to 0-1 range (cosine similarity is -1 to 1)
        similarity = (similarity + 1) / 2
        
        return float(similarity)
