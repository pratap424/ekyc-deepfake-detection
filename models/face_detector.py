"""
Face Detection - Separate from liveness (like GitHub repo)
Returns cropped faces, not just bboxes
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple

class FaceDetector:
    """Fast face detection that returns cropped faces"""
    
    def __init__(self, max_num_faces: int = 1):
        self.detector = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            static_image_mode=True
        )
    
    def __call__(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Detect faces and return cropped face arrays
        
        Args:
            image: BGR frame
            
        Returns:
            faces: List of cropped face arrays (BGR)
            boxes: List of bounding boxes [[x_min, y_min], [x_max, y_max]]
        """
        h, w = image.shape[:2]
        
        # Ensure contiguous
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect
        results = self.detector.process(image_rgb)
        
        faces = []
        boxes = []
        
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                pts = np.array(
                    [(pt.x * w, pt.y * h) for pt in landmarks.landmark],
                    dtype=np.float64
                )
                
                bbox = np.vstack([pts.min(axis=0), pts.max(axis=0)])
                bbox = np.round(bbox).astype(np.int32)
                
                x_min, y_min = bbox[0]
                x_max, y_max = bbox[1]
                
                # Extract face crop (BGR)
                face_arr = image[y_min:y_max, x_min:x_max]
                
                faces.append(face_arr)
                boxes.append(bbox)
        
        return faces, boxes
