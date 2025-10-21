"""
Optimized DeepPixBiS liveness detection model
Based on: https://github.com/ffletcherr/face-recognition-liveness
"""

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms as T
import mediapipe as mp

class LivenessDetector:
    """DeepPixBiS anti-spoofing detector - OPTIMIZED"""
    
    def __init__(self, model_path):
        # Load ONNX model
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        
        # Transform
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Face detector - ⚠️ FIX: Use static_image_mode=True for stability
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            static_image_mode=True,  # Changed from False
            min_detection_confidence=0.5,
            refine_landmarks=False  # Disable refinement for speed
        )
        
        print("✅ LivenessDetector loaded (DeepPixBiS - Optimized)")
    
    def detect_face(self, image):
        """Detect face and return crop + bbox"""
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None, None
        
        # Get first face
        landmarks = results.multi_face_landmarks[0]
        pts = np.array([(pt.x * w, pt.y * h) for pt in landmarks.landmark], dtype=np.float32)
        
        # Calculate bbox
        x_min, y_min = pts.min(axis=0).astype(np.int32)
        x_max, y_max = pts.max(axis=0).astype(np.int32)
        
        # ⚠️ FIX: Dynamic padding (10% of bbox size instead of fixed 20px)
        bbox_w, bbox_h = x_max - x_min, y_max - y_min
        padding_x = int(bbox_w * 0.1)
        padding_y = int(bbox_h * 0.1)
        
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(w, x_max + padding_x)
        y_max = min(h, y_max + padding_y)
        
        face = image[y_min:y_max, x_min:x_max]
        bbox = np.array([[x_min, y_min], [x_max, y_max]])
        
        return face, bbox
    
    def calculate_liveness(self, face_bgr):
        """Calculate liveness score for face crop"""
        if face_bgr is None or face_bgr.size == 0:
            return 0.0
        
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        face_tensor = self.transform(face_pil).unsqueeze(0).numpy().astype(np.float32)
        
        # Run ONNX model
        output_pixel, output_binary = self.session.run(
            ["output_pixel", "output_binary"],
            {"input": face_tensor}
        )
        
        # Average both outputs
        score = (np.mean(output_pixel) + np.mean(output_binary)) / 2.0
        return float(score)
    
    def predict(self, frame):
        """
        Predict liveness from frame
        
        Args:
            frame: BGR image (numpy array)
        
        Returns:
            dict with is_live, score, confidence, face_detected, bbox
        """
        face, bbox = self.detect_face(frame)
        
        if face is None or face.size == 0:
            return {
                'is_live': False,
                'score': 0.0,
                'confidence': 0.0,
                'face_detected': False,
                'bbox': None
            }
        
        score = self.calculate_liveness(face)
        
        # ⚠️ CRITICAL FIX: Use threshold of 0.03 (from config)
        from config import LIVENESS_THRESHOLD
        is_live = score > LIVENESS_THRESHOLD
        
        # Calculate confidence percentage (0.03 → 100%)
        confidence = min(100.0, (score / LIVENESS_THRESHOLD) * 100)
        
        return {
            'is_live': is_live,
            'score': score,
            'confidence': round(confidence, 2),
            'face_detected': True,
            'bbox': bbox
        }
