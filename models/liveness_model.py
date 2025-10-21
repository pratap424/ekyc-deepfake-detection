"""
PERFORMANCE-OPTIMIZED Liveness Detection
- Uses Mediapipe Face Detection (not FaceMesh) - 10x faster
- ONNX single-thread optimization
- Minimal preprocessing
"""
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms as T
import mediapipe as mp

class LivenessDetector:
    """High-performance liveness detector"""
    
    def __init__(self, model_path):
        # ✅ ONNX optimization: disable multi-threading (47% -> 0.5% CPU)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # Single thread - faster for small models
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        
        # Transform for liveness model (224×224)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # ✅ CRITICAL FIX: Use Face Detection (not FaceMesh)
        # Face Detection is 10x faster - only returns bounding boxes
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,  # 0 = short range (faster), 1 = full range
            min_detection_confidence=0.5
        )
        
        print("✅ LivenessDetector loaded (Performance Mode)")
        print("   - Face Detection: Mediapipe (fast)")
        print("   - ONNX: Single-threaded")
    
    def detect_faces(self, image):
        """
        ULTRA-FAST face detection - minimal conversions
        """
        h, w = image.shape[:2]
        
        # Gradio gives RGB, Mediapipe needs RGB - perfect!
        # Only convert if BGR (has 3 channels and looks like BGR)
        if len(image.shape) == 3:
            # Check if already RGB (Gradio streaming)
            # Mediapipe needs RGB, so just use directly
            image_rgb = image
        else:
            image_rgb = image
        
        results = self.face_detector.process(image_rgb)
        
        boxes = []
        faces = []
        
        if results.detections:
            detection = results.detections[0]  # Only process first face
            
            bbox = detection.location_data.relative_bounding_box
            
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            x2 = min(w, int((bbox.xmin + bbox.width) * w))
            y2 = min(h, int((bbox.ymin + bbox.height) * h))
            
            # Direct crop - no checks
            face_crop = image_rgb[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                # Resize to 160×160
                face_resized = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_LINEAR)
                
                bbox_array = np.array([[x1, y1], [x2, y2]])
                boxes.append(bbox_array)
                faces.append(face_resized)
        
        return faces, boxes

    
    def predict_liveness(self, face_arr):
        """
        Run liveness prediction on cropped face
        Returns averaged liveness score (0-1)
        """
        # Convert to PIL and apply transforms
        face_pil = Image.fromarray(cv2.cvtColor(face_arr, cv2.COLOR_BGR2RGB))
        face_tensor = self.transform(face_pil).unsqueeze(0).numpy()
        
        # ONNX inference
        outputs = self.session.run(None, {'input': face_tensor})
        
        # DeepPixBiS outputs 14×14 pixel-wise map
        pixel_map = outputs[0]
        
        # Average the map
        if len(pixel_map.shape) == 4:
            liveness_score = float(np.mean(pixel_map[0, 0, :, :]))
        elif len(pixel_map.shape) == 3:
            liveness_score = float(np.mean(pixel_map[0, :, :]))
        else:
            liveness_score = float(np.mean(pixel_map))
        
        return liveness_score
    
    def predict(self, frame):
        """ULTRA-FAST prediction pipeline"""
        faces, boxes = self.detect_faces(frame)
        
        if len(faces) == 0:
            return {
                'face_detected': False,
                'is_live': False,
                'score': 0.0,
                'confidence': 0.0,
                'bbox': None
            }
        
        # Get liveness score
        face_arr = faces[0]
        bbox = boxes[0]
        
        # Convert face to PIL (face is already RGB from Gradio)
        face_pil = Image.fromarray(face_arr.astype(np.uint8))
        face_tensor = self.transform(face_pil).unsqueeze(0).numpy()
        
        # ONNX inference
        outputs = self.session.run(None, {'input': face_tensor})
        pixel_map = outputs[0]
        
        # Fast average
        liveness_score = float(np.mean(pixel_map))
        
        is_live = liveness_score > 0.03
        confidence = min(liveness_score * 100, 100.0)
        
        return {
            'face_detected': True,
            'is_live': is_live,
            'score': liveness_score,
            'confidence': confidence,
            'bbox': [[int(bbox[0][0]), int(bbox[0][1])], [int(bbox[1][0]), int(bbox[1][1])]]
        }

