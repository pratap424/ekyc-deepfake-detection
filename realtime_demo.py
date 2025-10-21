"""
Real-time OpenCV demo (fastest performance)
Run: python realtime_demo.py
"""
import cv2
import time
from models import DeepfakeDetector, LivenessDetector
from utils.config import *

# Load models
deepfake = DeepfakeDetector(DEEPFAKE_MODEL, device=DEVICE)
liveness = LivenessDetector(LIVENESS_MODEL)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

fps_time = time.time()
fps_count = 0
fps = 0

print("✅ Press 'Q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    
    # Detect
    live_result = liveness.predict(frame)
    
    if live_result['face_detected']:
        deep_result = deepfake.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        verified = deep_result['prediction'] == 'REAL' and live_result['is_live']
        
        # Draw bbox
        if live_result['bbox'] is not None:
            color = (0, 255, 0) if live_result['is_live'] else (0, 0, 255)
            cv2.rectangle(canvas, tuple(live_result['bbox'][0]), 
                         tuple(live_result['bbox'][1]), color, 3)
        
        # Status
        status = "VERIFIED" if verified else "REJECTED"
        cv2.putText(canvas, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                   (0, 255, 0) if verified else (0, 0, 255), 3)
        
        # Scores
        cv2.putText(canvas, f"Liveness: {live_result['score']:.3f}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, f"Deepfake: {deep_result['prediction']}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # FPS
    fps_count += 1
    if time.time() - fps_time > 1:
        fps = fps_count
        fps_count = 0
        fps_time = time.time()
    
    cv2.putText(canvas, f"FPS: {fps}", (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("eKYC Verification - Press Q to quit", canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
