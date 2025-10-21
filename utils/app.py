"""
Gradio web interface for eKYC verification with Grad-CAM explainability
Run: python app.py
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image
from models.deepfake_model import DeepfakeDetector
from models.liveness_model import LivenessDetector
from config import *

# Load models
deepfake_detector = DeepfakeDetector(DEEPFAKE_MODEL, device=DEVICE)
liveness_detector = LivenessDetector(LIVENESS_MODEL)

print("✅ All models loaded successfully!")


def process_single_image(image):
    """Process single image with Grad-CAM explainability"""
    if image is None:
        return None, None, "⚠️ Please upload an image"
    
    # Convert to OpenCV format for liveness
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Liveness detection
    liveness_result = liveness_detector.predict(img_cv)
    
    # Deepfake detection WITH explainability
    deepfake_result = deepfake_detector.predict_with_explainability(image)
    
    # Verification logic
    verified = (
        deepfake_result['prediction'] == 'REAL' and
        liveness_result['is_live'] and
        liveness_result['face_detected']
    )
    
    # Build detailed report
    status = "✅ VERIFIED" if verified else "❌ REJECTED"
    
    report = f"""
## {status}

### Deepfake Detection
- **Prediction:** {deepfake_result['prediction']}
- **Real Probability:** {deepfake_result['real_probability']:.2%}
- **Fake Probability:** {deepfake_result['fake_probability']:.2%}
- **Match Score:** {deepfake_result['match_score']:.3f}
- **Model Liveness:** {deepfake_result['liveness_score']:.3f}

### Liveness Detection (DeepPixBiS)
- **Face Detected:** {'Yes' if liveness_result['face_detected'] else 'No'}
- **Is Live:** {'Yes ✅' if liveness_result['is_live'] else 'No ❌'}
- **Liveness Score:** {liveness_result['score']:.3f}

### Final Decision
**Status:** {status}
**Reason:** {'All checks passed' if verified else 'Failed one or more authenticity checks'}
"""
    
    # Return original image and Grad-CAM heatmap
    return image, deepfake_result['heatmap_pil'], report


def process_webcam(frame):
    """Process webcam stream with explainability"""
    if frame is None:
        return None, None, "⚠️ No frame"
    
    # Liveness detection
    liveness_result = liveness_detector.predict(frame)
    
    if not liveness_result['face_detected']:
        return frame, None, "⚠️ No face detected"
    
    # Deepfake detection with Grad-CAM
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    deepfake_result = deepfake_detector.predict_with_explainability(frame_rgb)
    
    # Verification
    verified = (
        deepfake_result['prediction'] == 'REAL' and
        liveness_result['is_live']
    )
    
    # Annotate frame
    annotated = frame.copy()
    bbox = liveness_result['bbox']
    
    if bbox is not None:
        color = (0, 255, 0) if liveness_result['is_live'] else (0, 0, 255)
        cv2.rectangle(annotated, tuple(bbox[0]), tuple(bbox[1]), color, 3)
    
    # Status overlay
    status_text = "VERIFIED" if verified else "REJECTED"
    status_color = (0, 255, 0) if verified else (0, 0, 255)
    cv2.putText(annotated, status_text, (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
    
    cv2.putText(annotated, f"Real: {deepfake_result['real_probability']:.1%}", 
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(annotated, f"Live: {liveness_result['score']:.3f}", 
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Report
    report = f"""
**Status:** {'✅ VERIFIED' if verified else '❌ REJECTED'}
**Deepfake:** {deepfake_result['prediction']} ({deepfake_result['real_probability']:.1%})
**Liveness:** {liveness_result['score']:.3f} {'✅' if liveness_result['is_live'] else '❌'}
"""
    
    # Convert Grad-CAM to PIL for display
    heatmap_pil = Image.fromarray(cv2.cvtColor(deepfake_result['grad_cam_heatmap'], cv2.COLOR_BGR2RGB))
    
    return annotated, heatmap_pil, report


# Build Gradio interface
with gr.Blocks(title="eKYC Deepfake-Proof Verification", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🔒 Deepfake-Proof eKYC System
    ## Multi-Task Detection with Grad-CAM Explainability
    **Models:** EfficientNet-B0 + DeepPixBiS + Grad-CAM
    """)
    
    with gr.Tabs():
        
        # Tab 1: Single Image Upload
        with gr.Tab("📷 Image Upload"):
            gr.Markdown("### Upload an image for verification with explainability")
            
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="pil", label="Upload Image")
                    img_btn = gr.Button("🔍 Verify Image", variant="primary")
                
                with gr.Column():
                    img_original = gr.Image(label="Original Image")
                    img_heatmap = gr.Image(label="Grad-CAM Explainability Heatmap")
            
            img_output = gr.Markdown(label="Verification Report")
            
            img_btn.click(
                process_single_image,
                inputs=[img_input],
                outputs=[img_original, img_heatmap, img_output]
            )
        
        # Tab 2: Webcam Stream
        with gr.Tab("🎥 Webcam Verification"):
            gr.Markdown("### Real-time verification with live explainability")
            
            with gr.Row():
                with gr.Column():
                    webcam_input = gr.Image(source="webcam", streaming=True, label="Webcam Feed")
                
                with gr.Column():
                    webcam_annotated = gr.Image(label="Detection Result")
                    webcam_heatmap = gr.Image(label="Grad-CAM Heatmap")
            
            webcam_output = gr.Markdown(label="Live Status")
            
            webcam_input.stream(
                process_webcam,
                inputs=[webcam_input],
                outputs=[webcam_annotated, webcam_heatmap, webcam_output],
                stream_every=1/STREAM_FPS
            )
        
        # Tab 3: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## System Architecture
            
            ### Multi-Task Deepfake Detection
            - **Backbone:** EfficientNet-B0 pre-trained on ImageNet
            - **Task 1:** Authenticity classification (Real vs Fake)
            - **Task 2:** Face matching score
            - **Task 3:** Liveness prediction
            
            ### Liveness Detection
            - **Model:** DeepPixBiS anti-spoofing
            - **Method:** Pixel-wise binary supervision
            - **Defense:** Protects against presentation attacks
            
            ### Explainable AI
            - **Method:** Grad-CAM (Gradient-weighted Class Activation Mapping)
            - **Purpose:** Visual explanation of model decisions
            - **Output:** Heatmap highlighting suspicious regions
            
            ### Performance
            - **Real-time capable:** 30+ FPS on GPU, 5-10 FPS on CPU
            - **Accuracy:** Trained on Sentinel-Faces-v1 dataset
            - **Robustness:** Multi-task learning improves generalization
            
            ---
            **Developed for:** ZenTej Season 3 Hackathon @ CAIR IIT Mandi
            """)
    
    gr.Markdown("""
    ---
    **Note:** Red regions in Grad-CAM indicate areas the model considers suspicious for deepfake detection.
    """)

# Launch
if __name__ == "__main__":
    demo.launch(
        server_port=GRADIO_PORT,
        share=False,
        show_error=True
    )
