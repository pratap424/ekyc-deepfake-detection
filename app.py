"""
Enhanced Gradio web interface for eKYC verification 
With Forgery-Type-Specific Attention Explainability + Advanced Identity Verification
Run: python app.py
"""
import threading
import queue
import os
os.environ['GLOG_minloglevel'] = '2'
import gradio as gr
import cv2
import numpy as np
from PIL import Image
from models.deepfake_model import DeepfakeDetector
from models.liveness_model import LivenessDetector
from identity_verification import IdentityVerifier
from frequency_analyzer import FrequencyAnalyzer
from attention_explainability import ForgeryTypeAttentionExplainer, EnhancedFaceVerifier
from config import *

captured_frame = None
capture_lock = threading.Lock()


import time
from collections import deque



class LivenessStreamOptimizer:
    def __init__(self):
        self.frame_counter = 0
        self.last_result = None
        self.last_bbox = None
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()
    
    def should_process(self, skip_frames=2):
        self.frame_counter += 1
        return self.frame_counter % skip_frames == 0

stream_optimizer = LivenessStreamOptimizer()


# Global variables for CV2 streaming
frame_queue = queue.Queue(maxsize=2)  # Small queue to avoid lag
stop_streaming = threading.Event()

# Load models
print("🚀 Loading models...")
deepfake_detector = DeepfakeDetector(DEEPFAKE_MODEL, device=DEVICE)
liveness_detector = LivenessDetector(LIVENESS_MODEL)
identity_verifier = IdentityVerifier(device=DEVICE, confidence_threshold=0.70)
frequency_analyzer = FrequencyAnalyzer(device='cpu')

# Initialize advanced explainability modules
attention_explainer = ForgeryTypeAttentionExplainer(deepfake_detector.model, device=DEVICE)
enhanced_verifier = EnhancedFaceVerifier(identity_verifier, threshold=0.70)

print("✅ All models loaded successfully!")
print("✅ Advanced Attention-Based Explainability Enabled")


# ===================== ADVANCED ANALYSIS FUNCTIONS =====================

def advanced_deepfake_analysis(image):
    """
    Advanced deepfake analysis with forgery-type-specific attention maps
    """
    if image is None:
        return None, None, "⚠️ Please upload an image"
    
    try:
        # Convert to OpenCV format for liveness
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Standard predictions
        liveness_result = liveness_detector.predict(img_cv)
        deepfake_result = deepfake_detector.predict_with_explainability(image)
        
        # Generate forgery-type-specific attention maps
        composite_viz, forgery_heatmaps = attention_explainer.create_composite_explanation(
            image, deepfake_result
        )
        
        # Extract suspicious regions
        suspicious_regions = attention_explainer.extract_manipulated_regions(image, threshold=0.6)
        
        # Build comprehensive report
        status = "✅ VERIFIED" if deepfake_result['prediction'] == 'REAL' else "❌ REJECTED"
        
        report = f"""
# {status}

---

## 🔍 Advanced Deepfake Detection Analysis

### Detection Result
- **Prediction:** {deepfake_result['prediction']}
- **Fake Probability:** {deepfake_result['fake_probability']:.2%}
- **Real Probability:** {deepfake_result['real_probability']:.2%}
- **Match Score:** {deepfake_result['match_score']:.3f}
- **Model Liveness:** {deepfake_result['liveness_score']:.3f}

---

## 🎯 Attention-Based Spatial Localization

### Suspicious Regions Detected: {len(suspicious_regions)}

**Multi-Layer Attention Analysis:**
- ✓ Shallow Artifacts (GAN noise, compression)
- ✓ Texture Inconsistencies (Face2Face, Neural Textures)
- ✓ Semantic Manipulation (Face Swap)
- ✓ Multi-Scale Fusion

"""
        
        if suspicious_regions:
            report += "\n### Top Suspicious Regions:\n"
            for i, region in enumerate(suspicious_regions[:3], 1):
                forgery_type = region['forgery_type'].replace('_', ' ').title()
                report += f"{i}. **{forgery_type}**: {region['confidence']:.1%} confidence (Area: {region['area']} px²)\n"
        else:
            report += "\n### No highly suspicious regions detected\n"
        
        report += f"""

---

## 🧬 Liveness Detection (DeepPixBiS)

- **Face Detected:** {'Yes ✅' if liveness_result['face_detected'] else 'No ❌'}
- **Is Live:** {'Yes ✅' if liveness_result['is_live'] else 'No ❌'}
- **Liveness Score:** {liveness_result['score']:.3f}

---

## 🔐 Final Decision

**Status:** {status}
**Reason:** {'All authenticity checks passed' if deepfake_result['prediction'] == 'REAL' else 'Deepfake manipulation detected'}
"""
        
        composite_pil = Image.fromarray(composite_viz)
        return composite_pil, deepfake_result['heatmap_pil'], report
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return None, None, error_msg


def enhanced_face_verification(image1, image2):
    """Enhanced face verification with KYC recommendations"""
    if image1 is None or image2 is None:
        return None, None, None, "⚠️ Please upload BOTH images"
    
    try:
        # Get enhanced verification
        verification_result = enhanced_verifier.verify_with_explanation(image1, image2)
        
        # Get deepfake analysis
        img1_cv = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
        img2_cv = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)
        
        deep1 = deepfake_detector.predict_with_explainability(image1)
        deep2 = deepfake_detector.predict_with_explainability(image2)
        
        live1 = liveness_detector.predict(img1_cv)
        live2 = liveness_detector.predict(img2_cv)
        
        all_checks_passed = (
            deep1['prediction'] == 'REAL' and
            deep2['prediction'] == 'REAL' and
            live1['is_live'] and
            live2['is_live'] and
            verification_result['match']
        )
        
        status = "✅ IDENTITY VERIFIED" if all_checks_passed else "❌ VERIFICATION FAILED"
        
        report = f"""
# {status}

---

## 👥 Identity Matching Analysis (FaceNet)

### Verification Result
- **Decision:** {verification_result['decision']}
- **Similarity Score:** {verification_result['similarity']:.2%}
- **Confidence Level:** {verification_result['confidence_level']}
- **Threshold Used:** {verification_result['threshold']:.0%}

### KYC Officer Recommendation
**{verification_result['recommendation']}**

---

## 📸 Reference Image Analysis

### Deepfake Detection
- **Prediction:** {deep1['prediction']} {'✅' if deep1['prediction']=='REAL' else '❌'}
- **Real Probability:** {deep1['real_probability']:.1%}

### Liveness Detection
- **Is Live:** {'YES ✅' if live1['is_live'] else 'NO ❌'}
- **Liveness Score:** {live1['score']:.3f}

---

## 📸 Query Image Analysis

### Deepfake Detection
- **Prediction:** {deep2['prediction']} {'✅' if deep2['prediction']=='REAL' else '❌'}
- **Real Probability:** {deep2['real_probability']:.1%}

### Liveness Detection
- **Is Live:** {'YES ✅' if live2['is_live'] else 'NO ❌'}
- **Liveness Score:** {live2['score']:.3f}

---

## 🔐 Final Security Assessment

**All Security Checks:** {'PASSED ✅' if all_checks_passed else 'FAILED ❌'}
"""
        
        failures = []
        if deep1['prediction'] != 'REAL':
            failures.append("- Reference image flagged as deepfake")
        if deep2['prediction'] != 'REAL':
            failures.append("- Query image flagged as deepfake")
        if not live1['is_live']:
            failures.append("- Reference failed liveness")
        if not live2['is_live']:
            failures.append("- Query failed liveness")
        if not verification_result['match']:
            failures.append(f"- Identity mismatch ({verification_result['similarity']:.1%})")
        
        if failures:
            report += "\n\n### Failure Reasons:\n" + "\n".join(failures)
        
        # Side-by-side comparison
        img1_resized = image1.resize((224, 224))
        img2_resized = image2.resize((224, 224))
        combined = np.hstack([np.array(img1_resized), np.array(img2_resized)])
        color = (0, 255, 0) if verification_result['match'] else (255, 0, 0)
        cv2.rectangle(combined, (0, 0), (combined.shape[1]-1, combined.shape[0]-1), color, 8)
        cv2.putText(combined, verification_result['decision'], (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        comparison_pil = Image.fromarray(combined)
        return comparison_pil, deep1['heatmap_pil'], deep2['heatmap_pil'], report
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return None, None, None, error_msg


# ===================== ORIGINAL FUNCTIONS =====================

def process_single_image(image):
    """Process single image with Grad-CAM explainability"""
    if image is None:
        return None, None, "⚠️ Please upload an image"
    
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    liveness_result = liveness_detector.predict(img_cv)
    deepfake_result = deepfake_detector.predict_with_explainability(image)
    
    verified = (
        deepfake_result['prediction'] == 'REAL' and
        liveness_result['is_live'] and
        liveness_result['face_detected']
    )
    
    status = "✅ VERIFIED" if verified else "❌ REJECTED"
    report = f"""
## {status}

### Deepfake Detection
- **Prediction:** {deepfake_result['prediction']}
- **Real Probability:** {deepfake_result['real_probability']:.2%}
- **Fake Probability:** {deepfake_result['fake_probability']:.2%}
- **Match Score:** {deepfake_result['match_score']:.3f}

### Liveness Detection
- **Is Live:** {'Yes ✅' if liveness_result['is_live'] else 'No ❌'}
- **Liveness Score:** {liveness_result['score']:.3f}

### Final Decision
**Status:** {status}
"""
    
    return image, deepfake_result['heatmap_pil'], report


def analyze_with_frequency(image):
    """Single image with frequency domain analysis"""
    if image is None:
        return None, None, "⚠️ Upload an image"
    
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    deep_result = deepfake_detector.predict_with_explainability(image)
    live_result = liveness_detector.predict(img_cv)
    freq_result = frequency_analyzer.analyze_deepfake_frequency(image)
    freq_spectrum = frequency_analyzer.visualize_frequency_spectrum(image)
    
    spatial_suspicious = deep_result['prediction'] == 'FAKE'
    frequency_suspicious = freq_result['is_suspicious']
    final_verdict = "❌ FAKE" if (spatial_suspicious or frequency_suspicious) else "✅ REAL"
    
    report = f"""
## {final_verdict} (Multi-Domain Analysis)

### Spatial Domain (CNN)
- **Prediction:** {deep_result['prediction']}
- **Confidence:** {deep_result['real_probability']:.1%}

### Frequency Domain (FFT/DCT)
- **Anomaly Score:** {freq_result['anomaly_score']:.1%}
- **Suspicious:** {'YES ❌' if freq_result['is_suspicious'] else 'NO ✅'}
- **High-Freq Ratio:** {freq_result['high_freq_ratio']:.4f}

### Liveness
- **Score:** {live_result['score']:.3f} {'✅' if live_result['is_live'] else '❌'}
"""
    
    return deep_result['heatmap_pil'], freq_spectrum, report


def process_webcam(frame):
    """Process webcam stream with explainability - FIXED FOR GRADIO"""
    if frame is None:
        return None, None, "⚠️ No frame"
    
    # Liveness detection
    liveness_result = liveness_detector.predict(frame)
    
    if not liveness_result['face_detected']:
        # Convert BGR to RGB for Gradio display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb, None, "⚠️ No face detected"
    
    # Deepfake detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    deepfake_result = deepfake_detector.predict_with_explainability(Image.fromarray(frame_rgb))
    
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
    
    # Convert BGR to RGB for Gradio
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    # Report
    report = f"""
**Status:** {'✅ VERIFIED' if verified else '❌ REJECTED'}
**Deepfake:** {deepfake_result['prediction']} ({deepfake_result['real_probability']:.1%})
**Liveness:** {liveness_result['score']:.3f} {'✅' if liveness_result['is_live'] else '❌'}
"""
    
    return annotated_rgb, deepfake_result['heatmap_pil'], report

# ==================== OPTIMIZED WEBCAM FUNCTION ====================
def process_webcam_liveness_only(frame):

    """
    ULTIMATE: Frame skipping + caching + frozen capture
    - Processes every 2nd frame only (FAST)
    - Saves frames for button (WORKS)
    """
    global captured_frame
    
    if frame is None:
        return frame, "⚠️ No frame"
    
    # ALWAYS save latest frame (minimal overhead - just a copy)
    with capture_lock:
        captured_frame = frame.copy() if isinstance(frame, np.ndarray) else np.array(frame)
    
    current_time = time.time()
    
    # FRAME SKIP: Only process every 2nd frame
    if not stream_optimizer.should_process(skip_frames=2):
        # Reuse cached result
        if stream_optimizer.last_result is not None:
            result = stream_optimizer.last_result
            bbox = stream_optimizer.last_bbox
            
            status_text = "LIVE" if result['is_live'] else "SPOOF"
            annotated = frame.copy()
            
            if bbox is not None:
                color = (0, 255, 0) if result['is_live'] else (255, 0, 0)
                cv2.rectangle(annotated, tuple(bbox[0]), tuple(bbox[1]), color, 3)
                
                cv2.putText(annotated, status_text, (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                cv2.putText(annotated, f"{result['score']:.3f}",
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # FPS counter
                elapsed = current_time - stream_optimizer.last_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                stream_optimizer.fps_history.append(fps)
                avg_fps = sum(stream_optimizer.fps_history) / len(stream_optimizer.fps_history)
                cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                avg_fps = 0 if not stream_optimizer.fps_history else sum(stream_optimizer.fps_history) / len(stream_optimizer.fps_history)
            
            stream_optimizer.last_time = current_time
            report = f"**{status_text}** | Score: {result['score']:.3f} | FPS: {avg_fps:.1f}"
            return annotated, report
        else:
            return frame, "⚡ Initializing..."
    
    # Process frame (only every 2nd frame)
    liveness_result = liveness_detector.predict(frame)
    stream_optimizer.last_result = liveness_result
    stream_optimizer.last_bbox = liveness_result['bbox']
    
    if not liveness_result['face_detected']:
        stream_optimizer.last_time = current_time
        return frame, "⚠️ No face detected"
    
    # Annotate
    annotated = frame.copy()
    bbox = liveness_result['bbox']
    color = (0, 255, 0) if liveness_result['is_live'] else (255, 0, 0)
    cv2.rectangle(annotated, tuple(bbox[0]), tuple(bbox[1]), color, 3)
    
    status_text = "LIVE" if liveness_result['is_live'] else "SPOOF"
    cv2.putText(annotated, status_text, (10, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(annotated, f"{liveness_result['score']:.3f}",
               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # FPS
    elapsed = current_time - stream_optimizer.last_time
    fps = 1.0 / elapsed if elapsed > 0 else 0
    stream_optimizer.fps_history.append(fps)
    avg_fps = sum(stream_optimizer.fps_history) / len(stream_optimizer.fps_history)
    cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 110),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    stream_optimizer.last_time = current_time
    report = f"**{status_text}** | Score: {liveness_result['score']:.3f} | FPS: {avg_fps:.1f}"
    
    return annotated, report







# ==================== OPTIONAL: CAPTURE & FULL ANALYSIS ====================

def capture_and_analyze_full():
    """
    Capture FROZEN frame and run full analysis
    NO INPUTS - uses globally captured frame
    """
    global captured_frame
    
    # Get frozen frame
    with capture_lock:
        if captured_frame is None:
            return None, """
⚠️ **No Frame Captured**

Please ensure webcam is streaming and try again.
"""
        frame = captured_frame.copy()
    
    try:
        # Ensure RGB format
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        # Run liveness detection
        liveness_result = liveness_detector.predict(frame)
        
        if not liveness_result['face_detected']:
            return None, "⚠️ **No Face Detected**\n\nCenter your face and try again."
        
        # Convert to PIL for deepfake detector
        frame_pil = Image.fromarray(frame)
        
        # Run deepfake detection with Grad-CAM
        deepfake_result = deepfake_detector.predict_with_explainability(frame_pil)
        
        # Verification
        verified = (
            deepfake_result['prediction'] == 'REAL' and 
            liveness_result['is_live'] and 
            liveness_result['face_detected']
        )
        
        status = "✅ VERIFIED" if verified else "❌ REJECTED"
        status_emoji = "✅" if verified else "❌"
        
        # Build report
        report = f"""
# {status}

---

## 🔍 Deepfake Detection

| Metric | Value |
|--------|-------|
| **Prediction** | {deepfake_result['prediction']} |
| **Real Probability** | {deepfake_result['real_probability']:.1%} |
| **Fake Probability** | {deepfake_result['fake_probability']:.1%} |

## 👤 Liveness Detection

| Metric | Value |
|--------|-------|
| **Status** | {'✅ LIVE' if liveness_result['is_live'] else '❌ SPOOF'} |
| **Liveness Score** | {liveness_result['score']:.4f} |

## 🎯 Final Verdict

**Status:** {status}

### Security Checks:
- {status_emoji} Deepfake: {'PASSED' if deepfake_result['prediction'] == 'REAL' else 'FAILED'}
- {status_emoji} Liveness: {'PASSED' if liveness_result['is_live'] else 'FAILED'}

**Model:** EfficientNet-B0 + DeepPixBiS
"""
        
        return deepfake_result['heatmap_pil'], report
        
    except Exception as e:
        import traceback
        error_msg = f"❌ **Error:** {str(e)}\n\n``````"
        print(f"Error: {e}")
        traceback.print_exc()
        return None, error_msg





# ===================== BUILD GRADIO INTERFACE =====================

with gr.Blocks(title="🛡️ Advanced eKYC with Attention Explainability", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🛡️ Advanced eKYC Verification System
    ## With Forgery-Type-Specific Attention Explainability
    
    **Revolutionary Features:**
    - ✅ **4 Forgery Type Detection**: Face Swap, GAN, Face2Face, Neural Textures
    - ✅ **Attention-Based Spatial Localization**: Shows WHERE manipulation occurred
    - ✅ **Multi-Layer Explainability**: Shallow, Texture, Semantic analysis
    - ✅ **Enhanced Face Verification**: Confidence levels + KYC recommendations
    - ✅ **Real-Time Webcam Verification**: Live deepfake + liveness detection
    ---
    """)
    
    with gr.Tabs():
        
        # ==================== TAB 1: ADVANCED DEEPFAKE ANALYSIS ====================
        with gr.Tab("🔬 Advanced Deepfake Analysis"):
            gr.Markdown("""
            ### Forgery-Type-Specific Attention Analysis
            Upload an image to see **WHERE** and **WHAT TYPE** of manipulation was detected.
            """)
            
            with gr.Row():
                adv_input = gr.Image(type="pil", label="📸 Upload Image")
                with gr.Column():
                    adv_composite = gr.Image(label="🎯 Forgery-Type-Specific Attention Maps")
                    adv_heatmap = gr.Image(label="🔥 Overall Attention Heatmap")
            
            adv_output = gr.Markdown(label="📋 Detailed Analysis Report")
            adv_btn = gr.Button("🔬 Run Advanced Analysis", variant="primary", size="lg")
            adv_btn.click(advanced_deepfake_analysis, inputs=[adv_input], 
                         outputs=[adv_composite, adv_heatmap, adv_output])
            
            gr.Markdown("""
            **Attention Map Legend:**
            - **Shallow Artifacts**: GAN noise, compression
            - **Texture Inconsistencies**: Face2Face, Neural Textures
            - **Semantic Manipulation**: Face Swap
            - Red/yellow = High manipulation probability
            """)
        
        # ==================== TAB 2: ENHANCED FACE VERIFICATION ====================
        with gr.Tab("👥 Enhanced Face Verification"):
            gr.Markdown("""
            ### Identity Verification with KYC Recommendations
            Compare ID card vs live selfie with actionable recommendations.
            """)
            
            with gr.Row():
                with gr.Column():
                    ver_input1 = gr.Image(type="pil", label="📄 Reference (ID Document)")
                    ver_input2 = gr.Image(type="pil", label="🤳 Query (Live Selfie)")
                    ver_btn = gr.Button("✓ Verify Identity", variant="primary", size="lg")
                
                with gr.Column():
                    ver_comparison = gr.Image(label="🔍 Face Comparison")
                    ver_heatmap1 = gr.Image(label="📄 Reference - Attention")
                    ver_heatmap2 = gr.Image(label="🤳 Query - Attention")
            
            ver_output = gr.Markdown(label="📋 Verification Report")
            ver_btn.click(enhanced_face_verification, inputs=[ver_input1, ver_input2],
                         outputs=[ver_comparison, ver_heatmap1, ver_heatmap2, ver_output])
        
        # ==================== TAB 3: REAL-TIME WEBCAM (OPTIMIZED) ====================
        with gr.Tab("🎥 Real-Time Liveness Check"):
            gr.Markdown("""
            ### ⚡ Fast Liveness-Only Mode
            **Optimized for smooth real-time performance (no deepfake check during streaming)**
            - Only runs liveness detection for instant feedback
            - For full deepfake analysis, capture frame and click "Analyze Full"
            - Use other tabs for comprehensive analysis
            """)
            
            with gr.Row():
                with gr.Column():
                    webcam_input = gr.Image(sources="webcam", streaming=True, label="📹 Webcam Feed")
                    
                with gr.Column():
                    webcam_annotated = gr.Image(label="🎯 Liveness Detection")
                    webcam_output = gr.Markdown(label="📊 Live Status")
            
            # Real-time streaming (liveness only)
            webcam_input.stream(
                process_webcam_liveness_only,
                inputs=[webcam_input],
                outputs=[webcam_annotated, webcam_output],
                stream_every=0.1 # 10 FPS target - adjust based on your system
            )
            
            gr.Markdown("---")
            gr.Markdown("### 🔍 Optional: Full Analysis on Captured Frame")
            
            with gr.Row():
                with gr.Column():
                    capture_btn = gr.Button("📸 Capture & Run Full Analysis", variant="secondary", size="lg")
                
                with gr.Column():
                    full_heatmap = gr.Image(label="🔥 Deepfake Heatmap")
                    full_output = gr.Markdown(label="📋 Full Report")
            
            # Full analysis on button click (not during streaming)
            # Streaming (fast + saves frames)
            webcam_input.stream(
                process_webcam_liveness_only,
                inputs=[webcam_input],
                outputs=[webcam_annotated, webcam_output],
                stream_every=0.1
            )
            
            # Button (uses frozen frame)
            capture_btn.click(
                capture_and_analyze_full,
                inputs=[],  # NO INPUTS
                outputs=[full_heatmap, full_output]
            )


        
        # ==================== TAB 4: SINGLE IMAGE ====================
        with gr.Tab("📸 Single Image Analysis"):
            gr.Markdown("### Standard deepfake + liveness with Grad-CAM")
            
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="pil", label="Upload Image")
                    img_btn = gr.Button("🔍 Analyze", variant="primary")
                with gr.Column():
                    img_original = gr.Image(label="Original")
                    img_heatmap = gr.Image(label="Grad-CAM")
            
            img_output = gr.Markdown(label="Report")
            img_btn.click(process_single_image, inputs=[img_input],
                         outputs=[img_original, img_heatmap, img_output])
        
        # ==================== TAB 5: FREQUENCY ANALYSIS ====================
        with gr.Tab("🌊 Frequency Domain"):
            gr.Markdown("### Multi-domain (Spatial + Frequency)")
            
            with gr.Row():
                with gr.Column():
                    freq_input = gr.Image(type="pil", label="Upload Image")
                    freq_btn = gr.Button("📊 Analyze", variant="primary")
                with gr.Column():
                    freq_heatmap = gr.Image(label="Spatial Heatmap")
                    freq_spectrum = gr.Image(label="Frequency Spectrum")
            
            freq_output = gr.Markdown(label="Report")
            freq_btn.click(analyze_with_frequency, inputs=[freq_input],
                          outputs=[freq_heatmap, freq_spectrum, freq_output])
        
        # ==================== TAB 6: SYSTEM INFO ====================
        with gr.Tab("ℹ️ System Architecture"):
            gr.Markdown("""
## Advanced System Components

### 1. Multi-Task Deepfake Detection
- **Backbone:** EfficientNet-B0
- **Tasks:** Authenticity + Matching + Liveness
- **Training:** Sentinel-Faces-v1

### 2. Attention-Based Forgery Localization (NEW!)
- **Method:** Multi-layer Grad-CAM++ with LayerCAM
- **Layers:** Shallow (GAN) → Mid (Texture) → Deep (Semantic)
- **Innovation:** Shows WHAT type and WHERE

### 3. Enhanced Face Verification
- **Model:** FaceNet (InceptionResnetV1)
- **Confidence Levels:** Very High / High / Medium / Low
- **KYC Recommendations:** APPROVE / REVIEW / REJECT

### 4. Liveness Detection
- **Model:** DeepPixBiS
- **Defends Against:** Print/replay/mask attacks

### 5. Frequency Domain Analysis
- **Methods:** FFT + DCT
- **Detects:** High-frequency GAN artifacts

---

**Developed for:** ZenTej Season 3 @ CAIR IIT Mandi  
**Problem:** Deepfake-Proof eKYC Challenge  
**Novelty:** Attention-based forgery localization
            """)

# Launch
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 Launching Advanced eKYC System")
    print("="*70 + "\n")
    
    demo.launch(
        server_port=GRADIO_PORT,
        share=True,
        show_error=True,
        server_name="0.0.0.0"
    )
