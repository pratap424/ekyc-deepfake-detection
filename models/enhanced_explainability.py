"""
Enhanced explainability wrapper for deepfake detection
Adds region-specific analysis and forgery type inference
"""

import numpy as np
from PIL import Image
import cv2

# Forgery type inference based on heatmap patterns
FORGERY_TYPES = {
    'FaceSwap': 'Entire face region replaced',
    'Face2Face': 'Facial expressions/mouth manipulated', 
    'Deepfake (GAN)': 'AI-generated synthetic face',
    'NeuralTextures': 'Texture/skin manipulated',
    'REAL': 'No manipulation detected'
}

class EnhancedExplainability:
    """
    Wrapper that adds advanced explainability to existing DeepfakeDetector
    Provides:
    1. Region-specific attention analysis
    2. Forgery type inference
    3. Confidence scores per facial region
    """
    
    def __init__(self, base_detector):
        """
        Args:
            base_detector: Instance of DeepfakeDetector with Grad-CAM
        """
        self.detector = base_detector
        print("✅ EnhancedExplainability wrapper loaded")
    
    def predict_with_enhanced_explainability(self, image):
        """
        Run prediction with enhanced regional analysis
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            dict with prediction, forgery_type, suspicious_regions, grad_cam, etc.
        """
        # Get base prediction with Grad-CAM
        result = self.detector.predict_with_explainability(image)
        
        # Convert heatmap to grayscale numpy array for analysis
        heatmap_pil = result['heatmap_pil']
        heatmap_gray = np.array(heatmap_pil.convert('L')).astype(np.float32) / 255.0
        
        # Analyze suspicious regions
        suspicious_regions = self._analyze_heatmap_regions(heatmap_gray)
        
        # Infer forgery type based on heatmap pattern
        forgery_type, forgery_confidence = self._infer_forgery_type(
            result['prediction'],
            result['fake_probability'],
            suspicious_regions
        )
        
        # Add enhanced info to result
        result['suspicious_regions'] = suspicious_regions
        result['forgery_type'] = forgery_type
        result['forgery_confidence'] = forgery_confidence
        result['forgery_description'] = FORGERY_TYPES.get(forgery_type, 'Unknown')
        
        return result
    
    def _analyze_heatmap_regions(self, heatmap):
        """
        Analyze Grad-CAM heatmap to identify manipulated facial regions
        
        Args:
            heatmap: 2D numpy array (grayscale, normalized 0-1)
        
        Returns:
            List of dicts with region name, confidence, max_activation
        """
        h, w = heatmap.shape
        
        # Define facial region masks (approximate locations)
        regions = {
            'eyes': heatmap[int(h*0.25):int(h*0.45), int(w*0.2):int(w*0.8)],
            'nose': heatmap[int(h*0.4):int(h*0.6), int(w*0.35):int(w*0.65)],
            'mouth': heatmap[int(h*0.55):int(h*0.75), int(w*0.25):int(w*0.75)],
            'chin': heatmap[int(h*0.7):int(h*0.95), int(w*0.3):int(w*0.7)],
            'forehead': heatmap[int(h*0.05):int(h*0.3), int(w*0.2):int(w*0.8)],
            'cheeks': heatmap[int(h*0.35):int(h*0.65), :],
        }
        
        suspicious = []
        threshold = 0.5  # High activation = suspicious
        
        for region_name, region_heatmap in regions.items():
            if region_heatmap.size == 0:
                continue
            
            avg_activation = np.mean(region_heatmap)
            max_activation = np.max(region_heatmap)
            
            # Flag as suspicious if average is high or has hotspots
            if avg_activation > threshold or max_activation > 0.75:
                suspicious.append({
                    'region': region_name.capitalize(),
                    'confidence': round(float(avg_activation * 100), 1),
                    'max_activation': round(float(max_activation * 100), 1)
                })
        
        # Sort by confidence descending
        suspicious.sort(key=lambda x: x['confidence'], reverse=True)
        
        return suspicious
    
    def _infer_forgery_type(self, prediction, fake_prob, suspicious_regions):
        """
        Infer forgery type based on prediction and heatmap patterns
        
        Args:
            prediction: 'REAL' or 'FAKE'
            fake_prob: Fake probability (0-1)
            suspicious_regions: List of suspicious regions
        
        Returns:
            (forgery_type: str, confidence: float)
        """
        if prediction == 'REAL':
            return 'REAL', 1.0 - fake_prob
        
        # Extract most suspicious regions
        top_regions = [r['region'].lower() for r in suspicious_regions[:3]]
        
        # Pattern-based forgery type inference
        if fake_prob > 0.85:
            # Very high confidence = likely GAN-generated
            forgery_type = 'Deepfake (GAN)'
        elif 'mouth' in top_regions or 'chin' in top_regions:
            # Mouth/chin manipulation = Face2Face reenactment
            forgery_type = 'Face2Face'
        elif len(suspicious_regions) >= 4:
            # Most regions flagged = likely full face swap
            forgery_type = 'FaceSwap'
        elif 'cheeks' in top_regions or 'forehead' in top_regions:
            # Texture-heavy regions = neural textures
            forgery_type = 'NeuralTextures'
        else:
            # Default to face swap
            forgery_type = 'FaceSwap'
        
        return forgery_type, fake_prob
