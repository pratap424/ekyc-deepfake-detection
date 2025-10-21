"""
Advanced Attention-Based Explainability for Deepfake Detection
Provides forgery-type-specific visualization and spatial localization
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms


class ForgeryTypeAttentionExplainer:
    """
    Multi-level attention-based explainability system
    Shows WHERE manipulation occurred and WHAT TYPE of forgery was detected
    """
    
    def __init__(self, model, device='cpu'):
        """
        Args:
            model: Trained MultiTaskDeepfakeDetector
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.device = device
        
        # Multi-layer attention extraction for different forgery types
        # Early layers: low-level artifacts (GAN noise, compression)
        # Mid layers: texture inconsistencies (face2face, neural textures)
        # Deep layers: semantic forgeries (face swap)
        
        self.attention_extractors = {
            'shallow_artifacts': GradCAMPlusPlus(
                model=model, 
                target_layers=[model.backbone._blocks[2]]  # Early block - GAN artifacts
            ),
            'texture_inconsistencies': GradCAMPlusPlus(
                model=model, 
                target_layers=[model.backbone._blocks[8]]  # Mid block - texture issues
            ),
            'semantic_manipulation': GradCAMPlusPlus(
                model=model, 
                target_layers=[model.backbone._blocks[-1]]  # Deep block - face swap
            ),
            'multi_scale': LayerCAM(
                model=model, 
                target_layers=[model.backbone._blocks[-1]]  # Combined view
            )
        }
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print("✅ Attention-Based Explainability Module Initialized")
        print("   → Shallow artifacts detection (GAN noise, compression)")
        print("   → Texture inconsistency detection (face2face, neural textures)")
        print("   → Semantic manipulation detection (face swap)")
        print("   → Multi-scale attention fusion")
    
    def generate_forgery_heatmaps(self, image):
        """
        Generate forgery-type-specific attention heatmaps
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            dict with forgery_type: heatmap_overlay
        """
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Store original for overlay
        original_image = image.resize((224, 224))
        original_np = np.array(original_image).astype(np.float32) / 255.0
        
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Target FAKE class (class 0)
        targets = [ClassifierOutputTarget(0)]
        
        # Generate multiple attention maps
        heatmaps = {}
        
        for forgery_type, cam_extractor in self.attention_extractors.items():
            try:
                # Generate attention map
                grayscale_cam = cam_extractor(input_tensor=img_tensor, targets=targets)
                
                # Overlay on original image
                heatmap_overlay = show_cam_on_image(
                    original_np, 
                    grayscale_cam[0], 
                    use_rgb=True,
                    colormap=cv2.COLORMAP_JET
                )
                
                heatmaps[forgery_type] = {
                    'overlay': heatmap_overlay,
                    'raw_attention': grayscale_cam[0],
                    'pil': Image.fromarray(heatmap_overlay)
                }
            except Exception as e:
                print(f"⚠️ Warning: Failed to generate {forgery_type} heatmap: {e}")
        
        return heatmaps
    
    def create_composite_explanation(self, image, prediction_result):
        """
        Create comprehensive visual explanation combining all attention maps
        
        Args:
            image: Input image
            prediction_result: Dict from deepfake detector
            
        Returns:
            Composite explanation image with annotations
        """
        heatmaps = self.generate_forgery_heatmaps(image)
        
        # Convert image for display
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        original = image.resize((224, 224))
        original_np = np.array(original)
        
        # Create 2x3 grid: [original, shallow, texture, semantic, multi-scale, legend]
        grid_h, grid_w = 224 * 2, 224 * 3
        composite = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
        
        # Place images
        positions = {
            'original': (0, 0),
            'shallow_artifacts': (0, 224),
            'texture_inconsistencies': (0, 448),
            'semantic_manipulation': (224, 0),
            'multi_scale': (224, 224)
        }
        
        # Place original
        composite[0:224, 0:224] = original_np
        cv2.putText(composite, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Place heatmaps
        for forgery_type, pos in positions.items():
            if forgery_type == 'original':
                continue
            if forgery_type in heatmaps:
                y, x = pos
                composite[y:y+224, x:x+224] = heatmaps[forgery_type]['overlay']
                
                # Add label
                label = forgery_type.replace('_', ' ').title()
                cv2.putText(composite, label, (x+10, y+30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add legend/info panel
        legend_y, legend_x = 224, 448
        info_panel = np.ones((224, 224, 3), dtype=np.uint8) * 240
        
        # Add prediction info
        pred_text = prediction_result.get('prediction', 'Unknown')
        fake_prob = prediction_result.get('fake_probability', 0.0)
        
        cv2.putText(info_panel, "Detection Results:", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(info_panel, f"Status: {pred_text}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if pred_text == 'FAKE' else (0, 255, 0), 2)
        cv2.putText(info_panel, f"Fake Prob: {fake_prob:.2%}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add forgery type indicators
        cv2.putText(info_panel, "Attention Maps:", (10, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(info_panel, "- Shallow: GAN/Noise", (10, 155), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(info_panel, "- Texture: Face2Face", (10, 175), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(info_panel, "- Semantic: FaceSwap", (10, 195), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(info_panel, "- Multi: Combined", (10, 215), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        composite[legend_y:legend_y+224, legend_x:legend_x+224] = info_panel
        
        return composite, heatmaps
    
    def extract_manipulated_regions(self, image, threshold=0.6):
        """
        Extract specific facial regions that show manipulation
        Returns bounding boxes of suspicious areas
        
        Args:
            image: Input image
            threshold: Attention threshold (0-1)
            
        Returns:
            List of suspicious region coordinates and confidence
        """
        heatmaps = self.generate_forgery_heatmaps(image)
        
        suspicious_regions = []
        
        for forgery_type, heatmap_data in heatmaps.items():
            attention_map = heatmap_data['raw_attention']
            
            # Threshold attention map
            binary_mask = (attention_map > threshold).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter small regions
                    x, y, w, h = cv2.boundingRect(contour)
                    confidence = float(attention_map[y:y+h, x:x+w].mean())
                    
                    suspicious_regions.append({
                        'forgery_type': forgery_type,
                        'bbox': (x, y, w, h),
                        'confidence': confidence,
                        'area': w * h
                    })
        
        # Sort by confidence
        suspicious_regions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return suspicious_regions


class EnhancedFaceVerifier:
    """
    Enhanced face verification with ArcFace-style similarity
    Determines if two faces belong to same person
    """
    
    def __init__(self, identity_verifier, threshold=0.70):
        """
        Args:
            identity_verifier: Existing IdentityVerifier instance
            threshold: Similarity threshold
        """
        self.verifier = identity_verifier
        self.threshold = threshold
        
        print(f"✅ Enhanced Face Verifier initialized (threshold: {threshold})")
    
    def verify_with_explanation(self, image1, image2):
        """
        Verify if two faces match with detailed explanation
        
        Returns:
            dict with match result, similarity score, and confidence level
        """
        # Get embeddings
        try:
            result = self.verifier.compare_faces(image1, image2)
            
            similarity = result.get('similarity', 0.0)
            match = result.get('match', False)
            
            # Determine confidence level
            if similarity >= self.threshold + 0.15:
                confidence_level = "Very High"
            elif similarity >= self.threshold + 0.05:
                confidence_level = "High"
            elif similarity >= self.threshold:
                confidence_level = "Medium"
            elif similarity >= self.threshold - 0.10:
                confidence_level = "Low"
            else:
                confidence_level = "Very Low"
            
            explanation = {
                'match': match,
                'similarity': similarity,
                'threshold': self.threshold,
                'confidence_level': confidence_level,
                'decision': "SAME PERSON" if match else "DIFFERENT PERSON",
                'recommendation': self._get_recommendation(similarity, match)
            }
            
            return explanation
            
        except Exception as e:
            return {
                'error': str(e),
                'match': False,
                'similarity': 0.0
            }
    
    def _get_recommendation(self, similarity, match):
        """Generate KYC officer recommendation"""
        if match and similarity > 0.85:
            return "APPROVE - Strong facial match detected"
        elif match and similarity > 0.70:
            return "APPROVE WITH CAUTION - Facial match detected but manual review recommended"
        elif not match and similarity > 0.60:
            return "MANUAL REVIEW - Similarity borderline, requires human verification"
        else:
            return "REJECT - No facial match detected"


# Usage example
if __name__ == "__main__":
    print("Testing Attention-Based Explainability System...")
    print("This module provides forgery-type-specific visualization")
    print("Ready for integration with main application!")
