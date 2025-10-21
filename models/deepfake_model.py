"""
Multi-task deepfake detection model with Grad-CAM explainability
Trained on Sentinel-Faces-v1 dataset
"""

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from attention_explainability import ForgeryTypeAttentionExplainer, EnhancedFaceVerifier


# Grad-CAM imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class MultiTaskDeepfakeDetector(nn.Module):
    """EfficientNet-B0 with 3 heads: authenticity, match score, liveness"""
    
    def __init__(self, backbone='efficientnet-b0', num_classes=2, dropout=0.6):
        super(MultiTaskDeepfakeDetector, self).__init__()
        
        # Backbone
        self.backbone = EfficientNet.from_pretrained(backbone)
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.5)
        )
        
        # Authenticity head (real vs fake)
        self.authenticity_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes)
        )
        
        # Match score head (similarity to reference) - ORIGINAL ARCHITECTURE
        self.match_score_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Liveness head (anti-spoofing) - ORIGINAL ARCHITECTURE
        self.liveness_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.feature_extractor(features)
        
        authenticity = self.authenticity_head(features)
        match_score = self.match_score_head(features)
        liveness = self.liveness_head(features)
        
        return authenticity, match_score, liveness


class DeepfakeDetector:
    """Wrapper for inference with Grad-CAM explainability"""
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = MultiTaskDeepfakeDetector()
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(device).eval()
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Initialize Grad-CAM on the last convolutional block
        target_layer = self.model.backbone._blocks[-1]
        self.grad_cam = GradCAM(model=self.model, target_layers=[target_layer])
        
        print(f"✅ DeepfakeDetector loaded on {device} with Grad-CAM explainability")
        self.attention_explainer = ForgeryTypeAttentionExplainer(self.model, device)
        print("✅ Advanced Attention-Based Explainability Enabled")


    
    def predict_with_advanced_explainability(self, image):
        """
        Prediction with forgery-type-specific attention maps
        
        Returns:
            dict with prediction + multiple attention heatmaps + suspicious regions
        """
        # Get basic prediction
        basic_result = self.predict(image)
        
        # Generate comprehensive explanation
        composite_viz, heatmaps = self.attention_explainer.create_composite_explanation(
            image, basic_result
        )
        
        # Extract suspicious regions
        suspicious_regions = self.attention_explainer.extract_manipulated_regions(image)
        
        return {
            **basic_result,
            'composite_visualization': Image.fromarray(composite_viz),
            'forgery_heatmaps': heatmaps,
            'suspicious_regions': suspicious_regions,
            'num_suspicious_regions': len(suspicious_regions)
        }
    
    def predict(self, image):
        """Basic prediction without explainability (for speed)"""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            auth_logits, match_score, liveness = self.model(img_tensor)
            probs = torch.softmax(auth_logits, dim=1)
        
        return {
            'prediction': 'REAL' if probs[0][1].item() > 0.5 else 'FAKE',
            'real_probability': probs[0][1].item(),
            'fake_probability': probs[0][0].item(),
            'match_score': match_score[0].item(),
            'liveness_score': liveness[0].item()
        }
    
    def predict_with_explainability(self, image, return_heatmap_only=False):
        """
        Prediction with Grad-CAM visualization
        Returns: dict with prediction + heatmap overlay image
        """
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Store original for overlay
        original_image = image.resize((224, 224))
        original_np = np.array(original_image).astype(np.float32) / 255.0
        
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Generate Grad-CAM for "FAKE" class (class 0)
        # This highlights regions model thinks are fake
        targets = [ClassifierOutputTarget(0)]
        grayscale_cam = self.grad_cam(input_tensor=img_tensor, targets=targets)
        
        # Overlay heatmap on image
        heatmap_overlay = show_cam_on_image(original_np, grayscale_cam[0], use_rgb=True)
        
        # Get prediction
        with torch.no_grad():
            auth_logits, match_score, liveness = self.model(img_tensor)
            probs = torch.softmax(auth_logits, dim=1)
        
        if return_heatmap_only:
            return heatmap_overlay
        
        return {
            'prediction': 'REAL' if probs[0][1].item() > 0.5 else 'FAKE',
            'real_probability': probs[0][1].item(),
            'fake_probability': probs[0][0].item(),
            'match_score': match_score[0].item(),
            'liveness_score': liveness[0].item(),
            'grad_cam_heatmap': heatmap_overlay,  # RGB numpy array
            'heatmap_pil': Image.fromarray(heatmap_overlay)  # PIL Image for Gradio
        }


# Backward compatibility
if __name__ == "__main__":
    print("Testing Grad-CAM explainability...")
    
    # Load model
    detector = DeepfakeDetector("best_forgery_multitask_model.pth", device='cpu')
    
    # Test with dummy image
    test_img = Image.new('RGB', (224, 224), color='red')
    result = detector.predict_with_explainability(test_img)
    
    print(f"Prediction: {result['prediction']}")
    print(f"Real prob: {result['real_probability']:.3f}")
    print(f"Heatmap shape: {result['grad_cam_heatmap'].shape}")
    print("✅ Grad-CAM working!")
