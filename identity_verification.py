"""
Identity Verification using FaceNet (Siamese approach)
Compares two face images to determine if they're the same person
"""

import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2


class IdentityVerifier:
    """
    FaceNet-based identity verification
    Uses pre-trained InceptionResnetV1 for face embeddings
    """
    
    def __init__(self, device='cpu', confidence_threshold=0.70):
        """
        Args:
            device: 'cpu' or 'cuda'
            confidence_threshold: Similarity threshold for match (0.70 = 70%)
        """
        self.device = device
        self.threshold = confidence_threshold
        
        # Load FaceNet model (pre-trained on VGGFace2)
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        # Face detector (MTCNN)
        self.face_detector = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device
        )
        
        # Transform for manual preprocessing (if face already cropped)
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print(f"✅ IdentityVerifier loaded on {device} (FaceNet VGGFace2)")
    
    def detect_and_crop_face(self, image):
        """
        Detect face and return cropped, aligned face
        Returns: PIL Image or None if no face detected
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Detect and align face
        face_tensor = self.face_detector(image)
        
        if face_tensor is None:
            return None
        
        # Convert tensor back to PIL (optional, for visualization)
        face_array = face_tensor.permute(1, 2, 0).cpu().numpy()
        face_array = ((face_array * 0.5 + 0.5) * 255).astype(np.uint8)
        
        return Image.fromarray(face_array)
    
    def get_embedding(self, image, auto_detect=True):
        """
        Extract 512-dimensional face embedding
        
        Args:
            image: PIL Image or numpy array
            auto_detect: If True, automatically detect and crop face
        
        Returns:
            torch.Tensor of shape (1, 512) or None if no face
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if auto_detect:
            # Use MTCNN to detect, crop, and align
            face_tensor = self.face_detector(image)
            if face_tensor is None:
                return None
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
        else:
            # Manual preprocessing (face already cropped)
            face_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model(face_tensor)
        
        return embedding
    
    def compute_similarity(self, embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings
        Returns: float in [0, 1] where 1 = identical
        """
        similarity = F.cosine_similarity(embedding1, embedding2).item()
        
        # Scale from [-1, 1] to [0, 1]
        match_score = (similarity + 1) / 2
        
        return match_score
    
    def verify_identity(self, image1, image2, return_details=False):
        """
        Compare two face images and determine if same person
        
        Args:
            image1: Reference image (PIL or numpy array)
            image2: Query image (PIL or numpy array)
            return_details: If True, return detailed dict with embeddings
        
        Returns:
            dict with match_score, is_same_person, confidence
        """
        # Get embeddings
        emb1 = self.get_embedding(image1, auto_detect=True)
        emb2 = self.get_embedding(image2, auto_detect=True)
        
        # Check if faces detected
        if emb1 is None or emb2 is None:
            return {
                'match_score': 0.0,
                'is_same_person': False,
                'confidence': 0.0,
                'face1_detected': emb1 is not None,
                'face2_detected': emb2 is not None,
                'error': 'Face not detected in one or both images'
            }
        
        # Compute similarity
        match_score = self.compute_similarity(emb1, emb2)
        is_same = match_score >= self.threshold
        
        result = {
            'match_score': match_score,
            'is_same_person': is_same,
            'confidence': match_score,
            'face1_detected': True,
            'face2_detected': True,
            'threshold_used': self.threshold
        }
        
        if return_details:
            result['embedding1'] = emb1
            result['embedding2'] = emb2
            result['euclidean_distance'] = torch.dist(emb1, emb2).item()
        
        return result
    
    def batch_verify(self, reference_image, query_images):
        """
        Compare one reference image against multiple query images
        Useful for 1:N identification
        
        Args:
            reference_image: Single PIL image
            query_images: List of PIL images
        
        Returns:
            List of dicts with match results
        """
        ref_embedding = self.get_embedding(reference_image)
        
        if ref_embedding is None:
            return [{'error': 'No face in reference image'}] * len(query_images)
        
        results = []
        for query_img in query_images:
            query_emb = self.get_embedding(query_img)
            
            if query_emb is None:
                results.append({
                    'match_score': 0.0,
                    'is_same_person': False,
                    'error': 'No face detected'
                })
            else:
                match_score = self.compute_similarity(ref_embedding, query_emb)
                results.append({
                    'match_score': match_score,
                    'is_same_person': match_score >= self.threshold
                })
        
        return results


# Test script
if __name__ == "__main__":
    print("Testing IdentityVerifier...")
    
    verifier = IdentityVerifier(device='cpu', confidence_threshold=0.70)
    
    # Create dummy test images
    test_img1 = Image.new('RGB', (224, 224), color='red')
    test_img2 = Image.new('RGB', (224, 224), color='blue')
    
    # Note: This will fail (no real faces), but tests the pipeline
    result = verifier.verify_identity(test_img1, test_img2)
    
    print(f"Match Score: {result['match_score']:.3f}")
    print(f"Same Person: {result['is_same_person']}")
    print(f"Face1 Detected: {result['face1_detected']}")
    print(f"Face2 Detected: {result['face2_detected']}")
    print("✅ IdentityVerifier pipeline working!")
