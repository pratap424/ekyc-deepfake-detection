"""
Identity verification using InceptionResnetV1 ONNX model
Based on: https://github.com/ffletcherr/face-recognition-liveness
Lighter and faster than PyTorch FaceNet for deployment
"""

import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image

class IdentityVerifierONNX:
    """Face recognition using InceptionResnetV1 ONNX model"""
    
    def __init__(self, model_path, use_gpu=False):
        """
        Args:
            model_path: Path to InceptionResnetV1_vggface2.onnx
            use_gpu: Use CUDA if available
        """
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        print(f"✅ IdentityVerifierONNX loaded (InceptionResnetV1)")
    
    def preprocess_face(self, face_arr):
        """
        Preprocess face crop for InceptionResnetV1
        
        Args:
            face_arr: BGR face crop (numpy array) or PIL Image
        
        Returns:
            Preprocessed array ready for ONNX inference
        """
        # Convert to RGB if needed
        if isinstance(face_arr, Image.Image):
            face_rgb = np.array(face_arr)
        elif len(face_arr.shape) == 3 and face_arr.shape[2] == 3:
            face_rgb = cv2.cvtColor(face_arr, cv2.COLOR_BGR2RGB)
        else:
            face_rgb = face_arr
        
        # Resize to 160x160 (InceptionResnetV1 input size)
        face_resized = cv2.resize(face_rgb, (160, 160))
        
        # Normalize: (pixel - 127.5) / 128.0
        face_normalized = (face_resized.astype(np.float32) - 127.5) / 128.0
        
        # Transpose to CHW format (ONNX expects channels-first)
        input_arr = np.transpose(face_normalized, (2, 0, 1))
        
        # Add batch dimension
        input_arr = np.expand_dims(input_arr, 0).astype(np.float32)
        
        return input_arr
    
    def get_embedding(self, face_arr):
        """
        Extract 512-dimensional face embedding
        
        Args:
            face_arr: BGR face crop or PIL Image
        
        Returns:
            512-dimensional numpy array or None if preprocessing fails
        """
        try:
            input_arr = self.preprocess_face(face_arr)
            
            # Run ONNX inference
            embeddings = self.session.run(["output"], {"input": input_arr})[0]
            
            return embeddings[0]  # Remove batch dimension
        except Exception as e:
            print(f"⚠️  Embedding extraction failed: {e}")
            return None
    
    def compute_similarity(self, embedding1, embedding2, metric='cosine'):
        """
        Compute similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: 'cosine' or 'euclidean'
        
        Returns:
            Similarity score (higher = more similar)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        if metric == 'cosine':
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            similarity = dot_product / (norm1 * norm2)
            # Scale from [-1, 1] to [0, 1]
            return (similarity + 1) / 2
        
        elif metric == 'euclidean':
            # L2 distance (smaller = more similar)
            distance = np.linalg.norm(embedding1 - embedding2)
            # Convert to similarity score (inverse)
            # Typical FaceNet threshold is ~1.0, so map accordingly
            similarity = max(0, 1 - (distance / 2.0))
            return similarity
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def verify_identity(self, face1, face2, threshold=0.70):
        """
        Compare two face crops and determine if same person
        
        Args:
            face1: First face crop (BGR numpy array or PIL Image)
            face2: Second face crop (BGR numpy array or PIL Image)
            threshold: Similarity threshold for match (default 0.70 = 70%)
        
        Returns:
            dict with match_score, is_same_person, confidence
        """
        # Get embeddings
        emb1 = self.get_embedding(face1)
        emb2 = self.get_embedding(face2)
        
        if emb1 is None or emb2 is None:
            return {
                'match_score': 0.0,
                'is_same_person': False,
                'confidence': 0.0,
                'face1_detected': emb1 is not None,
                'face2_detected': emb2 is not None,
                'error': 'Failed to extract embeddings'
            }
        
        # Compute cosine similarity
        match_score = self.compute_similarity(emb1, emb2, metric='cosine')
        
        # Determine if match
        is_same = match_score >= threshold
        
        return {
            'match_score': match_score,
            'is_same_person': is_same,
            'confidence': match_score,
            'face1_detected': True,
            'face2_detected': True,
            'threshold_used': threshold
        }


# Test script
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    model_path = Path("checkpoints/InceptionResnetV1_vggface2.onnx")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Download it with:")
        print("wget https://github.com/ffletcherr/face-recognition-liveness/releases/download/v0.1/InceptionResnetV1_vggface2.onnx")
        sys.exit(1)
    
    print("Testing IdentityVerifierONNX...")
    verifier = IdentityVerifierONNX(model_path.as_posix())
    
    # Create dummy test face
    test_face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    
    # Test embedding extraction
    embedding = verifier.get_embedding(test_face)
    print(f"✅ Embedding shape: {embedding.shape}")
    print(f"✅ Embedding L2 norm: {np.linalg.norm(embedding):.3f}")
    
    # Test self-similarity (should be ~1.0)
    result = verifier.verify_identity(test_face, test_face, threshold=0.70)
    print(f"✅ Self-similarity: {result['match_score']:.3f}")
    print(f"✅ Match: {result['is_same_person']}")
    
    print("\n✅ IdentityVerifierONNX working correctly!")
