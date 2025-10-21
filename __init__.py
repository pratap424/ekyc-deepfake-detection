"""
Models package for eKYC verification system
"""
from .deepfake_model import DeepfakeDetector
from .liveness_model import LivenessDetector

__all__ = ['DeepfakeDetector', 'LivenessDetector']
