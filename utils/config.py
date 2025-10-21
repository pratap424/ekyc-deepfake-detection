"""
Configuration file for eKYC system
"""
import os

# Paths
CHECKPOINT_DIR = "checkpoints"
DEEPFAKE_MODEL = os.path.join(CHECKPOINT_DIR, "best_forgery_multitask_model.pth")
LIVENESS_MODEL = os.path.join(CHECKPOINT_DIR, "OULU_Protocol_2_model_0_0.onnx")
IDENTITY_MODEL = os.path.join(CHECKPOINT_DIR, "InceptionResnetV1_vggface2.onnx")

# Thresholds
DEEPFAKE_THRESHOLD = 0.5  # Real probability > 0.5 = REAL
LIVENESS_THRESHOLD = 0.5  # Score > 0.5 = LIVE

# Identity verification threshold
IDENTITY_THRESHOLD = 1.0  # L2 distance < 1.0 = match

# Device
DEVICE = "cuda" if os.path.exists("/proc/driver/nvidia/version") else "cpu"

# Gradio settings
GRADIO_PORT = 7860
STREAM_FPS = 2  # Frames per second (stream_every = 1/FPS)
MAX_STREAM_TIME = 300  # 5 minutes max



