"""
Frequency Domain Analysis for Deepfake Detection
Detects artifacts in FFT/DCT frequency space that CNNs miss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2


class FrequencyAnalyzer:
    """
    Analyzes frequency domain features for deepfake detection
    Combines DCT (JPEG-like) and FFT (2D Fourier) analysis
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        print(f"✅ FrequencyAnalyzer loaded on {device}")
    
    def compute_fft_spectrum(self, image):
        """
        Compute 2D FFT magnitude spectrum
        Deepfakes show abnormal high-frequency patterns
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Compute 2D FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Log scale for visualization
        magnitude_log = np.log1p(magnitude)
        
        return magnitude, magnitude_log
    
    def compute_dct_coefficients(self, image):
        """
        Compute DCT (Discrete Cosine Transform)
        JPEG compression artifacts reveal deepfakes
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply DCT
        gray_float = np.float32(gray)
        dct = cv2.dct(gray_float)
        
        return dct
    
    def extract_frequency_features(self, image):
        """
        Extract frequency-domain features for deepfake detection
        Returns: dict with multiple frequency statistics
        """
        fft_mag, fft_log = self.compute_fft_spectrum(image)
        dct = self.compute_dct_coefficients(image)
        
        # High-frequency energy (outer regions of FFT)
        h, w = fft_mag.shape
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 4
        
        # Create mask for high frequencies
        y, x = np.ogrid[:h, :w]
        mask_high = ((x - center_w)**2 + (y - center_h)**2) > radius**2
        
        high_freq_energy = np.sum(fft_mag[mask_high])
        total_energy = np.sum(fft_mag)
        high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
        
        # DCT statistics (high coefficients indicate tampering)
        dct_high_freq = np.abs(dct[dct.shape[0]//2:, dct.shape[1]//2:])
        dct_high_mean = np.mean(dct_high_freq)
        dct_high_std = np.std(dct_high_freq)
        
        # Frequency entropy (irregularity measure)
        fft_flat = fft_mag.flatten()
        fft_flat = fft_flat / (np.sum(fft_flat) + 1e-8)
        freq_entropy = -np.sum(fft_flat * np.log2(fft_flat + 1e-8))
        
        return {
            'high_freq_ratio': high_freq_ratio,
            'dct_high_mean': dct_high_mean,
            'dct_high_std': dct_high_std,
            'freq_entropy': freq_entropy,
            'fft_magnitude': fft_mag,
            'fft_log_spectrum': fft_log,
            'dct_coefficients': dct
        }
    
    def analyze_deepfake_frequency(self, image, threshold_high_freq=0.15):
        """
        Analyze image for frequency-domain deepfake indicators
        
        Real faces: Smooth frequency distribution
        Deepfakes: Abnormal high-frequency spikes
        """
        features = self.extract_frequency_features(image)
        
        # Simple rule-based detection (can be replaced with ML)
        suspicious_indicators = []
        anomaly_score = 0.0
        
        # Check 1: High frequency ratio (deepfakes have more high-freq noise)
        if features['high_freq_ratio'] > threshold_high_freq:
            suspicious_indicators.append("Abnormal high-frequency energy")
            anomaly_score += 0.3
        
        # Check 2: DCT high-frequency variance (GAN artifacts)
        if features['dct_high_std'] > 50:  # Threshold from research
            suspicious_indicators.append("DCT coefficient irregularities")
            anomaly_score += 0.25
        
        # Check 3: Frequency entropy (unnatural smoothness or noise)
        if features['freq_entropy'] < 8.0 or features['freq_entropy'] > 12.0:
            suspicious_indicators.append("Abnormal frequency entropy")
            anomaly_score += 0.2
        
        # Normalize anomaly score
        anomaly_score = min(anomaly_score, 1.0)
        
        is_suspicious = anomaly_score > 0.4
        
        return {
            'is_suspicious': is_suspicious,
            'anomaly_score': anomaly_score,
            'suspicious_indicators': suspicious_indicators,
            'high_freq_ratio': features['high_freq_ratio'],
            'dct_high_mean': features['dct_high_mean'],
            'dct_high_std': features['dct_high_std'],
            'freq_entropy': features['freq_entropy']
        }
    
    def visualize_frequency_spectrum(self, image):
        """
        Generate visual representation of frequency analysis
        Returns: FFT spectrum image (for UI display)
        """
        features = self.extract_frequency_features(image)
        fft_log = features['fft_log_spectrum']
        
        # Normalize for visualization
        fft_vis = ((fft_log - fft_log.min()) / (fft_log.max() - fft_log.min() + 1e-8) * 255).astype(np.uint8)
        
        # Apply colormap
        fft_colored = cv2.applyColorMap(fft_vis, cv2.COLORMAP_JET)
        fft_rgb = cv2.cvtColor(fft_colored, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(fft_rgb)


class FrequencyAttentionModule(nn.Module):
    """
    Neural network module for frequency-aware feature enhancement
    Can be integrated into existing EfficientNet model
    """
    
    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels
        
        # Learnable frequency filters
        self.freq_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.amplitude_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Process input through frequency domain
        x: (B, C, H, W) tensor
        """
        # FFT
        x_fft = torch.fft.rfft2(x, norm='ortho')
        
        # Separate amplitude and phase
        amplitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        # Learn to enhance/suppress frequency bands
        amplitude_enhanced = self.freq_conv(amplitude)
        gate = self.amplitude_gate(amplitude)
        amplitude_gated = amplitude_enhanced * gate
        
        # Reconstruct with modified amplitude
        x_fft_enhanced = amplitude_gated * torch.exp(1j * phase)
        
        # Inverse FFT
        x_enhanced = torch.fft.irfft2(x_fft_enhanced, s=x.shape[-2:], norm='ortho')
        
        # Residual connection
        return x + x_enhanced


# Test script
if __name__ == "__main__":
    print("Testing FrequencyAnalyzer...")
    
    analyzer = FrequencyAnalyzer(device='cpu')
    
    # Test with dummy image
    test_img = Image.new('RGB', (224, 224), color='blue')
    
    # Analyze
    result = analyzer.analyze_deepfake_frequency(test_img)
    
    print(f"Suspicious: {result['is_suspicious']}")
    print(f"Anomaly Score: {result['anomaly_score']:.3f}")
    print(f"High Freq Ratio: {result['high_freq_ratio']:.4f}")
    print(f"DCT High Std: {result['dct_high_std']:.2f}")
    print(f"Freq Entropy: {result['freq_entropy']:.2f}")
    print(f"Indicators: {result['suspicious_indicators']}")
    
    # Generate spectrum visualization
    spectrum_img = analyzer.visualize_frequency_spectrum(test_img)
    print(f"Spectrum visualization size: {spectrum_img.size}")
    
    print("✅ FrequencyAnalyzer working!")
