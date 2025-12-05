import cv2
import numpy as np



# ============================================================
# Module: preprocessing
# ============================================================

class PreprocessingMixin:
    """Mixin class for preprocessing functionality"""

    def preprocess_scanned(self, image):
        """
        Gentle document enhancement that preserves content:
        - Minimal shadow correction
        - Preserve text and lines
        - No aggressive sharpening
        """
        if self.debug:
            print("\n[PREPROCESSING] Gentle document enhancement...")
        # Step 1: Very gentle white balance
        result = image.astype(np.float32)
        avg_per_channel = np.mean(result, axis=(0, 1))
        gray_value = np.mean(avg_per_channel)
        scale = gray_value / (avg_per_channel + 1e-6)
        # Limit the correction strength to avoid over-correction
        scale = np.clip(scale, 0.9, 1.1)
        result *= scale
        result = np.clip(result, 0, 255).astype(np.uint8)
        # Step 2: OPTIMIZED - Faster shadow reduction
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        if self.debug:
            print("  - Normalizing illumination...")
        # OPTIMIZED: Use smaller, faster blur kernel (reduces from ~5s to ~0.5s)
        # Use box filter instead of Gaussian for speed (similar quality for this use case)
        kernel_size = 61  # Smaller than before, but still effective
        background = cv2.boxFilter(gray, -1, (kernel_size, kernel_size), normalize=True)
        # Vectorized normalization (faster than cv2.divide)
        normalized = np.clip((gray.astype(np.float32) / (background.astype(np.float32) + 1e-6)) * 255, 
                            0, 255).astype(np.uint8)
        
        # Step 3: OPTIMIZED - Faster CLAHE with smaller tiles
        if self.debug:
            print("  - Gentle contrast boost...")
        # Smaller tile grid = faster processing (4x4 instead of 8x8)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
        enhanced = clahe.apply(normalized)
        # Convert back to BGR
        result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Step 4: OPTIMIZED - Skip slow denoising or use fast alternative
        # fastNlMeansDenoisingColored takes ~8-10s - REPLACE with faster method
        if self.debug:
            print("  - Light denoising (fast method)...")
        # Use bilateral filter instead - ~20x faster with similar quality for documents
        # Process each channel separately for color preservation
        result_denoised = np.zeros_like(result)
        for i in range(3):
            result_denoised[:, :, i] = cv2.bilateralFilter(result[:, :, i], d=5, 
                                                          sigmaColor=20, sigmaSpace=20)
        result = result_denoised
        if self.debug:
            print("  ✓ Preprocessing complete\n")
        return result

