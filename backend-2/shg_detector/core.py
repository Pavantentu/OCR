import cv2
import numpy as np
import os
from pathlib import Path
import multiprocessing as mp
import threading
from shg_detector.cache_utils import ImageDimensionsCache, BatchProcessor



# ============================================================
# Module: core
# ============================================================

class CoreMixin:
    """Mixin class for core functionality"""

    def __init__(self, debug=False, return_images=False, intersection_scale=1.0):  # Changed debug default to False
        self.debug = debug
        self.return_images = return_images
        self.result_folder = Path("result")
        self.training_folder = Path("debug-training")
        self.counter_file = self.training_folder / "counter.txt"
        self.size_profile = None  # Tracks current image scale/category info
        env_scale = os.getenv("SHG_INTERSECTION_SCALE")
        if intersection_scale is None and env_scale:
            try:
                intersection_scale = float(env_scale)
            except ValueError:
                intersection_scale = 1.0
        if intersection_scale is None or intersection_scale <= 0:
            intersection_scale = 1.0
        self.intersection_scale = float(intersection_scale)
        # Only create folders if needed
        if self.debug:
            self.result_folder.mkdir(parents=True, exist_ok=True)
        if self.return_images or self.debug:
            self.training_folder.mkdir(parents=True, exist_ok=True)
        # Load counter
        self.current_counter = self.load_counter()
        
        # Thread lock for thread-safe counter increment
        self._counter_lock = threading.Lock()
        
        # Detect GPU availability and CPU cores
        self.has_gpu = self._detect_gpu()
        self.num_workers = self._get_optimal_workers()
        
        # Initialize caching and batch processing
        self._img_cache = ImageDimensionsCache()
        self._batch_processor = BatchProcessor()
        
        if self.debug:
            print(f"Initialized SHG Form Detector")
            print(f"Debug mode: {debug}")
            print(f"Result folder: {self.result_folder}")
            print(f"Training folder: {self.training_folder}")
            print(f"Current counter: {self.current_counter}")
            print(f"GPU available: {self.has_gpu}")
            print(f"Parallel workers: {self.num_workers}")
    
    def _detect_gpu(self):
        """Detect if GPU-accelerated OpenCV is available"""
        try:
            # Check if OpenCV was built with CUDA support
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                return True
        except:
            pass
        # Check for OpenCL (alternative GPU acceleration)
        try:
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                return True
        except:
            pass
        return False
    
    def _get_optimal_workers(self):
        """Get optimal number of parallel workers based on CPU cores"""
        cpu_count = mp.cpu_count()
        # Use 75% of available cores, but at least 2 and at most 8
        optimal = max(2, min(8, int(cpu_count * 0.75)))
        return optimal
    
    def get_image_dimensions(self, image):
        """
        Get cached image dimensions: (h, w, diagonal, area)
        Uses caching to avoid repeated calculations
        """
        return self._img_cache.get_dimensions(image)

    def load_counter(self) -> int:
        """Load the current counter from file"""
        # Ensure directory exists before loading
        self.training_folder.mkdir(parents=True, exist_ok=True)
        if self.counter_file.exists():
            try:
                with open(self.counter_file, 'r') as f:
                    return int(f.read().strip())
            except:
                return 0
        return 0

    def save_counter(self):
        """Write counter only once when training is finished."""
        self.training_folder.mkdir(parents=True, exist_ok=True)
        with open(self.counter_file, 'w') as f:
            f.write(str(self.current_counter))

    def increment_counter(self):
        """Increase counter in memory only. Thread-safe. Do NOT write file here."""
        with self._counter_lock:
            self.current_counter += 1
            return self.current_counter

    def save_debug_image(self, image, filename, log_msg=""):
        """Save debug images with logging"""
        if not self.debug:
            return
        filepath = self.result_folder / filename
        cv2.imwrite(str(filepath), image)
        print(f"  [DEBUG] Saved: {filename}")
        if log_msg:
            print(f"          {log_msg}")

    def save_training_cell(self, image, cell_info: str):
        """Save cell to training folder with incremental naming"""
        counter = self.increment_counter()
        filename = f"img{counter}.jpg"
        filepath = self.training_folder / filename
        cv2.imwrite(str(filepath), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"    → Saved training cell: {filename} | {cell_info}")
        return filepath

    @staticmethod
    def _categorize_diagonal(diagonal: float):
        """Return a human-friendly size category based on image diagonal length."""
        if diagonal >= 5000:
            return 'large'
        if diagonal <= 2000:
            return 'small'
        return 'medium'

    def analyze_image_scale(self, image):
        """
        Inspect the input image, categorize its size, and optionally rescale it so
        downstream morphology uses a predictable working resolution.
        """
        h, w = image.shape[:2]
        diag = float(np.sqrt(h ** 2 + w ** 2))
        category = self._categorize_diagonal(diag)
        # Target diagonals chosen from prior experimentation
        target_diag = {
            'small': 2400.0,   # Upscale tiny scans for more line detail
            'medium': diag,    # Leave as-is
            'large': 4200.0    # Slightly downscale HDR captures to keep kernels manageable
        }[category]
        scale = target_diag / diag if category != 'medium' else 1.0
        # Clamp scale so we never blow up/down beyond reason
        scale = float(np.clip(scale, 0.6, 1.8))
        if abs(scale - 1.0) < 1e-2:
            resized = image.copy()
        else:
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
            if self.debug:
                print(f"[SCALE] {category.upper()} image detected (diag={diag:.0f}px) → "
                      f"rescaled by {scale:.2f} to {new_w}x{new_h}")
        # Preserve original_shape from first call (before cropping) if it exists
        # This allows us to call analyze_image_scale again after cropping/deskewing
        # without losing the original image dimensions
        original_shape = (self.size_profile['original_shape'] if self.size_profile 
                         else (h, w))
        self.size_profile = {
            'category': category,
            'original_shape': original_shape,  # Preserve original, not current input
            'working_shape': resized.shape[:2],
            'scale_factor': scale,
            'diag': diag
        }
        return resized

