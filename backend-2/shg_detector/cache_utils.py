"""
Caching utilities for optimizing repeated calculations
"""
import functools
import numpy as np
import cv2
from typing import Tuple, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def cached_property(func):
    """Cache property results"""
    @functools.wraps(func)
    def wrapper(self):
        cache_attr = f'_{func.__name__}_cached'
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, func(self))
        return getattr(self, cache_attr)
    return property(wrapper)


def lru_cache_by_shape(maxsize=128):
    """Cache function results based on image shape (h, w)"""
    def decorator(func):
        cache = {}
        cache_hits = [0]
        cache_misses = [0]
        
        @functools.wraps(func)
        def wrapper(self, image, *args, **kwargs):
            # Use image shape as cache key
            if isinstance(image, np.ndarray):
                shape_key = image.shape[:2]  # (h, w)
            elif isinstance(image, tuple) and len(image) >= 2:
                shape_key = image[:2]  # (h, w)
            else:
                # Fallback: call function without caching
                return func(self, image, *args, **kwargs)
            
            # Check cache
            cache_key = (shape_key, args, tuple(sorted(kwargs.items())))
            if cache_key in cache:
                cache_hits[0] += 1
                return cache[cache_key]
            
            # Call function and cache result
            result = func(self, image, *args, **kwargs)
            
            # Manage cache size
            if len(cache) >= maxsize:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            
            cache[cache_key] = result
            cache_misses[0] += 1
            return result
        
        wrapper.cache_info = lambda: {
            'hits': cache_hits[0],
            'misses': cache_misses[0],
            'size': len(cache),
            'hit_rate': cache_hits[0] / (cache_hits[0] + cache_misses[0]) if (cache_hits[0] + cache_misses[0]) > 0 else 0
        }
        wrapper.cache_clear = lambda: cache.clear()
        
        return wrapper
    return decorator


class ImageDimensionsCache:
    """Cache for image dimension calculations"""
    def __init__(self):
        self._cache = {}
    
    def get_dimensions(self, image: np.ndarray) -> Tuple[int, int, float, float]:
        """
        Get cached image dimensions: (h, w, diagonal, area)
        """
        shape = image.shape[:2]
        if shape not in self._cache:
            h, w = shape
            diagonal = np.sqrt(h**2 + w**2)
            area = h * w
            self._cache[shape] = (h, w, diagonal, area)
        return self._cache[shape]
    
    def clear(self):
        """Clear cache"""
        self._cache.clear()


class BatchProcessor:
    """Batch processing utilities with GPU/CPU detection"""
    
    def __init__(self):
        self.has_gpu = self._detect_gpu()
        self.available_memory = self._get_available_memory()
        self.batch_size = self._calculate_batch_size()
    
    def _detect_gpu(self) -> bool:
        """Detect GPU availability"""
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                return True
        except:
            pass
        try:
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                return True
        except:
            pass
        return False
    
    def _get_available_memory(self) -> int:
        """Get available system memory in MB"""
        if HAS_PSUTIL:
            try:
                memory = psutil.virtual_memory()
                # Use 70% of available memory to leave room for system
                available_mb = int(memory.available / (1024 * 1024) * 0.7)
                return available_mb
            except:
                pass
        # Fallback: assume 2GB available
        return 2048
    
    def _calculate_batch_size(self) -> int:
        """Calculate optimal batch size based on system resources"""
        # Estimate memory per cell image (assuming ~100x100 pixels, uint8)
        # Cell image: ~10KB, enhanced (2x upscaled): ~40KB, total ~50KB per cell
        memory_per_cell_mb = 0.05
        
        if self.has_gpu:
            # GPU: Can handle larger batches, but limit to avoid OOM
            # Use up to 50% of available memory for batch processing
            max_batch_mb = self.available_memory * 0.5
            batch_size = max(8, min(64, int(max_batch_mb / memory_per_cell_mb)))
        else:
            # CPU: Smaller batches to avoid system slowdown
            # Use up to 30% of available memory
            max_batch_mb = self.available_memory * 0.3
            batch_size = max(4, min(32, int(max_batch_mb / memory_per_cell_mb)))
        
        return batch_size
    
    def get_tile_size(self, image_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate optimal tile size for batch processing
        Returns: (tile_h, tile_w)
        """
        h, w = image_shape[:2]
        
        if self.has_gpu:
            # GPU: Larger tiles
            tile_h = min(512, h // 4)
            tile_w = min(512, w // 4)
        else:
            # CPU: Smaller tiles to avoid memory pressure
            tile_h = min(256, h // 8)
            tile_w = min(256, w // 8)
        
        # Ensure minimum size
        tile_h = max(64, tile_h)
        tile_w = max(64, tile_w)
        
        return (tile_h, tile_w)
    
    def process_in_batches(self, items, process_func, batch_size=None):
        """
        Process items in batches
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            batch_size: Optional batch size (uses calculated if None)
        
        Yields:
            Results from process_func
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            results = [process_func(item) for item in batch]
            yield from results

