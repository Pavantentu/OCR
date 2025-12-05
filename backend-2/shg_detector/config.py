import numpy as np
import functools



# ============================================================
# Module: config
# ============================================================

class ConfigMixin:
    """Mixin class for config functionality"""

    @functools.lru_cache(maxsize=32)
    def get_size_config(self, image_h, image_w):
        """
        Centralized size-adaptive configuration
        Returns tolerances that scale with image size
        All tolerance values are computed based on image diagonal to handle
        both portrait and landscape orientations properly.
        Formula guide:
        - Small tolerances (2-5px base): Use // 800-1000 (very conservative)
        - Medium tolerances (5-10px base): Use // 400-600 (moderate)
        - Large tolerances (10-20px base): Use // 200-300 (generous)
        """
        img_diagonal = np.sqrt(image_h**2 + image_w**2)
        # Always recalculate category based on the ACTUAL image dimensions passed to this function
        # (which should be the cropped/deskewed image, not the original image)
        # This ensures adaptive detection uses the correct size category for the actual working image
        size_category = self._categorize_diagonal(img_diagonal)
        max_dim = max(image_h, image_w)
        if size_category == 'large':
            h_scales = [16, 20, 24, 28, 32, 40]
            v_scales = [16, 20, 24, 28, 32, 40]
            h_close_iters = 1
            v_close_iters = 1
        elif size_category == 'small':
            h_scales = [8, 10, 12, 14, 16, 20]
            v_scales = [8, 10, 12, 14, 16, 20]
            h_close_iters = 3
            v_close_iters = 3
        else:
            h_scales = [12, 14, 18, 22, 26, 32]
            v_scales = [12, 14, 18, 22, 26, 32]
            h_close_iters = 2
            v_close_iters = 2
        min_h_kernel = max(20, int(image_w // 200))
        min_v_kernel = max(20, int(image_h // 200))
        horizontal_kernel_lengths = [max(min_h_kernel, int(image_w // s)) for s in h_scales]
        vertical_kernel_lengths = [max(min_v_kernel, int(image_h // s)) for s in v_scales]
        block_tolerance = max(6, int(min(image_w, image_h) // 150))
        # Adaptive intersection tolerances
        if size_category == 'large':
            base_intersection_tol = max(12, int(max_dim // 180))
        elif size_category == 'small':
            base_intersection_tol = max(8, int(max_dim // 120))
        else:
            base_intersection_tol = max(10, int(max_dim // 150))
        intersection_scale = getattr(self, 'intersection_scale', 1.0) or 1.0
        intersection_tolerance = max(6, int(round(base_intersection_tol * intersection_scale)))
        pix_tolerance = max(4, int(img_diagonal // 400))      # For cell corner matching
        intersection_roi_tolerance = max(pix_tolerance, int(round(intersection_tolerance * 1.25)))
        config = {
            # Line detection and clustering
            'intersection_tolerance': intersection_tolerance,
            'intersection_roi_tolerance': intersection_roi_tolerance,
            'pix_tolerance': pix_tolerance,
            'min_gap': max(5, int(image_h // 200)),                 # Minimum gap between lines
            # Cell validation
            'min_cell_w': max(5, int(image_w // 150)),              # Minimum cell width
            'min_cell_h': max(5, int(image_h // 150)),              # Minimum cell height
            # Line thickness detection
            'search_radius': max(2, int(img_diagonal // 800)),      # For line validation sampling
            # Cell extraction
            'padding_left': max(8, int(image_w // 200)),            # Left padding for cell extraction
            'padding_top': max(8, int(image_h // 200)),             # Top padding for cell extraction
            'padding_right': max(8, int(image_w // 200)),           # Right padding for cell extraction
            'padding_bottom': max(8, int(image_h // 200)),          # Bottom padding for cell extraction
            # Connectivity checks
            'x_tolerance': max(8, int(image_w // 100)),             # For merging vertical line fragments
            # Boundary detection helpers
            'size_category': size_category,
            'size_diagonal': img_diagonal,
            'horizontal_kernel_lengths': horizontal_kernel_lengths,
            'vertical_kernel_lengths': vertical_kernel_lengths,
            'h_close_iterations': h_close_iters,
            'v_close_iterations': v_close_iters,
            'blocker_edge_tolerance': block_tolerance
        }
        if self.debug:
            print(f"\n[SIZE CONFIG] Image: {image_w}x{image_h} (diagonal: {img_diagonal:.0f})")
            print(f"  Category: {size_category}")
            print(f"  Intersection tolerance: {config['intersection_tolerance']}px")
            print(f"  Intersection ROI tolerance: {config['intersection_roi_tolerance']}px")
            print(f"  Pixel tolerance: {config['pix_tolerance']}px")
            print(f"  Min gap: {config['min_gap']}px")
            print(f"  Search radius: {config['search_radius']}px")
            print(f"  Min cell size: {config['min_cell_w']}x{config['min_cell_h']}px")
            print(f"  X tolerance: {config['x_tolerance']}px")
            print(f"  Padding: L{config['padding_left']} T{config['padding_top']} "
                f"R{config['padding_right']} B{config['padding_bottom']}px")
            print(f"  Horizontal kernels: {horizontal_kernel_lengths}")
            print(f"  Vertical kernels: {vertical_kernel_lengths}")
        return config

