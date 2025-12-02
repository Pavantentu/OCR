import cv2
import numpy as np
import os
from pathlib import Path
import json
import argparse
from collections import defaultdict

class SHGFormDetector:
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
        
        if self.debug:
            print(f"Initialized SHG Form Detector")
            print(f"Debug mode: {debug}")
            print(f"Result folder: {self.result_folder}")
            print(f"Training folder: {self.training_folder}")
            print(f"Current counter: {self.current_counter}")

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
        """Increase counter in memory only. Do NOT write file here."""
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

    @staticmethod
    def _cluster_values(values, tolerance):
        """Cluster sorted scalar values so nearby entries collapse into one."""
        if not values:
            return []

        values = sorted(values)
        clusters = []
        current_cluster = [values[0]]

        for value in values[1:]:
            if abs(value - current_cluster[-1]) <= tolerance:
                current_cluster.append(value)
            else:
                clusters.append(int(np.mean(current_cluster)))
                current_cluster = [value]

        clusters.append(int(np.mean(current_cluster)))
        return clusters

    @staticmethod
    def _assign_to_cluster(value, clusters, tolerance):
        """Return the representative cluster value for a scalar."""
        for cluster_value in clusters:
            if abs(value - cluster_value) <= tolerance:
                return cluster_value
        return None

    @staticmethod
    def _edge_strength(mask, x1, y1, x2, y2, thickness=3):
        """Measure how strongly a line segment exists within the combined mask."""
        if mask is None:
            return 1.0

        h, w = mask.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        if y1 == y2:
            y_start = max(0, y1 - thickness)
            y_end = min(h, y1 + thickness + 1)
            x_start = min(x1, x2)
            x_end = max(x1, x2) + 1
            roi = mask[y_start:y_end, x_start:x_end]
        elif x1 == x2:
            x_start = max(0, x1 - thickness)
            x_end = min(w, x1 + thickness + 1)
            y_start = min(y1, y2)
            y_end = max(y1, y2) + 1
            roi = mask[y_start:y_end, x_start:x_end]
        else:
            return 0

        if roi.size == 0:
            return 0

        return cv2.countNonZero(roi) / roi.size

    def _line_exists_between(self, p1, p2, thickness=3, min_coverage=0.50, edge_type='unknown'):
        """Validate that a line between two points exists in the combined mask."""
        if self.line_validation_mask is None:
            return True, 1.0

        mask = self.line_validation_mask
        h, w = mask.shape[:2]
        
        # Get dynamic search_radius based on image size
        size_config = self.get_size_config(h, w)
        search_radius = size_config['search_radius']
        # For validation we keep the search radius fairly tight so we don't
        # accidentally "see" neighbouring, dilated structures as this line.
        # The horizontal masks were already dilated upstream to heal small
        # gaps; a large radius here would convert that dilation into phantom
        # line detections.
        search_radius = max(1, min(search_radius, 2))

        x1, y1 = map(int, p1)
        x2, y2 = map(int, p2)

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        # Calculate line length for adaptive sampling
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if line_length < 1:
            return False, 0.0
        
        # Sample points along the line (adaptive number based on length)
        num_samples = max(10, int(line_length / 2))  # Sample every ~2 pixels
        sample_points = []
        
        for i in range(num_samples + 1):
            t = i / num_samples
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            sample_points.append((x, y))
        
        # Check coverage using a small window around each sample point.
        # Instead of treating *any* white pixel as a hit (which becomes too
        # permissive once lines are dilated), we require a minimum local
        # density of line pixels inside the window.
        valid_samples = 0

        # Local density threshold for a sample to count as "on the line"
        if edge_type in ('top', 'bottom'):
            local_min_density = 0.20  # slightly more lenient for horizontals
        elif edge_type in ('left', 'right'):
            local_min_density = 0.25
        else:
            local_min_density = 0.25
        
        for x, y in sample_points:
            # Check small region around this point
            y_min = max(0, y - search_radius)
            y_max = min(h, y + search_radius + 1)
            x_min = max(0, x - search_radius)
            x_max = min(w, x + search_radius + 1)
            
            region = mask[y_min:y_max, x_min:x_max]
            
            if region.size == 0:
                continue

            white = cv2.countNonZero(region)
            density = white / float(region.size)

            # Only count this sample as valid if we have a reasonable local
            # density of line pixels, which prevents isolated specks from
            # creating phantom lines.
            if density >= local_min_density:
                valid_samples += 1
        
        coverage = valid_samples / len(sample_points)
        
        # Adaptive threshold based on edge type and line characteristics.
        # Use the SAME threshold for all four primary edges so that left/right
        # behave identically to top/bottom (0.40), matching the original
        # behaviour before the recent tuning.
        if edge_type in ('top', 'bottom', 'left', 'right'):
            adaptive_min = 0.40
        else:
            adaptive_min = min_coverage
        
        # Debug output
        if self.debug and coverage < 0.60:
            if edge_type != 'unknown':
                print(f"    [COVERAGE DEBUG] edge={edge_type}: ({x1},{y1})-({x2},{y2}) "
                      f"= {valid_samples}/{len(sample_points)} = {coverage:.2f} "
                      f"{'✓' if coverage >= adaptive_min else '✗'}")
        
        if coverage < adaptive_min:
            profile_category = self.size_profile['category'] if self.size_profile else 'medium'
            # For small images, fall back to pixel-accurate overlap to avoid missing faint lines
            if profile_category == 'small':
                overlap_cov = self._line_overlap_coverage(mask, (x1, y1), (x2, y2), thickness)
                coverage = max(coverage, overlap_cov)

        # If confidence is still low, re-evaluate coverage using the
        # *less dilated* combined mask (horizontal + vertical) to avoid
        # artifacts introduced by the extra validation dilation step.
        #
        # This is only used for genuinely borderline edges (coverage < 0.5)
        # so strong phantom lines (with already-high coverage) are unaffected.
        if coverage < 0.50 and getattr(self, "combined_mask", None) is not None:
            base_mask = self.combined_mask
            bh, bw = base_mask.shape[:2]

            base_radius = 1
            base_valid = 0

            for x, y in sample_points:
                y_min = max(0, y - base_radius)
                y_max = min(bh, y + base_radius + 1)
                x_min = max(0, x - base_radius)
                x_max = min(bw, x + base_radius + 1)

                region = base_mask[y_min:y_max, x_min:x_max]
                if region.size == 0:
                    continue

                white = cv2.countNonZero(region)
                density = white / float(region.size)

                if density >= local_min_density:
                    base_valid += 1

            alt_coverage = base_valid / len(sample_points)

            # Use the better of the two coverage estimates
            if alt_coverage > coverage:
                coverage = alt_coverage
        
        return coverage >= adaptive_min, coverage

    def _line_exists_between_unclean(self, p1, p2, edge_type='unknown'):
        """Check line coverage against the original combined_mask (unclean/filtered mask)."""
        if not hasattr(self, 'combined_mask') or self.combined_mask is None:
            return False, 0.0
        
        mask = self.combined_mask
        h, w = mask.shape[:2]
        
        x1, y1 = map(int, p1)
        x2, y2 = map(int, p2)
        
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_length < 1:
            return False, 0.0
        
        num_samples = max(10, int(line_length / 2))
        sample_points = []
        for i in range(num_samples + 1):
            t = i / num_samples
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            sample_points.append((x, y))
        
        search_radius = 1
        local_min_density = 0.25
        valid_samples = 0
        
        for x, y in sample_points:
            y_min = max(0, y - search_radius)
            y_max = min(h, y + search_radius + 1)
            x_min = max(0, x - search_radius)
            x_max = min(w, x + search_radius + 1)
            
            region = mask[y_min:y_max, x_min:x_max]
            if region.size == 0:
                continue
            
            white = cv2.countNonZero(region)
            density = white / float(region.size)
            
            if density >= local_min_density:
                valid_samples += 1
        
        coverage = valid_samples / len(sample_points) if sample_points else 0.0
        return coverage >= 0.25, coverage

    @staticmethod
    def _line_overlap_coverage(mask, p1, p2, thickness=3):
        """Compute exact coverage of a drawn line inside the mask."""
        line_canvas = np.zeros_like(mask, dtype=np.uint8)
        cv2.line(line_canvas, tuple(map(int, p1)), tuple(map(int, p2)), 255, thickness=thickness)
        overlap = cv2.bitwise_and(line_canvas, mask)
        line_pixels = cv2.countNonZero(line_canvas)
        if line_pixels == 0:
            return 0.0
        return cv2.countNonZero(overlap) / line_pixels
    
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
        
        # Step 2: Gentle shadow reduction using illumination normalization
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # Use bilateral filter to estimate background (preserves edges better)
        if self.debug:
            print("  - Normalizing illumination...")
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=30, sigmaY=30)
        
        # Normalize with safety limits
        normalized = np.zeros_like(gray, dtype=np.float32)
        cv2.divide(gray.astype(np.float32), background.astype(np.float32), 
                  normalized, scale=255)
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        
        # Step 3: Very mild CLAHE (just enough to help but not distort)
        if self.debug:
            print("  - Gentle contrast boost...")
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized)
        
        # Convert back to BGR
        result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Step 4: Light denoising only (preserve details)
        if self.debug:
            print("  - Light denoising...")
        result = cv2.fastNlMeansDenoisingColored(result, None, h=4, hColor=4, 
                                                templateWindowSize=7, searchWindowSize=21)
        
        if self.debug:
            print("  ✓ Preprocessing complete\n")
        
        return result

    def detect_table_boundary(self, image, img_name):
        """
        Step 1: Robust table boundary detection with gentle binarization
        """
        print("\n" + "="*70)
        print("STEP 1: DETECTING TABLE BOUNDARY")
        print("="*70)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.debug:
            self.save_debug_image(gray, f"{img_name}_01_grayscale.jpg", "Converted to grayscale")
        
        # Very light denoising to preserve structure
        denoised = cv2.fastNlMeansDenoising(gray, None, h=5, templateWindowSize=7, searchWindowSize=21)
        if self.debug:
            self.save_debug_image(denoised, f"{img_name}_02_denoised.jpg", "Applied denoising")
        
        h, w = gray.shape
        size_config = self.get_size_config(h, w)
        
        # Multi-method binarization approach
        print("  Creating binary masks using multiple methods...")
        
        # Method 1: Otsu (good for uniform lighting)
        _, binary_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Method 2: Gentle adaptive (good for shadows)
        # Use MEAN instead of GAUSSIAN for more stable results
        binary_adaptive = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 
            blockSize=99,  # Larger block = more stable
            C=20           # Higher C = less sensitive
        )
        
        # Method 3: Triangle threshold (good for bimodal distributions)
        _, binary_triangle = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
        
        # Combine methods: Use intersection for more conservative detection
        # This means a pixel must be detected by at least 2 methods
        binary_combined = cv2.bitwise_and(binary_otsu, binary_adaptive)
        binary = cv2.bitwise_or(binary_otsu, binary_otsu)
        
        if self.debug:
            self.save_debug_image(binary_otsu, f"{img_name}_03a_binary_otsu.jpg", "Otsu threshold")
            self.save_debug_image(binary_adaptive, f"{img_name}_03b_binary_adaptive.jpg", "Adaptive threshold")
            self.save_debug_image(binary, f"{img_name}_03_binary_final.jpg", "Combined binary (conservative)")
        
        # Detect lines with multiple kernel sizes for robustness
        print("  Detecting table lines with multi-scale approach...")
        
        # === HORIZONTAL LINES ===
        h_masks = []
        for kernel_size in size_config['horizontal_kernel_lengths']:
            if self.debug:
                print(f"    H-kernel size: {kernel_size}px")
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
            h_temp = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, h_kernel, iterations=1)
            h_masks.append(h_temp)
        
        # Combine all horizontal detections (union)
        horizontal = h_masks[0].copy()
        for mask in h_masks[1:]:
            horizontal = cv2.bitwise_or(horizontal, mask)
        
        # Close small gaps
        h_kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 2))
        horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, h_kernel_close, iterations=size_config['h_close_iterations'])
        
        # === VERTICAL LINES ===
        v_masks = []
        for kernel_size in size_config['vertical_kernel_lengths']:
            if self.debug:
                print(f"    V-kernel size: {kernel_size}px")
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
            v_temp = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, v_kernel, iterations=1)
            v_masks.append(v_temp)
        
        # Combine all vertical detections (union)
        vertical = v_masks[0].copy()
        for mask in v_masks[1:]:
            vertical = cv2.bitwise_or(vertical, mask)
        
        # Close small gaps
        v_kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 20))
        vertical = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, v_kernel_close, iterations=size_config['v_close_iterations'])
        
        def _rot_kernel(length, angle, thickness=2):
            size = length
            k = np.zeros((size, size), dtype=np.uint8)
            x2 = int(size//2 + np.cos(np.deg2rad(angle)) * length/2)
            y2 = int(size//2 + np.sin(np.deg2rad(angle)) * length/2)
            cv2.line(k, (size//2, size//2), (x2, y2), 1, thickness)
            return k

        angles = [-7, -5, -3, 3, 5, 7]   # small tilt angles for sensitive angled-line detection
        angled_masks = []

        for ang in angles:
            rk = _rot_kernel(45, ang)
            temp = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, rk)
            angled_masks.append(temp)

        angled_lines = angled_masks[0].copy()
        for m in angled_masks[1:]:
            angled_lines = cv2.bitwise_or(angled_lines, m)

        angled_lines = cv2.medianBlur(angled_lines, 3)

        # -- 2. Hough transform for clean angled lines --
        edges = cv2.Canny(binary_otsu, 50, 150)
        hough_mask = np.zeros_like(binary_otsu)

        lines = cv2.HoughLinesP(
            edges,
            1, np.pi/180, 60,
            minLineLength=40,
            maxLineGap=10
        )

        if lines is not None:
            for (x1, y1, x2, y2) in lines[:, 0]:
                # only include if angled (not near-vertical or near-horizontal)
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if 4 < angle < 86:
                    cv2.line(hough_mask, (x1, y1), (x2, y2), 255, 2)

        # -- 3. Merge with existing horizontal + vertical --
        if self.debug:
            self.save_debug_image(horizontal, f"{img_name}_04a_horizontal_lines.jpg", "Horizontal lines (multi-scale)")
            self.save_debug_image(vertical, f"{img_name}_04b_vertical_lines.jpg", "Vertical lines (multi-scale)")
            self.save_debug_image(angled_lines, f"{img_name}_04c_angled_lines_rot.jpg", "Angled lines (rot kernels)")
            self.save_debug_image(hough_mask, f"{img_name}_04d_angled_lines_hough.jpg", "Angled lines (Hough)")

        # Combine detected structures once so angled/hough lines are preserved
        table_structure = cv2.bitwise_or(horizontal, vertical)
        table_structure = cv2.bitwise_or(table_structure, angled_lines)
        table_structure = cv2.bitwise_or(table_structure, hough_mask)
        if self.debug:
            self.save_debug_image(table_structure, f"{img_name}_04_table_structure.jpg", "Combined table structure")
        
        # ============================================================
        # USE CLEANED EDGE DETECTION FOR CONTOUR DETECTION
        # ============================================================
        print("  Creating cleaned edge detection for contour detection...")
        
        # Step 1: Create cleaned Canny edges from denoised image
        # Use adaptive thresholds based on image size
        canny_low = 50
        canny_high = 150
        
        # Apply Gaussian blur to reduce noise before Canny
        blurred_for_edges = cv2.GaussianBlur(denoised, (5, 5), 0)
        clean_edges = cv2.Canny(blurred_for_edges, canny_low, canny_high)
        
        # Step 2: Combine with table structure to get table edges
        # Dilate table structure slightly to connect with edges
        table_dilated = cv2.dilate(table_structure, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        
        # Combine clean edges with table structure
        edge_combined = cv2.bitwise_or(clean_edges, table_dilated)
        
        # Step 3: Clean up noise - remove small isolated components
        # Remove small noise components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edge_combined, connectivity=8)
        min_component_area = (h * w) * 0.001  # Remove components smaller than 0.1% of image
        cleaned_edges = np.zeros_like(edge_combined)
        
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_component_area:
                cleaned_edges[labels == label] = 255
        
        # Step 4: Close gaps in edges to form continuous boundaries
        kernel_close_edges = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_edges = cv2.morphologyEx(cleaned_edges, cv2.MORPH_CLOSE, kernel_close_edges, iterations=2)
        
        # Step 5: Dilate slightly to ensure boundaries are connected
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        final_contour_mask = cv2.dilate(closed_edges, kernel_dilate, iterations=1)
        
        if self.debug:
            self.save_debug_image(clean_edges, f"{img_name}_05a_clean_edges.jpg", "Cleaned Canny edges")
            self.save_debug_image(cleaned_edges, f"{img_name}_05b_noise_removed.jpg", "Noise removed from edges")
            self.save_debug_image(closed_edges, f"{img_name}_05c_edges_closed.jpg", "Edges after closing gaps")
            self.save_debug_image(final_contour_mask, f"{img_name}_05_contour_mask.jpg", "Final contour detection mask")
        
        # Find contours using cleaned edge detection
        contours, _ = cv2.findContours(final_contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"  Found {len(contours)} contours")
        
        # Filter for table candidates
        img_area = w * h
        min_area = img_area * 0.15  # Very lenient minimum
        max_area = img_area * 0.95
        
        debug_contours = image.copy()
        valid_contours = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area < min_area or area > max_area:
                continue
                
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            x, y, w_box, h_box = cv2.boundingRect(contour)
            aspect_ratio = w_box / h_box if h_box > 0 else 0
            
            rect_area = w_box * h_box
            extent = area / rect_area if rect_area > 0 else 0
            
            print(f"  Contour {i}: area={area/img_area*100:.1f}%, aspect={aspect_ratio:.2f}, extent={extent:.2f}, points={len(approx)}")
            
            # Very lenient criteria - just needs to be roughly rectangular
            if (0.5 <= aspect_ratio <= 4.0 and  # Wide range
                extent >= 0.60):                 # Low extent requirement
                
                valid_contours.append((contour, area, approx, (x, y, w_box, h_box), extent))
                cv2.drawContours(debug_contours, [approx], -1, (0, 255, 0), 3)
        
        if self.debug:
            self.save_debug_image(debug_contours, f"{img_name}_06_valid_contours.jpg", 
                            f"Found {len(valid_contours)} valid candidates")
        
        if not valid_contours:
            print("  ✗ No contours found, using projection fallback...")

            # Fallback: line density projection
            h_projection = np.sum(horizontal, axis=1) / 255.0
            v_projection = np.sum(vertical, axis=0) / 255.0

            # ---- Pure OpenCV 1D Smoothing (replacement for gaussian_filter1d) ----
            def smooth_projection(proj, sigma=5):
                ksize = int(6 * sigma + 1)
                if ksize % 2 == 0:
                    ksize += 1
                smoothed = cv2.GaussianBlur(
                    proj.reshape(-1, 1), 
                    (1, ksize), 
                    sigma
                ).reshape(-1)
                return smoothed

            h_smooth = smooth_projection(h_projection, sigma=5)
            v_smooth = smooth_projection(v_projection, sigma=5)
            # ----------------------------------------------------------------------

            # Dynamic thresholding based on signal strength
            h_thresh = np.percentile(h_smooth[h_smooth > 0], 25) if np.any(h_smooth > 0) else 0
            v_thresh = np.percentile(v_smooth[v_smooth > 0], 25) if np.any(v_smooth > 0) else 0

            h_regions = np.where(h_smooth > h_thresh)[0]
            v_regions = np.where(v_smooth > v_thresh)[0]

            if len(h_regions) > 0 and len(v_regions) > 0:
                y_min, y_max = h_regions[0], h_regions[-1]
                x_min, x_max = v_regions[0], v_regions[-1]

                # Add padding so edges are not cropped
                pad = 5
                bbox = (
                    max(0, x_min - pad),
                    max(0, y_min - pad),
                    min(w, (x_max - x_min) + 2 * pad),
                    min(h, (y_max - y_min) + 2 * pad)
                )

                print(f"  ✓ Fallback successful: bbox={bbox}")

                debug_final = image.copy()
                x, y, w_box, h_box = bbox
                cv2.rectangle(debug_final, (x, y), (x + w_box, y + h_box), (0, 255, 0), 3)
                cv2.putText(
                    debug_final, "TABLE (PROJECTION)", (x + 10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
                )
                if self.debug:
                    self.save_debug_image(
                        debug_final, f"{img_name}_07_table_boundary.jpg",
                        "Final boundary (fallback)"
                    )

                return bbox
            
            print("  ✗ Fallback also failed")
            return None
        
        # Select best contour by area
        valid_contours.sort(key=lambda x: x[1], reverse=True)
            
        table_contour, area, approx, bbox, extent = valid_contours[0]
        x, y, w_box, h_box = bbox

        # Ensure exactly 4 points for the selected contour
        if len(approx) != 4:
            # Try to get 4 points using convex hull
            hull = cv2.convexHull(table_contour)
            if len(hull) >= 4:
                peri_hull = cv2.arcLength(hull, True)
                for hull_eps in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
                    hull_approx = cv2.approxPolyDP(hull, hull_eps * peri_hull, True)
                    if len(hull_approx) == 4:
                        approx = hull_approx
                        break
            
            # If still not 4, use bounding box corners
            if len(approx) != 4:
                approx = np.array([
                    [[x, y]],
                    [[x + w_box, y]],
                    [[x + w_box, y + h_box]],
                    [[x, y + h_box]]
                ], dtype=np.int32)

        # Add a small padding so table edges are not tight to the image border
        pad = 5
        x_p = max(0, x - pad)
        y_p = max(0, y - pad)
        w_p = min(w, w_box + 2 * pad)
        h_p = min(h, h_box + 2 * pad)

        print(f"  ✓ Selected table: bbox=({x_p}, {y_p}, {w_p}, {h_p})")
        print(f"    Coverage: {area/img_area*100:.1f}% of image, extent: {extent:.2f}, points: {len(approx)}")
        
        # Draw final selection with exactly 4 lines
        debug_final = image.copy()
        if len(approx) == 4:
            # Draw as 4 lines connecting the 4 points
            pts = approx.reshape(4, 2)
            for i in range(4):
                pt1 = tuple(pts[i].astype(int))
                pt2 = tuple(pts[(i + 1) % 4].astype(int))
                cv2.line(debug_final, pt1, pt2, (0, 255, 0), 3)
            # Also draw the points
            for i, pt in enumerate(pts):
                cv2.circle(debug_final, tuple(pt.astype(int)), 8, (0, 0, 255), -1)
                cv2.putText(debug_final, str(i), tuple(pt.astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            # Fallback: draw as contour
            cv2.drawContours(debug_final, [approx], -1, (0, 255, 0), 3)
        
        cv2.rectangle(debug_final, (x_p, y_p), (x_p + w_p, y_p + h_p), (255, 0, 0), 2)
        cv2.putText(debug_final, "TABLE BOUNDARY", (x_p + 10, y_p + 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        if self.debug:
            self.save_debug_image(debug_final, f"{img_name}_07_table_boundary.jpg", "Final table boundary (4 points)")
        
        return (x_p, y_p, w_p, h_p)
    
    def crop_and_deskew_table(self, image, bbox, img_name):
        """
        Step 2: Crop to table boundary with padding applied BEFORE cropping, then deskew.
        Preserves your original perspective-correction and debug saves (uses _08a_perspective_corrected).
        """
        print("\n" + "="*70)
        print("STEP 2: CROPPING AND DESKEWING")
        print("="*70)

        x, y, w, h = bbox

        # ----------------------------
        # Apply padding to the full image BEFORE cropping
        # ----------------------------
        # Calculate padding as percentage of bbox dimensions (1% of width/height)
        padding_percent = 0.01  # 1% padding
        outer_padding_w = max(20, int(w * padding_percent))
        outer_padding_h = max(20, int(h * padding_percent))
        outer_padding = max(outer_padding_w, outer_padding_h)  # Use the larger value
        print(f"  Calculated outer padding: {outer_padding}px (1% of bbox: {w}x{h})")
        
        padded_full = cv2.copyMakeBorder(
            image,
            outer_padding, outer_padding, outer_padding, outer_padding,
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )

        # shift bbox into padded image coordinates
        x += outer_padding
        y += outer_padding

        # Add percentage-based safety padding to avoid cropping table content (0.5% of dimensions)
        safety_pad_percent = 0.005  # 0.5% padding
        safety_pad_w = max(15, int(w * safety_pad_percent))
        safety_pad_h = max(15, int(h * safety_pad_percent))
        safety_pad = max(safety_pad_w, safety_pad_h)  # Use the larger value
        print(f"  Calculated safety padding: {safety_pad}px (0.5% of bbox: {w}x{h})")
        
        x1 = max(0, x - safety_pad)
        y1 = max(0, y - safety_pad)
        x2 = min(padded_full.shape[1], x + w + safety_pad)
        y2 = min(padded_full.shape[0], y + h + safety_pad)

        cropped = padded_full[y1:y2, x1:x2].copy()
        print(f"  Cropped from padded image: ({x1}, {y1}) -> ({x2}, {y2})")
        if self.debug:
            self.save_debug_image(cropped, f"{img_name}_08_cropped.jpg", "Cropped from padded image")

        # helper to order quad points
        def _order_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  # top-left
            rect[2] = pts[np.argmax(s)]  # bottom-right
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  # top-right
            rect[3] = pts[np.argmax(diff)]  # bottom-left
            return rect

        def _expand_quad(quad_pts, expand_px, img_shape):
            """Push each quad corner outward by expand_px while staying within image."""
            if quad_pts is None:
                return None
            h_img, w_img = img_shape[:2]
            center = np.mean(quad_pts, axis=0)
            expanded = []
            for pt in quad_pts:
                vec = pt - center
                length = np.linalg.norm(vec)
                if length < 1e-3:
                    new_pt = pt.copy()
                else:
                    scale = (length + expand_px) / length
                    new_pt = center + vec * scale
                new_pt[0] = np.clip(new_pt[0], 0, w_img - 1)
                new_pt[1] = np.clip(new_pt[1], 0, h_img - 1)
                expanded.append(new_pt)
            return np.array(expanded, dtype=np.float32)

        # Convert to grayscale and prepare for contour detection
        gray_c = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        h, w = gray_c.shape
        
        # ============================================================
        # STEP 1: Separate paper/document from wood table background
        # ============================================================
        # Paper is typically brighter/whiter than wood table
        # Use brightness analysis to identify paper region
        
        # Method 1: Find bright regions (paper is brighter than wood)
        # Use adaptive thresholding to separate bright paper from darker wood
        blur = cv2.GaussianBlur(gray_c, (5, 5), 0)
        
        # Calculate statistics to determine paper brightness threshold
        mean_intensity = np.mean(blur)
        std_intensity = np.std(blur)
        
        # Paper should be in the upper brightness range
        # Use percentile-based threshold (paper is typically in top 30-40% of brightness)
        paper_threshold = np.percentile(blur, 60)  # Top 40% brightest pixels
        
        # Create mask for bright regions (likely paper)
        _, paper_mask = cv2.threshold(blur, paper_threshold, 255, cv2.THRESH_BINARY)
        
        # Clean up the mask - remove small noise, fill holes
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN, kernel_clean, iterations=2)
        paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, kernel_clean, iterations=3)
        
        # Find the largest bright region (should be the paper)
        paper_contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if paper_contours:
            # Get the largest contour (paper)
            largest_paper_contour = max(paper_contours, key=cv2.contourArea)
            paper_area = cv2.contourArea(largest_paper_contour)
            image_area = h * w
            
            # Only use paper mask if the detected paper is substantial (at least 30% of image)
            if paper_area > 0.3 * image_area:
                # Create refined paper mask from largest contour
                refined_paper_mask = np.zeros_like(paper_mask)
                cv2.drawContours(refined_paper_mask, [largest_paper_contour], -1, 255, -1)
                paper_mask = refined_paper_mask
            else:
                # Paper detection failed, use full image
                paper_mask = np.ones_like(paper_mask) * 255
        
        if self.debug:
            self.save_debug_image(paper_mask, f"{img_name}_08b_paper_mask.jpg", "Paper region mask (bright regions)")
        
        # Method 2: Enhanced edge detection to find fine lines of page/table
        # Use multiple Canny thresholds to capture both strong and fine edges
        edges_low = cv2.Canny(blur, 30, 100)  # Lower threshold for fine lines
        edges_medium = cv2.Canny(blur, 50, 150)  # Medium threshold
        edges_high = cv2.Canny(blur, 100, 200)  # Higher threshold for strong edges
        
        # Combine all edge detections
        edges_full = cv2.bitwise_or(edges_low, cv2.bitwise_or(edges_medium, edges_high))
        
        # Dilate paper mask slightly to include edges at paper boundary
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        paper_mask_dilated = cv2.dilate(paper_mask, kernel_dilate, iterations=2)
        
        # Focus edge detection on paper region and its immediate boundary
        edges_focused = cv2.bitwise_and(edges_full, paper_mask_dilated)
        
        # Also detect edges that are transitions from paper to background
        # Invert paper mask to get background
        background_mask = cv2.bitwise_not(paper_mask)
        # Find edges at the boundary between paper and background
        paper_edges = cv2.Canny(paper_mask, 50, 150)
        
        # Combine: edges within paper + edges at paper boundary
        edges_combined = cv2.bitwise_or(edges_focused, paper_edges)
        
        # Method 2b: Use Hough lines to find long boundary lines and filter short noise
        # This helps identify the main 4 boundary lines vs internal grid lines
        min_line_length = int(max(h, w) * 0.15)  # At least 15% of image dimension
        hough_lines = cv2.HoughLinesP(
            edges_combined,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=min_line_length,
            maxLineGap=30
        )
        
        # Filter lines: keep only long lines that are likely boundaries
        # Remove short peripheral lines (noise)
        boundary_lines = []
        if hough_lines is not None:
            for line in hough_lines[:, 0]:
                x1, y1, x2, y2 = line
                length = np.hypot(x2 - x1, y2 - y1)
                
                # Filter: keep lines that are long enough and not too close to image edges (unless they span most of the dimension)
                edge_threshold = 0.05  # 5% from edge
                min_dist_from_edge = min(h, w) * edge_threshold
                
                # Check if line is near image boundary (likely a boundary line)
                near_top = min(y1, y2) < min_dist_from_edge
                near_bottom = max(y1, y2) > (h - min_dist_from_edge)
                near_left = min(x1, x2) < min_dist_from_edge
                near_right = max(x1, x2) > (w - min_dist_from_edge)
                
                # Keep if: long line OR line near boundary (likely page edge)
                is_boundary_candidate = (near_top or near_bottom or near_left or near_right)
                
                if length >= min_line_length * 0.7 or (is_boundary_candidate and length >= min_line_length * 0.5):
                    boundary_lines.append(line)
        
        # Create image with only boundary lines (filtered)
        edges_boundary_only = np.zeros_like(edges_combined)
        for line in boundary_lines:
            x1, y1, x2, y2 = line
            cv2.line(edges_boundary_only, (x1, y1), (x2, y2), 255, 2)
        
        # NOISE REMOVAL: Clean up the edges
        # Step 1: Remove small isolated components (noise)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges_combined, connectivity=8)
        min_component_area = (h * w) * 0.001  # Remove components smaller than 0.1% of image
        edges_cleaned = np.zeros_like(edges_combined)
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_component_area:
                # Keep this component
                component_mask = (labels == i).astype(np.uint8) * 255
                edges_cleaned = cv2.bitwise_or(edges_cleaned, component_mask)
        
        # Step 2: Morphological operations to clean up edges
        # Close small gaps in lines
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_cleaned = cv2.morphologyEx(edges_cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        # Remove small noise spots
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges_cleaned = cv2.morphologyEx(edges_cleaned, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # Step 3: Also clean the boundary-only edges
        edges_boundary_cleaned = np.zeros_like(edges_boundary_only)
        num_labels_boundary, labels_boundary, stats_boundary, _ = cv2.connectedComponentsWithStats(
            edges_boundary_only, connectivity=8)
        
        for i in range(1, num_labels_boundary):
            area = stats_boundary[i, cv2.CC_STAT_AREA]
            if area >= min_component_area:
                component_mask = (labels_boundary == i).astype(np.uint8) * 255
                edges_boundary_cleaned = cv2.bitwise_or(edges_boundary_cleaned, component_mask)
        
        # Apply morphological operations to boundary edges too
        edges_boundary_cleaned = cv2.morphologyEx(edges_boundary_cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        edges_boundary_cleaned = cv2.morphologyEx(edges_boundary_cleaned, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        if self.debug:
            self.save_debug_image(edges_low, f"{img_name}_08b_edges_low_thresh.jpg", "Edges (low threshold - fine lines)")
            self.save_debug_image(edges_medium, f"{img_name}_08b_edges_medium_thresh.jpg", "Edges (medium threshold)")
            self.save_debug_image(edges_high, f"{img_name}_08b_edges_high_thresh.jpg", "Edges (high threshold - strong lines)")
            self.save_debug_image(edges_full, f"{img_name}_08b_edges_full.jpg", "All edges (combined)")
            self.save_debug_image(edges_focused, f"{img_name}_08b_edges_focused.jpg", "Edges focused on paper region")
            self.save_debug_image(paper_edges, f"{img_name}_08b_paper_boundary_edges.jpg", "Paper boundary edges")
            self.save_debug_image(edges_combined, f"{img_name}_08b_edges_combined.jpg", "Combined edges (paper-focused)")
            self.save_debug_image(edges_cleaned, f"{img_name}_08b_edges_cleaned.jpg", 
                                 f"Edges after noise removal ({num_labels-1} components -> {len([i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_component_area])} kept)")
            self.save_debug_image(edges_boundary_only, f"{img_name}_08b_edges_boundary_filtered.jpg", 
                                 f"Filtered boundary lines only ({len(boundary_lines)} lines)")
            self.save_debug_image(edges_boundary_cleaned, f"{img_name}_08b_edges_boundary_cleaned.jpg", 
                                 "Boundary edges after noise removal")
        
        # Use cleaned edges for detection (prefer boundary-cleaned, fallback to general cleaned)
        edges_combined = edges_boundary_cleaned if np.sum(edges_boundary_cleaned) > 0 else edges_cleaned
        
        # Method 3: Traditional thresholding for fallback
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply paper mask to threshold result to focus on paper region
        th_masked = cv2.bitwise_and(th, paper_mask)
        
        if self.debug:
            self.save_debug_image(th, f"{img_name}_08b_threshold_full.jpg", "Binary threshold (full)")
            self.save_debug_image(th_masked, f"{img_name}_08b_threshold_masked.jpg", "Binary threshold (paper-masked)")

        # Helper function to identify shape type (rectangle vs trapezoid)
        def identify_shape(quad_pts):
            """Identify if quadrilateral is rectangle or trapezoid
            Handles trapezoids with different sized sides and angles"""
            rect = _order_points(quad_pts)
            (tl, tr, br, bl) = rect
            
            # Calculate angles at each corner
            def angle_between_vectors(v1, v2):
                """Calculate angle between two vectors in degrees"""
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                return np.degrees(np.arccos(cos_angle))
            
            # Vectors for each edge
            top_edge = tr - tl
            right_edge = br - tr
            bottom_edge = bl - br
            left_edge = tl - bl
            
            # Calculate angles at each corner
            angle_tl = angle_between_vectors(-left_edge, top_edge)
            angle_tr = angle_between_vectors(-top_edge, right_edge)
            angle_br = angle_between_vectors(-right_edge, bottom_edge)
            angle_bl = angle_between_vectors(-bottom_edge, left_edge)
            
            angles = [angle_tl, angle_tr, angle_br, angle_bl]
            avg_angle = np.mean(angles)
            angle_deviation = np.std(angles)
            
            # Check if top and bottom edges are parallel (trapezoid characteristic)
            # Normalize vectors
            top_norm = top_edge / (np.linalg.norm(top_edge) + 1e-6)
            bottom_norm = bottom_edge / (np.linalg.norm(bottom_edge) + 1e-6)
            parallel_score_top_bottom = abs(np.dot(top_norm, bottom_norm))
            
            # Check if left and right edges are parallel
            left_norm = left_edge / (np.linalg.norm(left_edge) + 1e-6)
            right_norm = right_edge / (np.linalg.norm(right_edge) + 1e-6)
            parallel_score_left_right = abs(np.dot(left_norm, right_norm))
            
            # Check side length differences (trapezoid can have different sized sides)
            top_len = np.linalg.norm(top_edge)
            bottom_len = np.linalg.norm(bottom_edge)
            left_len = np.linalg.norm(left_edge)
            right_len = np.linalg.norm(right_edge)
            
            # Rectangle: all angles ~90 degrees, low deviation, opposite sides equal
            is_rectangle = (avg_angle > 85 and avg_angle < 95 and angle_deviation < 10 and
                          abs(top_len - bottom_len) / max(top_len, bottom_len) < 0.1 and
                          abs(left_len - right_len) / max(left_len, right_len) < 0.1)
            
            # Trapezoid: at least one pair of opposite sides parallel (allow for different sizes)
            # Allow for angles and different side lengths
            is_trapezoid = (parallel_score_top_bottom > 0.9 or parallel_score_left_right > 0.9)
            
            if is_rectangle:
                return "rectangle", angles
            elif is_trapezoid:
                return "trapezoid", angles
            else:
                # Still return as trapezoid if it's close (handles angled/irregular shapes)
                return "trapezoid", angles
        
        # Helper function to detect quadrilateral from contours
        def detect_quad_from_contours(contours, min_area_ratio=0.4, target="paper"):
            """Detect quadrilateral from contours, ensuring exactly 4 points"""
            if not contours:
                return None, None, None
            
            bbox_area = w * h
            # Sort contours by area (largest first)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # For paper detection, look for larger contours (top 3)
            # For table detection, look for medium-sized contours (top 5)
            num_to_check = 3 if target == "paper" else 5
            
            for idx, contour in enumerate(sorted_contours[:num_to_check]):
                area = cv2.contourArea(contour)
                area_ratio = area / bbox_area if bbox_area > 0 else 0
                
                # Paper should be larger, table can be smaller
                min_ratio = 0.6 if target == "paper" else min_area_ratio
                
                if area_ratio < min_ratio:
                    continue
                
                peri = cv2.arcLength(contour, True)
                if peri < 100:
                    continue
                
                # Try multiple epsilon values to find best quad approximation
                # Force to exactly 4 points
                epsilon_values = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25]
                
                for eps_factor in epsilon_values:
                    approx = cv2.approxPolyDP(contour, eps_factor * peri, True)
                    
                    # Force to exactly 4 points if we have 3-6 points
                    if len(approx) >= 3 and len(approx) <= 6:
                        # If we have more than 4, use convex hull and approximate again
                        if len(approx) > 4:
                            # Try to get 4 points by using convex hull
                            hull = cv2.convexHull(contour)
                            if len(hull) >= 4:
                                # Try different epsilon values on hull to get exactly 4
                                peri_hull = cv2.arcLength(hull, True)
                                for hull_eps in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
                                    approx_hull = cv2.approxPolyDP(hull, hull_eps * peri_hull, True)
                                    if len(approx_hull) == 4:
                                        quad_pts = approx_hull.reshape(4, 2).astype(np.float32)
                                        # Validate quad dimensions
                                        rect = _order_points(quad_pts)
                                        widthA = np.linalg.norm(rect[2] - rect[3])
                                        widthB = np.linalg.norm(rect[1] - rect[0])
                                        heightA = np.linalg.norm(rect[1] - rect[2])
                                        heightB = np.linalg.norm(rect[0] - rect[3])
                                        max_w = max(widthA, widthB)
                                        max_h = max(heightA, heightB)
                                        
                                        if max_w > 50 and max_h > 50:
                                            shape_type, angles = identify_shape(quad_pts)
                                            return quad_pts, shape_type, angles
                        elif len(approx) == 4:
                            quad_pts = approx.reshape(4, 2).astype(np.float32)
                            
                            # Validate quad dimensions
                            rect = _order_points(quad_pts)
                            widthA = np.linalg.norm(rect[2] - rect[3])
                            widthB = np.linalg.norm(rect[1] - rect[0])
                            heightA = np.linalg.norm(rect[1] - rect[2])
                            heightB = np.linalg.norm(rect[0] - rect[3])
                            max_w = max(widthA, widthB)
                            max_h = max(heightA, heightB)
                            
                            if max_w > 50 and max_h > 50:
                                shape_type, angles = identify_shape(quad_pts)
                                return quad_pts, shape_type, angles
                        elif len(approx) == 3:
                            # If we have 3 points, construct 4th point to form a rectangle/trapezoid
                            # Assume the missing corner based on the pattern
                            pts = approx.reshape(-1, 2).astype(np.float32)
                            # Find the bounding box and use its corners
                            x_coords = pts[:, 0]
                            y_coords = pts[:, 1]
                            min_x, max_x = np.min(x_coords), np.max(x_coords)
                            min_y, max_y = np.min(y_coords), np.max(y_coords)
                            
                            # Construct 4 corners from bounding box (assume rectangle/trapezoid)
                            quad_pts = np.array([
                                [min_x, min_y],  # top-left
                                [max_x, min_y],  # top-right
                                [max_x, max_y],  # bottom-right
                                [min_x, max_y]   # bottom-left
                            ], dtype=np.float32)
                            
                            # Validate quad dimensions
                            rect = _order_points(quad_pts)
                            widthA = np.linalg.norm(rect[2] - rect[3])
                            widthB = np.linalg.norm(rect[1] - rect[0])
                            heightA = np.linalg.norm(rect[1] - rect[2])
                            heightB = np.linalg.norm(rect[0] - rect[3])
                            max_w = max(widthA, widthB)
                            max_h = max(heightA, heightB)
                            
                            if max_w > 50 and max_h > 50:
                                shape_type, angles = identify_shape(quad_pts)
                                return quad_pts, shape_type, angles
                
                # If direct approximation failed, try convex hull and force to 4 points
                hull = cv2.convexHull(contour)
                if len(hull) >= 4:
                    peri_hull = cv2.arcLength(hull, True)
                    for eps_factor in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
                        approx_hull = cv2.approxPolyDP(hull, eps_factor * peri_hull, True)
                        if len(approx_hull) == 4:
                            quad_pts = approx_hull.reshape(4, 2).astype(np.float32)
                            shape_type, angles = identify_shape(quad_pts)
                            return quad_pts, shape_type, angles
                        elif len(approx_hull) >= 3 and len(approx_hull) <= 6:
                            # Force to 4 points by using bounding box
                            pts = approx_hull.reshape(-1, 2).astype(np.float32)
                            x_coords = pts[:, 0]
                            y_coords = pts[:, 1]
                            min_x, max_x = np.min(x_coords), np.max(x_coords)
                            min_y, max_y = np.min(y_coords), np.max(y_coords)
                            
                            quad_pts = np.array([
                                [min_x, min_y],
                                [max_x, min_y],
                                [max_x, max_y],
                                [min_x, max_y]
                            ], dtype=np.float32)
                            
                            shape_type, angles = identify_shape(quad_pts)
                            return quad_pts, shape_type, angles
            
            return None, None, None
        
        # Helper function to detect quadrilateral from cleaned edges using contour detection
        def detect_quad_from_cleaned_edges(edges_image, target="paper"):
            """Detect quadrilateral from cleaned edges using contour detection
            
            Args:
                edges_image: Cleaned edge image (after noise removal)
                target: "paper" or "table"
            """
            if edges_image is None or np.sum(edges_image) == 0:
                return None, None, None
            
            h, w = edges_image.shape
            bbox_area = w * h
            
            # Find contours from cleaned edges
            contours, _ = cv2.findContours(edges_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, None, None
            
            # Sort contours by area (largest first)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # For paper detection, look for larger contours (top 3)
            # For table detection, prioritize contours that are:
            # 1. Large enough (at least 40% of image)
            # 2. Close to image edges (likely the outer table boundary)
            num_to_check = 3 if target == "paper" else 10  # Check more for table
            
            best_quad = None
            best_score = -1
            
            for idx, contour in enumerate(sorted_contours[:num_to_check]):
                area = cv2.contourArea(contour)
                area_ratio = area / bbox_area if bbox_area > 0 else 0
                
                # Paper should be larger, table should be substantial
                min_ratio = 0.4 if target == "paper" else 0.35  # Increased from 0.2 to 0.35
                
                if area_ratio < min_ratio:
                    continue
                
                # For table detection, prioritize contours near image edges
                # This helps find the outer table boundary, not internal structures
                if target == "table":
                    # Get bounding box of contour
                    x_cont, y_cont, w_cont, h_cont = cv2.boundingRect(contour)
                    
                    # Calculate how close contour is to image edges
                    edge_margin = min(w, h) * 0.05  # 5% margin
                    near_top = y_cont < edge_margin
                    near_bottom = (y_cont + h_cont) > (h - edge_margin)
                    near_left = x_cont < edge_margin
                    near_right = (x_cont + w_cont) > (w - edge_margin)
                    
                    # Score: higher for contours that touch more edges
                    edge_score = sum([near_top, near_bottom, near_left, near_right])
                    
                    # Also prefer larger contours
                    size_score = area_ratio
                    
                    # Combined score
                    combined_score = edge_score * 0.6 + size_score * 0.4
                    
                    # Only consider if it touches at least 2 edges (likely outer boundary)
                    if edge_score < 2:
                        continue
                else:
                    combined_score = area_ratio
                
                peri = cv2.arcLength(contour, True)
                if peri < 100:
                    continue
                
                # Try multiple epsilon values to find best quad approximation
                # Force to exactly 4 points
                epsilon_values = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25]
                
                for eps_factor in epsilon_values:
                    approx = cv2.approxPolyDP(contour, eps_factor * peri, True)
                    
                    # Force to exactly 4 points if we have 3-6 points
                    if len(approx) >= 3 and len(approx) <= 6:
                        # If we have more than 4, use convex hull and approximate again
                        if len(approx) > 4:
                            # Try to get 4 points by using convex hull
                            hull = cv2.convexHull(contour)
                            if len(hull) >= 4:
                                # Try different epsilon values on hull to get exactly 4
                                peri_hull = cv2.arcLength(hull, True)
                                for hull_eps in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
                                    approx_hull = cv2.approxPolyDP(hull, hull_eps * peri_hull, True)
                                    if len(approx_hull) == 4:
                                        quad_pts = approx_hull.reshape(4, 2).astype(np.float32)
                                        # Validate quad dimensions
                                        rect = _order_points(quad_pts)
                                        widthA = np.linalg.norm(rect[2] - rect[3])
                                        widthB = np.linalg.norm(rect[1] - rect[0])
                                        heightA = np.linalg.norm(rect[1] - rect[2])
                                        heightB = np.linalg.norm(rect[0] - rect[3])
                                        max_w = max(widthA, widthB)
                                        max_h = max(heightA, heightB)
                                        
                                        if max_w > 50 and max_h > 50:
                                            shape_type, angles = identify_shape(quad_pts)
                                            
                                            # Store best quad based on score
                                            if combined_score > best_score:
                                                best_score = combined_score
                                                best_quad = (quad_pts, shape_type, angles, approx_hull)
                                            
                                            if self.debug:
                                                debug_img = cropped.copy()
                                                cv2.drawContours(debug_img, [approx_hull], -1, (0, 255, 0), 3)
                                                for i, pt in enumerate(quad_pts):
                                                    cv2.circle(debug_img, tuple(pt.astype(int)), 8, (0, 0, 255), -1)
                                                    cv2.putText(debug_img, str(i), tuple(pt.astype(int)), 
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                                                self.save_debug_image(debug_img, f"{img_name}_08f_edges_contour_detection.jpg", 
                                                                     f"Contour detection from cleaned edges ({target})")
                                            
                                            # For paper, return first valid one
                                            if target == "paper":
                                                return quad_pts, shape_type, angles
                        elif len(approx) == 4:
                            quad_pts = approx.reshape(4, 2).astype(np.float32)
                            
                            # Validate quad dimensions
                            rect = _order_points(quad_pts)
                            widthA = np.linalg.norm(rect[2] - rect[3])
                            widthB = np.linalg.norm(rect[1] - rect[0])
                            heightA = np.linalg.norm(rect[1] - rect[2])
                            heightB = np.linalg.norm(rect[0] - rect[3])
                            max_w = max(widthA, widthB)
                            max_h = max(heightA, heightB)
                            
                            if max_w > 50 and max_h > 50:
                                shape_type, angles = identify_shape(quad_pts)
                                
                                # Store best quad based on score
                                if combined_score > best_score:
                                    best_score = combined_score
                                    best_quad = (quad_pts, shape_type, angles, approx)
                                
                                if self.debug:
                                    debug_img = cropped.copy()
                                    cv2.drawContours(debug_img, [approx], -1, (0, 255, 0), 3)
                                    for i, pt in enumerate(quad_pts):
                                        cv2.circle(debug_img, tuple(pt.astype(int)), 8, (0, 0, 255), -1)
                                        cv2.putText(debug_img, str(i), tuple(pt.astype(int)), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                                    self.save_debug_image(debug_img, f"{img_name}_08f_edges_contour_detection.jpg", 
                                                         f"Contour detection from cleaned edges ({target})")
                                
                                # For paper, return first valid one
                                if target == "paper":
                                    return quad_pts, shape_type, angles
                        elif len(approx) == 3:
                            # If we have 3 points, construct 4th point to form a rectangle/trapezoid
                            pts = approx.reshape(-1, 2).astype(np.float32)
                            x_coords = pts[:, 0]
                            y_coords = pts[:, 1]
                            min_x, max_x = np.min(x_coords), np.max(x_coords)
                            min_y, max_y = np.min(y_coords), np.max(y_coords)
                            
                            # Construct 4 corners from bounding box
                            quad_pts = np.array([
                                [min_x, min_y],
                                [max_x, min_y],
                                [max_x, max_y],
                                [min_x, max_y]
                            ], dtype=np.float32)
                            
                            # Validate quad dimensions
                            rect = _order_points(quad_pts)
                            widthA = np.linalg.norm(rect[2] - rect[3])
                            widthB = np.linalg.norm(rect[1] - rect[0])
                            heightA = np.linalg.norm(rect[1] - rect[2])
                            heightB = np.linalg.norm(rect[0] - rect[3])
                            max_w = max(widthA, widthB)
                            max_h = max(heightA, heightB)
                            
                            if max_w > 50 and max_h > 50:
                                shape_type, angles = identify_shape(quad_pts)
                                
                                # Store best quad based on score
                                if combined_score > best_score:
                                    best_score = combined_score
                                    best_quad = (quad_pts, shape_type, angles, approx)
                                
                                if self.debug:
                                    debug_img = cropped.copy()
                                    cv2.drawContours(debug_img, [approx], -1, (0, 255, 0), 3)
                                    for i, pt in enumerate(quad_pts):
                                        cv2.circle(debug_img, tuple(pt.astype(int)), 8, (0, 0, 255), -1)
                                        cv2.putText(debug_img, str(i), tuple(pt.astype(int)), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                                    self.save_debug_image(debug_img, f"{img_name}_08f_edges_contour_detection.jpg", 
                                                         f"Contour detection from cleaned edges ({target})")
                                
                                # For paper, return first valid one
                                if target == "paper":
                                    return quad_pts, shape_type, angles
                
                # If direct approximation failed, try convex hull and force to 4 points
                hull = cv2.convexHull(contour)
                if len(hull) >= 4:
                    peri_hull = cv2.arcLength(hull, True)
                    for eps_factor in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
                        approx_hull = cv2.approxPolyDP(hull, eps_factor * peri_hull, True)
                        if len(approx_hull) == 4:
                            quad_pts = approx_hull.reshape(4, 2).astype(np.float32)
                            shape_type, angles = identify_shape(quad_pts)
                            
                            if self.debug:
                                debug_img = cropped.copy()
                                cv2.drawContours(debug_img, [approx_hull], -1, (0, 255, 0), 3)
                                for i, pt in enumerate(quad_pts):
                                    cv2.circle(debug_img, tuple(pt.astype(int)), 8, (0, 0, 255), -1)
                                    cv2.putText(debug_img, str(i), tuple(pt.astype(int)), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                                self.save_debug_image(debug_img, f"{img_name}_08f_edges_contour_detection.jpg", 
                                                     f"Contour detection from cleaned edges ({target})")
                            
                            return quad_pts, shape_type, angles
                        elif len(approx_hull) >= 3 and len(approx_hull) <= 6:
                            # Force to 4 points by using bounding box
                            pts = approx_hull.reshape(-1, 2).astype(np.float32)
                            x_coords = pts[:, 0]
                            y_coords = pts[:, 1]
                            min_x, max_x = np.min(x_coords), np.max(x_coords)
                            min_y, max_y = np.min(y_coords), np.max(y_coords)
                            
                            quad_pts = np.array([
                                [min_x, min_y],
                                [max_x, min_y],
                                [max_x, max_y],
                                [min_x, max_y]
                            ], dtype=np.float32)
                            
                            shape_type, angles = identify_shape(quad_pts)
                            
                            if self.debug:
                                debug_img = cropped.copy()
                                cv2.drawContours(debug_img, [approx_hull], -1, (0, 255, 0), 3)
                                for i, pt in enumerate(quad_pts):
                                    cv2.circle(debug_img, tuple(pt.astype(int)), 8, (0, 0, 255), -1)
                                    cv2.putText(debug_img, str(i), tuple(pt.astype(int)), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                                self.save_debug_image(debug_img, f"{img_name}_08f_edges_contour_detection.jpg", 
                                                     f"Contour detection from cleaned edges ({target})")
                            
                            # Store best quad based on score
                            if combined_score > best_score:
                                best_score = combined_score
                                best_quad = (quad_pts, shape_type, angles, approx_hull)
                            
                            # For paper, return first valid one
                            if target == "paper":
                                return quad_pts, shape_type, angles
            
            # Return the best quad found (for table detection, this ensures we get the outer boundary)
            if best_quad is not None:
                quad_pts, shape_type, angles, approx = best_quad
                if self.debug:
                    debug_img = cropped.copy()
                    cv2.drawContours(debug_img, [approx], -1, (0, 255, 0), 3)
                    for i, pt in enumerate(quad_pts):
                        cv2.circle(debug_img, tuple(pt.astype(int)), 8, (0, 0, 255), -1)
                        cv2.putText(debug_img, str(i), tuple(pt.astype(int)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    self.save_debug_image(debug_img, f"{img_name}_08f_best_quad_{target}.jpg", 
                                         f"Best quad selected (score: {best_score:.2f})")
                return quad_pts, shape_type, angles
            
            return None, None, None
        
        # Helper function to detect quadrilateral using horizontal + vertical + angled lines
        def detect_quad_from_line_masks(image_gray, target="paper"):
            """Detect quadrilateral using horizontal, vertical, and angled line masks"""
            h, w = image_gray.shape
            
            # Step 1: Detect horizontal, vertical, and angled lines (similar to detect_table_boundary)
            blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
            _, binary_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Apply paper mask if available to focus on paper region
            if 'paper_mask' in locals() and paper_mask is not None:
                binary_otsu = cv2.bitwise_and(binary_otsu, paper_mask)
            
            # === HORIZONTAL LINES ===
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 15, 1))
            horizontal = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, h_kernel, iterations=1)
            h_kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 2))
            horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, h_kernel_close, iterations=2)
            
            # === VERTICAL LINES ===
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 15))
            vertical = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, v_kernel, iterations=1)
            v_kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 20))
            vertical = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, v_kernel_close, iterations=2)
            
            # === ANGLED LINES ===
            def _rot_kernel(length, angle, thickness=2):
                size = length
                k = np.zeros((size, size), dtype=np.uint8)
                x2 = int(size//2 + np.cos(np.deg2rad(angle)) * length/2)
                y2 = int(size//2 + np.sin(np.deg2rad(angle)) * length/2)
                cv2.line(k, (size//2, size//2), (x2, y2), 1, thickness)
                return k
            
            angles = [-7, -5, -3, 3, 5, 7]  # Small tilt angles for angled-line detection
            angled_masks = []
            for ang in angles:
                rk = _rot_kernel(45, ang)
                temp = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, rk)
                angled_masks.append(temp)
            
            angled_lines = angled_masks[0].copy()
            for m in angled_masks[1:]:
                angled_lines = cv2.bitwise_or(angled_lines, m)
            angled_lines = cv2.medianBlur(angled_lines, 3)
            
            # Also use Hough for angled lines
            edges = cv2.Canny(binary_otsu, 50, 150)
            hough_mask = np.zeros_like(binary_otsu)
            hough_lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60, minLineLength=40, maxLineGap=10)
            
            if hough_lines is not None:
                for (x1, y1, x2, y2) in hough_lines[:, 0]:
                    angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                    if 4 < angle < 86:  # Angled lines
                        cv2.line(hough_mask, (x1, y1), (x2, y2), 255, 2)
            
            # Combine angled lines
            angled_combined = cv2.bitwise_or(angled_lines, hough_mask)
            
            if self.debug:
                self.save_debug_image(horizontal, f"{img_name}_08h_horizontal_lines.jpg", f"Horizontal lines ({target})")
                self.save_debug_image(vertical, f"{img_name}_08h_vertical_lines.jpg", f"Vertical lines ({target})")
                self.save_debug_image(angled_combined, f"{img_name}_08h_angled_lines.jpg", f"Angled lines ({target})")
            
            # Step 2: Extract line segments from masks using HoughLinesP
            min_line_length = int(max(h, w) * 0.15)  # At least 15% of image dimension
            
            all_lines = []
            
            # Extract horizontal lines
            h_lines = cv2.HoughLinesP(horizontal, 1, np.pi/180, threshold=30, 
                                     minLineLength=min_line_length, maxLineGap=30)
            if h_lines is not None:
                for line in h_lines[:, 0]:
                    all_lines.append(('horizontal', line))
            
            # Extract vertical lines
            v_lines = cv2.HoughLinesP(vertical, 1, np.pi/180, threshold=30,
                                     minLineLength=min_line_length, maxLineGap=30)
            if v_lines is not None:
                for line in v_lines[:, 0]:
                    all_lines.append(('vertical', line))
            
            # Extract angled lines
            a_lines = cv2.HoughLinesP(angled_combined, 1, np.pi/180, threshold=30,
                                      minLineLength=int(min_line_length * 0.7), maxLineGap=40)
            if a_lines is not None:
                for line in a_lines[:, 0]:
                    all_lines.append(('angled', line))
            
            if len(all_lines) < 4:
                return None, None, None
            
            # Step 3: Group lines and find boundary lines
            horizontal_lines = []
            vertical_lines = []
            diagonal_lines = []
            
            for line_type, (x1, y1, x2, y2) in all_lines:
                dx = x2 - x1
                dy = y2 - y1
                length = np.hypot(dx, dy)
                if length < min_line_length * 0.5:
                    continue
                
                angle = abs(np.degrees(np.arctan2(dy, dx)))
                if angle > 90:
                    angle = 180 - angle
                
                if line_type == 'horizontal' or angle < 15:  # Near horizontal
                    horizontal_lines.append((x1, y1, x2, y2))
                elif line_type == 'vertical' or angle > 75:  # Near vertical
                    vertical_lines.append((x1, y1, x2, y2))
                else:  # Diagonal/angled
                    diagonal_lines.append((x1, y1, x2, y2))
            
            # Step 4: Find the four boundary lines
            def find_boundary_line(lines, orientation):
                """Find the outermost line in a given orientation"""
                if not lines:
                    return None
                
                if orientation == 'top':
                    # Topmost line (minimum y)
                    return min(lines, key=lambda l: min(l[1], l[3]))
                elif orientation == 'bottom':
                    # Bottommost line (maximum y)
                    return max(lines, key=lambda l: max(l[1], l[3]))
                elif orientation == 'left':
                    # Leftmost line (minimum x)
                    return min(lines, key=lambda l: min(l[0], l[2]))
                elif orientation == 'right':
                    # Rightmost line (maximum x)
                    return max(lines, key=lambda l: max(l[0], l[2]))
                return None
            
            # Find boundary lines (prefer horizontal/vertical, fallback to diagonal)
            top_line = find_boundary_line(horizontal_lines, 'top')
            if top_line is None and diagonal_lines:
                top_line = find_boundary_line(diagonal_lines, 'top')
            
            bottom_line = find_boundary_line(horizontal_lines, 'bottom')
            if bottom_line is None and diagonal_lines:
                bottom_line = find_boundary_line(diagonal_lines, 'bottom')
            
            left_line = find_boundary_line(vertical_lines, 'left')
            if left_line is None and diagonal_lines:
                left_line = find_boundary_line(diagonal_lines, 'left')
            
            right_line = find_boundary_line(vertical_lines, 'right')
            if right_line is None and diagonal_lines:
                right_line = find_boundary_line(diagonal_lines, 'right')
            
            # Step 5: Calculate intersection points
            def line_intersection(line1, line2):
                """Find intersection point of two line segments"""
                x1, y1, x2, y2 = line1
                x3, y3, x4, y4 = line2
                
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-10:
                    return None
                
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
                
                # Extend lines if needed (allow intersection outside segments for boundary detection)
                if -10 <= t <= 11 and -10 <= u <= 11:  # Allow some extension
                    px = x1 + t * (x2 - x1)
                    py = y1 + t * (y2 - y1)
                    return (px, py)
                return None
            
            # Find 4 corners - construct missing corners using table structure assumptions
            corners = []
            corner_positions = {}  # Store corners by position: 'tl', 'tr', 'bl', 'br'
            
            # Find intersections where lines exist
            if top_line and left_line:
                corner = line_intersection(top_line, left_line)
                if corner:
                    corners.append(corner)
                    corner_positions['tl'] = corner
            if top_line and right_line:
                corner = line_intersection(top_line, right_line)
                if corner:
                    corners.append(corner)
                    corner_positions['tr'] = corner
            if bottom_line and left_line:
                corner = line_intersection(bottom_line, left_line)
                if corner:
                    corners.append(corner)
                    corner_positions['bl'] = corner
            if bottom_line and right_line:
                corner = line_intersection(bottom_line, right_line)
                if corner:
                    corners.append(corner)
                    corner_positions['br'] = corner
            
            # If we have at least 2 corners, construct the missing ones using table structure
            if len(corners) >= 2:
                # Extend lines to image boundaries to find missing corners
                def extend_line_to_boundary(line, h, w):
                    """Extend line to image boundaries"""
                    x1, y1, x2, y2 = line
                    dx = x2 - x1
                    dy = y2 - y1
                    if abs(dx) < 1e-6:  # Vertical line
                        return (x1, 0, x1, h)
                    if abs(dy) < 1e-6:  # Horizontal line
                        return (0, y1, w, y1)
                    
                    # Extend line to boundaries
                    # Find intersections with image boundaries
                    t_values = []
                    # Top boundary (y=0)
                    if dy != 0:
                        t = -y1 / dy
                        x = x1 + t * dx
                        if 0 <= x <= w:
                            t_values.append((t, x, 0))
                    # Bottom boundary (y=h)
                    if dy != 0:
                        t = (h - y1) / dy
                        x = x1 + t * dx
                        if 0 <= x <= w:
                            t_values.append((t, x, h))
                    # Left boundary (x=0)
                    if dx != 0:
                        t = -x1 / dx
                        y = y1 + t * dy
                        if 0 <= y <= h:
                            t_values.append((t, 0, y))
                    # Right boundary (x=w)
                    if dx != 0:
                        t = (w - x1) / dx
                        y = y1 + t * dy
                        if 0 <= y <= h:
                            t_values.append((t, w, y))
                    
                    if len(t_values) >= 2:
                        # Get the two points with min and max t
                        t_values.sort(key=lambda x: x[0])
                        return (int(t_values[0][1]), int(t_values[0][2]), 
                                int(t_values[-1][1]), int(t_values[-1][2]))
                    return line
                
                # Construct missing corners
                # If we have top_line, use it to find top corners
                if top_line and 'tl' not in corner_positions:
                    # Extend top_line and find intersection with left boundary or left_line
                    if left_line:
                        extended_top = extend_line_to_boundary(top_line, h, w)
                        corner = line_intersection(extended_top, left_line)
                        if corner:
                            corner_positions['tl'] = corner
                    elif 'tr' in corner_positions:
                        # Use top-right corner and assume parallel sides (trapezoid/rectangle)
                        tr = corner_positions['tr']
                        # Estimate top-left based on top-right and image structure
                        if 'bl' in corner_positions:
                            bl = corner_positions['bl']
                            # Assume rectangle: top-left y = top-right y, top-left x = bottom-left x
                            corner_positions['tl'] = (bl[0], tr[1])
                        else:
                            # Use image boundary
                            corner_positions['tl'] = (0, tr[1])
                
                if top_line and 'tr' not in corner_positions:
                    if right_line:
                        extended_top = extend_line_to_boundary(top_line, h, w)
                        corner = line_intersection(extended_top, right_line)
                        if corner:
                            corner_positions['tr'] = corner
                    elif 'tl' in corner_positions:
                        tl = corner_positions['tl']
                        if 'br' in corner_positions:
                            br = corner_positions['br']
                            corner_positions['tr'] = (br[0], tl[1])
                        else:
                            corner_positions['tr'] = (w, tl[1])
                
                if bottom_line and 'bl' not in corner_positions:
                    if left_line:
                        extended_bottom = extend_line_to_boundary(bottom_line, h, w)
                        corner = line_intersection(extended_bottom, left_line)
                        if corner:
                            corner_positions['bl'] = corner
                    elif 'br' in corner_positions:
                        br = corner_positions['br']
                        if 'tl' in corner_positions:
                            tl = corner_positions['tl']
                            corner_positions['bl'] = (tl[0], br[1])
                        else:
                            corner_positions['bl'] = (0, br[1])
                
                if bottom_line and 'br' not in corner_positions:
                    if right_line:
                        extended_bottom = extend_line_to_boundary(bottom_line, h, w)
                        corner = line_intersection(extended_bottom, right_line)
                        if corner:
                            corner_positions['br'] = corner
                    elif 'bl' in corner_positions:
                        bl = corner_positions['bl']
                        if 'tr' in corner_positions:
                            tr = corner_positions['tr']
                            corner_positions['br'] = (tr[0], bl[1])
                        else:
                            corner_positions['br'] = (w, bl[1])
                
                # If we still don't have 4 corners, use image boundaries
                if len(corner_positions) < 4:
                    # Use detected corners and image boundaries to construct missing ones
                    if 'tl' not in corner_positions:
                        if 'tr' in corner_positions and 'bl' in corner_positions:
                            corner_positions['tl'] = (corner_positions['bl'][0], corner_positions['tr'][1])
                        elif 'tr' in corner_positions:
                            corner_positions['tl'] = (0, corner_positions['tr'][1])
                        elif 'bl' in corner_positions:
                            corner_positions['tl'] = (corner_positions['bl'][0], 0)
                        else:
                            corner_positions['tl'] = (0, 0)
                    
                    if 'tr' not in corner_positions:
                        if 'tl' in corner_positions and 'br' in corner_positions:
                            corner_positions['tr'] = (corner_positions['br'][0], corner_positions['tl'][1])
                        elif 'tl' in corner_positions:
                            corner_positions['tr'] = (w, corner_positions['tl'][1])
                        elif 'br' in corner_positions:
                            corner_positions['tr'] = (corner_positions['br'][0], 0)
                        else:
                            corner_positions['tr'] = (w, 0)
                    
                    if 'bl' not in corner_positions:
                        if 'tl' in corner_positions and 'br' in corner_positions:
                            corner_positions['bl'] = (corner_positions['tl'][0], corner_positions['br'][1])
                        elif 'tl' in corner_positions:
                            corner_positions['bl'] = (corner_positions['tl'][0], h)
                        elif 'br' in corner_positions:
                            corner_positions['bl'] = (0, corner_positions['br'][1])
                        else:
                            corner_positions['bl'] = (0, h)
                    
                    if 'br' not in corner_positions:
                        if 'tr' in corner_positions and 'bl' in corner_positions:
                            corner_positions['br'] = (corner_positions['tr'][0], corner_positions['bl'][1])
                        elif 'tr' in corner_positions:
                            corner_positions['br'] = (corner_positions['tr'][0], h)
                        elif 'bl' in corner_positions:
                            corner_positions['br'] = (w, corner_positions['bl'][1])
                        else:
                            corner_positions['br'] = (w, h)
                
                # Ensure we have exactly 4 corners in correct order: tl, tr, br, bl
                if len(corner_positions) == 4:
                    quad_pts = np.array([
                        corner_positions['tl'],
                        corner_positions['tr'],
                        corner_positions['br'],
                        corner_positions['bl']
                    ], dtype=np.float32)
                    
                    shape_type, angles = identify_shape(quad_pts)
                    
                    if self.debug:
                        # Draw detected lines and corners
                        debug_img = cropped.copy()
                        if top_line:
                            cv2.line(debug_img, (int(top_line[0]), int(top_line[1])), 
                                    (int(top_line[2]), int(top_line[3])), (0, 255, 0), 2)
                        if bottom_line:
                            cv2.line(debug_img, (int(bottom_line[0]), int(bottom_line[1])), 
                                    (int(bottom_line[2]), int(bottom_line[3])), (0, 255, 0), 2)
                        if left_line:
                            cv2.line(debug_img, (int(left_line[0]), int(left_line[1])), 
                                    (int(left_line[2]), int(left_line[3])), (255, 0, 0), 2)
                        if right_line:
                            cv2.line(debug_img, (int(right_line[0]), int(right_line[1])), 
                                    (int(right_line[2]), int(right_line[3])), (255, 0, 0), 2)
                        for i, (cx, cy) in enumerate(quad_pts):
                            cv2.circle(debug_img, (int(cx), int(cy)), 8, (0, 0, 255), -1)
                            cv2.putText(debug_img, str(i), (int(cx), int(cy)), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        self.save_debug_image(debug_img, f"{img_name}_08i_line_masks_detection.jpg", 
                                             f"Line masks detection ({target})")
                    
                    return quad_pts, shape_type, angles
            
            return None, None, None
        
        # Helper function to detect quadrilateral using edge detection and Hough lines
        def detect_quad_from_edges(image_gray, edges_image=None, target="paper"):
            """Detect quadrilateral using Canny edges and Hough lines
            Enhanced to filter peripheral noise and find main 4 boundary lines
            
            Args:
                image_gray: Grayscale image
                edges_image: Pre-computed edges (if None, will compute from image_gray)
                target: "paper" or "table"
            """
            h, w = image_gray.shape
            
            # Use provided edges or compute new ones
            if edges_image is not None:
                # Use the paper-focused edges (already computed and filtered)
                working_edges = edges_image.copy()
            else:
                # Fallback: compute edges from image with multiple thresholds
                blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
                edges_low = cv2.Canny(blur, 30, 100)
                edges_medium = cv2.Canny(blur, 50, 150)
                edges_high = cv2.Canny(blur, 100, 200)
                working_edges = cv2.bitwise_or(edges_low, cv2.bitwise_or(edges_medium, edges_high))
            
            # Dilate edges slightly to connect broken lines
            kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            working_edges = cv2.dilate(working_edges, kernel_edge, iterations=1)
            
            if self.debug:
                self.save_debug_image(working_edges, f"{img_name}_08e_working_edges.jpg", f"Working edges for {target}")
            
            # Minimum line length (at least 15% of image dimension for boundary lines)
            min_line_length = int(max(h, w) * 0.15)
            
            # HoughLinesP to detect line segments from the working edges
            all_lines = []
            lines = cv2.HoughLinesP(
                working_edges,
                rho=1,
                theta=np.pi/180,
                threshold=50,
                minLineLength=min_line_length,
                maxLineGap=30
            )
            
            if lines is not None:
                all_lines.extend(lines[:, 0].tolist())
            
            # Filter lines: prioritize long boundary lines, filter short peripheral noise
            filtered_lines = []
            edge_threshold = 0.05  # 5% from edge
            min_dist_from_edge = min(h, w) * edge_threshold
            
            for line in all_lines:
                x1, y1, x2, y2 = line
                length = np.hypot(x2 - x1, y2 - y1)
                
                # Check if line is near image boundary (likely a boundary line)
                near_top = min(y1, y2) < min_dist_from_edge
                near_bottom = max(y1, y2) > (h - min_dist_from_edge)
                near_left = min(x1, x2) < min_dist_from_edge
                near_right = max(x1, x2) > (w - min_dist_from_edge)
                
                is_boundary_candidate = (near_top or near_bottom or near_left or near_right)
                
                # Keep if: long line OR boundary candidate with reasonable length
                if length >= min_line_length * 0.8 or (is_boundary_candidate and length >= min_line_length * 0.5):
                    filtered_lines.append(line)
            
            all_lines = filtered_lines
            
            # Also try with different thresholds if we have few lines
            if len(all_lines) < 4:
                # Try with lower threshold to get more lines
                lines2 = cv2.HoughLinesP(
                    working_edges,
                    rho=1,
                    theta=np.pi/180,
                    threshold=30,  # Lower threshold
                    minLineLength=int(min_line_length * 0.7),  # Shorter lines
                    maxLineGap=30
                )
                if lines2 is not None:
                    all_lines.extend(lines2[:, 0].tolist())
            
            if not all_lines:
                return None, None, None
            
            # Group lines by orientation (horizontal, vertical, diagonal)
            horizontal_lines = []
            vertical_lines = []
            diagonal_lines = []
            
            for x1, y1, x2, y2 in all_lines:
                dx = x2 - x1
                dy = y2 - y1
                length = np.hypot(dx, dy)
                if length < min_line_length:
                    continue
                
                angle = abs(np.degrees(np.arctan2(dy, dx)))
                
                # Normalize angle to 0-90 range
                if angle > 90:
                    angle = 180 - angle
                
                if angle < 15:  # Near horizontal
                    horizontal_lines.append((x1, y1, x2, y2))
                elif angle > 75:  # Near vertical
                    vertical_lines.append((x1, y1, x2, y2))
                else:  # Diagonal (important for trapezoids)
                    diagonal_lines.append((x1, y1, x2, y2))
            
            # Find the four boundary lines
            def find_boundary_lines(lines, orientation):
                """Find the outermost lines in a given orientation"""
                if not lines:
                    return None
                
                if orientation == 'horizontal':
                    # Find topmost and bottommost lines
                    lines_sorted = sorted(lines, key=lambda l: min(l[1], l[3]))
                    top_line = lines_sorted[0]
                    lines_sorted = sorted(lines, key=lambda l: max(l[1], l[3]), reverse=True)
                    bottom_line = lines_sorted[0]
                    return top_line, bottom_line
                elif orientation == 'vertical':
                    # Find leftmost and rightmost lines
                    lines_sorted = sorted(lines, key=lambda l: min(l[0], l[2]))
                    left_line = lines_sorted[0]
                    lines_sorted = sorted(lines, key=lambda l: max(l[0], l[2]), reverse=True)
                    right_line = lines_sorted[0]
                    return left_line, right_line
                return None
            
            # Try to find 4 boundary lines
            # For trapezoids, we need diagonal lines too
            top_line = bottom_line = left_line = right_line = None
            
            if horizontal_lines:
                top_line, bottom_line = find_boundary_lines(horizontal_lines, 'horizontal')
            if vertical_lines:
                left_line, right_line = find_boundary_lines(vertical_lines, 'vertical')
            
            # If we don't have 4 lines, try using diagonal lines
            if (top_line is None or bottom_line is None) and diagonal_lines:
                # Find topmost and bottommost diagonal lines
                diag_sorted = sorted(diagonal_lines, key=lambda l: min(l[1], l[3]))
                if top_line is None:
                    top_line = diag_sorted[0]
                diag_sorted = sorted(diagonal_lines, key=lambda l: max(l[1], l[3]), reverse=True)
                if bottom_line is None:
                    bottom_line = diag_sorted[0]
            
            if (left_line is None or right_line is None) and diagonal_lines:
                # Find leftmost and rightmost diagonal lines
                diag_sorted = sorted(diagonal_lines, key=lambda l: min(l[0], l[2]))
                if left_line is None:
                    left_line = diag_sorted[0]
                diag_sorted = sorted(diagonal_lines, key=lambda l: max(l[0], l[2]), reverse=True)
                if right_line is None:
                    right_line = diag_sorted[0]
            
            # Calculate intersection points
            def line_intersection(line1, line2):
                """Find intersection point of two line segments"""
                x1, y1, x2, y2 = line1
                x3, y3, x4, y4 = line2
                
                # Convert to line equations: ax + by = c
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-10:
                    return None
                
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
                
                # Check if intersection is within both line segments
                if 0 <= t <= 1 and 0 <= u <= 1:
                    px = x1 + t * (x2 - x1)
                    py = y1 + t * (y2 - y1)
                    return (px, py)
                return None
            
            # Try to find 4 corners - construct missing corners using table structure assumptions
            corners = []
            corner_positions = {}  # Store corners by position: 'tl', 'tr', 'bl', 'br'
            
            # Find intersections where lines exist
            if top_line and left_line:
                corner = line_intersection(top_line, left_line)
                if corner:
                    corners.append(corner)
                    corner_positions['tl'] = corner
            if top_line and right_line:
                corner = line_intersection(top_line, right_line)
                if corner:
                    corners.append(corner)
                    corner_positions['tr'] = corner
            if bottom_line and left_line:
                corner = line_intersection(bottom_line, left_line)
                if corner:
                    corners.append(corner)
                    corner_positions['bl'] = corner
            if bottom_line and right_line:
                corner = line_intersection(bottom_line, right_line)
                if corner:
                    corners.append(corner)
                    corner_positions['br'] = corner
            
            # If we have at least 2 corners, construct the missing ones using table structure
            if len(corners) >= 2:
                # Extend lines to image boundaries to find missing corners
                def extend_line_to_boundary(line, h, w):
                    """Extend line to image boundaries"""
                    x1, y1, x2, y2 = line
                    dx = x2 - x1
                    dy = y2 - y1
                    if abs(dx) < 1e-6:  # Vertical line
                        return (x1, 0, x1, h)
                    if abs(dy) < 1e-6:  # Horizontal line
                        return (0, y1, w, y1)
                    
                    # Extend line to boundaries
                    t_values = []
                    if dy != 0:
                        t = -y1 / dy
                        x = x1 + t * dx
                        if 0 <= x <= w:
                            t_values.append((t, x, 0))
                        t = (h - y1) / dy
                        x = x1 + t * dx
                        if 0 <= x <= w:
                            t_values.append((t, x, h))
                    if dx != 0:
                        t = -x1 / dx
                        y = y1 + t * dy
                        if 0 <= y <= h:
                            t_values.append((t, 0, y))
                        t = (w - x1) / dx
                        y = y1 + t * dy
                        if 0 <= y <= h:
                            t_values.append((t, w, y))
                    
                    if len(t_values) >= 2:
                        t_values.sort(key=lambda x: x[0])
                        return (int(t_values[0][1]), int(t_values[0][2]), 
                                int(t_values[-1][1]), int(t_values[-1][2]))
                    return line
                
                # Construct missing corners (same logic as line_masks function)
                if top_line and 'tl' not in corner_positions:
                    if left_line:
                        extended_top = extend_line_to_boundary(top_line, h, w)
                        corner = line_intersection(extended_top, left_line)
                        if corner:
                            corner_positions['tl'] = corner
                    elif 'tr' in corner_positions:
                        tr = corner_positions['tr']
                        if 'bl' in corner_positions:
                            bl = corner_positions['bl']
                            corner_positions['tl'] = (bl[0], tr[1])
                        else:
                            corner_positions['tl'] = (0, tr[1])
                
                if top_line and 'tr' not in corner_positions:
                    if right_line:
                        extended_top = extend_line_to_boundary(top_line, h, w)
                        corner = line_intersection(extended_top, right_line)
                        if corner:
                            corner_positions['tr'] = corner
                    elif 'tl' in corner_positions:
                        tl = corner_positions['tl']
                        if 'br' in corner_positions:
                            br = corner_positions['br']
                            corner_positions['tr'] = (br[0], tl[1])
                        else:
                            corner_positions['tr'] = (w, tl[1])
                
                if bottom_line and 'bl' not in corner_positions:
                    if left_line:
                        extended_bottom = extend_line_to_boundary(bottom_line, h, w)
                        corner = line_intersection(extended_bottom, left_line)
                        if corner:
                            corner_positions['bl'] = corner
                    elif 'br' in corner_positions:
                        br = corner_positions['br']
                        if 'tl' in corner_positions:
                            tl = corner_positions['tl']
                            corner_positions['bl'] = (tl[0], br[1])
                        else:
                            corner_positions['bl'] = (0, br[1])
                
                if bottom_line and 'br' not in corner_positions:
                    if right_line:
                        extended_bottom = extend_line_to_boundary(bottom_line, h, w)
                        corner = line_intersection(extended_bottom, right_line)
                        if corner:
                            corner_positions['br'] = corner
                    elif 'bl' in corner_positions:
                        bl = corner_positions['bl']
                        if 'tr' in corner_positions:
                            tr = corner_positions['tr']
                            corner_positions['br'] = (tr[0], bl[1])
                        else:
                            corner_positions['br'] = (w, bl[1])
                
                # If we still don't have 4 corners, use image boundaries
                if len(corner_positions) < 4:
                    if 'tl' not in corner_positions:
                        if 'tr' in corner_positions and 'bl' in corner_positions:
                            corner_positions['tl'] = (corner_positions['bl'][0], corner_positions['tr'][1])
                        elif 'tr' in corner_positions:
                            corner_positions['tl'] = (0, corner_positions['tr'][1])
                        elif 'bl' in corner_positions:
                            corner_positions['tl'] = (corner_positions['bl'][0], 0)
                        else:
                            corner_positions['tl'] = (0, 0)
                    
                    if 'tr' not in corner_positions:
                        if 'tl' in corner_positions and 'br' in corner_positions:
                            corner_positions['tr'] = (corner_positions['br'][0], corner_positions['tl'][1])
                        elif 'tl' in corner_positions:
                            corner_positions['tr'] = (w, corner_positions['tl'][1])
                        elif 'br' in corner_positions:
                            corner_positions['tr'] = (corner_positions['br'][0], 0)
                        else:
                            corner_positions['tr'] = (w, 0)
                    
                    if 'bl' not in corner_positions:
                        if 'tl' in corner_positions and 'br' in corner_positions:
                            corner_positions['bl'] = (corner_positions['tl'][0], corner_positions['br'][1])
                        elif 'tl' in corner_positions:
                            corner_positions['bl'] = (corner_positions['tl'][0], h)
                        elif 'br' in corner_positions:
                            corner_positions['bl'] = (0, corner_positions['br'][1])
                        else:
                            corner_positions['bl'] = (0, h)
                    
                    if 'br' not in corner_positions:
                        if 'tr' in corner_positions and 'bl' in corner_positions:
                            corner_positions['br'] = (corner_positions['tr'][0], corner_positions['bl'][1])
                        elif 'tr' in corner_positions:
                            corner_positions['br'] = (corner_positions['tr'][0], h)
                        elif 'bl' in corner_positions:
                            corner_positions['br'] = (w, corner_positions['bl'][1])
                        else:
                            corner_positions['br'] = (w, h)
                
                # Ensure we have exactly 4 corners in correct order: tl, tr, br, bl
                if len(corner_positions) == 4:
                    quad_pts = np.array([
                        corner_positions['tl'],
                        corner_positions['tr'],
                        corner_positions['br'],
                        corner_positions['bl']
                    ], dtype=np.float32)
                    
                    shape_type, angles = identify_shape(quad_pts)
                    
                    if self.debug:
                        # Draw detected lines and corners
                        debug_img = cropped.copy()
                        if top_line:
                            cv2.line(debug_img, (int(top_line[0]), int(top_line[1])), 
                                    (int(top_line[2]), int(top_line[3])), (0, 255, 0), 2)
                        if bottom_line:
                            cv2.line(debug_img, (int(bottom_line[0]), int(bottom_line[1])), 
                                    (int(bottom_line[2]), int(bottom_line[3])), (0, 255, 0), 2)
                        if left_line:
                            cv2.line(debug_img, (int(left_line[0]), int(left_line[1])), 
                                    (int(left_line[2]), int(left_line[3])), (255, 0, 0), 2)
                        if right_line:
                            cv2.line(debug_img, (int(right_line[0]), int(right_line[1])), 
                                    (int(right_line[2]), int(right_line[3])), (255, 0, 0), 2)
                        for i, (cx, cy) in enumerate(quad_pts):
                            cv2.circle(debug_img, (int(cx), int(cy)), 8, (0, 0, 255), -1)
                            cv2.putText(debug_img, str(i), (int(cx), int(cy)), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        self.save_debug_image(debug_img, f"{img_name}_08e_edge_detection.jpg", 
                                             f"Edge detection ({target})")
                    
                    return quad_pts, shape_type, angles
            
            return None, None, None
        
        # Helper function using corner detection
        def detect_quad_from_corners(image_gray, target="paper"):
            """Detect quadrilateral using corner detection (Harris/Shi-Tomasi)"""
            # Detect corners
            corners = cv2.goodFeaturesToTrack(
                image_gray,
                maxCorners=100,
                qualityLevel=0.01,
                minDistance=30,
                blockSize=3,
                useHarrisDetector=False
            )
            
            if corners is None or len(corners) < 4:
                return None, None, None
            
            corners = corners.reshape(-1, 2)
            
            # Find the 4 outermost corners
            # Top-left: min(x+y)
            # Top-right: max(x-y) or min(y-x)
            # Bottom-right: max(x+y)
            # Bottom-left: min(x-y) or max(y-x)
            
            sums = corners[:, 0] + corners[:, 1]
            diffs = corners[:, 0] - corners[:, 1]
            
            tl_idx = np.argmin(sums)
            br_idx = np.argmax(sums)
            tr_idx = np.argmax(diffs)
            bl_idx = np.argmin(diffs)
            
            quad_pts = np.array([
                corners[tl_idx],
                corners[tr_idx],
                corners[br_idx],
                corners[bl_idx]
            ], dtype=np.float32)
            
            # Validate quad
            rect = _order_points(quad_pts)
            widthA = np.linalg.norm(rect[2] - rect[3])
            widthB = np.linalg.norm(rect[1] - rect[0])
            heightA = np.linalg.norm(rect[1] - rect[2])
            heightB = np.linalg.norm(rect[0] - rect[3])
            max_w = max(widthA, widthB)
            max_h = max(heightA, heightB)
            
            bbox_area = w * h
            area = cv2.contourArea(quad_pts)
            area_ratio = area / bbox_area if bbox_area > 0 else 0
            
            if max_w > 50 and max_h > 50 and area_ratio > 0.4:
                shape_type, angles = identify_shape(quad_pts)
                
                if self.debug:
                    debug_img = cropped.copy()
                    for i, pt in enumerate(quad_pts):
                        cv2.circle(debug_img, tuple(pt.astype(int)), 8, (0, 255, 0), -1)
                        cv2.putText(debug_img, str(i), tuple(pt.astype(int)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    self.save_debug_image(debug_img, f"{img_name}_08g_corner_detection.jpg", 
                                         f"Corner detection ({target})")
                
                return quad_pts, shape_type, angles
            
            return None, None, None
        
        # Stage 1: Try to detect PAGE/DOCUMENT boundary first (assume rectangle if page detected)
        print("  Stage 1: Detecting page/document boundary using cleaned edges...")
        
        # Initialize variables
        quad = None
        shape_type = None
        angles = None
        page_detected = False
        
        # Use contour detection from cleaned edges (noise-removed edges)
        print("    Contour detection from cleaned edges (noise removed)...")
        quad, shape_type, angles = detect_quad_from_cleaned_edges(edges_combined, target="paper")
        
        if quad is not None:
            page_detected = True
            print(f"  ✓ Found page/document boundary: {shape_type} (angles: {[f'{a:.1f}°' for a in angles]})")
            shape_type, angles = identify_shape(quad)
            if shape_type != "rectangle":
                print(f"    Assuming rectangle shape for page detection")
                shape_type = "rectangle"
        
        if not page_detected:
            # Stage 2: Page not detected, try to detect TABLE boundary
            print("  Stage 2: Page detection failed, trying table boundary using cleaned edges...")
            
            # Use contour detection from cleaned edges
            print("    Contour detection from cleaned edges (noise removed)...")
            quad, shape_type, angles = detect_quad_from_cleaned_edges(edges_combined, target="table")
            
            if quad is not None:
                print(f"  ✓ Found table boundary: {shape_type} (angles: {[f'{a:.1f}°' for a in angles]})")
            else:
                print("  ✗ No valid quadrilateral found (neither page nor table)")
        
        if quad is not None:
            if self.debug:
                # Draw the detected quad on the cropped image
                debug_img = cropped.copy()
                rect_ordered = _order_points(quad)
                quad_int = rect_ordered.astype(np.int32)
                cv2.drawContours(debug_img, [quad_int], -1, (0, 255, 0), 3)
                for i, pt in enumerate(rect_ordered):
                    pt_int = tuple(pt.astype(np.int32))
                    cv2.circle(debug_img, pt_int, 8, (255, 0, 0), -1)
                    label = f"{i} ({shape_type[0].upper()})"
                    cv2.putText(debug_img, label, pt_int, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                self.save_debug_image(debug_img, f"{img_name}_08d_detected_quad.jpg", 
                                      f"Detected {shape_type} boundary")

        # ============================================================
        # STEP 1: DESKEWING (Skew Correction)
        # Apply to the cropped image first
        # Use cleaned edge detection approach like in test_backup.txt
        # ============================================================
        print("\n  Applying deskewing (skew correction) to cropped image...")
        working = cropped
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise before edge detection
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Create cleaned Canny edges (same approach as contour detection)
        canny_low = 50
        canny_high = 150
        clean_edges = cv2.Canny(blur, canny_low, canny_high)
        
        # Clean up noise - remove small isolated components
        h_img, w_img = working.shape[:2]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_edges, connectivity=8)
        min_component_area = (h_img * w_img) * 0.001  # Remove components smaller than 0.1% of image
        cleaned_edges = np.zeros_like(clean_edges)
        
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_component_area:
                cleaned_edges[labels == label] = 255
        
        # Close small gaps in edges
        kernel_close_edges = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed_edges = cv2.morphologyEx(cleaned_edges, cv2.MORPH_CLOSE, kernel_close_edges, iterations=1)
        
        if self.debug:
            self.save_debug_image(clean_edges, f"{img_name}_09a_clean_edges.jpg", "Cleaned Canny edges for skew detection")
            self.save_debug_image(cleaned_edges, f"{img_name}_09b_noise_removed.jpg", "Noise removed from edges")
            self.save_debug_image(closed_edges, f"{img_name}_09c_edges_closed.jpg", "Edges after closing gaps")
        
        # Use cleaned edges for Hough line detection
        # Hough lines for dominant angle
        lines = cv2.HoughLines(closed_edges, 1, np.pi / 180, max(100, int(min(working.shape[:2]) / 2)))

        angles = []
        if lines is not None:
            # iterate up to first 100 lines (if available)
            for rho, theta in lines[:100, 0]:
                angle = np.degrees(theta) - 90
                # normalize angle to [-90, 90]
                if angle > 90:
                    angle -= 180
                if angle < -90:
                    angle += 180
                # accept moderate skews only
                if abs(angle) <= 15:
                    angles.append(angle)

        if angles:
            median_angle = np.median(angles)
            print(f"  Detected skew angle: {median_angle:.2f}°")
        else:
            median_angle = 0
            print(f"  No skew detected")

        # Rotate if needed
        if abs(median_angle) > 0.25:
            center = (w_img // 2, h_img // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

            # compute new bounding dims to avoid cropping during rotation
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h_img * sin) + (w_img * cos))
            new_h = int((h_img * cos) + (w_img * sin))

            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]

            deskewed = cv2.warpAffine(
                working, M, (new_w, new_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            print(f"  ✓ Deskewing complete: Applied rotation of {median_angle:.2f}°")
            print(f"    Output size: {new_w}x{new_h}")
            if self.debug:
                self.save_debug_image(deskewed, f"{img_name}_09_deskewed.jpg", "Deskewed image")
            deskewed_image = deskewed
        else:
            print("  ✓ No deskewing needed (skew angle < 0.25°)")
            deskewed_image = working

        # ============================================================
        # STEP 2: PERSPECTIVE CORRECTION
        # Re-detect quad from deskewed image for better accuracy
        # ============================================================
        warped = None
        # Add generous padding to deskewed image before quad detection to prevent edge cutoff
        # Use 2% of image dimensions with higher minimum to ensure edges are fully captured
        h_deskewed_orig, w_deskewed_orig = deskewed_image.shape[:2]
        padding_percent_deskewed = 0.04  # Increase padding to keep full table context
        padding_w_deskewed = max(60, int(w_deskewed_orig * padding_percent_deskewed))  # Min 60px
        padding_h_deskewed = max(60, int(h_deskewed_orig * padding_percent_deskewed))  # Min 60px
        padding_deskewed = max(padding_w_deskewed, padding_h_deskewed)
        
        deskewed_padded = cv2.copyMakeBorder(
            deskewed_image,
            padding_deskewed, padding_deskewed, padding_deskewed, padding_deskewed,
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        print(f"  Added padding to deskewed image: {padding_deskewed}px (2% of {w_deskewed_orig}x{h_deskewed_orig}, min 50px)")
        
        # Re-detect quad from padded deskewed image
        print("  Re-detecting quadrilateral from padded deskewed image...")
        deskewed_gray = cv2.cvtColor(deskewed_padded, cv2.COLOR_BGR2GRAY)
        deskewed_blur = cv2.GaussianBlur(deskewed_gray, (5, 5), 0)
        deskewed_edges = cv2.Canny(deskewed_blur, 50, 150)
        
        # Clean up noise in edges
        h_deskewed, w_deskewed = deskewed_padded.shape[:2]
        num_labels_d, labels_d, stats_d, centroids_d = cv2.connectedComponentsWithStats(deskewed_edges, connectivity=8)
        min_component_area_d = (h_deskewed * w_deskewed) * 0.001
        cleaned_edges_d = np.zeros_like(deskewed_edges)
        for label in range(1, num_labels_d):
            area = stats_d[label, cv2.CC_STAT_AREA]
            if area >= min_component_area_d:
                cleaned_edges_d[labels_d == label] = 255
        
        # Close small gaps
        kernel_close_d = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed_edges_d = cv2.morphologyEx(cleaned_edges_d, cv2.MORPH_CLOSE, kernel_close_d, iterations=1)
        
        # Try to detect quad from deskewed image using the same detection function
        # We need to call detect_quad_from_cleaned_edges with the deskewed edges
        # But that function is defined inside detect_table_boundary, so we'll do a simpler detection here
        contours_d, _ = cv2.findContours(closed_edges_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        quad_deskewed = None
        
        if contours_d:
            # Find largest contour
            largest_contour = max(contours_d, key=cv2.contourArea)
            peri = cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
            
            if len(approx) >= 4:
                # Use convex hull if needed
                if len(approx) > 4:
                    hull = cv2.convexHull(approx)
                    peri_hull = cv2.arcLength(hull, True)
                    for eps_factor in [0.05, 0.1, 0.15, 0.2]:
                        approx_hull = cv2.approxPolyDP(hull, eps_factor * peri_hull, True)
                        if len(approx_hull) == 4:
                            quad_deskewed = approx_hull.reshape(4, 2).astype(np.float32)
                            break
                else:
                    quad_deskewed = approx.reshape(4, 2).astype(np.float32)
        
        # Use re-detected quad if available, otherwise fall back to transformed original quad
        if quad_deskewed is not None:
            print("  ✓ Quad re-detected from padded deskewed image")
            # Quad coordinates are in padded image space, adjust them back to original deskewed image space
            # by subtracting the padding offset
            quad_deskewed_adjusted = quad_deskewed.copy()
            quad_deskewed_adjusted[:, 0] -= padding_deskewed  # Adjust x coordinates
            quad_deskewed_adjusted[:, 1] -= padding_deskewed  # Adjust y coordinates
            
            # Ensure coordinates are within valid bounds (not negative, not beyond image)
            quad_deskewed_adjusted[:, 0] = np.clip(quad_deskewed_adjusted[:, 0], 0, w_deskewed_orig - 1)
            quad_deskewed_adjusted[:, 1] = np.clip(quad_deskewed_adjusted[:, 1], 0, h_deskewed_orig - 1)
            
            quad = quad_deskewed_adjusted
            print(f"    Adjusted quad coordinates to original image space (bounds: 0-{w_deskewed_orig-1}, 0-{h_deskewed_orig-1})")
            if self.debug:
                debug_img = deskewed_padded.copy()
                rect_ordered = _order_points(quad_deskewed)
                quad_int = rect_ordered.astype(np.int32)
                cv2.drawContours(debug_img, [quad_int], -1, (0, 255, 0), 3)
                for i, pt in enumerate(rect_ordered):
                    pt_int = tuple(pt.astype(np.int32))
                    cv2.circle(debug_img, pt_int, 8, (255, 0, 0), -1)
                    cv2.putText(debug_img, str(i), pt_int, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                self.save_debug_image(debug_img, f"{img_name}_08d_quad_deskewed.jpg", 
                                      "Quad re-detected from padded deskewed image")
        elif quad is not None:
            print("  ⚠ Using original quad (transformed to deskewed space)")
            # Transform quad coordinates from cropped image space to deskewed image space
            if abs(median_angle) > 0.25:
                h_img, w_img = cropped.shape[:2]
                center = (w_img // 2, h_img // 2)
                M_rot = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                cos = np.abs(M_rot[0, 0])
                sin = np.abs(M_rot[0, 1])
                new_w = int((h_img * sin) + (w_img * cos))
                new_h = int((h_img * cos) + (w_img * sin))
                M_rot[0, 2] += (new_w / 2) - center[0]
                M_rot[1, 2] += (new_h / 2) - center[1]
                quad_reshaped = quad.reshape(-1, 1, 2).astype(np.float32)
                quad_transformed = cv2.transform(quad_reshaped, M_rot)
                quad = quad_transformed.reshape(-1, 2).astype(np.float32)
        
        if quad is not None:
            print("  Applying perspective correction to deskewed image...")
            expand_px = np.clip(int(0.005 * max(w_deskewed_orig, h_deskewed_orig)), 5, 10)
            expanded_quad = _expand_quad(quad, expand_px, deskewed_image.shape)
            rect_source = expanded_quad if expanded_quad is not None else quad
            rect = _order_points(rect_source)
            (tl, tr, br, bl) = rect

            # compute dimensions of perspective-corrected image
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxWidth = int(max(widthA, widthB))
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxHeight = int(max(heightA, heightB))

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(
                deskewed_image, M, (maxWidth, maxHeight),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

            # keep the small padding around the warped image to avoid edges flush with border
            pad_debug = max(25, int(max(maxWidth, maxHeight) * 0.02))
            warped_padded = cv2.copyMakeBorder(
                warped, pad_debug, pad_debug, pad_debug, pad_debug,
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )

            print(f"  ✓ Perspective correction complete: {warped_padded.shape[1]}x{warped_padded.shape[0]}")
            if self.debug:
                # same debug filename as you requested: _08a_perspective_corrected
                self.save_debug_image(
                    warped_padded,
                    f"{img_name}_08a_perspective_corrected.jpg",
                    "Perspective corrected (quadrilateral) — padded"
                )

            final_image = warped_padded
        else:
            # If no quad found, use the deskewed image but still add a small pad so edges aren't flush
            print("  ⚠ No quadrilateral detected - skipping perspective correction")
            print("  Using deskewed image (no perspective correction)")
            pad_debug = max(25, int(max(deskewed_image.shape[:2]) * 0.02))
            deskewed_padded = cv2.copyMakeBorder(
                deskewed_image, pad_debug, pad_debug, pad_debug, pad_debug,
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
            final_image = deskewed_padded

            if self.debug:
                # same debug filename used here too
                self.save_debug_image(
                    final_image,
                    f"{img_name}_08a_perspective_corrected.jpg",
                    "No quad found — using deskewed image (padded)"
                )
        
        # Update size_profile to reflect the cropped/deskewed image dimensions
        # This ensures all subsequent adaptive detection uses the correct image size
        # analyze_image_scale will recalculate category and potentially resize if needed
        final_image = self.analyze_image_scale(final_image)
        
        if self.debug:
            print(f"  Updated size_profile after cropping/deskewing: "
                  f"{self.size_profile['working_shape'][1]}x{self.size_profile['working_shape'][0]} "
                  f"(category: {self.size_profile['category']})")
        
        return final_image
    
    def verify_table_shape(self, image, img_name):
        """
        Step 3: Verify the table has the expected SHG form structure
        """
        print("\n" + "="*70)
        print("STEP 3: VERIFYING TABLE SHAPE")
        print("="*70)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        h, w = binary.shape

        # Bridge small gaps first so broken lines become continuous
        kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bridged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_bridge, iterations=2)
        if self.debug:
            self.save_debug_image(bridged, f"{img_name}_03b_bridged.jpg", "Bridged small gaps")

        # ----- Horizontal line detection (morphological multi-scale) -----
        h_counts_positions = []
        h_masks = []
        for scale in [60, 40, 30]:
            kx = max(3, w // scale)
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
            h_mask = cv2.morphologyEx(bridged, cv2.MORPH_OPEN, h_kernel, iterations=1)
            # close tiny gaps in the detected horizontal lines
            h_mask = cv2.morphologyEx(h_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)), iterations=1)
            h_masks.append(h_mask)
            # connected components to get positions
            num, labels, stats, cents = cv2.connectedComponentsWithStats(h_mask, connectivity=8)
            for label in range(1, num):
                width = stats[label, cv2.CC_STAT_WIDTH]
                y_pos = int(cents[label][1])
                if width > w * 0.25:
                    h_counts_positions.append(y_pos)
        if self.debug:
            combined_h = h_masks[0].copy()
            for m in h_masks[1:]:
                combined_h = cv2.bitwise_or(combined_h, m)
            self.save_debug_image(combined_h, f"{img_name}_03a_horiz_lines_multi.jpg", "Multi-scale horizontal lines")

        # Cluster horizontal positions
        h_positions = self._cluster_values(h_counts_positions, tolerance=max(4, h // 200)) if h_counts_positions else []

        # ----- Vertical line detection (morphological multi-scale) -----
        v_counts_positions = []
        v_masks = []
        for scale in [60, 40, 30]:
            ky = max(3, h // scale)
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky))
            v_mask = cv2.morphologyEx(bridged, cv2.MORPH_OPEN, v_kernel, iterations=1)
            v_mask = cv2.morphologyEx(v_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)), iterations=1)
            v_masks.append(v_mask)
            num, labels, stats, cents = cv2.connectedComponentsWithStats(v_mask, connectivity=8)
            for label in range(1, num):
                height = stats[label, cv2.CC_STAT_HEIGHT]
                x_pos = int(cents[label][0])
                if height > h * 0.18:
                    v_counts_positions.append(x_pos)
        if self.debug:
            combined_v = v_masks[0].copy()
            for m in v_masks[1:]:
                combined_v = cv2.bitwise_or(combined_v, m)
            self.save_debug_image(combined_v, f"{img_name}_03b_vert_lines_multi.jpg", "Multi-scale vertical lines")

        # Cluster vertical positions
        v_positions = self._cluster_values(v_counts_positions, tolerance=max(4, w // 200)) if v_counts_positions else []

        # ----- Hough fallback for missing verticals/horizontals -----
        edges = cv2.Canny(bridged, 50, 150)
        # Hough P to detect line segments
        min_line_len = int(max(h, w) * 0.25)
        linesP = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=min_line_len, maxLineGap=20)
        hough_h = []
        hough_v = []
        if linesP is not None:
            for x1, y1, x2, y2 in linesP[:, 0]:
                dx = x2 - x1
                dy = y2 - y1
                length = np.hypot(dx, dy)
                if length < min_line_len:
                    continue
                angle = abs(np.degrees(np.arctan2(dy, dx)))
                if angle < 10:  # near-horizontal
                    hough_h.append(int((y1 + y2) / 2))
                elif angle > 80:  # near-vertical
                    hough_v.append(int((x1 + x2) / 2))
        # Cluster Hough results
        h_hough_positions = self._cluster_values(hough_h, tolerance=max(4, h // 200)) if hough_h else []
        v_hough_positions = self._cluster_values(hough_v, tolerance=max(4, w // 200)) if hough_v else []

        # Merge morphological and Hough detections (union)
        final_h_positions = sorted(set(h_positions) | set(h_hough_positions))
        final_v_positions = sorted(set(v_positions) | set(v_hough_positions))

        h_count = len(final_h_positions)
        v_count = len(final_v_positions)

        print(f"  Horizontal lines (detected): {h_count}")
        print(f"  Vertical lines (detected): {v_count}")

        # Debug overlay: show detected lines
        debug_verify = image.copy()
        for ypos in final_h_positions:
            cv2.line(debug_verify, (0, ypos), (w - 1, ypos), (255, 0, 0), 1)
        for xpos in final_v_positions:
            cv2.line(debug_verify, (xpos, 0), (xpos, h - 1), (0, 0, 255), 1)
        cv2.putText(debug_verify, f"H-Lines: {h_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if h_count >= 10 else (0, 0, 255), 2)
        cv2.putText(debug_verify, f"V-Lines: {v_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if v_count >= 12 else (0, 0, 255), 2)
        if self.debug:
            self.save_debug_image(debug_verify, f"{img_name}_10_shape_verification.jpg", f"H={h_count}, V={v_count}")

        # SHG form should have 10+ rows and 12+ columns
        expected_h_min = 10
        expected_v_min = 12

        if h_count < expected_h_min or v_count < expected_v_min:
            print(f"  ✗ ERROR: Table shape verification FAILED!")
            print(f"          Expected: {expected_h_min}+ horizontal, {expected_v_min}+ vertical lines")
            print(f"          Found: {h_count} horizontal, {v_count} vertical lines")
            return False

        print(f"  ✓ Table shape verified successfully")
        return True
    
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
    
    def detect_lines_and_intersections(self, image, img_name):
        """
        Step 4: Detect horizontal and vertical lines, find intersection points
        """
        print("\n" + "="*70)
        print("STEP 4: DETECTING LINES AND INTERSECTIONS")
        print("="*70)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Get size-adaptive configuration  
        size_config = self.get_size_config(h, w)
        intersection_tolerance = size_config['intersection_tolerance']
        intersection_roi_tolerance = size_config['intersection_roi_tolerance']
        search_radius = size_config['search_radius']
        x_tolerance = size_config['x_tolerance']
        min_gap = size_config['min_gap']
        
        print(f"  Dynamic tolerances: intersection={intersection_tolerance}px, "
            f"roi={intersection_roi_tolerance}px, search_radius={search_radius}px, "
            f"x_tolerance={x_tolerance}px")
        
        # Enhanced binary for line detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Mask-based adaptive sharpening with blur detection
        print("  Applying mask-based adaptive sharpening...")
        
        # Step 1: Global blur detection
        laplacian_global = cv2.Laplacian(enhanced, cv2.CV_64F)
        global_blur_variance = float(laplacian_global.var())
        
        # Reject if image is too blurry before even trying to sharpen
        # Absolute minimum threshold: if variance is too low, no sharpening can help
        MIN_ACCEPTABLE_BLUR_VARIANCE = 20.0
        if global_blur_variance < MIN_ACCEPTABLE_BLUR_VARIANCE:
            raise ValueError(f"Image is too blurry to process. Blur score: {global_blur_variance:.2f} "
                           f"(minimum required: {MIN_ACCEPTABLE_BLUR_VARIANCE:.2f}). "
                           f"Please use a sharper image for accurate line detection.")
        
        print(f"    Global blur detection: variance={global_blur_variance:.2f}")
        
        # Step 2: Create local blur map (sliding window approach)
        print("    Creating local blur mask...")
        window_size = max(15, min(h, w) // 20)  # Adaptive window size based on image dimensions
        window_size = window_size if window_size % 2 == 1 else window_size + 1  # Ensure odd
        
        # Compute local blur variance using sliding window
        blur_map = np.zeros((h, w), dtype=np.float32)
        half_window = window_size // 2
        
        # Compute Laplacian once
        laplacian_local = cv2.Laplacian(enhanced, cv2.CV_64F)
        
        # Use convolution to compute local variance efficiently
        # Variance = mean(x^2) - mean(x)^2
        laplacian_squared = laplacian_local ** 2
        kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size * window_size)
        
        mean_squared = cv2.filter2D(laplacian_squared, -1, kernel)
        mean_val = cv2.filter2D(laplacian_local, -1, kernel)
        blur_map = mean_squared - (mean_val ** 2)
        
        # Normalize blur map to 0-1 range for mask creation
        blur_map_max = blur_map.max()
        blur_map_min = blur_map.min()
        if blur_map_max > blur_map_min:
            blur_map_normalized = (blur_map - blur_map_min) / (blur_map_max - blur_map_min)
        else:
            blur_map_normalized = np.ones_like(blur_map) * 0.5
        
        # Step 3: Create sharpening mask - areas with low variance need more sharpening
        # Invert: low variance (blurry) = high mask value (more sharpening needed)
        sharpening_mask = 1.0 - blur_map_normalized
        
        # Smooth the mask to avoid artifacts
        blur_radius_mask = max(3, window_size // 4)
        blur_radius_mask = blur_radius_mask if blur_radius_mask % 2 == 1 else blur_radius_mask + 1
        sharpening_mask = cv2.GaussianBlur(sharpening_mask, (blur_radius_mask, blur_radius_mask), 0)
        
        # Threshold: only sharpen areas that actually need it
        blur_threshold = 0.3  # Only sharpen if local blur is above this threshold
        sharpening_mask = np.clip((sharpening_mask - blur_threshold) / (1.0 - blur_threshold), 0, 1)
        
        # Enhance the sharpening mask values to make sharpening more aggressive
        # Apply power function to boost mask values (gamma correction)
        gamma = 0.7  # Values < 1.0 boost lower values more (makes mask more aggressive)
        sharpening_mask_enhanced = np.power(sharpening_mask, gamma)
        
        # Further boost by adding a base level to ensure minimum sharpening in mask areas
        min_sharpening_level = 0.3  # Minimum sharpening even for areas with low mask values
        sharpening_mask_enhanced = np.clip(sharpening_mask_enhanced + min_sharpening_level, 0, 1)
        
        # Normalize back to full range to maximize effect
        if sharpening_mask_enhanced.max() > 0:
            sharpening_mask_enhanced = sharpening_mask_enhanced / sharpening_mask_enhanced.max()
        
        print(f"    Sharpening mask enhanced: original_max={sharpening_mask.max():.3f}, "
              f"enhanced_max={sharpening_mask_enhanced.max():.3f}")
        
        # Adjust sharpening strength based on image size (INCREASED VALUES)
        image_area = h * w
        if image_area > 2000000:  # Very large images
            base_max_strength = 5.0
            kernel_size = 5
        elif image_area > 1000000:  # Large images
            base_max_strength = 4.5
            kernel_size = 3
        elif image_area > 500000:  # Medium images
            base_max_strength = 4.0
            kernel_size = 3
        else:  # Small images
            base_max_strength = 3.5
            kernel_size = 3
        
        # Scale enhanced mask by maximum strength to get variable sharpening
        variable_strength_mask = sharpening_mask_enhanced * base_max_strength
        
        if self.debug:
            # Save blur map and mask for debugging
            blur_map_vis = (blur_map_normalized * 255).astype(np.uint8)
            self.save_debug_image(blur_map_vis, f"{img_name}_10a_blur_map.jpg", "Local blur map")
            mask_vis = (sharpening_mask * 255).astype(np.uint8)
            self.save_debug_image(mask_vis, f"{img_name}_10b_sharpening_mask_original.jpg", "Original sharpening mask")
            mask_enhanced_vis = (sharpening_mask_enhanced * 255).astype(np.uint8)
            self.save_debug_image(mask_enhanced_vis, f"{img_name}_10b2_sharpening_mask_enhanced.jpg", "Enhanced sharpening mask")
        
        # Step 4: Apply adaptive sharpening using the enhanced mask
        print("    Applying aggressive adaptive sharpening based on enhanced mask...")
        
        # Apply variable unsharp masking with multiple passes for heavily blurred areas
        enhanced_float = enhanced.astype(np.float32)
        
        # Determine number of passes based on global blur level
        if global_blur_variance < 100:
            num_passes = 2  # Very blurry images get 2 passes
        elif global_blur_variance < 200:
            num_passes = 2  # Blurry images get 2 passes
        else:
            num_passes = 1  # Less blurry images get 1 pass
        
        print(f"    Sharpening passes: {num_passes} (based on blur level)")
        
        for pass_num in range(num_passes):
            # Adjust blur radius based on pass (use larger radius for first pass on very blurry images)
            if global_blur_variance < 100 and pass_num == 0:
                blur_radius = 5  # Larger blur radius for very blurry images
            else:
                blur_radius = 3
            
            # Create blurred version for unsharp masking
            blurred = cv2.GaussianBlur(enhanced_float, (0, 0), sigmaX=blur_radius, sigmaY=blur_radius)
            
            # Unsharp mask formula with variable strength per pixel
            # sharpened = original + (original - blurred) * strength_per_pixel
            unsharp_diff = enhanced_float - blurred
            
            # For second pass, slightly reduce strength to avoid over-sharpening
            pass_strength_multiplier = 1.0 if pass_num == 0 else 0.8
            variable_strength_pass = variable_strength_mask * pass_strength_multiplier
            
            enhanced_sharpened = enhanced_float + unsharp_diff * variable_strength_pass
            enhanced_float = np.clip(enhanced_sharpened, 0, 255).astype(np.float32)
            
            if pass_num > 0:
                print(f"      Pass {pass_num + 1}/{num_passes} completed")
        
        enhanced = enhanced_float.astype(np.uint8)
        
        # Step 5: Verify improvement and reject if still too blurry
        laplacian_after = cv2.Laplacian(enhanced, cv2.CV_64F)
        blur_variance_after = float(laplacian_after.var())
        improvement = ((blur_variance_after - global_blur_variance) / max(global_blur_variance, 1.0)) * 100
        
        # Minimum acceptable variance after sharpening
        MIN_VARIANCE_AFTER_SHARPENING = 150.0
        
        print(f"    Sharpening applied: max_strength={base_max_strength:.1f}, passes={num_passes}, "
              f"enhanced_mask=True")
        print(f"    Improvement: {improvement:+.1f}% (variance: {global_blur_variance:.1f} -> {blur_variance_after:.1f})")
        
        if blur_variance_after < MIN_VARIANCE_AFTER_SHARPENING:
            raise ValueError(f"Image is too blurry to process even after sharpening. "
                           f"Final blur score: {blur_variance_after:.2f} "
                           f"(minimum required: {MIN_VARIANCE_AFTER_SHARPENING:.2f}). "
                           f"Please use a sharper image for accurate line detection.")
        
        # Check improvement threshold - if we couldn't improve enough, reject
        MIN_IMPROVEMENT_PERCENT = 10.0  # Must improve by at least 10%
        if improvement < MIN_IMPROVEMENT_PERCENT and global_blur_variance < 100:
            raise ValueError(f"Image blur could not be improved sufficiently. "
                           f"Improvement: {improvement:.1f}% (minimum: {MIN_IMPROVEMENT_PERCENT:.1f}%). "
                           f"Please use a sharper image for accurate line detection.")
        
        if self.debug:
            # Create visualization of sharpening effect
            diff_vis = np.abs(enhanced.astype(np.int16) - gray.astype(np.int16)).astype(np.uint8)
            diff_vis = cv2.applyColorMap(diff_vis, cv2.COLORMAP_HOT)
            self.save_debug_image(diff_vis, f"{img_name}_10c_sharpening_effect.jpg", "Sharpening effect visualization")
            
            self.save_debug_image(enhanced, f"{img_name}_10_sharpened.jpg", 
                                f"Sharpened image (improvement: {improvement:+.1f}%)")
        
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if self.debug:
            self.save_debug_image(binary, f"{img_name}_11_binary_for_lines.jpg", "Binary for line detection")
        
        # DIRECTIONAL DILATION - only dilate in line directions
        print("  Applying directional dilation for lines...")

        # === Gentle Vertical Dilation ===
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        v_strong = cv2.dilate(binary, v_kernel, iterations=1)
        binary_v_dilated = cv2.addWeighted(binary, 0.8, v_strong, 0.2, 0)

        # === Gentle Horizontal Dilation ===
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        h_strong = cv2.dilate(binary, h_kernel, iterations=1)
        binary_h_dilated = cv2.addWeighted(binary, 0.8, h_strong, 0.2, 0)

        # Combine both
        binary_working = cv2.bitwise_or(binary_v_dilated, binary_h_dilated)

        if self.debug:
            self.save_debug_image(binary_working, f"{img_name}_11a_binary_dilated.jpg", "Directional dilated binary")
        
        # Detect horizontal lines with adaptive gap closing (similar to vertical)
        print("\n  Detecting horizontal lines...")
        h_kernel_width = max(3, size_config['min_cell_w'] * 2)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_width, 1))
        print(f"    Horizontal kernel size: {(h_kernel_width, 1)}")
        h_mask = cv2.morphologyEx(binary_working, cv2.MORPH_OPEN, h_kernel, iterations=1)
        
        # Analyze horizontal line fragmentation
        h_labeled_pre = cv2.connectedComponentsWithStats(h_mask, connectivity=8)
        h_num_labels_pre, _, h_stats_pre, _ = h_labeled_pre
        
        # Quick fragmentation check for horizontal lines
        h_small_fragments = sum(1 for label in range(1, h_num_labels_pre) 
                              if h_stats_pre[label, cv2.CC_STAT_WIDTH] < w * 0.05)
        h_fragmentation_ratio = h_small_fragments / max(h_num_labels_pre - 1, 1) if h_num_labels_pre > 1 else 0
        
        # Use backup's base approach (20x3 close kernel) but enhance to connect broken lines
        base_h_close_width = 20
        base_h_close_height = 3
        
        # Always apply base close first
        h_kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (base_h_close_width, base_h_close_height))
        h_mask = cv2.morphologyEx(h_mask, cv2.MORPH_CLOSE, h_kernel_close)
        
        if h_fragmentation_ratio > 0.3:  # Lower threshold to catch more cases
            # Medium to high fragmentation - apply additional pass with larger kernel
            print(f"    Horizontal fragmentation detected ({h_fragmentation_ratio:.1%}) - applying enhanced gap closing")
            # Use larger kernel to connect more broken pieces
            enhanced_width = max(35, int(w * 0.025))  # Larger kernel for better connection
            h_kernel_close2 = cv2.getStructuringElement(cv2.MORPH_RECT, (enhanced_width, base_h_close_height))
            h_mask = cv2.morphologyEx(h_mask, cv2.MORPH_CLOSE, h_kernel_close2)
        
        h_mask = cv2.bitwise_and(h_mask, binary_working)
        
        # Gap dilation - more aggressive if fragmentation is high
        if h_fragmentation_ratio > 0.3:
            # Use larger gap dilation to connect broken horizontal lines
            h_gap_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))  # Larger vertical dilation
            h_mask = cv2.dilate(h_mask, h_gap_kernel, iterations=2)  # More iterations
        else:
            # Normal case - backup's approach
            h_gap_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
            h_mask = cv2.dilate(h_mask, h_gap_kernel, iterations=1)
        
        # Final safety trim
        h_mask = cv2.bitwise_and(h_mask, binary_working)
        
        # DEBUG: Check h_mask before saving
        h_nonzero = cv2.countNonZero(h_mask)
        print(f"    h_mask non-zero pixels: {h_nonzero}")
        
        if self.debug:
            self.save_debug_image(h_mask, f"{img_name}_12_horizontal_lines.jpg", "Horizontal line mask")
        
        # Detect vertical lines with MORE AGGRESSIVE gap closing
        print("  Detecting vertical lines...")
        # 1) Extract thin vertical strokes only
        v_kernel_height = max(10, size_config['min_cell_h'] * 2)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_height))
        v_mask = cv2.morphologyEx(binary_working, cv2.MORPH_OPEN, v_kernel)

        # 2) ADAPTIVE gap closing - use backup's simple approach with light adaptive enhancement
        # First analyze fragmentation to decide if we need extra help
        v_labeled_pre = cv2.connectedComponentsWithStats(v_mask, connectivity=8)
        v_num_labels_pre, _, v_stats_pre, _ = v_labeled_pre
        
        # Quick fragmentation check
        small_fragments = sum(1 for label in range(1, v_num_labels_pre) 
                            if v_stats_pre[label, cv2.CC_STAT_HEIGHT] < h * 0.05)
        fragmentation_ratio = small_fragments / max(v_num_labels_pre - 1, 1) if v_num_labels_pre > 1 else 0
        
        # Use backup's simple approach as base (kernel=15, gentle)
        base_close_size = 15
        
        # Only add extra closing if fragmentation is very high (backup approach + small boost)
        if fragmentation_ratio > 0.5:
            # High fragmentation - apply base close + one more with larger kernel
            print(f"    High fragmentation detected ({fragmentation_ratio:.1%}) - applying enhanced gap closing")
            v_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, base_close_size))
            v_mask = cv2.morphologyEx(v_mask, cv2.MORPH_CLOSE, v_close_kernel)
            # Second pass with slightly larger kernel for high fragmentation
            v_close_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(h * 0.018))))
            v_mask = cv2.morphologyEx(v_mask, cv2.MORPH_CLOSE, v_close_kernel2)
        else:
            # Normal case - use backup's simple approach
            v_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, base_close_size))
            v_mask = cv2.morphologyEx(v_mask, cv2.MORPH_CLOSE, v_close_kernel)

        # 3) Limit to original binary so it can't invade cells
        v_mask = cv2.bitwise_and(v_mask, binary_working)

        # 4) Gentle dilation ONLY upwards/downwards (backup approach)
        v_gap_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        v_mask = cv2.dilate(v_mask, v_gap_kernel, iterations=1)

        # final safety trim
        v_mask = cv2.bitwise_and(v_mask, binary_working)
        
        if self.debug:
            self.save_debug_image(v_mask, f"{img_name}_13_vertical_lines.jpg", "Vertical line mask")

        # ---------------- SIMPLE BINARY CLEANING ----------------

        # Step 1: Pure binary - make everything either black (0) or white (255)
        print("  Converting to pure binary...")
        h_mask = cv2.threshold(h_mask, 127, 255, cv2.THRESH_BINARY)[1]
        v_mask = cv2.threshold(v_mask, 127, 255, cv2.THRESH_BINARY)[1]

        if self.debug:
            self.save_debug_image(h_mask, f"{img_name}_h_mask_binary.jpg", "Horizontal mask - pure binary")
            self.save_debug_image(v_mask, f"{img_name}_v_mask_binary.jpg", "Vertical mask - pure binary")

        # Step 2: Estimate minimum cell height from horizontal lines
        h_labeled = cv2.connectedComponentsWithStats(h_mask, connectivity=8)
        h_num_labels, h_labels, h_stats, h_centroids = h_labeled

        h_line_positions = []
        for label in range(1, h_num_labels):
            y_pos = int(h_centroids[label][1])
            width = h_stats[label, cv2.CC_STAT_WIDTH]
            if width > w * 0.10:  # Accept fragments (for merged cells)
                h_line_positions.append(y_pos)

        h_line_positions = sorted(h_line_positions)

        # Calculate minimum cell height
        if len(h_line_positions) >= 2:
            gaps = [h_line_positions[i+1] - h_line_positions[i] for i in range(len(h_line_positions)-1)]
            min_cell_height = min(gaps) if gaps else h // 10
            print(f"  Minimum cell height detected: {min_cell_height} pixels")
        else:
            min_cell_height = h // 10
            print(f"  Using default minimum cell height: {min_cell_height} pixels")

        # Step 3: Filter vertical lines - only keep those that connect to horizontal lines
        print("  Filtering vertical lines based on horizontal line connections...")
        v_mask_merged = cv2.dilate(v_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)
        v_labeled = cv2.connectedComponentsWithStats(v_mask_merged, connectivity=8)
        v_num_labels, v_labels, v_stats, v_centroids = v_labeled

        filtered_v_mask = np.zeros_like(v_mask)

        # Dilate horizontal lines slightly for connection detection
        h_dilated = cv2.dilate(h_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

        for label in range(1, v_num_labels):
            # Get vertical line component
            component_mask = ((v_labels == label).astype(np.uint8) * 255)
            component_mask = cv2.bitwise_and(component_mask, v_mask)  # restore original thin line shape
            
            x = v_stats[label, cv2.CC_STAT_LEFT]
            y = v_stats[label, cv2.CC_STAT_TOP]
            width = v_stats[label, cv2.CC_STAT_WIDTH]
            height = v_stats[label, cv2.CC_STAT_HEIGHT]
            
            # Check if this vertical line intersects with any horizontal line
            intersection = cv2.bitwise_and(component_mask, h_dilated)
            has_connection = cv2.countNonZero(intersection) > 0
            
            # Check if vertical line is at least min_cell_height or near image edges
            is_significant_height = height >= min_cell_height * 0.8
            is_edge = x < w * 0.05 or (x + width) > w * 0.95
            
            # Keep if: connects to horizontal line AND (is tall enough OR at edge)
            if has_connection and (is_significant_height or is_edge):
                filtered_v_mask[v_labels == label] = 255
                print(f"    ✓ Kept vertical line at x={x}: height={height}, connected={has_connection}")
            else:
                print(f"    ✗ Removed vertical line at x={x}: height={height}, connected={has_connection}, significant={is_significant_height}")

        if self.debug:
            self.save_debug_image(filtered_v_mask, f"{img_name}_v_mask_filtered.jpg", "Vertical mask - filtered")

        # Final dilation to make lines thicker/wider (same as backup)
        print("  Dilating vertical lines...")
        v_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))  # Horizontal dilation only
        filtered_v_mask = cv2.dilate(filtered_v_mask, v_dilate_kernel, iterations=2)  # Same as backup

        if self.debug:
            self.save_debug_image(filtered_v_mask, f"{img_name}_v_mask_dilated.jpg", "Vertical mask - dilated")

        # Filter horizontal lines - only keep those that connect to vertical lines (same as vertical filtering)
        print("  Filtering horizontal lines based on vertical line connections...")
        h_mask_merged = cv2.dilate(h_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)), iterations=1)
        h_clean_labeled = cv2.connectedComponentsWithStats(h_mask_merged, connectivity=8)
        h_clean_num_labels, h_clean_labels, h_clean_stats, h_clean_centroids = h_clean_labeled
        
        filtered_h_mask = np.zeros_like(h_mask)
        
        # Dilate ORIGINAL vertical lines slightly for connection detection (not filtered_v_mask)
        # This matches how vertical lines use original h_mask for connection detection
        v_dilated = cv2.dilate(v_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        
        # Estimate minimum cell width from vertical lines (same approach as min_cell_height for vertical)
        v_labeled_for_width = cv2.connectedComponentsWithStats(v_mask, connectivity=8)
        v_num_labels_for_width, _, v_stats_for_width, v_centroids_for_width = v_labeled_for_width
        
        v_line_positions = []
        for label in range(1, v_num_labels_for_width):
            x_pos = int(v_centroids_for_width[label][0])
            height = v_stats_for_width[label, cv2.CC_STAT_HEIGHT]
            if height > h * 0.10:  # Accept fragments (for merged cells)
                v_line_positions.append(x_pos)
        
        v_line_positions = sorted(v_line_positions)
        
        # Calculate minimum cell width
        if len(v_line_positions) >= 2:
            gaps = [v_line_positions[i+1] - v_line_positions[i] for i in range(len(v_line_positions)-1)]
            min_cell_width = min(gaps) if gaps else w // 10
            print(f"  Minimum cell width detected: {min_cell_width} pixels")
        else:
            min_cell_width = w // 10
            print(f"  Using default minimum cell width: {min_cell_width} pixels")
        
        for label in range(1, h_clean_num_labels):
            # Get horizontal line component
            component_mask = ((h_clean_labels == label).astype(np.uint8) * 255)
            component_mask = cv2.bitwise_and(component_mask, h_mask)  # restore original thin line shape
            
            x = h_clean_stats[label, cv2.CC_STAT_LEFT]
            y = h_clean_stats[label, cv2.CC_STAT_TOP]
            width = h_clean_stats[label, cv2.CC_STAT_WIDTH]
            height = h_clean_stats[label, cv2.CC_STAT_HEIGHT]
            
            # Check if this horizontal line intersects with any vertical line
            intersection = cv2.bitwise_and(component_mask, v_dilated)
            has_connection = cv2.countNonZero(intersection) > 0
            
            # Check if horizontal line is at least min_cell_width or near image edges
            is_significant_width = width >= min_cell_width * 0.8
            is_edge = y < h * 0.05 or (y + height) > h * 0.95
            
            # Keep if: connects to vertical line AND (is wide enough OR at edge)
            if has_connection and (is_significant_width or is_edge):
                filtered_h_mask[h_clean_labels == label] = 255
                print(f"    ✓ Kept horizontal line at y={y}: width={width}, connected={has_connection}")
            else:
                print(f"    ✗ Removed horizontal line at y={y}: width={width}, connected={has_connection}, significant={is_significant_width}")
        
        if self.debug:
            self.save_debug_image(filtered_h_mask, f"{img_name}_h_mask_filtered.jpg", "Horizontal mask - filtered")
        
        # Final dilation to make lines thicker/wider (same as vertical)
        print("  Dilating horizontal lines...")
        h_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))  # Vertical dilation only
        filtered_h_mask = cv2.dilate(filtered_h_mask, h_dilate_kernel, iterations=2)  # Same as vertical
        
        if self.debug:
            self.save_debug_image(filtered_h_mask, f"{img_name}_h_mask_dilated.jpg", "Horizontal mask - dilated")

        # Step 4: Combine filtered masks
        final_mask = cv2.bitwise_or(filtered_h_mask, filtered_v_mask)

        # Make sure it's pure binary
        final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)[1]

        if self.debug:
            self.save_debug_image(final_mask, f"{img_name}_final_cleaned.jpg", "Final Cleaned Mask")
            print(f"  Final mask non-zero pixels: {cv2.countNonZero(final_mask)}")

        # ---------------- STORE CLEANED MASKS ----------------
        self.horizontal_mask = filtered_h_mask
        self.vertical_mask = filtered_v_mask
        self.combined_mask = final_mask

        validation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.line_validation_mask = cv2.dilate(final_mask, validation_kernel, iterations=1)

        if self.debug:
            self.save_debug_image(final_mask, f"{img_name}_14_cleaned_lines.jpg", "Cleaned combined line mask")
        
        # Find horizontal line positions (use filtered mask)
        h_lines = []
        h_labeled = cv2.connectedComponentsWithStats(filtered_h_mask, connectivity=8)
        h_num_labels, h_labels, h_stats, h_centroids = h_labeled
        
        print(f"\n  [H-LINE DEBUG] Total horizontal components: {h_num_labels - 1}")
        h_line_candidates = []
        for label in range(1, h_num_labels):
            y_pos = int(h_centroids[label][1])
            x = h_stats[label, cv2.CC_STAT_LEFT]
            width = h_stats[label, cv2.CC_STAT_WIDTH]
            h_line_candidates.append({'position': y_pos, 'start': x, 'end': x + width, 'width': width})
            
            width_pct = (width / w) * 100
            size_status = "✓" if width > w * 0.10 else "✗"
            print(f"    {size_status} Component at y={y_pos}: x_range=[{x}-{x+width}], width={width}px ({width_pct:.1f}%)")
            
            if width > w * 0.10:  # Accept fragments down to 10% (for merged cells)
                h_lines.append({'position': y_pos, 'start': x, 'end': x + width})
        
        print(f"  [H-LINE DEBUG] After width filter: {len(h_lines)}/{len(h_line_candidates)} accepted")
        h_lines = sorted(h_lines, key=lambda x: x['position'])
        clustered_h_lines = []
        min_gap = size_config['min_gap']
        for line in h_lines:
            if not clustered_h_lines or abs(line['position'] - clustered_h_lines[-1]['position']) > min_gap:
                clustered_h_lines.append(line)
            else:
                existing = clustered_h_lines[-1]
                existing['position'] = int((existing['position'] + line['position']) / 2)
                existing['start'] = min(existing['start'], line['start'])
                existing['end'] = max(existing['end'], line['end'])
        h_lines = clustered_h_lines
        print(f"  Found {len(h_lines)} horizontal lines")
        
        # Find vertical line positions - NOW USING CLEANED filtered_v_mask
        print("  Detecting vertical line fragments from CLEANED mask...")
        v_lines = []
        v_labeled = cv2.connectedComponentsWithStats(filtered_v_mask, connectivity=8)  # CHANGED
        v_num_labels, v_labels, v_stats, v_centroids = v_labeled
        
        # Collect ALL vertical fragments (very low threshold)
        v_fragments = []
        for label in range(1, v_num_labels):
            x_pos = int(v_centroids[label][0])
            y = v_stats[label, cv2.CC_STAT_TOP]
            height = v_stats[label, cv2.CC_STAT_HEIGHT]
            width = v_stats[label, cv2.CC_STAT_WIDTH]
            
            # VERY LOW threshold to catch even small fragments
            if height > h * 0.03 and width < w * 0.1:  # Added width check to avoid false positives
                v_fragments.append({
                    'position': x_pos, 
                    'start': y, 
                    'end': y + height,
                    'height': height
                })
        
        print(f"  Found {len(v_fragments)} vertical line fragments before merging")

        v_line_positions = sorted(frag['position'] for frag in v_fragments)
        if len(v_line_positions) >= 2:
            x_gaps = [v_line_positions[i + 1] - v_line_positions[i] for i in range(len(v_line_positions) - 1)]
            if x_gaps:
                min_cell_width = min(x_gaps)
            print(f"  Minimum cell width detected: {min_cell_width} pixels")
        else:
            print(f"  Using default minimum cell width: {min_cell_width} pixels")
        
        # ====================================================================
        # IMPROVED GROUPING LOGIC FOR VERTICAL LINE FRAGMENTS
        # ====================================================================
        # Groups fragments that are:
        # 1. Horizontally aligned (within x_tolerance)
        # 2. Vertically overlapping or with small gaps (continuation of same line)
        #
        # ALL ADJUSTABLE PARAMETERS ARE MARKED BELOW WITH "ADJUST:" COMMENT
        # Find the section starting with "===== ADJUSTABLE PARAMETERS ====="
        # ====================================================================
        
        x_tolerance = size_config['x_tolerance']
        
        # ===== ADJUSTABLE PARAMETERS - Fine-tune here for accuracy =====
        # LOCATION: Lines ~1633-1660
        # 
        # QUICK REFERENCE - What to adjust:
        # 1. min_x_percent (line ~1636): Horizontal tolerance - increase to merge fragments further apart horizontally
        # 2. max_gap_ratio (line ~1640): Vertical gap tolerance - increase to merge fragments with larger vertical gaps
        # 3. small_gap_percent (line ~1644): Small gap threshold - increase to merge fragments with slightly larger gaps
        # 4. very_close_pixel_threshold (line ~1647): "Very close" distance - increase to treat more fragments as close
        # 5. merge_threshold_close/normal (line ~1721): Merge strictness - decrease to merge more, increase to merge less
        # 6. x_score_boost (line ~1690): Boost for close fragments - increase for more aggressive merging
        # 7. overlap_boost (line ~1697): Boost for overlapping fragments - increase for more aggressive merging
        # 8. tiny_gap_score (line ~1703): Score for small gaps - increase to merge small gap fragments more
        # 9. x_weight_close/v_weight_close (line ~1713): Score weighting - adjust priorities
        # =====
        # For small pixel differences (3-4px), be more aggressive in merging
        # ADJUST: min_x_percent - Increase to allow wider X tolerance (default: 0.003 = 0.3% of width)
        min_x_percent = 0.003  # Increase this (e.g., 0.004, 0.005) to merge fragments further apart horizontally
        min_x_tolerance = max(6, int(w * min_x_percent))  # At least 0.3% of width, minimum 6px
        effective_x_tolerance = max(x_tolerance, min_x_tolerance)
        
        # ADJUST: max_gap_ratio - Increase to allow larger gaps between fragments (default: 0.18 = 18% of height)
        max_gap_ratio = 0.18  # Increase this (e.g., 0.20, 0.25) to merge fragments with larger vertical gaps
        max_gap_pixels = max(int(h * max_gap_ratio), int(h * 0.06))  # At least 6% of height
        
        # ADJUST: small_gap_percent - Increase to merge fragments with larger small gaps (default: 0.015 = 1.5% of height)
        small_gap_percent = 0.015  # Increase this (e.g., 0.02, 0.025) to merge fragments with slightly larger gaps
        small_gap_threshold = max(12, int(h * small_gap_percent))  # 1.5% of height or 12px, whichever is larger
        
        # ADJUST: very_close_pixel_threshold - Pixel distance considered "very close" (default: 5px)
        very_close_pixel_threshold = 5  # Increase this (e.g., 6, 7, 8) to treat more fragments as "very close"
        # ===== END ADJUSTABLE PARAMETERS =====
        
        print(f"    Grouping tolerances: x_tolerance={x_tolerance}px, effective_x_tolerance={effective_x_tolerance}px, "
              f"small_gap_threshold={small_gap_threshold}px, max_gap={max_gap_pixels}px, "
              f"very_close_threshold={very_close_pixel_threshold}px")
        
        # Sort fragments by X position first, then by Y start position
        v_fragments = sorted(v_fragments, key=lambda x: (x['position'], x['start']))
        
        merged_v_lines = []
        
        for frag in v_fragments:
            best_match = None
            best_score = 0
            
            # Find the best matching existing line group
            for existing in merged_v_lines:
                x_diff = abs(frag['position'] - existing['position'])
                
                # Check horizontal alignment - use effective tolerance which is more permissive
                if x_diff > effective_x_tolerance:
                    continue
                
                # Special handling for very close fragments - uses very_close_pixel_threshold
                # ADJUST: Change very_close_pixel_threshold above to modify this behavior
                is_very_close = x_diff <= very_close_pixel_threshold
                
                # Calculate vertical relationship (overlap or gap)
                # Check if fragments overlap vertically
                if frag['end'] > existing['start'] and frag['start'] < existing['end']:
                    # Fragments overlap
                    overlap_start = max(frag['start'], existing['start'])
                    overlap_end = min(frag['end'], existing['end'])
                    overlap_height = overlap_end - overlap_start
                    min_gap = 0
                else:
                    # Fragments don't overlap - calculate gap
                    overlap_height = 0
                    if frag['end'] <= existing['start']:
                        # Fragment is above existing line
                        min_gap = existing['start'] - frag['end']
                    else:  # frag['start'] >= existing['end']
                        # Fragment is below existing line
                        min_gap = frag['start'] - existing['end']
                
                # Score based on X alignment and vertical continuity
                x_score = 1.0 - (x_diff / effective_x_tolerance) if effective_x_tolerance > 0 else 1.0
                
                # Boost score for very close fragments
                # ADJUST: x_score_boost - Increase to give more weight to close fragments (default: 1.3)
                x_score_boost = 1.3  # Increase this (e.g., 1.4, 1.5) to merge close fragments more aggressively
                if is_very_close:
                    x_score = min(1.0, x_score * x_score_boost)
                
                if overlap_height > 0:
                    # Overlapping fragments get high score
                    overlap_ratio = overlap_height / min(frag['height'], existing['height'])
                    vertical_score = overlap_ratio
                    # Very close overlapping fragments should always merge
                    # ADJUST: overlap_boost - Increase to merge overlapping close fragments more (default: 1.4)
                    overlap_boost = 1.4  # Increase this (e.g., 1.5, 1.6) for more aggressive merging
                    if is_very_close:
                        vertical_score = min(1.0, vertical_score * overlap_boost)
                elif min_gap <= small_gap_threshold:
                    # Very small gaps - almost certain continuation
                    # ADJUST: tiny_gap_score - Increase to merge fragments with small gaps more (default: 0.98)
                    tiny_gap_score = 0.98  # Increase this (up to 1.0) to merge small gap fragments more
                    vertical_score = tiny_gap_score
                elif min_gap <= max_gap_pixels:
                    # Small gaps are acceptable (line continuation)
                    gap_ratio = 1.0 - (min_gap / max_gap_pixels)
                    vertical_score = gap_ratio * 0.8  # Slightly lower for larger gaps
                else:
                    # Too much gap - not a continuation
                    continue
                
                # Combined score: both horizontal and vertical alignment matter
                # ADJUST: x_weight_close, v_weight_close - Adjust weighting for very close fragments
                x_weight_close = 0.65  # Increase to prioritize X alignment for close fragments (default: 0.65)
                v_weight_close = 0.35  # Decrease if X alignment is more important (default: 0.35)
                x_weight_normal = 0.4  # Weight for normal fragments - X alignment (default: 0.4)
                v_weight_normal = 0.6  # Weight for normal fragments - Vertical alignment (default: 0.6)
                
                if is_very_close:
                    combined_score = (x_score * x_weight_close + vertical_score * v_weight_close)
                else:
                    combined_score = (x_score * x_weight_normal + vertical_score * v_weight_normal)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = existing
            
            # Merge with best match if score is good enough, otherwise create new group
            # ADJUST: merge_threshold_close and merge_threshold_normal control merging strictness
            # Lower values = more aggressive merging, Higher values = stricter merging
            merge_threshold_close = 0.15  # Threshold for very close fragments (lower = merge more)
            merge_threshold_normal = 0.25  # Threshold for normal fragments (lower = merge more)
            
            is_frag_very_close_to_best = (best_match is not None and 
                                          abs(frag['position'] - best_match['position']) <= very_close_pixel_threshold)
            merge_threshold = merge_threshold_close if is_frag_very_close_to_best else merge_threshold_normal
            
            if best_match and best_score > merge_threshold:
                # Weighted average of X position (weighted by fragment total height)
                existing_total = best_match.get('total_fragment_height', best_match['height'])
                total_height = existing_total + frag['height']
                weight_existing = existing_total / total_height
                weight_new = frag['height'] / total_height
                
                best_match['position'] = int(best_match['position'] * weight_existing + frag['position'] * weight_new)
                
                # Extend vertical span
                best_match['start'] = min(best_match['start'], frag['start'])
                best_match['end'] = max(best_match['end'], frag['end'])
                best_match['height'] = best_match['end'] - best_match['start']
                
                # Track fragment count for better averaging
                best_match['fragment_count'] = best_match.get('fragment_count', 1) + 1
                best_match['total_fragment_height'] = best_match.get('total_fragment_height', best_match['height']) + frag['height']
            else:
                # Create new line group
                new_line = frag.copy()
                new_line['fragment_count'] = 1
                new_line['total_fragment_height'] = frag['height']
                merged_v_lines.append(new_line)
        
        # Sort merged lines by X position
        merged_v_lines = sorted(merged_v_lines, key=lambda x: x['position'])
        print(f"  After improved merging: {len(merged_v_lines)} vertical line groups "
              f"(from {len(v_fragments)} fragments)")
        
        # Now filter merged lines that span significant height OR are at edges
        v_lines = []
        for line in merged_v_lines:
            # Accept if line spans at least 10% of height OR is near edges
            is_significant = line['height'] > h * 0.10  # Lowered from 0.15
            is_edge = line['position'] < w * 0.05 or line['position'] > w * 0.95
            
            if is_significant or is_edge:
                v_lines.append(line)
                print(f"    ✓ Line at x={line['position']}: height={line['height']} ({100*line['height']/h:.1f}% of image height)")
            else:
                print(f"    ✗ Rejected line at x={line['position']}: height={line['height']} ({100*line['height']/h:.1f}% - too short)")
        
        v_lines = sorted(v_lines, key=lambda x: x['position'])
        
        # Additional clustering for nearby vertical lines
        # Use width-based min_gap for vertical lines (columns should be further apart)
        v_min_gap = max(5, int(w // 200))  # Use width instead of height for vertical lines
        print(f"  Before clustering: {len(v_lines)} vertical lines, min_gap={v_min_gap}px (width-based)")
        clustered_v_lines = []
        for line in v_lines:
            if not clustered_v_lines:
                clustered_v_lines.append(line)
            else:
                last_pos = clustered_v_lines[-1]['position']
                distance = abs(line['position'] - last_pos)
                if distance > v_min_gap:
                    clustered_v_lines.append(line)
                else:
                    existing = clustered_v_lines[-1]
                    existing['position'] = int((existing['position'] + line['position']) / 2)
                    existing['start'] = min(existing['start'], line['start'])
                    existing['end'] = max(existing['end'], line['end'])
        v_lines = clustered_v_lines
        
        print(f"  Final vertical lines: {len(v_lines)}")
        
        # Find intersections - NOW USING CLEANED MASKS
        print("\n  Finding intersection points...")
        max_dim_current = max(h, w)
        small_form = max_dim_current <= 1400
        kernel_scale = 1.5 if small_form else 1.2
        kernel_length = max(10, int(round(intersection_roi_tolerance * kernel_scale)))
        h_iterations = 3 if small_form else 2
        v_iterations = 3 if small_form else 2
        
        # More aggressive dilation for intersection detection - USE CLEANED MASKS
        h_intersection_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        v_intersection_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        h_intersection = cv2.dilate(h_mask, h_intersection_kernel, iterations=h_iterations)
        v_intersection = cv2.dilate(filtered_v_mask, v_intersection_kernel, iterations=v_iterations)
        intersection_mask = cv2.bitwise_and(h_intersection, v_intersection)
        
        if self.debug:
            self.save_debug_image(intersection_mask, f"{img_name}_14a_intersections.jpg", "Intersection mask")

        cluster_tolerance = intersection_tolerance
        roi_tolerance = int(intersection_roi_tolerance)
        print(f"  Using intersection tolerances: cluster={cluster_tolerance}px, roi={roi_tolerance}px (adaptive)")
        row_positions = [line['position'] for line in h_lines]
        col_positions = [line['position'] for line in v_lines]
        clustered_rows = self._cluster_values(row_positions, cluster_tolerance)
        clustered_cols = self._cluster_values(col_positions, cluster_tolerance)

        def collect_intersections(roi_tol: int):
            hits = []
            found = []
            if roi_tol <= 0:
                return found, hits
            for y in clustered_rows:
                for x in clustered_cols:
                    y1 = max(0, y - roi_tol)
                    y2 = min(h, y + roi_tol)
                    x1 = max(0, x - roi_tol)
                    x2 = min(w, x + roi_tol)
                    roi = intersection_mask[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    nonzero = cv2.countNonZero(roi)
                    if nonzero <= 0:
                        continue
                    hits.append(nonzero)
                    h_ref = next((
                        line for line in h_lines
                        if abs(line['position'] - y) <= roi_tol and
                        line['start'] - roi_tol <= x <= line['end'] + roi_tol
                    ), None)
                    v_ref = next((
                        line for line in v_lines
                        if abs(line['position'] - x) <= roi_tol and
                        line['start'] - roi_tol <= y <= line['end'] + roi_tol
                    ), None)
                    found.append({
                        'x': x,
                        'y': y,
                        'h_line': h_ref,
                        'v_line': v_ref
                    })
            return found, hits

        intersections, roi_hits = collect_intersections(roi_tolerance)
        min_expected_intersections = max(4, len(clustered_rows))
        if len(intersections) < min_expected_intersections:
            boosted_roi_tolerance = int(round(roi_tolerance * 1.5))
            if boosted_roi_tolerance > roi_tolerance:
                print(f"  ⚠ Low intersection count ({len(intersections)} < {min_expected_intersections}). "
                      f"Retrying with ROI tolerance {boosted_roi_tolerance}px")
                boosted_intersections, boosted_hits = collect_intersections(boosted_roi_tolerance)
                if len(boosted_intersections) > len(intersections):
                    intersections = boosted_intersections
                    roi_hits = boosted_hits
                    roi_tolerance = boosted_roi_tolerance
                    print(f"    ✓ Boosted ROI tolerance recovered {len(intersections)} intersections")

        if self.debug:
            if roi_hits:
                hits_sorted = sorted(roi_hits)
                mid = len(hits_sorted) // 2
                if len(hits_sorted) % 2 == 0 and mid > 0:
                    median_hit = int(round((hits_sorted[mid - 1] + hits_sorted[mid]) / 2))
                else:
                    median_hit = hits_sorted[mid]
                print(f"  ROI hit pixels stats: min={hits_sorted[0]}, median={median_hit}, "
                      f"max={hits_sorted[-1]} (samples={len(hits_sorted)})")
            else:
                print("  ROI hit pixels stats: no positive intersections detected in sampled windows")

        print(f"  Found {len(intersections)} intersection points (ROI tolerance used: {roi_tolerance}px)")
        
        # Draw debug image with better visualization
        debug_lines = image.copy()
        
        # Draw horizontal lines in blue
        for h_line in h_lines:
            cv2.line(debug_lines, (h_line['start'], h_line['position']), 
                    (h_line['end'], h_line['position']), (255, 0, 0), 2)
        
        # Draw vertical lines in red with start/end markers
        for i, v_line in enumerate(v_lines):
            cv2.line(debug_lines, (v_line['position'], v_line['start']), 
                    (v_line['position'], v_line['end']), (0, 0, 255), 2)
            # Draw markers at start and end
            cv2.circle(debug_lines, (v_line['position'], v_line['start']), 6, (0, 255, 255), -1)
            cv2.circle(debug_lines, (v_line['position'], v_line['end']), 6, (255, 0, 255), -1)
            # Label the line
            cv2.putText(debug_lines, f"V{i}", (v_line['position'] + 5, v_line['start'] + 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw intersections in green
        for inter in intersections:
            cv2.circle(debug_lines, (inter['x'], inter['y']), 4, (0, 255, 0), -1)
        
        if self.debug:
            self.save_debug_image(debug_lines, f"{img_name}_15_lines_and_intersections.jpg", 
                                f"H:{len(h_lines)} V:{len(v_lines)} Int:{len(intersections)}")
        
        return h_lines, v_lines, intersections
    
    def trace_cells_from_intersections(self, h_lines, v_lines, intersections, img_name, image):
        """
        Step 5: Trace cells from intersection points
        """
        print("\n" + "="*70)
        print("STEP 5: TRACING CELLS FROM INTERSECTIONS")
        print("="*70)
    
        if not intersections:
            print(" No intersections available to trace cells")
            return []

        # Get size-adaptive configuration
        img_h, img_w = image.shape[:2]
        size_config = self.get_size_config(img_h, img_w)
        PIX_TOL = size_config['pix_tolerance']
        MIN_CELL_W = size_config['min_cell_w']
        MIN_CELL_H = size_config['min_cell_h']
        blocker_tolerance = size_config['blocker_edge_tolerance']
        
        print(f" Using dynamic tolerances: PIX_TOL={PIX_TOL}px, "
            f"MIN_CELL_W={MIN_CELL_W}px, MIN_CELL_H={MIN_CELL_H}px")

        for idx, inter in enumerate(intersections):
            inter['_idx'] = idx

        debug_img = image.copy()
        log_lines = []

        def log(message=""):
            if self.debug:
                log_lines.append(str(message))

        cells = []
        processed_cells = {}
        next_cell_id = 0

        h_map = defaultdict(list)
        v_map = defaultdict(list)
        for inter in intersections:
            h_key = int(round(inter['y']))
            v_key = int(round(inter['x']))
            h_map[h_key].append(inter)
            v_map[v_key].append(inter)

        for row in h_map.values():
            row.sort(key=lambda p: p['x'])
        for col in v_map.values():
            col.sort(key=lambda p: p['y'])

        def h_line_covers_point(h_line, x, tol=PIX_TOL):
            # Use a more lenient tolerance for point coverage
            # Line detection and intersection detection may have slight differences
            # Allow points slightly outside the line's recorded range
            # Use 2x tolerance or 10px, whichever is larger (but cap at reasonable max)
            extended_tol = min(max(tol * 2, 10), 20)  # Between 10-20px
            return h_line and (h_line['start'] - extended_tol <= x <= h_line['end'] + extended_tol)

        def _mask_hits(mask, x, y, radius=3):
            """Check if mask contains line pixels near (x, y)."""
            if mask is None:
                return False
            h_mask, w_mask = mask.shape[:2]
            xi = int(round(x))
            yi = int(round(y))
            x1 = max(0, xi - radius)
            y1 = max(0, yi - radius)
            x2 = min(w_mask, xi + radius + 1)
            y2 = min(h_mask, yi + radius + 1)
            if x1 >= x2 or y1 >= y2:
                return False
            roi = mask[y1:y2, x1:x2]
            return cv2.countNonZero(roi) > 0

        def v_line_covers_point(v_line, y, tol=PIX_TOL):
            # Use a more lenient tolerance for point coverage
            # Line detection and intersection detection may have slight differences
            # Allow points slightly outside the line's recorded range
            # Use 2x tolerance or 10px, whichever is larger (but cap at reasonable max)
            extended_tol = min(max(tol * 2, 10), 20)  # Between 10-20px
            return v_line and (v_line['start'] - extended_tol <= y <= v_line['end'] + extended_tol)

        def are_h_connected(a, b, tol=PIX_TOL):
            ha = a.get('h_line')
            hb = b.get('h_line')
            if not ha or not hb:
                log(f"      [H-CONN DEBUG] No h_line: a={ha is not None}, b={hb is not None}")
                fallback_radius = max(3, tol // 2)
                if all(_mask_hits(getattr(self, 'horizontal_mask', None), pt['x'], pt['y'], fallback_radius)
                       for pt in (a, b)):
                    edge_type = 'top' if a['y'] <= b['y'] else 'bottom'
                    fallback_ok, fallback_cov = self._line_exists_between(
                        (a['x'], a['y']),
                        (b['x'], b['y']),
                        edge_type=edge_type,
                        min_coverage=0.40
                    )
                    if fallback_ok:
                        log(f"      [H-CONN DEBUG] Fallback coverage succeeded ({fallback_cov:.2f}) for {edge_type} edge")
                        return True
                return False
            
            # Check if points are covered by their respective lines (use more lenient tolerance for coverage)
            coverage_tol = max(tol * 2, 10)  # More lenient for point coverage
            a_covered = h_line_covers_point(ha, a['x'], coverage_tol)
            b_covered = h_line_covers_point(hb, b['x'], coverage_tol)
            
            # Check if lines overlap in X direction (use more lenient tolerance)
            overlap_tol = max(tol * 2, 10)
            left = max(min(a['x'], b['x']), max(ha['start'], hb['start']) - overlap_tol)
            right = min(max(a['x'], b['x']), min(ha['end'], hb['end']) + overlap_tol)
            x_overlap = right >= left
            
            # Calculate Y position difference
            y_diff = abs(ha['position'] - hb['position'])
            
            # More lenient check: if lines overlap in X and points are covered, 
            # allow larger Y differences (up to 100px or 6% of image height) as lines may be slightly misaligned
            # For 1902x1227 image: 6% = ~74px, so max would be 100px
            max_y_tolerance = max(100, int(img_h * 0.06))  # At least 100px or 6% of image height
            
            if y_diff > tol:
                # If lines overlap and points are covered, be more lenient with Y tolerance
                if x_overlap and a_covered and b_covered and y_diff <= max_y_tolerance:
                    log(f"      [H-CONN DEBUG] Y mismatch but accepting: ha_pos={ha['position']}, hb_pos={hb['position']}, diff={y_diff}px (lenient: max={max_y_tolerance}px, overlap={x_overlap}, covered={a_covered}/{b_covered})")
                    return True
                else:
                    log(f"      [H-CONN DEBUG] Y mismatch: ha_pos={ha['position']}, hb_pos={hb['position']}, diff={y_diff}px (max allowed: {max_y_tolerance}px, overlap={x_overlap}, covered={a_covered}/{b_covered})")
                    return False
            
            if not a_covered or not b_covered:
                log(f"      [H-CONN DEBUG] Point not in line: ha covers a[{a['x']}]? {a_covered}, hb covers b[{b['x']}]? {b_covered}")
                log(f"                     ha_range=[{ha['start']}-{ha['end']}], hb_range=[{hb['start']}-{hb['end']}]")
                return False
            
            if not x_overlap:
                log(f"      [H-CONN DEBUG] No overlap: left={left}, right={right}, a_x={a['x']}, b_x={b['x']}")
            return x_overlap

        def are_v_connected(a, b, tol=PIX_TOL):
            va = a.get('v_line')
            vb = b.get('v_line')
            if not va or not vb:
                log(f"      [V-CONN DEBUG] No v_line: a={va is not None}, b={vb is not None}")
                fallback_radius = max(3, tol // 2)
                if all(_mask_hits(getattr(self, 'vertical_mask', None), pt['x'], pt['y'], fallback_radius)
                       for pt in (a, b)):
                    edge_type = 'left' if a['x'] <= b['x'] else 'right'
                    fallback_ok, fallback_cov = self._line_exists_between(
                        (a['x'], a['y']),
                        (b['x'], b['y']),
                        edge_type=edge_type,
                        min_coverage=0.40
                    )
                    if fallback_ok:
                        log(f"      [V-CONN DEBUG] Fallback coverage succeeded ({fallback_cov:.2f}) for {edge_type} edge")
                        return True
                return False
            if abs(va['position'] - vb['position']) > tol:
                log(f"      [V-CONN DEBUG] X mismatch: va_pos={va['position']}, vb_pos={vb['position']}, diff={abs(va['position'] - vb['position'])}")
                return False
            if not v_line_covers_point(va, a['y'], tol) or not v_line_covers_point(vb, b['y'], tol):
                log(f"      [V-CONN DEBUG] Point not in line: va covers a[{a['y']}]? {v_line_covers_point(va, a['y'], tol)}, vb covers b[{b['y']}]? {v_line_covers_point(vb, b['y'], tol)}")
                log(f"                     va_range=[{va['start']}-{va['end']}], vb_range=[{vb['start']}-{vb['end']}]")
                return False
            top = max(min(a['y'], b['y']), max(va['start'], vb['start']) - tol)
            bottom = min(max(a['y'], b['y']), min(va['end'], vb['end']) + tol)
            overlap = bottom >= top
            if not overlap:
                log(f"      [V-CONN DEBUG] No overlap: top={top}, bottom={bottom}, a_y={a['y']}, b_y={b['y']}")
            return overlap
        
        def has_intermediate_line_blocking_cell(tl, tr, bl, br, h_lines, v_lines, tolerance=5):
            """
            Check if there are intermediate lines that block cell formation.
            Tests BOTH horizontal and vertical lines to determine which actually exists.
            VALIDATES that detected lines actually exist in the mask (not false positives).
            
            A blocked cell is one where:
            - There's a STRONG horizontal line crossing through the middle (suggests merged rows)
            - There's a STRONG vertical line crossing through the middle (suggests merged columns)
            
            Returns: (is_blocked, line_type, details)
            """
            # Get cell bounds
            left_x = min(tl['x'], bl['x'], tr['x'], br['x'])
            right_x = max(tl['x'], bl['x'], tr['x'], br['x'])
            top_y = min(tl['y'], tr['y'], bl['y'], br['y'])
            bottom_y = max(tl['y'], tr['y'], bl['y'], br['y'])
            
            cell_width = right_x - left_x
            cell_height = bottom_y - top_y
            
            # Interior range where blocking lines would appear (stricter interior check)
            interior_y_min = top_y + tolerance
            interior_y_max = bottom_y - tolerance
            interior_x_min = left_x + tolerance
            interior_x_max = right_x - tolerance
            
            edge_tolerance = max(tolerance, blocker_tolerance)

            # Check for intermediate HORIZONTAL lines
            # IMPORTANT: Check lines close to edges too (within tolerance)
            # because a line at the exact boundary can block cells
            h_lines_found = []
            for h_line in h_lines:
                if top_y < h_line['position'] < bottom_y:
                    # Check if line spans across the cell width
                    if h_line['start'] <= right_x and h_line['end'] >= left_x:
                        # Measure how much of the cell width it covers
                        overlap_start = max(h_line['start'], left_x)
                        overlap_end = min(h_line['end'], right_x)
                        overlap_ratio = (overlap_end - overlap_start) / cell_width if cell_width > 0 else 0
                        
                        # CRITICAL: Only consider STRONG lines (>=70% coverage) to block cells
                        # This filters out weak artifacts and noise
                        touches_left = h_line['start'] <= left_x + edge_tolerance
                        touches_right = h_line['end'] >= right_x - edge_tolerance

                        if overlap_ratio >= 0.70 and touches_left and touches_right:
                            # VALIDATE: Check if this line actually exists in the mask
                            is_real, actual_coverage = self._line_exists_between(
                                (overlap_start, h_line['position']),
                                (overlap_end, h_line['position']),
                                edge_type='top',
                                min_coverage=0.50  # More lenient for intermediate line validation
                            )
                            
                            # Only block if line is REALLY there in the mask
                            if is_real and actual_coverage >= 0.60:
                                h_lines_found.append({
                                    'position': h_line['position'],
                                    'start': h_line['start'],
                                    'end': h_line['end'],
                                    'overlap_ratio': overlap_ratio,
                                    'span': overlap_end - overlap_start,
                                    'actual_coverage': actual_coverage
                                })
            
            # Check for intermediate VERTICAL lines
            # IMPORTANT: Check lines very close to edges too (within tolerance)
            # because a line at the exact edge blocks cells that span it
            v_lines_found = []
            for v_line in v_lines:
                # Include lines that are interior OR very close to edges (within tolerance+1)
                if left_x < v_line['position'] < right_x:
                    # Check if line spans across the cell height
                    if v_line['start'] <= bottom_y and v_line['end'] >= top_y:
                        # Measure how much of the cell height it covers
                        overlap_start = max(v_line['start'], top_y)
                        overlap_end = min(v_line['end'], bottom_y)
                        overlap_ratio = (overlap_end - overlap_start) / cell_height if cell_height > 0 else 0
                        
                        # CRITICAL: Only consider STRONG lines (>=70% coverage) to block cells
                        # This filters out weak artifacts and noise
                        touches_top = v_line['start'] <= top_y + edge_tolerance
                        touches_bottom = v_line['end'] >= bottom_y - edge_tolerance

                        if overlap_ratio >= 0.70 and touches_top and touches_bottom:
                            # VALIDATE: Check if this line actually exists in the mask
                            is_real, actual_coverage = self._line_exists_between(
                                (v_line['position'], overlap_start),
                                (v_line['position'], overlap_end),
                                edge_type='left',
                                min_coverage=0.50  # More lenient for intermediate line validation
                            )
                            
                            # Only block if line is REALLY there in the mask
                            if is_real and actual_coverage >= 0.60:
                                v_lines_found.append({
                                    'position': v_line['position'],
                                    'start': v_line['start'],
                                    'end': v_line['end'],
                                    'overlap_ratio': overlap_ratio,
                                    'span': overlap_end - overlap_start,
                                    'actual_coverage': actual_coverage
                                })
            
            # Determine if blocked and which line type
            if h_lines_found and v_lines_found:
                # Both types found - choose the one with strongest evidence (most overlap)
                best_h = max(h_lines_found, key=lambda x: x['actual_coverage'])
                best_v = max(v_lines_found, key=lambda x: x['actual_coverage'])
                
                if best_h['actual_coverage'] > best_v['actual_coverage']:
                    return True, 'horizontal', {
                        'count': len(h_lines_found),
                        'best': best_h,
                        'lines': h_lines_found
                    }
                else:
                    return True, 'vertical', {
                        'count': len(v_lines_found),
                        'best': best_v,
                        'lines': v_lines_found
                    }
            elif h_lines_found:
                return True, 'horizontal', {
                    'count': len(h_lines_found),
                    'best': max(h_lines_found, key=lambda x: x['actual_coverage']),
                    'lines': h_lines_found
                }
            elif v_lines_found:
                return True, 'vertical', {
                    'count': len(v_lines_found),
                    'best': max(v_lines_found, key=lambda x: x['actual_coverage']),
                    'lines': v_lines_found
                }
            
            return False, None, {}

        def find_intersection_at(x, y, tol=PIX_TOL):
            for inter in intersections:
                if abs(inter['x'] - x) <= tol and abs(inter['y'] - y) <= tol:
                    return inter
            return None

        def row_list_for(inter):
            key = int(round(inter['y']))
            return h_map.get(key, [])

        def col_list_for(inter):
            key = int(round(inter['x']))
            return v_map.get(key, [])

        def find_tr_br_candidates_for_tl(tl, bl_override=None):
            """Return all (tr, br) candidate pairs to the right of TL for a given BL."""
            row = row_list_for(tl)
            bl = bl_override
            if bl is None:
                col = col_list_for(tl)
                if col:
                    idx = min(range(len(col)), key=lambda i: abs(col[i]['y'] - tl['y']))
                    for j in range(idx + 1, len(col)):
                        if col[j]['y'] > tl['y']:
                            bl = col[j]
                            break
            if not bl:
                return []
            candidates = [p for p in row if p['x'] > tl['x']]
            candidates.sort(key=lambda p: p['x'])
            pairs = []
            for tr in candidates:
                br = find_intersection_at(tr['x'], bl['y'])
                if br:
                    pairs.append((tr, br))
            return pairs

        def infer_br(tr, bl):
            vline = tr.get('v_line')
            hline = bl.get('h_line')
            if vline and hline:
                if (vline['start'] - PIX_TOL <= bl['y'] <= vline['end'] + PIX_TOL and
                        hline['start'] - PIX_TOL <= tr['x'] <= hline['end'] + PIX_TOL):
                    return {'x': tr['x'], 'y': bl['y'], 'h_line': hline, 'v_line': vline, '_synthetic': True}
            return find_intersection_at(tr['x'], bl['y'])

        def find_first_valid_tr_for_tl(tl):
            row = row_list_for(tl)
            col = col_list_for(tl)
            bl = None
            if col:
                idx = min(range(len(col)), key=lambda i: abs(col[i]['y'] - tl['y']))
                for j in range(idx + 1, len(col)):
                    if col[j]['y'] > tl['y']:
                        bl = col[j]
                        break
            if not bl:
                return None, None
            candidates = [p for p in row if p['x'] > tl['x']]
            candidates.sort(key=lambda p: p['x'])
            for tr in candidates:
                br = find_intersection_at(tr['x'], bl['y'])
                if br:
                    return tr, br
            return None, None

        def find_first_valid_tl_for_tr(tr):
            row = row_list_for(tr)
            col = col_list_for(tr)
            br = None
            if col:
                idx = min(range(len(col)), key=lambda i: abs(col[i]['y'] - tr['y']))
                for j in range(idx + 1, len(col)):
                    if col[j]['y'] > tr['y']:
                        br = col[j]
                        break
            if not br:
                return None, None
            candidates = [p for p in row if p['x'] < tr['x']]
            candidates.sort(key=lambda p: -p['x'])
            for tl in candidates:
                # Find bottom-left at (tl.x, br.y)
                bl = find_intersection_at(tl['x'], br['y'])
                if bl:
                    return tl, bl
            return None, None

        def find_first_valid_br_for_bl(bl):
            row = row_list_for(bl)
            col = col_list_for(bl)
            tl = None
            if col:
                idx = min(range(len(col)), key=lambda i: abs(col[i]['y'] - bl['y']))
                for j in range(idx - 1, -1, -1):
                    if col[j]['y'] < bl['y']:
                        tl = col[j]
                        break
            if not tl:
                return None, None
            candidates = [p for p in row if p['x'] > bl['x']]
            candidates.sort(key=lambda p: p['x'])
            for br in candidates:
                # Find top-right at (br.x, tl.y)
                tr = find_intersection_at(br['x'], tl['y'])
                if tr:
                    return tr, br
            return None, None

        def find_first_valid_tl_for_br(br):
            row = row_list_for(br)
            col = col_list_for(br)
            tr = None
            if col:
                idx = min(range(len(col)), key=lambda i: abs(col[i]['y'] - br['y']))
                for j in range(idx - 1, -1, -1):
                    if col[j]['y'] < br['y']:
                        tr = col[j]
                        break
            if not tr:
                return None, None, None
            candidates = [p for p in row if p['x'] < br['x']]
            candidates.sort(key=lambda p: -p['x'])
            for tl in candidates:
                # Find bottom-left at (tl.x, br.y)
                bl = find_intersection_at(tl['x'], br['y'])
                if bl:
                    return tl, tr, bl
            return None, None, None

        def log_intersection(prefix, inter):
            return f"{prefix}#{inter.get('_idx', -1):03d} (x={inter['x']}, y={inter['y']})" if inter else f"{prefix}None"

        def create_cell(tl, tr, bl, br, context):
            nonlocal next_cell_id
            img_h, img_w = image.shape[:2]
            
            # Extract attempt number from context for visualization
            import re
            attempt_match = re.search(r'\[Attempt (\d+)\]', context)
            attempt_num = int(attempt_match.group(1)) if attempt_match else 0
            
            log("")
            log(context)
            log(f"  TL: {log_intersection('', tl)}")
            log(f"  TR: {log_intersection('', tr)}")
            log(f"  BL: {log_intersection('', bl)}")
            log(f"  BR: {log_intersection('', br)}")
            if not all([tl, tr, bl, br]):
                log("  Result: rejected — missing corners")
                return False
            top_conn = are_h_connected(tl, tr)
            bottom_conn = are_h_connected(bl, br)
            left_conn = are_v_connected(tl, bl)
            right_conn = are_v_connected(tr, br)
            log(f"  Connectivity top/bottom/left/right: {top_conn}/{bottom_conn}/{left_conn}/{right_conn}")
            
            # Additional debug for right side if it failed
            if not right_conn:
                tr_vline = tr.get('v_line')
                br_vline = br.get('v_line')
                if tr_vline and br_vline:
                    log(f"  [RIGHT-SIDE DEBUG] TR v_line: pos={tr_vline['position']}, range=[{tr_vline['start']}-{tr_vline['end']}]")
                    log(f"  [RIGHT-SIDE DEBUG] BR v_line: pos={br_vline['position']}, range=[{br_vline['start']}-{br_vline['end']}]")
                    log(f"  [RIGHT-SIDE DEBUG] TR point: y={tr['y']}, covered? {v_line_covers_point(tr_vline, tr['y'], PIX_TOL)}")
                    log(f"  [RIGHT-SIDE DEBUG] BR point: y={br['y']}, covered? {v_line_covers_point(br_vline, br['y'], PIX_TOL)}")
                    log(f"  [RIGHT-SIDE DEBUG] Position diff: {abs(tr_vline['position'] - br_vline['position'])} (tolerance: {PIX_TOL})")
                else:
                    log(f"  [RIGHT-SIDE DEBUG] Missing v_line: TR={tr_vline is not None}, BR={br_vline is not None}")
            connectivity_ok = top_conn and bottom_conn and left_conn and right_conn

            is_blocked, line_type, block_details = has_intermediate_line_blocking_cell(tl, tr, bl, br, h_lines, v_lines)
            if is_blocked:
                if line_type == 'horizontal':
                    log(f"  Result: rejected — {block_details['count']} intermediate horizontal line(s) detected "
                        f"(overlap: {block_details['best']['overlap_ratio']:.1%}) (merged rows)")
                elif line_type == 'vertical':
                    log(f"  Result: rejected — {block_details['count']} intermediate vertical line(s) detected "
                        f"(overlap: {block_details['best']['overlap_ratio']:.1%}) (merged columns)")
                return False
            
            x1, y1 = tl['x'], tl['y']
            x2, y2 = tr['x'], tr['y']
            x3, y3 = bl['x'], bl['y']
            width = x2 - x1
            height = y3 - y1
            log(f"  Size: width={width}, height={height}")
            if width < MIN_CELL_W or height < MIN_CELL_H:
                log("  Result: rejected — too small")
                return False
            top_ok, top_cov = self._line_exists_between((x1, y1), (x2, y1), edge_type='top')
            bottom_ok, bottom_cov = self._line_exists_between((x1, y3), (x2, y3), edge_type='bottom')
            left_ok, left_cov = self._line_exists_between((x1, y1), (x1, y3), edge_type='left')
            right_ok, right_cov = self._line_exists_between((x2, y1), (x2, y3), edge_type='right')
            log(f"  Coverage top/bottom/left/right: "
                f"{top_cov:.2f}/{bottom_cov:.2f}/{left_cov:.2f}/{right_cov:.2f}")

            # Size-sensitive coverage thresholds - smaller cells need higher coverage
            # Calculate cell area for size-based threshold
            cell_area = width * height
            img_area = img_w * img_h
            cell_area_ratio = cell_area / img_area if img_area > 0 else 1.0
            
            # Smaller cells need higher coverage thresholds
            if cell_area_ratio < 0.001:  # Very small cells (< 0.1% of image)
                min_coverage_threshold = 0.50  # Require 50% for very small cells
            elif cell_area_ratio < 0.005:  # Small cells (< 0.5% of image)
                min_coverage_threshold = 0.45  # Require 45% for small cells
            else:  # Normal/large cells
                min_coverage_threshold = 0.40  # 40% for normal cells
            
            # Special handling for weak left vertical lines
            if not left_ok and top_ok and bottom_ok and right_ok:
                # If top, bottom, right are good but left is weak, check if left edge is near image boundary
                if x1 < img_w * 0.10:
                    log(f"  Left edge near boundary (x={x1} < {img_w*0.10:.0f}) - accepting despite coverage={left_cov:.2f}")
                    left_ok = True
                # For very low coverage (< 0.30), check against original/unclean mask and be stricter
                elif left_cov < 0.30:
                    # Check against original combined_mask (unclean) if available
                    if hasattr(self, 'combined_mask') and self.combined_mask is not None:
                        left_ok_unclean, left_cov_unclean = self._line_exists_between_unclean((x1, y1), (x1, y3), edge_type='left')
                        if left_ok_unclean and left_cov_unclean >= min_coverage_threshold:
                            log(f"  Left edge passed unclean mask check: coverage={left_cov_unclean:.2f} (filtered={left_cov:.2f}) - accepting")
                            left_ok = True
                        else:
                            log(f"  Left edge failed unclean mask check: coverage={left_cov_unclean:.2f} < {min_coverage_threshold:.2f} (filtered={left_cov:.2f}) - rejecting")
                            left_ok = False
                    else:
                        log(f"  Left edge coverage too low ({left_cov:.2f} < 0.30) and no unclean mask - rejecting")
                        left_ok = False
                # Or check if there's connectivity (line exists but may be slightly misaligned) - but require higher threshold for small cells
                elif left_conn and left_cov >= max(0.30, min_coverage_threshold * 0.75):  # At least 30% or 75% of min threshold
                    log(f"  Left edge has connectivity and partial coverage ({left_cov:.2f}) - accepting")
                    left_ok = True
                # Or check if there's a significant line detected (size-sensitive minimum)
                elif left_cov >= min_coverage_threshold:
                    log(f"  Left edge has strong partial coverage ({left_cov:.2f}) >= {min_coverage_threshold:.2f} - accepting")
                    left_ok = True
                else:
                    log(f"  Left edge coverage too weak ({left_cov:.2f} < {min_coverage_threshold:.2f}) - rejecting")
            elif left_ok and left_cov < 0.40:
                # If left_ok from coverage check but coverage is extremely weak (<40%), be more strict
                # Only accept if at boundary
                if x1 < img_w * 0.10:
                    log(f"  Left edge at boundary with weak coverage ({left_cov:.2f}) - accepting")
                else:
                    log(f"  Left edge has insufficient coverage ({left_cov:.2f}) even though initially passed - rejecting")
                    left_ok = False

            # Special handling for weak right vertical lines
            if not right_ok and top_ok and bottom_ok and left_ok:
                # If top, bottom, left are good but right is weak, check if right edge is near image boundary
                if x2 > img_w * 0.90:
                    log(f"  Right edge near boundary (x={x2} > {img_w*0.90:.0f}) - accepting despite coverage={right_cov:.2f}")
                    right_ok = True
                # For very low coverage (< 0.30), check against original/unclean mask and be stricter
                elif right_cov < 0.30:
                    # Check against original combined_mask (unclean) if available
                    if hasattr(self, 'combined_mask') and self.combined_mask is not None:
                        right_ok_unclean, right_cov_unclean = self._line_exists_between_unclean((x2, y1), (x2, y3), edge_type='right')
                        if right_ok_unclean and right_cov_unclean >= min_coverage_threshold:
                            log(f"  Right edge passed unclean mask check: coverage={right_cov_unclean:.2f} (filtered={right_cov:.2f}) - accepting")
                            right_ok = True
                        else:
                            log(f"  Right edge failed unclean mask check: coverage={right_cov_unclean:.2f} < {min_coverage_threshold:.2f} (filtered={right_cov:.2f}) - rejecting")
                            right_ok = False
                    else:
                        log(f"  Right edge coverage too low ({right_cov:.2f} < 0.30) and no unclean mask - rejecting")
                        right_ok = False
                # Or check if there's connectivity (line exists but may be slightly misaligned) - but require higher threshold for small cells
                elif right_conn and right_cov >= max(0.30, min_coverage_threshold * 0.75):  # At least 30% or 75% of min threshold
                    log(f"  Right edge has connectivity and partial coverage ({right_cov:.2f}) - accepting")
                    right_ok = True
                # Or check if there's a significant line detected (size-sensitive minimum)
                elif right_cov >= min_coverage_threshold:
                    log(f"  Right edge has strong partial coverage ({right_cov:.2f}) >= {min_coverage_threshold:.2f} - accepting")
                    right_ok = True
                else:
                    log(f"  Right edge coverage too weak ({right_cov:.2f} < {min_coverage_threshold:.2f}) - rejecting")
            elif right_ok and right_cov < 0.40:
                # If right_ok from coverage check but coverage is extremely weak (<40%), be more strict
                # Only accept if at boundary
                if x2 > img_w * 0.90:
                    log(f"  Right edge at boundary with weak coverage ({right_cov:.2f}) - accepting")
                else:
                    log(f"  Right edge has insufficient coverage ({right_cov:.2f}) even though initially passed - rejecting")
                    right_ok = False

            if not all([top_ok, bottom_ok, left_ok, right_ok]):
                log("  Result: rejected — insufficient mask coverage")
                return False
            if not connectivity_ok:
                log("  Result: rejected — connectivity failed (likely merged cells)")
                return False
            cell_key = tuple(sorted([(x1, y1), (x2, y2), (x3, y3), (br['x'], br['y'])]))
            if cell_key in processed_cells:
                log(f"  Result: rejected — duplicate of cell {processed_cells[cell_key]}")
                return False
            cell = {
                'id': next_cell_id,
                'x': x1,
                'y': y1,
                'width': width,
                'height': height,
                'bbox': (x1, y1, width, height),
                'corners': {
                    'top_left': (x1, y1),
                    'top_right': (x2, y2),
                    'bottom_left': (x3, y3),
                    'bottom_right': (br['x'], br['y'])
                }
            }
            cells.append(cell)
            processed_cells[cell_key] = next_cell_id
            pts = np.array([
                (x1, y1),
                (x2, y2),
                (br['x'], br['y']),
                (x3, y3)
            ], np.int32)
            cv2.polylines(debug_img, [pts], True, (0, 255, 0), 2)
            log(f"  Result: ACCEPTED — cell id {next_cell_id}")
            next_cell_id += 1
            return True

        attempt = 0
        used_tl_bl_pairs = set()  # Track (TL_idx, BL_idx) pairs to prevent duplicate cell attempts
        
        MAX_BL_CANDIDATES = 3  # limit how many BL candidates we try per TL

        for inter in intersections:
            attempt += 1
            context = f"[Attempt {attempt:04d}] TL role starting at #{inter['_idx']:03d}"
            bl_candidates = []
            col = col_list_for(inter)
            if col:
                idx = min(range(len(col)), key=lambda i: abs(col[i]['y'] - inter['y']))
                for j in range(idx + 1, len(col)):
                    if col[j]['y'] > inter['y']:
                        bl_candidates.append(col[j])
                        if len(bl_candidates) >= MAX_BL_CANDIDATES:
                            break
            
            if not bl_candidates:
                log(f"{context} — missing corner(s)")
            else:
                accepted = False
                for bl in bl_candidates:
                    # Skip if TL-BL pair already produced a cell
                    if (inter['_idx'], bl['_idx']) in used_tl_bl_pairs:
                        log(f"{context} — TL-BL pair ({inter['_idx']},{bl['_idx']}) already used")
                        continue
                    tr_br_candidates = find_tr_br_candidates_for_tl(inter, bl)
                    if not tr_br_candidates:
                        continue
                    for tr, br in tr_br_candidates:
                        if create_cell(inter, tr, bl, br, context):
                            used_tl_bl_pairs.add((inter['_idx'], bl['_idx']))
                            accepted = True
                            break
                    if accepted:
                        break
                if not accepted:
                    log(f"{context} — no TR candidate accepted")

            attempt += 1
            context = f"[Attempt {attempt:04d}] TR role starting at #{inter['_idx']:03d}"
            tl, bl = find_first_valid_tl_for_tr(inter)
            br = None
            col = col_list_for(inter)
            if col:
                idx = min(range(len(col)), key=lambda i: abs(col[i]['y'] - inter['y']))
                for j in range(idx + 1, len(col)):
                    if col[j]['y'] > inter['y']:
                        br = col[j]
                        break
            if tl and bl and br:
                create_cell(tl, inter, bl, br, context)
            else:
                log(f"{context} — missing corners")

            attempt += 1
            context = f"[Attempt {attempt:04d}] BL role starting at #{inter['_idx']:03d}"
            tr, br = find_first_valid_br_for_bl(inter)
            tl = None
            col = col_list_for(inter)
            if col:
                idx = min(range(len(col)), key=lambda i: abs(col[i]['y'] - inter['y']))
                for j in range(idx - 1, -1, -1):
                    if col[j]['y'] < inter['y']:
                        tl = col[j]
                        break
            if tl and tr and br:
                create_cell(tl, tr, inter, br, context)
            else:
                log(f"{context} — missing corners")

            attempt += 1
            context = f"[Attempt {attempt:04d}] BR role starting at #{inter['_idx']:03d}"
            tl, tr, bl = find_first_valid_tl_for_br(inter)
            if tl and tr and bl:
                create_cell(tl, tr, bl, inter, context)
            else:
                log(f"{context} — missing corners")

        usage_count = {id(inter): 0 for inter in intersections}
        for cell in cells:
            for corner in cell['corners'].values():
                inter = find_intersection_at(*corner)
                if inter:
                    usage_count[id(inter)] += 1

        usage_img = debug_img.copy()
        for inter in intersections:
            x, y = inter['x'], inter['y']
            count = usage_count[id(inter)]
            color = (0, 255, 0) if count > 0 else (0, 0, 255)
            cv2.circle(usage_img, (x, y), 4, color, -1)
            cv2.putText(usage_img, f"{inter['_idx']}:{count}", (x + 4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        if self.debug:
            self.save_debug_image(debug_img, f"{img_name}_16_traced_cells.jpg",
                                  f"Traced {len(cells)} cells")
            self.save_debug_image(usage_img, f"{img_name}_16a_cell_usage.jpg",
                                  "Intersection usage")
            # Create cell ID visualization
            cell_id_img = image.copy()
            for cell in cells:
                x, y, w, h = cell['bbox']
                corners = cell['corners']
                
                # Draw cell rectangle
                cv2.rectangle(cell_id_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Get corner intersection IDs
                tl = find_intersection_at(*corners['top_left'])
                tr = find_intersection_at(*corners['top_right'])
                bl = find_intersection_at(*corners['bottom_left'])
                br = find_intersection_at(*corners['bottom_right'])
                
                tl_id = tl['_idx'] if tl else -1
                tr_id = tr['_idx'] if tr else -1
                bl_id = bl['_idx'] if bl else -1
                br_id = br['_idx'] if br else -1
                
                # Store IDs in cell (CRITICAL - this persists the IDs!)
                cell['corner_ids'] = {
                    'tl': tl_id,
                    'tr': tr_id,
                    'bl': bl_id,
                    'br': br_id
                }
                
                # Draw TL intersection ID prominently in center of cell
                text_x = x + w // 2 - 10
                text_y = y + h // 2
                cv2.putText(cell_id_img, str(tl_id), (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                # Optionally draw corner IDs (smaller)
                if self.debug:
                    cv2.putText(cell_id_img, str(tl_id), corners['top_left'],
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                    cv2.putText(cell_id_img, str(tr_id), corners['top_right'],
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                    cv2.putText(cell_id_img, str(bl_id), corners['bottom_left'],
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                    cv2.putText(cell_id_img, str(br_id), corners['bottom_right'],
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            
            self.save_debug_image(cell_id_img, f"{img_name}_16b_cell_ids.jpg",
                                f"Cell TL intersection IDs")

        if self.debug:
            debug_path = self.result_folder / f"{img_name}_cell_trace_debug.txt"
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(log_lines) if log_lines else "No cell trace entries recorded.")
            
            # Write summary statistics
            summary_path = self.result_folder / f"{img_name}_trace_summary.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("CELL DETECTION SUMMARY\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Intersections found: {len(intersections)}\n")
                f.write(f"Cells created: {len(cells)}\n")
                f.write(f"Cell creation attempts: {attempt}\n")
                f.write(f"Approximate attempts per intersection: {attempt // len(intersections) if intersections else 0} (typically 3-4 roles per point)\n\n")
                
                f.write("WHY SO MANY ATTEMPTS?\n")
                f.write("-" * 80 + "\n")
                f.write("The algorithm tries each intersection point in multiple roles:\n")
                f.write("  1. TL (Top-Left) role - looks for TR to the right, BL below\n")
                f.write("  2. TR (Top-Right) role - looks for TL to the left, BR below\n")
                f.write("  3. BL (Bottom-Left) role - looks for TL above, BR to the right\n")
                f.write("  4. BR (Bottom-Right) role - looks for BL above, TR to the left\n\n")
                
                f.write("This multi-role approach ensures cells can be detected from any corner,\n")
                f.write("but it means N intersections generate ~4N attempts.\n\n")
                
                f.write("WHY DO SOME CELLS SHOW AS TRIANGLES IN DEBUG IMAGES?\n")
                f.write("-" * 80 + "\n")
                f.write("Triangles appear when an attempt finds 3 valid corners but fails on the 4th.\n")
                f.write("This is correct behavior - cells must be 4-sided rectangles.\n")
                f.write("Examples:\n")
                f.write("  - Missing BR corner → Triangle with TL, TR, BL\n")
                f.write("  - BR exists but connectivity fails → Shows 3 connected + 1 failing\n")
                f.write("  - Bottom line broken → BL-BR connection shows red X\n\n")
                
                f.write("CONNECTIVITY VS COVERAGE\n")
                f.write("-" * 80 + "\n")
                f.write("These are checked separately:\n\n")
                
                f.write("CONNECTIVITY (uses h_line/v_line references):\n")
                f.write("  - Checks if intersection points have line references\n")
                f.write("  - Validates lines overlap between corners\n")
                f.write("  - Detects missing internal lines (merged cells)\n")
                f.write("  - MUST PASS: All 4 edges connected\n\n")
                
                f.write("COVERAGE (checks actual pixels in mask):\n")
                f.write("  - Samples pixels along expected edge lines\n")
                f.write("  - Requires 40% for horizontal, 35% for vertical edges\n")
                f.write("  - More forgiving than connectivity\n")
                f.write("  - MUST PASS: All 4 edges >= threshold\n\n")
                
                f.write("Issue in your example:\n")
                f.write("  - CONNECTIVITY: BOTTOM edge FAILED (True/False/True/True)\n")
                f.write("  - COVERAGE: All edges 1.00 (perfect coverage)\n")
                f.write("  - Result: REJECTED because connectivity is mandatory\n")
                f.write("  - This means: Corners exist, lines detected, but NOT properly connected\n\n")
                
                f.write("DEBUGGING THE FAILED BOTTOM EDGE:\n")
                f.write("-" * 80 + "\n")
                f.write("When an edge fails connectivity, check:\n")
                f.write("  1. Are both corners on the SAME line? (Y position for horizontal)\n")
                f.write("  2. Is the line broken/segmented? (Check visualized line ranges)\n")
                f.write("  3. Do the line spans overlap? (Check [start-end] ranges)\n")
                f.write("  4. Is the tolerance sufficient? (Currently 5 pixels)\n\n")
                
                f.write("Look for debug logging:\n")
                f.write("  [H-CONN DEBUG] - Shows exact reason for horizontal connection failure\n")
                f.write("  [V-CONN DEBUG] - Shows exact reason for vertical connection failure\n\n")
                
                f.write("CONNECTIVITY CHECK VISUALIZATION\n")
                f.write("-" * 80 + "\n")
                f.write("Each attempt saves: connectivity_check_attempt_XXXX.jpg\n")
                f.write("  - Green lines/corners: Connected successfully\n")
                f.write("  - Red lines/corners: Connection failed\n")
                f.write("  - Line labels: Show [position] and (start-end) ranges\n")
                f.write("  - Status: 'ALL CONNECTED ✓' (green) or 'FAILED ✗' (red)\n\n")
                
                f.write("NEXT STEPS TO IMPROVE CELL DETECTION\n")
                f.write("-" * 80 + "\n")
                f.write("If cells are being rejected:\n")
                f.write("  1. Check the connectivity debug images\n")
                f.write("  2. Read the debug log for [H-CONN DEBUG] or [V-CONN DEBUG] lines\n")
                f.write("  3. Verify line detection (check _14_cleaned_lines.jpg)\n")
                f.write("  4. Check if lines are broken or segmented\n")
                f.write("  5. Consider increasing PIX_TOL (currently 5 pixels) if lines are slightly offset\n")
                f.write("  6. Consider lowering coverage thresholds if lines are faint\n\n")
                
                f.write("="*80 + "\n")
            
            usage_path = self.result_folder / f"{img_name}_intersection_usage.txt"
            with open(usage_path, 'w', encoding='utf-8') as f:
                for inter in intersections:
                    count = usage_count[id(inter)]
                    f.write(f"{inter['_idx']:04d}: (x={inter['x']}, y={inter['y']}), used={count}\n")
            print(f"  Detailed cell trace debug saved to {debug_path}")
            print(f"  Intersection usage summary saved to {usage_path}")

        print(f"  Traced {len(cells)} cells")

        self._assign_cell_grid_positions(cells)
        return cells
        if self.debug:
            self.save_debug_image(debug_img, f"{img_name}_trace_cells.jpg", f"cells:{len(cells)}")
            
            # Create detailed debug visualization with coordinates and corner IDs
            debug_detailed = image.copy()
            for cell in cells:
                x, y, w, h = cell['bbox']
                tl_id = cell['corner_ids']['tl']
                tr_id = cell['corner_ids']['tr']
                bl_id = cell['corner_ids']['bl']
                br_id = cell['corner_ids']['br']
                
                # Draw cell rectangle
                cv2.rectangle(debug_detailed, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw corners as circles
                corners = cell['corners']
                cv2.circle(debug_detailed, corners['top_left'], 5, (255, 0, 0), -1)      # Blue = TL
                cv2.circle(debug_detailed, corners['top_right'], 5, (0, 255, 0), -1)     # Green = TR
                cv2.circle(debug_detailed, corners['bottom_left'], 5, (0, 0, 255), -1)   # Red = BL
                cv2.circle(debug_detailed, corners['bottom_right'], 5, (255, 255, 0), -1) # Cyan = BR
                
                # Label corners with their intersection IDs
                cv2.putText(debug_detailed, f"TL#{tl_id}", (x - 20, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                cv2.putText(debug_detailed, f"TR#{tr_id}", (x + w + 5, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(debug_detailed, f"BL#{bl_id}", (x - 20, y + h + 15), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                cv2.putText(debug_detailed, f"BR#{br_id}", (x + w + 5, y + h + 15), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Cell ID and dimensions in center
                cv2.putText(debug_detailed, f"Cell#{cell['id']} ({w}x{h})", (x + 5, y + h // 2), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            self.save_debug_image(debug_detailed, f"{img_name}_trace_cells_detailed.jpg", 
                                f"cells_detailed:{len(cells)}")
            
            try:
                p = self.result_folder
                p.mkdir(parents=True, exist_ok=True)
                with open(p / f"{img_name}_points_connection.txt", 'w', encoding='utf-8') as f:
                    for d in decisions:
                        f.write(d + "\n")
            except Exception:
                pass

        # Assign grid positions
        try:
            self._assign_cell_grid_positions(cells)
        except Exception:
            pass

        print(f"  ✓ Found {len(cells)} cells")
        return cells
    
    def _assign_cell_grid_positions(self, cells, tolerance=5):
        """Assign row/column indices to cells based on their top-left positions."""
        if not cells:
            return

        rows = []
        for cell in sorted(cells, key=lambda c: (c['y'], c['x'])):
            placed = False
            for row in rows:
                if abs(row['y'] - cell['y']) <= tolerance:
                    row['cells'].append(cell)
                    placed = True
                    break
            if not placed:
                rows.append({'y': cell['y'], 'cells': [cell]})

        rows.sort(key=lambda r: r['y'])
        for row_idx, row in enumerate(rows):
            row['cells'].sort(key=lambda c: c['x'])
            for col_idx, cell in enumerate(row['cells']):
                cell['row'] = row_idx
                cell['col'] = col_idx

    def extract_shg_id_field(self, image, cells, img_name):
        """
        Step 6: Extract SHG MBK ID field from Row 1, Column 1
        """
        print("\n" + "="*70)
        print("STEP 6: EXTRACTING SHG MBK ID FIELD")
        print("="*70)

        if not cells:
            print("  ✗ No cells available to locate SHG ID field")
            return None
        
        # Find cell at row=1, col=1 (second row, second column)
        shg_cell = None
        for cell in cells:
            if cell['row'] == 1 and cell['col'] == 1:
                shg_cell = cell
                break
        
        if not shg_cell:
            print("  ✗ Could not find SHG ID field (row=1, col=1)")
            return None
        
        x, y, w, h = shg_cell['bbox']
        print(f"  Found SHG ID field at row=1, col=1: bbox=({x}, {y}, {w}, {h})")
        
        # Extract with percentage-based padding (0.5% of cell dimensions)
        padding_percent = 0.005  # 0.5% padding
        padding_w = max(5, int(w * padding_percent))
        padding_h = max(5, int(h * padding_percent))
        padding = max(padding_w, padding_h)  # Use the larger value
        
        x1 = max(0, x + padding)
        y1 = max(0, y + padding)
        x2 = min(image.shape[1], x + w - padding)
        y2 = min(image.shape[0], y + h - padding)
        
        shg_img = image[y1:y2, x1:x2].copy()
        
        # Enhance for OCR
        enhanced = self.enhance_cell_for_ocr(shg_img)
        
        # Save to training folder
        filepath = None
        if self.return_images:
            cell_info = f"SHG_MBK_ID | Row 1, Col 1"
            filepath = self.save_training_cell(enhanced, cell_info)
        
        if self.debug:
            debug_shg = image.copy()
            cv2.rectangle(debug_shg, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(debug_shg, "SHG MBK ID", (x, y - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if self.debug:
                self.save_debug_image(debug_shg, f"{img_name}_17_shg_id_field.jpg", "SHG ID field")
        
        # Return enhanced image if return_images is True, else filepath
        return enhanced if self.return_images else filepath
    
    def get_cells_by_debug_order(self, cells):
        """
        Get cells ordered by their position (top-to-bottom, left-to-right)
        and assign them sequential debug IDs that match the debug visualization.
        
        Returns:
            dict: Mapping of debug_id -> cell
        """
        # Sort cells by position (row, then column)
        sorted_cells = sorted(cells, key=lambda c: (c['y'], c['x']))
        
        # Create mapping
        debug_id_to_cell = {}
        for idx, cell in enumerate(sorted_cells):
            debug_id = idx  # 0-based index matching debug visualization
            cell['debug_id'] = debug_id
            debug_id_to_cell[debug_id] = cell
        
        return debug_id_to_cell
    
    def filter_cells_by_intersection_ids(self, cells, tl_ids):
        """
        Filter cells by their top-left intersection IDs
        
        DEPRECATED: Use debug IDs instead for easier configuration
        
        Args:
            cells: List of cell dictionaries
            tl_ids: List of top-left intersection IDs or tuples for ranges
                    Examples: [5, 10, 15] or [(5, 15), 20, (30, 40)]
                    
        Returns:
            List of filtered cells
        """
        print("⚠ Warning: Using intersection IDs is deprecated. Consider using debug IDs instead.")
        
        # Expand ranges
        expanded_ids = set()
        for item in tl_ids:
            if isinstance(item, tuple) and len(item) == 2:
                # Range: (start, end) inclusive
                start, end = item
                expanded_ids.update(range(start, end + 1))
            else:
                # Single ID
                expanded_ids.add(item)
        
        # Filter cells
        filtered = []
        for cell in cells:
            tl_id = cell.get('corner_ids', {}).get('tl', -1)
            if tl_id in expanded_ids:
                filtered.append(cell)
        
        return filtered
    
    def filter_and_extract_cells(self, image, cells, img_name):
        """
        Step 7: Filter cells and extract with dynamic padding
        """
        if self.debug:
            print("\n" + "="*70)
            print("STEP 7: FILTERING AND EXTRACTING CELLS")
            print("="*70)

        if not cells:
            if self.debug:
                print("  âœ— No cells available to filter")
            return []
        
        # Get size-adaptive configuration
        img_h, img_w = image.shape[:2]
        size_config = self.get_size_config(img_h, img_w)
        
        # Get dynamic padding values
        padding_left = size_config['padding_left']
        padding_top = size_config['padding_top']
        padding_right = size_config['padding_right']
        padding_bottom = size_config['padding_bottom']
        
        print(f"  Using dynamic padding: left={padding_left}px, top={padding_top}px, "
          f"right={padding_right}px, bottom={padding_bottom}px")

        # Remove table lines using cell edges (smart mask based on image size)
        if self.debug:
            print("  Removing table lines using cell edges...")
        image_clean = self.remove_lines_using_cell_edges(image, cells, img_name)
        
        # Save the cleaned "no lines" image for debugging FIRST
        if self.debug:
            self.save_debug_image(image_clean, f"{img_name}_18_lines_removed.jpg", "Table with lines removed using cell edges")
        
        # ============================================================
        # STEP 1: Create debug ID mapping (matches debug visualization)
        # ============================================================
        debug_id_map = self.get_cells_by_debug_order(cells)
        
        if self.debug:
            print(f"  Created debug ID mapping for {len(debug_id_map)} cells")
            # Save a reference image showing debug IDs
            debug_ref = image.copy()
            for debug_id, cell in debug_id_map.items():
                x, y, w, h = cell['bbox']
                cv2.rectangle(debug_ref, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Draw debug ID in center
                text_x = x + w // 2 - 10
                text_y = y + h // 2
                cv2.putText(debug_ref, str(debug_id), (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            self.save_debug_image(debug_ref, f"{img_name}_18a_debug_ids.jpg", 
                                "Debug IDs (use these for cell_debug_ids)")
        
        # ============================================================
        # CONFIGURE HERE: Which cells to extract by DEBUG ID
        # ============================================================
        # These IDs match the numbers shown in the debug visualization images
        # Look at the _18a_debug_ids.jpg image to see which cells to extract
        # Order matters! Cells will be saved in the order specified here.

        def build_cell_debug_ids(num_rows):
            result = [2]  # SHG MBK ID always at front

            # Base row = your Row 2 pattern
            base_A = 25
            base_B = 26
            base_C = 27
            base_E = 29
            base_F = 36

            # The pattern increases by +18 each row
            step = 17

            for i in range(num_rows):
                offset = i * step

                A = base_A + offset
                B = base_B + offset
                C = base_C + offset
                E = base_E + offset
                F = base_F + offset

                result += [A, B, C, (E, F)]

            return result
        
        cell_debug_ids = build_cell_debug_ids(15)
        
        # ============================================================
        
        # Expand the ID list to individual IDs while preserving order
        if self.debug:
            print(f"  Filtering by debug cell IDs (preserving order)")
        
        ordered_ids = []
        for item in cell_debug_ids:
            if isinstance(item, tuple) and len(item) == 2:
                start, end = item
                ordered_ids.extend(range(start, end + 1))
            else:
                ordered_ids.append(item)
        
        # Extract cells IN THE ORDER specified
        valid_cells = []
        missing_ids = []
        
        for debug_id in ordered_ids:
            if debug_id in debug_id_map:
                cell = debug_id_map[debug_id]
                valid_cells.append(cell)
                if self.debug:
                    tl_id = cell.get('corner_ids', {}).get('tl', -1)
                    print(f"    ✓ Debug ID {debug_id}: TL_ID={tl_id}, row={cell.get('row')}, col={cell.get('col')}")
            else:
                missing_ids.append(debug_id)
                if self.debug:
                    print(f"    ✗ Debug ID {debug_id}: NOT FOUND")
        
        if self.debug:
            print(f"  Requested {len(ordered_ids)} debug IDs")
            print(f"  Found {len(valid_cells)} matching cells")
            if missing_ids:
                print(f"  ⚠ Missing {len(missing_ids)} cells: {missing_ids}")
        
        if self.debug:
            print(f"  Filtered to {len(valid_cells)} valid cells")
            print("\n  Extracting cell images...")
        
        # Extract cells IN ORDER (already ordered by debug IDs)
        extracted_cells = []

        for cell in valid_cells:
            x, y, w, h = cell['bbox']
            col = cell.get('col', -1)
            tl_id = cell.get('corner_ids', {}).get('tl', -1)
            debug_id = cell.get('debug_id', -1)

            # ============================================
            # 1) MEASURE LINE THICKNESS
            # ============================================
            if hasattr(self, 'horizontal_mask') and self.horizontal_mask is not None:
                # Horizontal line thickness
                h_mask = self.horizontal_mask
                h_labeled = cv2.connectedComponentsWithStats(h_mask, connectivity=8)
                h_stats = h_labeled[2]
                h_thicknesses = [h_stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, h_labeled[0])]
                line_thickness_h = int(np.median(h_thicknesses)) if h_thicknesses else 3

                # Vertical line thickness
                v_mask = self.vertical_mask
                v_labeled = cv2.connectedComponentsWithStats(v_mask, connectivity=8)
                v_stats = v_labeled[2]
                v_thicknesses = [v_stats[i, cv2.CC_STAT_WIDTH] for i in range(1, v_labeled[0])]
                line_thickness_v = int(np.median(v_thicknesses)) if v_thicknesses else 3
            else:
                line_thickness_h = 3
                line_thickness_v = 3

            # ============================================
            # 2) COMPUTE CROP COORDINATES WITH PADDING
            # ============================================
            # Calculate percentage-based padding (0.5% of cell dimensions)
            cell_padding_percent = 0.005  # 0.5% padding for cells
            cell_padding_w = max(padding_left, int(w * cell_padding_percent))
            cell_padding_h = max(padding_top, int(h * cell_padding_percent))
            
            # Use percentage-based padding, but ensure minimum values from size_config
            padding_left_final = max(padding_left, cell_padding_w)
            padding_top_final = max(padding_top, cell_padding_h)
            padding_right_final = max(padding_right, int(w * cell_padding_percent))
            padding_bottom_final = max(padding_bottom, int(h * cell_padding_percent))

            # Calculate crop region with percentage-based padding
            x1 = max(0, x + line_thickness_v - padding_left_final)       # Move LEFT
            y1 = max(0, y + line_thickness_h - padding_top_final)        # Move UP
            x2 = min(image_clean.shape[1], x + w - line_thickness_v + padding_right_final)   # Move RIGHT
            y2 = min(image_clean.shape[0], y + h - line_thickness_h + padding_bottom_final)  # Move DOWN

            # Check if cell is large enough after accounting for line thickness
            if x2 - x1 < 10 or y2 - y1 < 10:
                if self.debug:
                    print(f"    ⚠ Skipping cell {debug_id}: too small after line removal")
                continue

            # ============================================
            # 3) CROP THE CELL
            # ============================================
            cell_img = image_clean[y1:y2, x1:x2].copy()

            # ============================================
            # 4) OCR ENHANCEMENT
            # ============================================
            enhanced = self.enhance_cell_for_ocr(cell_img)

            # ============================================
            # 5) SAVE OR STORE
            # ============================================
            if self.return_images:
                cell_info = (
                    f"DebugID={debug_id}, Cell_ID={cell['id']}, TL_ID={tl_id}, "
                    f"Row={cell.get('row')}, Col={col}, Image={img_name}"
                )
                filepath = self.save_training_cell(enhanced, cell_info)

                extracted_cells.append({
                    'debug_id': debug_id,
                    'cell_id': cell['id'],
                    'tl_id': tl_id,
                    'row': cell.get('row', -1),
                    'col': col,
                    'y': y,
                    'x': x,
                    'bbox': (x, y, w, h),
                    'image': enhanced,
                    'path': str(filepath)
                })
            else:
                extracted_cells.append({
                    'debug_id': debug_id,
                    'cell_id': cell['id'],
                    'tl_id': tl_id,
                    'row': cell.get('row', -1),
                    'col': col,
                    'y': y,
                    'x': x,
                    'bbox': (x, y, w, h),
                    'image': enhanced
                })
        
        if self.debug:
            print(f"\n  ✓ Extracted {len(extracted_cells)} cells")
            
            # Draw debug image with extraction order
            debug_extracted = image.copy()
            for idx, cell in enumerate(valid_cells):
                x, y, w, h = cell['bbox']
                cv2.rectangle(debug_extracted, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Show both debug ID and extraction order
                debug_id = cell.get('debug_id', -1)
                cv2.putText(debug_extracted, f"#{idx+1}", (x + 5, y + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(debug_extracted, f"D{debug_id}", (x + 5, y + 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            self.save_debug_image(debug_extracted, f"{img_name}_19_extracted_cells.jpg", 
                                f"Extracted {len(extracted_cells)} cells in order")
        
        return extracted_cells
    
    def remove_table_lines(self, image):
        """
        Remove table lines precisely using mask-based width detection.
        Measures actual line thickness from masks to avoid removing text.
        """
        if self.combined_mask is None:
            return image
        
        # Work with color image
        if len(image.shape) == 2:
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_color = image.copy()
        
        if self.debug:
            print("  Measuring line widths from masks...")
        
        # === MEASURE HORIZONTAL LINE THICKNESS ===
        h_mask = self.horizontal_mask
        h_labeled = cv2.connectedComponentsWithStats(h_mask, connectivity=8)
        h_num_labels, h_labels, h_stats, h_centroids = h_labeled
        
        h_thicknesses = []
        for label in range(1, h_num_labels):
            height = h_stats[label, cv2.CC_STAT_HEIGHT]
            h_thicknesses.append(height)
        
        if h_thicknesses:
            h_median_thickness = int(np.median(h_thicknesses))
            h_removal_thickness = max(h_median_thickness + 2, 3)  # Add 2px safety margin
        else:
            h_removal_thickness = 3
        
        if self.debug:
            print(f"    Horizontal line thickness: {h_median_thickness}px → removing {h_removal_thickness}px")
        
        # === MEASURE VERTICAL LINE THICKNESS ===
        v_mask = self.vertical_mask
        v_labeled = cv2.connectedComponentsWithStats(v_mask, connectivity=8)
        v_num_labels, v_labels, v_stats, v_centroids = v_labeled
        
        v_thicknesses = []
        for label in range(1, v_num_labels):
            width = v_stats[label, cv2.CC_STAT_WIDTH]
            v_thicknesses.append(width)
        
        if v_thicknesses:
            v_median_thickness = int(np.median(v_thicknesses))
            v_removal_thickness = max(v_median_thickness + 2, 3)  # Add 2px safety margin
        else:
            v_removal_thickness = 3
        
        if self.debug:
            print(f"    Vertical line thickness: {v_median_thickness}px → removing {v_removal_thickness}px")
        
        # === CREATE PRECISE REMOVAL MASK ===
        # Dilate only by measured thickness
        h_removal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h_removal_thickness))
        h_removal_mask = cv2.dilate(h_mask, h_removal_kernel, iterations=1)
        
        v_removal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (v_removal_thickness, 1))
        v_removal_mask = cv2.dilate(v_mask, v_removal_kernel, iterations=1)
        
        # Combine masks
        final_removal_mask = cv2.bitwise_or(h_removal_mask, v_removal_mask)
        
        if self.debug:
            self.save_debug_image(final_removal_mask, f"line_removal_mask.jpg", 
                                f"Precise removal mask (H:{h_removal_thickness}px, V:{v_removal_thickness}px)")
        
        # === INPAINT TO REMOVE LINES ===
        # Use smaller inpaint radius to preserve text
        result = cv2.inpaint(image_color, final_removal_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
        
        if self.debug:
            print(f"    ✓ Lines removed with precise masking")
        
        return result
    
    def remove_lines_using_cell_edges(self, image, cells, img_name):
        """
        Remove table lines using cell edges to create a smart mask based on image size.
        Only removes lines at cell boundaries, preserving content inside cells.
        """
        if not cells:
            if self.debug:
                print("  No cells provided, cannot use cell-edge-based removal")
            return image
        
        # Work with color image
        if len(image.shape) == 2:
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_color = image.copy()
        
        img_h, img_w = image.shape[:2]
        
        if self.debug:
            print(f"  Creating smart line removal mask from {len(cells)} cell edges...")
        
        # === CALCULATE SMART LINE THICKNESS BASED ON IMAGE SIZE ===
        # Use image diagonal to determine appropriate line thickness
        img_diagonal = np.sqrt(img_h**2 + img_w**2)

        # Compute a size-aware cap so 1920x1217px images produce ~5-6px masks (never thicker)
        # Adjust the divisor (400) to change how aggressively thickness grows with image size,
        # and tweak the clamp (min(..., 6)) to enforce the maximum width.
        size_target = max(2, int(round(img_diagonal / 550)))

        # Defaults unless refined by actual mask measurements
        h_removal_thickness = size_target
        v_removal_thickness = size_target
        
        if hasattr(self, 'horizontal_mask') and self.horizontal_mask is not None:
            h_mask = self.horizontal_mask
            h_labeled = cv2.connectedComponentsWithStats(h_mask, connectivity=8)
            h_num_labels, h_labels, h_stats, h_centroids = h_labeled
            
            h_thicknesses = []
            for label in range(1, h_num_labels):
                height = h_stats[label, cv2.CC_STAT_HEIGHT]
                h_thicknesses.append(height)
            
            if h_thicknesses:
                h_median_thickness = int(round(np.median(h_thicknesses)))
                # Clamp to size_target so we never exceed the desired maximum for this image
                h_removal_thickness = max(1, min(size_target, h_median_thickness))
        
        if hasattr(self, 'vertical_mask') and self.vertical_mask is not None:
            v_mask = self.vertical_mask
            v_labeled = cv2.connectedComponentsWithStats(v_mask, connectivity=8)
            v_num_labels, v_labels, v_stats, v_centroids = v_labeled
            
            v_thicknesses = []
            for label in range(1, v_num_labels):
                width = v_stats[label, cv2.CC_STAT_WIDTH]
                v_thicknesses.append(width)
            
            if v_thicknesses:
                v_median_thickness = int(round(np.median(v_thicknesses)))
                v_removal_thickness = max(1, min(size_target, v_median_thickness))
        
        if self.debug:
            print(f"    Smart line thickness: H={h_removal_thickness}px, V={v_removal_thickness}px (image size: {img_w}x{img_h})")
        
        # === CREATE REMOVAL MASK FROM CELL EDGES ===
        # Create a mask that marks lines at cell boundaries
        removal_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        # Collect all unique edge positions from cells
        horizontal_edges = set()  # y positions
        vertical_edges = set()      # x positions
        
        for cell in cells:
            corners = cell.get('corners', {})
            if not corners:
                # Fallback to bbox if corners not available
                x, y, w, h = cell['bbox']
                corners = {
                    'top_left': (int(x), int(y)),
                    'top_right': (int(x + w), int(y)),
                    'bottom_left': (int(x), int(y + h)),
                    'bottom_right': (int(x + w), int(y + h))
                }
            
            tl = corners.get('top_left', (0, 0))
            tr = corners.get('top_right', (0, 0))
            bl = corners.get('bottom_left', (0, 0))
            br = corners.get('bottom_right', (0, 0))
            
            # Add horizontal edges (top and bottom)
            horizontal_edges.add(int(round(tl[1])))  # top y
            horizontal_edges.add(int(round(bl[1])))  # bottom y
            
            # Add vertical edges (left and right)
            vertical_edges.add(int(round(tl[0])))    # left x
            vertical_edges.add(int(round(tr[0])))    # right x
        
        # Draw horizontal lines at cell boundaries with smart thickness
        for y_pos in horizontal_edges:
            y_pos = max(0, min(img_h - 1, y_pos))
            half_thick = h_removal_thickness // 2
            y_start = max(0, y_pos - half_thick)
            y_end = min(img_h, y_pos + half_thick + 1)
            removal_mask[y_start:y_end, :] = 255
        
        # Draw vertical lines at cell boundaries with smart thickness
        for x_pos in vertical_edges:
            x_pos = max(0, min(img_w - 1, x_pos))
            half_thick = v_removal_thickness // 2
            x_start = max(0, x_pos - half_thick)
            x_end = min(img_w, x_pos + half_thick + 1)
            removal_mask[:, x_start:x_end] = 255
        
        # Refine mask: only keep pixels that overlap with actual line masks (if available)
        # This ensures we only remove actual table lines, not random content
        has_h_mask = hasattr(self, 'horizontal_mask') and self.horizontal_mask is not None
        has_v_mask = hasattr(self, 'vertical_mask') and self.vertical_mask is not None
        
        if has_h_mask and has_v_mask:
            # Both masks exist - use intersection for precision
            h_mask_refined = cv2.bitwise_and(removal_mask, self.horizontal_mask)
            v_mask_refined = cv2.bitwise_and(removal_mask, self.vertical_mask)
            final_removal_mask = cv2.bitwise_or(h_mask_refined, v_mask_refined)
        elif has_h_mask:
            # Only horizontal mask exists
            final_removal_mask = cv2.bitwise_and(removal_mask, self.horizontal_mask)
        elif has_v_mask:
            # Only vertical mask exists
            final_removal_mask = cv2.bitwise_and(removal_mask, self.vertical_mask)
        else:
            # No masks available - use cell edges directly
            if self.debug:
                print("    Warning: No line masks available, using cell edges directly")
            final_removal_mask = removal_mask
        
        # Dilate slightly to ensure complete line removal
        if h_removal_thickness > 0 or v_removal_thickness > 0:
            kernel_size = max(h_removal_thickness, v_removal_thickness)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            final_removal_mask = cv2.dilate(final_removal_mask, kernel, iterations=1)
        
        if self.debug:
            self.save_debug_image(final_removal_mask, f"{img_name}_18b_cell_edge_mask.jpg", 
                                f"Cell-edge removal mask (H:{h_removal_thickness}px, V:{v_removal_thickness}px, {len(horizontal_edges)}H edges, {len(vertical_edges)}V edges)")
        
        # === DRAW WHITE LINES AS OVERLAY TO REMOVE BLACK LINES ===
        # Instead of inpaint (which smudges), add a white overlay wherever the mask is active.
        result = image_color.copy()
        white_overlay = np.full_like(result, 255)
        cv2.add(result, white_overlay, result, mask=final_removal_mask)
        
        if self.debug:
            print(f"    ✓ Lines removed using white overlay on {len(cells)} cell boundaries (no smudging)")
        
        return result
    
    def enhance_cell_for_ocr(self, cell_img):
        """Enhance cell image for OCR"""
        if len(cell_img.shape) == 3:
            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_img.copy()
        
        # Add border
        border = 10
        gray = cv2.copyMakeBorder(gray, border, border, border, border,
                                  cv2.BORDER_CONSTANT, value=255)
        
        # Gentle denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, h=5, templateWindowSize=7, searchWindowSize=21)
        
        # Upscale 2x
        h, w = denoised.shape
        upscaled = cv2.resize(denoised, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            upscaled, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10
        )
        
        return binary
    
    def save_for_training(self, image, cells, img_name):
        """
        Save cells for training - delegates to filter_and_extract_cells
        Counter is automatically maintained across runs
        """
        print("\n" + "="*70)
        print("TRAINING SAVE MODE")
        print("="*70)
        print(f"  Starting from counter: {self.current_counter}")
        
        # filter_and_extract_cells handles everything:
        # - Filtering, sorting, extracting, enhancing, and saving
        extracted = self.filter_and_extract_cells(image, cells, img_name)
        
        print(f"  ✓ Saved {len(extracted)} training cells")
        print(f"  ✓ Counter now at: {self.current_counter}")
        print(f"  ✓ Next image will start from: {self.current_counter + 1}")
        print("="*70)
    
    def process_image(self, image_path, training_mode=False):
        """Main processing pipeline"""
        print("\n" + "#"*70)
        print(f"# SHG FORM DETECTOR - Processing: {image_path}")
        print("#"*70)
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"\n✗ ERROR: Could not read image: {image_path}")
            return None
        
        image = self.analyze_image_scale(image)
        
        img_name = Path(image_path).stem
        print(f"\nImage: {image_path}")
        print(f"Original size: {self.size_profile['original_shape'][1]}x{self.size_profile['original_shape'][0]} pixels")
        print(f"Working size: {image.shape[1]}x{image.shape[0]} pixels ({self.size_profile['category']})")
        # Preprocess
        image = self.preprocess_scanned(image)
        if self.debug:
            self.save_debug_image(image, f"{img_name}_00_preprocessed.jpg", "Document scanner preprocessing")
        if image is None:
            print(f"\n✗ ERROR: Could not read image: {image_path}")
            return None
        
        # Step 1: Detect table boundary
        bbox = self.detect_table_boundary(image, img_name)
        if bbox is None:
            print("\n✗ FAILED: Could not detect table boundary")
            return None
        
        # Step 2: Crop and deskew
        cropped = self.crop_and_deskew_table(image, bbox, img_name)
        
        # Step 3: Verify table shape
        if not self.verify_table_shape(cropped, img_name):
            print("\n✗ FAILED: Wrong table shape - not an SHG form")
            return None
        
        # Step 4: Detect lines and intersections
        h_lines, v_lines, intersections = self.detect_lines_and_intersections(cropped, img_name)
        
        # Step 5: Trace cells
        cells = self.trace_cells_from_intersections(h_lines, v_lines, intersections, img_name, cropped)
    
        if training_mode:
            self.save_for_training(cropped, cells, img_name)
            
            # Save counter ONCE at the very end
            self.save_counter()

            return {
                'training_mode': True,
                'counter': self.current_counter
            }
        
        # Normal mode: extract and return
        shg_id_resource = self.extract_shg_id_field(cropped, cells, img_name)
        extracted_cells = self.filter_and_extract_cells(cropped, cells, img_name)
        shg_id_path = None
        if isinstance(shg_id_resource, (str, Path)):
            shg_id_path = str(shg_id_resource)
        
        # Save summary
        summary = {
            'image': str(image_path),
            'table_bbox': bbox,
            'total_cells': len(cells),
            'extracted_cells': len(extracted_cells),
            'shg_id_path': str(shg_id_path) if shg_id_path else None,
            'training_counter': self.current_counter
        }
        
        json_path = self.result_folder / f"{img_name}_summary.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*70)
        print("PROCESSING COMPLETE")
        print("="*70)
        print(f"✓ Table detected and verified")
        print(f"✓ Total cells found: {len(cells)}")
        print(f"✓ Data cells extracted: {len(extracted_cells)}")
        print(f"✓ Training images: {self.current_counter}")
        print(f"✓ Results saved to: {self.result_folder}")
        print(f"✓ Training cells in: {self.training_folder}")
        print("="*70 + "\n")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='SHG Form Table Detector')
    parser.add_argument('image', help='Path to SHG form image')
    parser.add_argument('--debug', action='store_true', help='Enable debug output and save debug images')
    parser.add_argument('--training-save', action='store_true', 
                       help='Save cells for training (SHG ID + all data cells)')
    parser.add_argument('--return-images', action='store_true',
                       help='Return cell images in output (for integration)')
    parser.add_argument('--intersection-scale', type=float, default=None,
                       help='Multiplier applied to intersection tolerance (overrides env)')
    
    args = parser.parse_args()
    
    detector = SHGFormDetector(
        debug=args.debug,
        return_images=args.return_images or args.training_save,
        intersection_scale=args.intersection_scale
    )
    
    if not os.path.exists(args.image):
        print(f"✗ Error: Image file not found: {args.image}")
        return 1
    
    result = detector.process_image(args.image, training_mode=args.training_save)
    
    if result:
        return 0
    else:
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())