import cv2
import numpy as np
import os
from pathlib import Path
import json
import argparse
from collections import defaultdict

class SHGFormDetector:
    def __init__(self, debug=False, return_images=False):  # Changed debug default to False
        self.debug = debug
        self.return_images = return_images
        self.result_folder = Path("result")
        self.training_folder = Path("debug-training")
        self.counter_file = self.training_folder / "counter.txt"
        
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

    def _line_exists_between(self, p1, p2, thickness=3, min_coverage=0.55, edge_type='unknown'):
        """Validate that a line between two points exists in the combined mask."""
        if self.line_validation_mask is None:
            return True, 1.0

        mask = self.line_validation_mask
        h, w = mask.shape[:2]

        x1, y1 = map(int, p1)
        x2, y2 = map(int, p2)

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        line_canvas = np.zeros_like(mask)
        cv2.line(line_canvas, (x1, y1), (x2, y2), 255, thickness=thickness)

        overlap = cv2.bitwise_and(line_canvas, mask)
        line_pixels = cv2.countNonZero(line_canvas)
        if line_pixels == 0:
            return False, 0.0

        coverage = cv2.countNonZero(overlap) / line_pixels
        
        # Adaptive threshold based on edge type
        if edge_type == 'right' or edge_type == 'left':
            # More lenient for vertical edges (especially right side which is often weak)
            adaptive_min = 0.35  # Reduced from 0.55
        else:
            adaptive_min = min_coverage
        
        return coverage >= adaptive_min, coverage
    
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
        for scale_factor in [12, 15, 20, 25, 30, 35, 40]:
            kernel_size = max(w // scale_factor, 25)
            if self.debug and scale_factor == 25:
                print(f"    H-kernel size: {kernel_size}px (scale 1/{scale_factor})")
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
            h_temp = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, h_kernel, iterations=1)
            h_masks.append(h_temp)
        
        # Combine all horizontal detections (union)
        horizontal = h_masks[0].copy()
        for mask in h_masks[1:]:
            horizontal = cv2.bitwise_or(horizontal, mask)
        
        # Close small gaps
        h_kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 2))
        horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, h_kernel_close, iterations=2)
        
        # === VERTICAL LINES ===
        v_masks = []
        for scale_factor in [12, 15, 20, 25, 30, 35, 40]:
            kernel_size = max(h // scale_factor, 25)
            if self.debug and scale_factor == 25:
                print(f"    V-kernel size: {kernel_size}px (scale 1/{scale_factor})")
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
            v_temp = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, v_kernel, iterations=1)
            v_masks.append(v_temp)
        
        # Combine all vertical detections (union)
        vertical = v_masks[0].copy()
        for mask in v_masks[1:]:
            vertical = cv2.bitwise_or(vertical, mask)
        
        # Close small gaps
        v_kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 20))
        vertical = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, v_kernel_close, iterations=2)
        
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

        # Combine
        table_structure = cv2.bitwise_or(horizontal, vertical)
        table_structure = cv2.bitwise_or(table_structure, angled_lines)
        table_structure = cv2.bitwise_or(table_structure, hough_mask)
        
        # Combine to get table structure
        table_structure = cv2.bitwise_or(horizontal, vertical)
        if self.debug:
            self.save_debug_image(table_structure, f"{img_name}_04_table_structure.jpg", "Combined table structure")
        
        # Connect nearby components
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        table_structure = cv2.dilate(table_structure, kernel_connect, iterations=2)
        
        # Close to form solid boundary
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        closed = cv2.morphologyEx(table_structure, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        if self.debug:
            self.save_debug_image(closed, f"{img_name}_05_closed.jpg", "Morphological closing")
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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
            
            print(f"  Contour {i}: area={area/img_area*100:.1f}%, aspect={aspect_ratio:.2f}, extent={extent:.2f}")
            
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

        # Add a small padding so table edges are not tight to the image border
        pad = 5
        x_p = max(0, x - pad)
        y_p = max(0, y - pad)
        w_p = min(w, w_box + 2 * pad)
        h_p = min(h, h_box + 2 * pad)

        print(f"  ✓ Selected table: bbox=({x_p}, {y_p}, {w_p}, {h_p})")
        print(f"    Coverage: {area/img_area*100:.1f}% of image, extent: {extent:.2f}")
        
        # Draw final selection
        debug_final = image.copy()
        cv2.drawContours(debug_final, [approx], -1, (0, 255, 0), 3)
        cv2.rectangle(debug_final, (x_p, y_p), (x_p + w_p, y_p + h_p), (255, 0, 0), 2)
        cv2.putText(debug_final, "TABLE BOUNDARY", (x_p + 10, y_p + 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        if self.debug:
            self.save_debug_image(debug_final, f"{img_name}_07_table_boundary.jpg", "Final table boundary")
        
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
        # Increased outer padding to ensure table edges aren't clipped when cropping
        outer_padding = 20  # Increased from 5 to 20
        padded_full = cv2.copyMakeBorder(
            image,
            outer_padding, outer_padding, outer_padding, outer_padding,
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )

        # shift bbox into padded image coordinates
        x += outer_padding
        y += outer_padding

        # Add generous safety padding to avoid cropping table content
        safety_pad = 15  # Increased from 0
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

        # Convert to grayscale and prepare for contour detection
        gray_c = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_c, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        if self.debug:
            self.save_debug_image(th, f"{img_name}_08b_threshold.jpg", "Binary threshold")

        # Use DILATION instead of erosion to connect table borders and preserve content
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        th_processed = cv2.dilate(th, kernel, iterations=2)

        if self.debug:
            self.save_debug_image(th_processed, f"{img_name}_08c_dilated.jpg", "Dilated threshold")

        # find largest contour (table outline) in the cropped image
        contours, _ = cv2.findContours(th_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        quad = None
        
        if contours:
            # Get the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            bbox_area = w * h
            area_ratio = area / bbox_area if bbox_area > 0 else 0
            
            print(f"  Largest contour area: {area} ({100*area_ratio:.1f}% of bbox area)")
            
            # Require contour to be at least 50% of bbox area to avoid false detections
            if area >= 0.5 * bbox_area:
                peri = cv2.arcLength(largest_contour, True)
                # Use smaller epsilon for more accurate quad detection
                approx = cv2.approxPolyDP(largest_contour, 0.01 * peri, True)
                
                print(f"  Contour has {len(approx)} vertices after approximation")
                
                if len(approx) == 4:
                    quad = approx.reshape(4, 2).astype(np.float32)
                    print(f"  ✓ Found valid quadrilateral")
                    
                    if self.debug:
                        # Draw the detected quad on the cropped image
                        debug_img = cropped.copy()
                        cv2.drawContours(debug_img, [approx], -1, (0, 255, 0), 3)
                        for i, pt in enumerate(approx):
                            cv2.circle(debug_img, tuple(pt[0]), 8, (255, 0, 0), -1)
                            cv2.putText(debug_img, str(i), tuple(pt[0]), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        self.save_debug_image(debug_img, f"{img_name}_08d_detected_quad.jpg", 
                                              "Detected quadrilateral")
                else:
                    print(f"  ✗ Contour is not a quadrilateral (has {len(approx)} vertices)")
            else:
                print(f"  ✗ Contour too small ({100*area_ratio:.1f}% of bbox area)")
        else:
            print("  ✗ No contours found")

        warped = None
        if quad is not None:
            rect = _order_points(quad)
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
                cropped, M, (maxWidth, maxHeight),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

            # keep the small padding around the warped image to avoid edges flush with border
            pad_debug = 10
            warped_padded = cv2.copyMakeBorder(
                warped, pad_debug, pad_debug, pad_debug, pad_debug,
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )

            if self.debug:
                # same debug filename as you requested: _08a_perspective_corrected
                self.save_debug_image(
                    warped_padded,
                    f"{img_name}_08a_perspective_corrected.jpg",
                    "Perspective corrected (quadrilateral) — padded"
                )

            warped = warped_padded
        else:
            # If no quad found, use the cropped image but still add a small pad so edges aren't flush
            print("  Using full cropped image (no perspective correction)")
            pad_debug = 10
            cropped_padded = cv2.copyMakeBorder(
                cropped, pad_debug, pad_debug, pad_debug, pad_debug,
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
            warped = cropped_padded

            if self.debug:
                # same debug filename used here too
                self.save_debug_image(
                    warped,
                    f"{img_name}_08a_perspective_corrected.jpg",
                    "No quad found — using cropped image (padded)"
                )

        # Continue with skew detection on the warped image
        working = warped if warped is not None else cropped
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Hough lines for dominant angle
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, max(100, int(min(working.shape[:2]) / 2)))

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
            (h_img, w_img) = working.shape[:2]
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
            print(f"  Applied rotation: {median_angle:.2f}°")
            if self.debug:
                self.save_debug_image(deskewed, f"{img_name}_09_deskewed.jpg", "Deskewed image")
            return deskewed

        return working
    
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
    
    def detect_lines_and_intersections(self, image, img_name):
        """
        Step 4: Detect horizontal and vertical lines, find intersection points
        """
        print("\n" + "="*70)
        print("STEP 4: DETECTING LINES AND INTERSECTIONS")
        print("="*70)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Enhanced binary for line detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
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
        
        # Detect horizontal lines
        print("\n  Detecting horizontal lines...")
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 40), 1))
        h_mask = cv2.morphologyEx(binary_working, cv2.MORPH_OPEN, h_kernel, iterations=1)
        h_kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        h_mask = cv2.morphologyEx(h_mask, cv2.MORPH_CLOSE, h_kernel_close)
        h_mask = cv2.bitwise_and(h_mask, binary_working)
        h_gap_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        h_mask = cv2.dilate(h_mask, h_gap_kernel, iterations=1)
        if self.debug:
            self.save_debug_image(h_mask, f"{img_name}_12_horizontal_lines.jpg", "Horizontal line mask")
        
        # Detect vertical lines with MORE AGGRESSIVE gap closing
        print("  Detecting vertical lines...")
        # 1) Extract thin vertical strokes only
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 60)))
        v_mask = cv2.morphologyEx(binary_working, cv2.MORPH_OPEN, v_kernel)

        # 2) Gentle CLOSE — vertical only, no horizontal spread
        v_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        v_mask = cv2.morphologyEx(v_mask, cv2.MORPH_CLOSE, v_close_kernel)

        # 3) Limit to original binary so it can't invade cells
        v_mask = cv2.bitwise_and(v_mask, binary_working)

        # 4) Gentle dilation ONLY upwards/downwards (subtle effect)
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
            if width > w * 0.3:  # Significant horizontal line
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
        v_labeled = cv2.connectedComponentsWithStats(v_mask, connectivity=8)
        v_num_labels, v_labels, v_stats, v_centroids = v_labeled

        filtered_v_mask = np.zeros_like(v_mask)

        # Dilate horizontal lines slightly for connection detection
        h_dilated = cv2.dilate(h_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

        for label in range(1, v_num_labels):
            # Get vertical line component
            component_mask = (v_labels == label).astype(np.uint8) * 255
            
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

        # NEW: Dilate vertical lines to make them thicker/wider
        print("  Dilating vertical lines...")
        v_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))  # Horizontal dilation only
        filtered_v_mask = cv2.dilate(filtered_v_mask, v_dilate_kernel, iterations=2)  # Adjust iterations for more/less thickness

        if self.debug:
            self.save_debug_image(filtered_v_mask, f"{img_name}_v_mask_dilated.jpg", "Vertical mask - dilated")

        # Step 4: Combine filtered masks
        final_mask = cv2.bitwise_or(h_mask, filtered_v_mask)

        # Make sure it's pure binary
        final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)[1]

        if self.debug:
            self.save_debug_image(final_mask, f"{img_name}_final_cleaned.jpg", "Final Cleaned Mask")
            print(f"  Final mask non-zero pixels: {cv2.countNonZero(final_mask)}")

        # ---------------- STORE CLEANED MASKS ----------------
        self.horizontal_mask = h_mask
        self.vertical_mask = filtered_v_mask
        self.combined_mask = final_mask

        validation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.line_validation_mask = cv2.dilate(final_mask, validation_kernel, iterations=1)

        if self.debug:
            self.save_debug_image(final_mask, f"{img_name}_14_cleaned_lines.jpg", "Cleaned combined line mask")
        
        # Find horizontal line positions
        h_lines = []
        h_labeled = cv2.connectedComponentsWithStats(h_mask, connectivity=8)
        h_num_labels, h_labels, h_stats, h_centroids = h_labeled
        
        for label in range(1, h_num_labels):
            y_pos = int(h_centroids[label][1])
            x = h_stats[label, cv2.CC_STAT_LEFT]
            width = h_stats[label, cv2.CC_STAT_WIDTH]
            
            if width > w * 0.3:  # Significant width
                h_lines.append({'position': y_pos, 'start': x, 'end': x + width})
        
        h_lines = sorted(h_lines, key=lambda x: x['position'])
        clustered_h_lines = []
        min_gap = max(5, h // 200)
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
        
        # Merge fragments that are vertically aligned (same X position)
        x_tolerance = max(8, w // 100)  # Increased tolerance
        merged_v_lines = []
        
        v_fragments = sorted(v_fragments, key=lambda x: x['position'])
        
        for frag in v_fragments:
            merged = False
            for existing in merged_v_lines:
                # Check if fragments are at similar X position
                if abs(frag['position'] - existing['position']) <= x_tolerance:
                    # Merge by extending the span
                    existing['start'] = min(existing['start'], frag['start'])
                    existing['end'] = max(existing['end'], frag['end'])
                    existing['height'] = existing['end'] - existing['start']
                    # Average the X position
                    existing['position'] = int((existing['position'] + frag['position']) / 2)
                    merged = True
                    break
            
            if not merged:
                merged_v_lines.append(frag.copy())
        
        print(f"  After merging: {len(merged_v_lines)} vertical lines")
        
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
        clustered_v_lines = []
        for line in v_lines:
            if not clustered_v_lines or abs(line['position'] - clustered_v_lines[-1]['position']) > min_gap:
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
        intersections = []
        
        # More aggressive dilation for intersection detection - USE CLEANED MASKS
        h_intersection = cv2.dilate(h_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10)), iterations=2)
        v_intersection = cv2.dilate(filtered_v_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1)), iterations=2)  # CHANGED
        intersection_mask = cv2.bitwise_and(h_intersection, v_intersection)
        
        if self.debug:
            self.save_debug_image(intersection_mask, f"{img_name}_14a_intersections.jpg", "Intersection mask")

        tolerance = max(12, min(h, w) // 80)  # Increased tolerance even more
        row_positions = [line['position'] for line in h_lines]
        col_positions = [line['position'] for line in v_lines]
        clustered_rows = self._cluster_values(row_positions, tolerance)
        clustered_cols = self._cluster_values(col_positions, tolerance)

        for y in clustered_rows:
            for x in clustered_cols:
                y1 = max(0, y - tolerance)
                y2 = min(h, y + tolerance)
                x1 = max(0, x - tolerance)
                x2 = min(w, x + tolerance)
                roi = intersection_mask[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                if cv2.countNonZero(roi) > 0:
                    h_ref = None
                    for line in h_lines:
                        if (abs(line['position'] - y) <= tolerance and
                                line['start'] - tolerance <= x <= line['end'] + tolerance):
                            h_ref = line
                            break

                    v_ref = None
                    for line in v_lines:
                        if (abs(line['position'] - x) <= tolerance and
                                line['start'] - tolerance <= y <= line['end'] + tolerance):
                            v_ref = line
                            break

                    intersections.append({
                        'x': x,
                        'y': y,
                        'h_line': h_ref,
                        'v_line': v_ref
                    })

        print(f"  Found {len(intersections)} intersection points")
        
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
            print("  ✗ No intersections available to trace cells")
            return []

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
        PIX_TOL = 5
        MIN_CELL_W = 5
        MIN_CELL_H = 5

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
            return h_line and (h_line['start'] - tol <= x <= h_line['end'] + tol)

        def v_line_covers_point(v_line, y, tol=PIX_TOL):
            return v_line and (v_line['start'] - tol <= y <= v_line['end'] + tol)

        def are_h_connected(a, b, tol=PIX_TOL):
            ha = a.get('h_line')
            hb = b.get('h_line')
            if not ha or not hb:
                return False
            if abs(ha['position'] - hb['position']) > tol:
                return False
            if not h_line_covers_point(ha, a['x'], tol) or not h_line_covers_point(hb, b['x'], tol):
                return False
            left = max(min(a['x'], b['x']), max(ha['start'], hb['start']) - tol)
            right = min(max(a['x'], b['x']), min(ha['end'], hb['end']) + tol)
            return right >= left

        def are_v_connected(a, b, tol=PIX_TOL):
            va = a.get('v_line')
            vb = b.get('v_line')
            if not va or not vb:
                return False
            if abs(va['position'] - vb['position']) > tol:
                return False
            if not v_line_covers_point(va, a['y'], tol) or not v_line_covers_point(vb, b['y'], tol):
                return False
            top = max(min(a['y'], b['y']), max(va['start'], vb['start']) - tol)
            bottom = min(max(a['y'], b['y']), min(va['end'], vb['end']) + tol)
            return bottom >= top

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

        def find_tr_br_candidates_for_tl(tl):
            """Return all (tr, br) candidate pairs to the right of the given TL."""
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
            connectivity_ok = top_conn and bottom_conn and left_conn and right_conn
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

            # Special handling for weak right vertical lines
            if not right_ok and top_ok and bottom_ok and left_ok:
                # If top, bottom, left are good but right is weak, check if right edge is near image boundary
                if x2 > img_w * 0.90:  # RIGHT EDGE NEAR BOUNDARY - Changed 'w' to 'img_w'
                    log(f"  Right edge near boundary - accepting despite low coverage")
                    right_ok = True
                # Or check if there's a vertical line nearby
                elif right_cov > 0.15:  # At least some line detected
                    log(f"  Right edge has partial coverage ({right_cov:.2f}) - accepting")
                    right_ok = True

            if not all([top_ok, bottom_ok, left_ok, right_ok]):
                log("  Result: rejected — insufficient mask coverage")
                return False
            if not connectivity_ok:
                log("  Connectivity failed but mask coverage is sufficient — accepting")
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
        for inter in intersections:
            attempt += 1
            context = f"[Attempt {attempt:04d}] TL role starting at #{inter['_idx']:03d}"
            candidates = find_tr_br_candidates_for_tl(inter)
            bl = None
            col = col_list_for(inter)
            if col:
                idx = min(range(len(col)), key=lambda i: abs(col[i]['y'] - inter['y']))
                for j in range(idx + 1, len(col)):
                    if col[j]['y'] > inter['y']:
                        bl = col[j]
                        break
            if bl and candidates:
                accepted = False
                for tr, br in candidates:
                    if create_cell(inter, tr, bl, br, context):
                        accepted = True
                        break
                if not accepted:
                    log(f"{context} — no TR candidate accepted")
            else:
                log(f"{context} — missing corner(s)")

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
        
        # Extract with padding
        padding = 5
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
        Step 7: Filter cells and extract
        
        **EDIT THE cell_debug_ids VARIABLE BELOW TO CHANGE WHICH CELLS TO EXTRACT**
        """
        if self.debug:
            print("\n" + "="*70)
            print("STEP 7: FILTERING AND EXTRACTING CELLS")
            print("="*70)

        if not cells:
            if self.debug:
                print("  ✗ No cells available to filter")
            return []
        
        # Remove line mask from image
        if self.debug:
            print("  Removing table lines from image...")
        image_clean = image
        if self.debug:
            self.save_debug_image(self.remove_table_lines(image_clean), f"{img_name}_18_lines_removed.jpg", "Lines removed")
        
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
            padding_left = 8      # More padding on left
            padding_top = 8       # More padding on top
            padding_right = 8     # Normal padding on right
            padding_bottom = 8    # Normal padding on bottom

            # Calculate crop region with asymmetric padding
            x1 = max(0, x + line_thickness_v - padding_left)       # Move LEFT more
            y1 = max(0, y + line_thickness_h - padding_top)        # Move UP more
            x2 = min(image_clean.shape[1], x + w - line_thickness_v + padding_right)   # Move RIGHT normally
            y2 = min(image_clean.shape[0], y + h - line_thickness_h + padding_bottom)  # Move DOWN normally

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
        
        img_name = Path(image_path).stem
        print(f"\nImage: {image_path}")
        print(f"Size: {image.shape[1]}x{image.shape[0]} pixels")
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
        shg_id = self.extract_shg_id_field(cropped, cells, img_name)
        extracted_cells = self.filter_and_extract_cells(cropped, cells, img_name)
        
        # Step 6: Extract SHG ID field
        shg_id_path = self.extract_shg_id_field(cropped, cells, img_name)
        
        # Step 7: Filter and extract cells
        extracted_cells = self.filter_and_extract_cells(cropped, cells, img_name)
        
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
    
    args = parser.parse_args()
    
    detector = SHGFormDetector(
        debug=args.debug,
        return_images=args.return_images or args.training_save
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