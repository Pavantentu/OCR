import cv2
import numpy as np
import os
from pathlib import Path
import json
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed



# ============================================================
# Module: table_detection
# ============================================================

class TableDetectionMixin:
    """Mixin class for table_detection functionality"""

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
        # OPTIMIZED: Run binarization methods in parallel
        def compute_otsu(denoised_img):
            _, binary = cv2.threshold(denoised_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            return binary
        
        def compute_adaptive(denoised_img):
            binary = cv2.adaptiveThreshold(
                denoised_img, 255, 
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV, 
                blockSize=99,  # Larger block = more stable
                C=20           # Higher C = less sensitive
            )
            return binary
        
        def compute_triangle(denoised_img):
            _, binary = cv2.threshold(denoised_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
            return binary
        
        # Parallel binarization
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=3) as executor:
                otsu_future = executor.submit(compute_otsu, denoised)
                adaptive_future = executor.submit(compute_adaptive, denoised)
                triangle_future = executor.submit(compute_triangle, denoised)
                
                binary_otsu = otsu_future.result()
                binary_adaptive = adaptive_future.result()
                binary_triangle = triangle_future.result()
        else:
            # Sequential fallback
            _, binary_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            binary_adaptive = cv2.adaptiveThreshold(
                denoised, 255, 
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV, 
                blockSize=99,
                C=20
            )
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
        
        # OPTIMIZED: Parallel horizontal and vertical line detection
        def detect_horizontal_lines(binary_img, kernel_lengths, close_iterations):
            """Detect horizontal lines with multiple kernel sizes"""
            h_masks = []
            for kernel_size in kernel_lengths:
                if self.debug:
                    print(f"    H-kernel size: {kernel_size}px")
                h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
                h_temp = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, h_kernel, iterations=1)
                h_masks.append(h_temp)
            horizontal = h_masks[0].copy()
            for mask in h_masks[1:]:
                horizontal = cv2.bitwise_or(horizontal, mask)
            h_kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 2))
            horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, h_kernel_close, iterations=close_iterations)
            return horizontal
        
        def detect_vertical_lines(binary_img, kernel_lengths, close_iterations):
            """Detect vertical lines with multiple kernel sizes"""
            v_masks = []
            for kernel_size in kernel_lengths:
                if self.debug:
                    print(f"    V-kernel size: {kernel_size}px")
                v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
                v_temp = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, v_kernel, iterations=1)
                v_masks.append(v_temp)
            vertical = v_masks[0].copy()
            for mask in v_masks[1:]:
                vertical = cv2.bitwise_or(vertical, mask)
            v_kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 20))
            vertical = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, v_kernel_close, iterations=close_iterations)
            return vertical
        
        # Run horizontal and vertical detection in parallel
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=2) as executor:
                h_future = executor.submit(
                    detect_horizontal_lines, 
                    binary_otsu, 
                    size_config['horizontal_kernel_lengths'],
                    size_config['h_close_iterations']
                )
                v_future = executor.submit(
                    detect_vertical_lines,
                    binary_otsu,
                    size_config['vertical_kernel_lengths'],
                    size_config['v_close_iterations']
                )
                
                horizontal = h_future.result()
                vertical = v_future.result()
        else:
            # Sequential fallback
            horizontal = detect_horizontal_lines(
                binary_otsu, 
                size_config['horizontal_kernel_lengths'],
                size_config['h_close_iterations']
            )
            vertical = detect_vertical_lines(
                binary_otsu,
                size_config['vertical_kernel_lengths'],
                size_config['v_close_iterations']
            )
        
        # Old sequential code removed - now using parallel functions above
        
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

