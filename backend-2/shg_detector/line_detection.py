import cv2
import numpy as np
import os
from pathlib import Path
import json
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor



# ============================================================
# Module: line_detection
# ============================================================

class LineDetectionMixin:
    """Mixin class for line_detection functionality"""

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
        # Enhanced binary for line detection with adaptive sharpening
        print("  Applying mask-based adaptive sharpening...")
        # Use consolidated sharpening method with strict validation for line detection
        enhanced_bgr = self._apply_adaptive_sharpening(image, img_name=img_name, strict_validation=True)
        # Convert back to grayscale for line detection
        if len(enhanced_bgr.shape) == 3:
            enhanced = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        else:
            enhanced = enhanced_bgr
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
        # OPTIMIZED: Parallel horizontal and vertical line detection
        def detect_horizontal_lines_parallel(binary_img, size_cfg, img_w, img_h, img_name):
            """Detect horizontal lines with adaptive gap closing"""
            print("\n  Detecting horizontal lines...")
            h_kernel_width = max(3, size_cfg['min_cell_w'] * 2)
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_width, 1))
            print(f"    Horizontal kernel size: {(h_kernel_width, 1)}")
            h_mask = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, h_kernel, iterations=1)
            # OPTIMIZED: Analyze horizontal line fragmentation (vectorized)
            h_labeled_pre = cv2.connectedComponentsWithStats(h_mask, connectivity=8)
            h_num_labels_pre, _, h_stats_pre, _ = h_labeled_pre
            # Vectorized fragmentation check (faster than loop)
            if h_num_labels_pre > 1:
                widths = h_stats_pre[1:h_num_labels_pre, cv2.CC_STAT_WIDTH]
                h_small_fragments = np.sum(widths < img_w * 0.05)
                h_fragmentation_ratio = h_small_fragments / (h_num_labels_pre - 1)
            else:
                h_fragmentation_ratio = 0
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
                enhanced_width = max(35, int(img_w * 0.025))  # Larger kernel for better connection
                h_kernel_close2 = cv2.getStructuringElement(cv2.MORPH_RECT, (enhanced_width, base_h_close_height))
                h_mask = cv2.morphologyEx(h_mask, cv2.MORPH_CLOSE, h_kernel_close2)
            h_mask = cv2.bitwise_and(h_mask, binary_img)
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
            h_mask = cv2.bitwise_and(h_mask, binary_img)
            # DEBUG: Check h_mask before saving
            h_nonzero = cv2.countNonZero(h_mask)
            print(f"    h_mask non-zero pixels: {h_nonzero}")
            if self.debug:
                self.save_debug_image(h_mask, f"{img_name}_12_horizontal_lines.jpg", "Horizontal line mask")
            return h_mask
        
        def detect_vertical_lines_parallel(binary_img, size_cfg, img_w, img_h, img_name):
            """Detect vertical lines with MORE AGGRESSIVE gap closing"""
            print("  Detecting vertical lines...")
            # 1) Extract thin vertical strokes only
            v_kernel_height = max(10, size_cfg['min_cell_h'] * 2)
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_height))
            v_mask = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, v_kernel)
            # 2) ADAPTIVE gap closing - use backup's simple approach with light adaptive enhancement
            # First analyze fragmentation to decide if we need extra help
            v_labeled_pre = cv2.connectedComponentsWithStats(v_mask, connectivity=8)
            v_num_labels_pre, _, v_stats_pre, _ = v_labeled_pre
            # OPTIMIZED: Vectorized fragmentation check
            if v_num_labels_pre > 1:
                heights = v_stats_pre[1:v_num_labels_pre, cv2.CC_STAT_HEIGHT]
                small_fragments = np.sum(heights < img_h * 0.05)
                fragmentation_ratio = small_fragments / (v_num_labels_pre - 1)
            else:
                fragmentation_ratio = 0
            # Use backup's simple approach as base (kernel=15, gentle)
            base_close_size = 15
            # Only add extra closing if fragmentation is very high (backup approach + small boost)
            if fragmentation_ratio > 0.5:
                # High fragmentation - apply base close + one more with larger kernel
                print(f"    High fragmentation detected ({fragmentation_ratio:.1%}) - applying enhanced gap closing")
                v_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, base_close_size))
                v_mask = cv2.morphologyEx(v_mask, cv2.MORPH_CLOSE, v_close_kernel)
                # Second pass with slightly larger kernel for high fragmentation
                v_close_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(img_h * 0.018))))
                v_mask = cv2.morphologyEx(v_mask, cv2.MORPH_CLOSE, v_close_kernel2)
            else:
                # Normal case - use backup's simple approach
                v_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, base_close_size))
                v_mask = cv2.morphologyEx(v_mask, cv2.MORPH_CLOSE, v_close_kernel)
            # 3) Limit to original binary so it can't invade cells
            v_mask = cv2.bitwise_and(v_mask, binary_img)
            # 4) Gentle dilation ONLY upwards/downwards (backup approach)
            v_gap_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
            v_mask = cv2.dilate(v_mask, v_gap_kernel, iterations=1)
            # final safety trim
            v_mask = cv2.bitwise_and(v_mask, binary_img)
            if self.debug:
                self.save_debug_image(v_mask, f"{img_name}_13_vertical_lines.jpg", "Vertical line mask")
            return v_mask
        
        # Run horizontal and vertical detection in parallel
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=2) as executor:
                h_future = executor.submit(
                    detect_horizontal_lines_parallel,
                    binary_working, size_config, w, h, img_name
                )
                v_future = executor.submit(
                    detect_vertical_lines_parallel,
                    binary_working, size_config, w, h, img_name
                )
                
                h_mask = h_future.result()
                v_mask = v_future.result()
        else:
            # Sequential fallback
            h_mask = detect_horizontal_lines_parallel(binary_working, size_config, w, h, img_name)
            v_mask = detect_vertical_lines_parallel(binary_working, size_config, w, h, img_name)
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

