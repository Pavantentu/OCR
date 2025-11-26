import cv2
import numpy as np
import os
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import json

# Import the main detector
from test import SHGFormDetector


class SHGImageValidator:
    """
    Pre-validation checks for SHG form images before processing.
    Checks for:
    1. Image quality issues (blur, noise, shadows)
    2. Correct table structure (exactly 298 cells)
    """
    
    def __init__(self, debug=False):
        self.debug = debug
        self.validation_folder = Path("validation-results")
        if self.debug:
            self.validation_folder.mkdir(parents=True, exist_ok=True)
    
    def make_json_serializable(self, obj):
        """
        Convert numpy types to native Python types for JSON serialization.
        Compatible with NumPy 2.0+
        """
        if isinstance(obj, dict):
            return {key: self.make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.make_json_serializable(item) for item in obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def save_debug_image(self, image, filename, log_msg=""):
        """Save validation debug images"""
        if not self.debug:
            return
        
        filepath = self.validation_folder / filename
        cv2.imwrite(str(filepath), image)
        if log_msg:
            print(f"  [DEBUG] Saved: {filename} - {log_msg}")
    
    def check_blur(self, image: np.ndarray) -> Tuple[bool, float, str]:
        """
        Check if image is too blurry using Laplacian variance method.
        
        Returns:
            (is_acceptable, blur_score, message)
        """
        print("\n[CHECK 1/4] Blur Detection...")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate Laplacian variance (edge sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(laplacian.var())
        
        # Threshold: Lower variance = more blur
        # Typical values: <100 = very blurry, >500 = sharp
        blur_threshold = 100.0
        is_acceptable = bool(variance >= blur_threshold)
        
        status = "✓ PASS" if is_acceptable else "✗ FAIL"
        message = f"{status} - Blur score: {variance:.2f} (threshold: {blur_threshold:.2f})"
        
        print(f"  {message}")
        
        if self.debug:
            # Visualize blur regions
            edges = cv2.Canny(gray, 50, 150)
            self.save_debug_image(edges, "01_edge_detection.jpg", "Edge detection for blur analysis")
        
        return is_acceptable, variance, message
    
    def check_noise(self, image: np.ndarray) -> Tuple[bool, float, str]:
        """
        Check for excessive noise in the image.
        
        Returns:
            (is_acceptable, noise_score, message)
        """
        print("\n[CHECK 2/4] Noise Detection...")
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Estimate noise using standard deviation of Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_score = float(laplacian.std())
        
        # Threshold: Higher std = more noise
        # Typical values: <30 = clean, >80 = very noisy
        noise_threshold = 80.0
        is_acceptable = bool(noise_score <= noise_threshold)
        
        status = "✓ PASS" if is_acceptable else "✗ FAIL"
        message = f"{status} - Noise score: {noise_score:.2f} (threshold: {noise_threshold:.2f})"
        
        print(f"  {message}")
        
        if self.debug:
            # Visualize noise
            noise_map = np.abs(laplacian).astype(np.uint8)
            self.save_debug_image(noise_map, "02_noise_map.jpg", "Noise intensity map")
        
        return is_acceptable, noise_score, message
    
    def check_shadows(self, image: np.ndarray) -> Tuple[bool, float, str]:
        """
        Check for dark shadows that might affect detection.
        Uses SPECIFIC VALUE CHECK - if any region has severe shadow, fail immediately.
        No averaging - direct threshold check on worst regions.
        
        Returns:
            (is_acceptable, shadow_score, message)
        """
        print("\n[CHECK 3/4] Shadow Detection...")
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        
        # ==== SPECIFIC VALUE SEARCH (No Averaging) ====
        print("  Searching for severe shadow values in table regions...")
        
        # Divide image into grid (approximate table structure)
        grid_rows, grid_cols = 16, 13  # SHG table structure
        cell_h, cell_w = h // grid_rows, w // grid_cols
        
        severe_shadows = []  # Regions with truly problematic shadows
        
        # Direct threshold - if ANY pixel in region is this dark, it's a problem
        CRITICAL_DARKNESS = 50  # Absolute darkness threshold
        CRITICAL_REGION_MEAN = 70  # If region average is this low, it's too dark
        
        for r in range(grid_rows):
            for c in range(grid_cols):
                y1 = r * cell_h
                y2 = min((r + 1) * cell_h, h)
                x1 = c * cell_w
                x2 = min((c + 1) * cell_w, w)
                
                region = gray[y1:y2, x1:x2]
                if region.size == 0:
                    continue
                
                # Check for SPECIFIC severe darkness
                min_value = float(region.min())  # Darkest pixel in region
                region_mean = float(region.mean())
                
                # Count how many pixels are severely dark
                severely_dark_pixels = int(np.sum(region < CRITICAL_DARKNESS))
                severely_dark_pct = float((severely_dark_pixels / region.size) * 100)
                
                # FAIL CONDITIONS (specific, not averaged):
                # 1. Region has very dark pixels (>30% below critical threshold)
                # 2. OR region mean is critically low (lines will disappear)
                if severely_dark_pct > 30 or region_mean < CRITICAL_REGION_MEAN:
                    severe_shadows.append({
                        'row': int(r),
                        'col': int(c),
                        'min_value': min_value,
                        'mean_brightness': region_mean,
                        'dark_pixels_pct': severely_dark_pct,
                        'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                        'reason': 'severe_darkness' if severely_dark_pct > 30 else 'low_mean'
                    })
        
        # PASS/FAIL: If ANY severe shadow found, FAIL
        is_acceptable = bool(len(severe_shadows) == 0)
        
        # Build message
        status = "✓ PASS" if is_acceptable else "✗ FAIL"
        
        if not is_acceptable:
            message = f"{status} - Found {len(severe_shadows)} regions with SEVERE shadows"
            print(f"  {message}")
            print(f"  ⚠ CRITICAL: Detected {len(severe_shadows)} severely dark regions that will break detection")
            print(f"  🔍 Severe shadow locations:")
            
            for idx, shadow in enumerate(severe_shadows[:5], 1):  # Show first 5
                reason = "pixels too dark" if shadow['reason'] == 'severe_darkness' else "region too dim"
                print(f"      {idx}. Row {shadow['row']}, Col {shadow['col']}: "
                      f"Mean={shadow['mean_brightness']:.1f}, "
                      f"Darkest={shadow['min_value']}, "
                      f"Critical dark pixels={shadow['dark_pixels_pct']:.1f}% ({reason})")
        else:
            message = f"{status} - No severe shadows detected (checked {grid_rows*grid_cols} regions)"
            print(f"  {message}")
        
        if self.debug:
            # Visualize ONLY severe shadows (no moderate - binary pass/fail)
            shadow_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Draw severe shadows in RED with details
            for shadow in severe_shadows:
                x, y, w, h = shadow['bbox']
                cv2.rectangle(shadow_vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(shadow_vis, f"SEVERE", (x + 5, y + 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                cv2.putText(shadow_vis, f"Mean:{shadow['mean_brightness']:.0f}", (x + 5, y + 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                cv2.putText(shadow_vis, f"Min:{shadow['min_value']}", (x + 5, y + 42),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            
            self.save_debug_image(shadow_vis, "03_severe_shadows.jpg", 
                                f"Severe shadow detection ({len(severe_shadows)} found)")
            
            # Also save a heatmap showing brightness values
            # Normalize and colorize for better visualization
            normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            self.save_debug_image(heatmap, "03_brightness_heatmap.jpg", 
                                "Brightness heatmap (blue=dark, red=bright)")
        
        # Return the shadow score (number of severe shadows)
        shadow_score = float(len(severe_shadows))
        
        return is_acceptable, shadow_score, message
    
    def diagnose_cell_detection_failure(self, image_path: str, cropped, h_lines, v_lines, 
                                        intersections, cells, expected_count=298) -> Dict:
        """
        Diagnose WHY cell detection failed by analyzing specific issues.
        
        Returns detailed diagnostic information about what went wrong.
        """
        diagnostics = {
            'issues_found': [],
            'severity': 'none',
            'recommendations': []
        }
        
        if len(image_path.shape) == 3:
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped
        
        h, w = gray.shape
        detected_count = len(cells)
        missing_cells = expected_count - detected_count
        
        # 1. Check for localized shadows in table cells
        print("  🔍 Analyzing localized shadows in table area...")
        
        # Divide image into grid and check each region
        grid_rows, grid_cols = 16, 13  # Approximate SHG table structure
        cell_h, cell_w = h // grid_rows, w // grid_cols
        
        shadow_regions = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                y1 = r * cell_h
                y2 = min((r + 1) * cell_h, h)
                x1 = c * cell_w
                x2 = min((c + 1) * cell_w, w)
                
                region = gray[y1:y2, x1:x2]
                if region.size > 0:
                    mean_brightness = float(region.mean())
                    if mean_brightness < 100:  # Dark region threshold
                        shadow_regions.append({
                            'row': int(r),
                            'col': int(c),
                            'brightness': mean_brightness,
                            'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1))
                        })
        
        if shadow_regions:
            severity = 'critical' if len(shadow_regions) > 5 else 'moderate'
            diagnostics['issues_found'].append({
                'type': 'localized_shadows',
                'severity': severity,
                'count': len(shadow_regions),
                'regions': shadow_regions,
                'description': f"Found {len(shadow_regions)} dark regions in table cells"
            })
            diagnostics['recommendations'].append(
                f"Localized shadows detected in {len(shadow_regions)} cell regions - "
                "try rescanning with better lighting or adjusting document position"
            )
        
        # 2. Check for missing/weak lines
        print("  🔍 Analyzing line detection quality...")
        
        expected_h_lines = 16  # Approximate for SHG form
        expected_v_lines = 13
        
        h_line_deficit = max(0, expected_h_lines - len(h_lines))
        v_line_deficit = max(0, expected_v_lines - len(v_lines))
        
        if h_line_deficit > 0 or v_line_deficit > 0:
            diagnostics['issues_found'].append({
                'type': 'missing_lines',
                'severity': 'critical',
                'h_missing': int(h_line_deficit),
                'v_missing': int(v_line_deficit),
                'h_detected': int(len(h_lines)),
                'v_detected': int(len(v_lines)),
                'description': f"Missing {h_line_deficit} horizontal and {v_line_deficit} vertical lines"
            })
            diagnostics['recommendations'].append(
                f"Line detection incomplete ({len(h_lines)}H/{len(v_lines)}V detected) - "
                "this may be caused by faded lines, shadows obscuring borders, or poor contrast"
            )
        
        # 3. Check intersection quality
        print("  🔍 Analyzing intersection detection...")
        
        expected_intersections = expected_h_lines * expected_v_lines
        intersection_deficit = max(0, expected_intersections - len(intersections))
        
        if intersection_deficit > 20:  # Significant missing
            diagnostics['issues_found'].append({
                'type': 'missing_intersections',
                'severity': 'critical',
                'missing': int(intersection_deficit),
                'detected': int(len(intersections)),
                'expected': int(expected_intersections),
                'description': f"Missing {intersection_deficit} intersection points"
            })
            diagnostics['recommendations'].append(
                f"Only {len(intersections)}/{expected_intersections} intersections detected - "
                "table structure is incomplete, possibly due to shadows or damaged lines"
            )
        
        # 4. Analyze cell distribution (check for gaps)
        print("  🔍 Analyzing cell distribution...")
        
        if cells:
            # Group cells by row
            row_counts = {}
            for cell in cells:
                row = cell.get('row', -1)
                row_counts[row] = row_counts.get(row, 0) + 1
            
            # Find rows with significantly fewer cells
            avg_cells_per_row = detected_count / len(row_counts) if row_counts else 0
            sparse_rows = []
            
            for row, count in row_counts.items():
                if count < avg_cells_per_row * 0.5:  # Less than 50% of average
                    sparse_rows.append({'row': int(row), 'count': int(count)})
            
            if sparse_rows:
                diagnostics['issues_found'].append({
                    'type': 'sparse_rows',
                    'severity': 'moderate',
                    'rows': sparse_rows,
                    'description': f"Found {len(sparse_rows)} rows with incomplete cell detection"
                })
                diagnostics['recommendations'].append(
                    f"{len(sparse_rows)} rows have incomplete detection - "
                    "likely caused by shadows or poor contrast in those areas"
                )
        
        # 5. Check contrast issues in specific regions
        print("  🔍 Analyzing contrast distribution...")
        
        # Calculate local contrast using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        contrast_map = np.abs(laplacian)
        
        # Find regions with very low contrast
        low_contrast_regions = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                y1 = r * cell_h
                y2 = min((r + 1) * cell_h, h)
                x1 = c * cell_w
                x2 = min((c + 1) * cell_w, w)
                
                region = contrast_map[y1:y2, x1:x2]
                if region.size > 0:
                    mean_contrast = float(region.mean())
                    if mean_contrast < 5:  # Very low contrast
                        low_contrast_regions.append({
                            'row': int(r),
                            'col': int(c),
                            'contrast': mean_contrast,
                            'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1))
                        })
        
        if low_contrast_regions:
            diagnostics['issues_found'].append({
                'type': 'low_contrast',
                'severity': 'moderate',
                'count': len(low_contrast_regions),
                'regions': low_contrast_regions[:10],  # Limit to first 10
                'description': f"Found {len(low_contrast_regions)} regions with poor contrast"
            })
            diagnostics['recommendations'].append(
                f"{len(low_contrast_regions)} regions have poor contrast - "
                "lines may be too faint or washed out"
            )
        
        # Determine overall severity
        severities = [issue['severity'] for issue in diagnostics['issues_found']]
        if 'critical' in severities:
            diagnostics['severity'] = 'critical'
        elif 'moderate' in severities:
            diagnostics['severity'] = 'moderate'
        else:
            diagnostics['severity'] = 'minor'
        
        return diagnostics
    
    def check_cell_count(
        self,
        image: np.ndarray,
        image_path: str,
        capture_state: bool = False
    ) -> Dict[str, object]:
        """
        Run the main detector to verify exactly 298 cells are detected.
        Optionally capture the detector state so downstream processing can
        continue without repeating the heavy work.
        """
        print("\n[CHECK 4/4] Table Structure Validation...")
        print("  Running SHG detector to count cells (using debug ID mapping)...")
        
        temp_detector = SHGFormDetector(debug=self.debug, return_images=False)
        pipeline_state = None
        
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return {
                    'passed': False,
                    'count': 0,
                    'message': "✗ FAIL - Could not read image",
                    'state': None
                }
            
            img_name = Path(image_path).stem
            img = temp_detector.preprocess_scanned(img)
            
            bbox = temp_detector.detect_table_boundary(img, img_name)
            if bbox is None:
                return {
                    'passed': False,
                    'count': 0,
                    'message': "✗ FAIL - Could not detect table boundary",
                    'state': None
                }
            
            cropped = temp_detector.crop_and_deskew_table(img, bbox, img_name)
            
            if not temp_detector.verify_table_shape(cropped, img_name):
                return {
                    'passed': False,
                    'count': 0,
                    'message': "✗ FAIL - Wrong table shape",
                    'state': None
                }
            
            h_lines, v_lines, intersections = temp_detector.detect_lines_and_intersections(
                cropped, img_name
            )
            
            cells = temp_detector.trace_cells_from_intersections(
                h_lines, v_lines, intersections, img_name, cropped
            )
            
            debug_id_map = temp_detector.get_cells_by_debug_order(cells)
            cell_count = int(len(debug_id_map))
            expected_count = 298
            is_acceptable = bool(cell_count == expected_count)
            
            status = "✓ PASS" if is_acceptable else "✗ FAIL"
            message = f"{status} - Detected {cell_count} cells (expected: {expected_count})"
            print(f"  {message}")
            
            if not is_acceptable:
                diff = cell_count - expected_count
                if diff > 0:
                    print(f"  ⚠ {diff} extra cells detected - table may have modifications")
                else:
                    print(f"  ⚠ {abs(diff)} cells missing - running diagnostic analysis...\n")
                
                diagnostics = self.diagnose_cell_detection_failure(
                    cropped, cropped, h_lines, v_lines, intersections, cells, expected_count
                )
                
                print("  " + "="*66)
                print("  DIAGNOSTIC REPORT - Why Cell Detection Failed")
                print("  " + "="*66)
                
                if diagnostics['issues_found']:
                    print(f"  Overall Severity: {diagnostics['severity'].upper()}\n")
                    
                    for idx, issue in enumerate(diagnostics['issues_found'], 1):
                        print(f"  Issue #{idx}: {issue['type'].upper().replace('_', ' ')}")
                        print(f"    Severity: {issue['severity']}")
                        print(f"    Details: {issue['description']}")
                        
                        if issue['type'] == 'localized_shadows':
                            print(f"    Shadow regions detected: {issue['count']}")
                            if self.debug and issue['regions'][:3]:
                                print("    Sample locations:")
                                for region in issue['regions'][:3]:
                                    print(f"      - Row {region['row']}, Col {region['col']}, "
                                          f"Brightness: {region['brightness']:.1f}")
                        
                        elif issue['type'] == 'missing_lines':
                            print(f"    Horizontal lines: {issue['h_detected']}/{16} "
                                  f"(missing {issue['h_missing']})")
                            print(f"    Vertical lines: {issue['v_detected']}/{13} "
                                  f"(missing {issue['v_missing']})")
                        
                        elif issue['type'] == 'missing_intersections':
                            print(f"    Intersections: {issue['detected']}/{issue['expected']} "
                                  f"(missing {issue['missing']})")
                        
                        elif issue['type'] == 'sparse_rows':
                            print("    Rows with incomplete detection:")
                            for sparse in issue['rows'][:5]:
                                print(f"      - Row {sparse['row']}: {sparse['count']} cells")
                        
                        print()
                    
                    print("  RECOMMENDATIONS:")
                    for idx, rec in enumerate(diagnostics['recommendations'], 1):
                        print(f"    {idx}. {rec}")
                    
                    print("  " + "="*66 + "\n")
                    
                    if self.debug:
                        self.save_diagnostic_visualization(
                            cropped, diagnostics, img_name
                        )
                else:
                    print("  No specific issues identified - may be table modification")
                    print("  " + "="*66 + "\n")
            
            if self.debug and debug_id_map:
                debug_ref = cropped.copy()
                for debug_id, cell in debug_id_map.items():
                    x, y, w, h = cell['bbox']
                    cv2.rectangle(debug_ref, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    text_x = x + w // 2 - 10
                    text_y = y + h // 2
                    cv2.putText(debug_ref, str(debug_id), (text_x, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                self.save_debug_image(debug_ref, "04_cell_count_verification.jpg", 
                                    f"Verified {cell_count} cells with debug IDs")
            
            result = {
                'passed': bool(is_acceptable),
                'count': int(cell_count),
                'message': message,
                'state': None
            }
            
            if capture_state and is_acceptable:
                pipeline_state = {
                    'detector': temp_detector,
                    'img_name': img_name,
                    'bbox': bbox,
                    'cropped': cropped,
                    'h_lines': h_lines,
                    'v_lines': v_lines,
                    'intersections': intersections,
                    'cells': cells,
                    'image_path': str(image_path)
                }
                result['state'] = pipeline_state
            
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'passed': False,
                'count': 0,
                'message': f"✗ FAIL - Error during validation: {str(e)}",
                'state': None
            }
    
    def save_diagnostic_visualization(self, image, diagnostics, img_name):
        """Save visualization of detected issues"""
        vis = image.copy()
        
        for issue in diagnostics['issues_found']:
            if issue['type'] == 'localized_shadows':
                # Draw red boxes around shadow regions
                for region in issue['regions']:
                    x, y, w, h = region['bbox']
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(vis, f"Shadow", (x + 5, y + 15),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            elif issue['type'] == 'low_contrast':
                # Draw yellow boxes around low contrast regions
                for region in issue.get('regions', [])[:10]:
                    x, y, w, h = region['bbox']
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        self.save_debug_image(vis, "04_diagnostic_visualization.jpg",
                            "Problem areas highlighted")

    
    def validate_image(self, image_path: str, capture_state: bool = False):
        """
        Run all validation checks on the image.
        
        Returns:
            Dictionary with validation results and overall pass/fail status
        """
        print("\n" + "="*70)
        print("SHG FORM IMAGE VALIDATION")
        print("="*70)
        print(f"Image: {image_path}")
        
        # Check if file exists
        if not os.path.exists(image_path):
            return {
                'valid': False,
                'image': str(image_path),
                'error': 'Image file not found',
                'checks': {}
            }
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return {
                'valid': False,
                'image': str(image_path),
                'error': 'Could not read image file',
                'checks': {}
            }
        
        print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")
        
        # Run all checks
        checks = {}
        cell_result = None
        
        # 1. Blur check
        blur_pass, blur_score, blur_msg = self.check_blur(image)
        checks['blur'] = {
            'passed': bool(blur_pass),
            'score': float(blur_score),
            'message': str(blur_msg)
        }
        
        # 2. Noise check
        noise_pass, noise_score, noise_msg = self.check_noise(image)
        checks['noise'] = {
            'passed': bool(noise_pass),
            'score': float(noise_score),
            'message': str(noise_msg)
        }
        
        # 3. Shadow check
        shadow_pass, shadow_score, shadow_msg = self.check_shadows(image)
        checks['shadows'] = {
            'passed': bool(shadow_pass),
            'score': float(shadow_score),
            'message': str(shadow_msg)
        }
        
        # 4. Cell count check (only if quality checks passed)
        quality_passed = blur_pass and noise_pass and shadow_pass
        
        if quality_passed:
            cell_result = self.check_cell_count(
                image,
                image_path,
                capture_state=capture_state
            )
            checks['cell_count'] = {
                'passed': bool(cell_result['passed']),
                'count': int(cell_result['count']),
                'message': str(cell_result['message'])
            }
            cell_pass = cell_result['passed']
        else:
            print("\n[CHECK 4/4] Table Structure Validation...")
            print("  ⚠ SKIPPED - Image quality checks failed")
            checks['cell_count'] = {
                'passed': False,
                'count': 0,
                'message': '⚠ SKIPPED - Image quality insufficient'
            }
            cell_pass = False
        
        # Overall validation result
        all_passed = all(check['passed'] for check in checks.values())
        
        result = {
            'valid': bool(all_passed),
            'image': str(image_path),
            'checks': checks
        }
        
        # Print summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        for check_name, check_result in checks.items():
            status = "✓ PASS" if check_result['passed'] else "✗ FAIL"
            print(f"{status} - {check_name.upper()}: {check_result['message']}")
        
        print("="*70)
        if all_passed:
            print("✓ ALL CHECKS PASSED - Image is ready for processing")
        else:
            print("✗ VALIDATION FAILED - Image cannot be processed")
            failed_checks = [name for name, check in checks.items() if not check['passed']]
            print(f"  Failed checks: {', '.join(failed_checks)}")
        print("="*70 + "\n")
        
        # Save validation report (with JSON serialization fix)
        if self.debug:
            try:
                report_path = self.validation_folder / f"{Path(image_path).stem}_validation.json"
                with open(report_path, 'w', encoding='utf-8') as f:
                    # Convert numpy types to native Python types before JSON serialization
                    json_safe_result = self.make_json_serializable(result)
                    json.dump(json_safe_result, f, indent=2)
                print(f"Validation report saved to: {report_path}\n")
            except Exception as e:
                print(f"Warning: Could not save validation report: {e}\n")
        
        # Also make result JSON-safe before returning
        json_safe_result = self.make_json_serializable(result)
        
        if capture_state:
            pipeline_state = cell_result.get('state') if (cell_result and cell_result.get('passed')) else None
            return json_safe_result, pipeline_state
        
        return json_safe_result


def process_with_validation(image_path: str, debug=False, training_mode=False, 
                           return_images=False) -> Optional[Dict]:
    """
    Validate image first, then process if validation passes.
    Returns cells and all processing data for backend integration.
    
    Args:
        image_path: Path to the image file
        debug: Enable debug output
        training_mode: Save cells for training
        return_images: Return cell images in output
    
    Returns:
        Dict containing:
        - validation: validation results
        - cells: extracted cell data with images
        - shg_id: SHG MBK ID field
        - summary: processing summary
        Or None if validation failed
    """
    # Step 1: Validate image
    validator = SHGImageValidator(debug=debug)
    capture_state = not return_images
    validation_output = validator.validate_image(image_path, capture_state=capture_state)
    
    if isinstance(validation_output, tuple):
        validation_result, pipeline_state = validation_output
    else:
        validation_result = validation_output
        pipeline_state = None
    
    # Step 2: Only proceed if validation passed
    if not validation_result['valid']:
        print("\n⚠ Processing ABORTED due to failed validation checks")
        return {
            'success': False,
            'validation': validation_result,
            'error': 'Validation failed',
            'cells': None,
            'shg_id': None
        }
    
    print("\n" + "="*70)
    print("VALIDATION PASSED - Starting SHG Form Detection")
    print("="*70 + "\n")
    
    # Step 3: Continue processing without restarting detector logic
    resume_state = pipeline_state if (capture_state and pipeline_state and pipeline_state.get('cells')) else None
    
    detector = None
    if resume_state:
        detector = resume_state['detector']
    else:
        detector = SHGFormDetector(debug=debug, return_images=return_images)
    
    try:
        if resume_state:
            img_name = resume_state['img_name']
            bbox = resume_state['bbox']
            cropped = resume_state['cropped']
            h_lines = resume_state['h_lines']
            v_lines = resume_state['v_lines']
            intersections = resume_state['intersections']
            cells = resume_state['cells']
        else:
            # Read and preprocess image
            img = cv2.imread(str(image_path))
            if img is None:
                return {
                    'success': False,
                    'validation': validation_result,
                    'error': 'Could not read image',
                    'cells': None,
                    'shg_id': None
                }
            
            img_name = Path(image_path).stem
            
            img = detector.preprocess_scanned(img)
            bbox = detector.detect_table_boundary(img, img_name)
            
            if bbox is None:
                return {
                    'success': False,
                    'validation': validation_result,
                    'error': 'Could not detect table boundary',
                    'cells': None,
                    'shg_id': None
                }
            
            cropped = detector.crop_and_deskew_table(img, bbox, img_name)
            
            if not detector.verify_table_shape(cropped, img_name):
                return {
                    'success': False,
                    'validation': validation_result,
                    'error': 'Wrong table shape',
                    'cells': None,
                    'shg_id': None
                }
            
            h_lines, v_lines, intersections = detector.detect_lines_and_intersections(
                cropped, img_name
            )
            
            cells = detector.trace_cells_from_intersections(
                h_lines, v_lines, intersections, img_name, cropped
            )
        
        # Extract SHG ID field
        shg_id = detector.extract_shg_id_field(cropped, cells, img_name)
        
        # Extract and filter cells (returns cell images)
        extracted_cells = detector.filter_and_extract_cells(cropped, cells, img_name)
        
        # Build comprehensive result for backend
        result = {
            'success': True,
            'validation': validation_result,
            'image_path': str(image_path),
            'table_bbox': bbox,
            'total_cells_detected': len(cells),
            'extracted_cells_count': len(extracted_cells),
            'cells': extracted_cells,  # List of dicts with cell data and images
            'shg_id': shg_id,  # SHG MBK ID field image
            'processing_summary': {
                'horizontal_lines': len(h_lines),
                'vertical_lines': len(v_lines),
                'intersections': len(intersections),
                'total_cells': len(cells)
            }
        }
        
        print("\n" + "="*70)
        print("PROCESSING COMPLETE - Ready for Backend")
        print("="*70)
        print(f"✓ Validation passed")
        print(f"✓ Total cells detected: {len(cells)}")
        print(f"✓ Data cells extracted: {len(extracted_cells)}")
        print("✓ Cell images included: True (kept in memory)")
        print("="*70 + "\n")
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'validation': validation_result,
            'error': f'Processing error: {str(e)}',
            'cells': None,
            'shg_id': None
        }


def main():
    parser = argparse.ArgumentParser(
        description='SHG Form Validator - Pre-checks before processing'
    )
    parser.add_argument('image', help='Path to SHG form image')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug output and save debug images')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation checks, skip processing')
    parser.add_argument('--training-save', action='store_true', 
                       help='Save cells for training (if validation passes)')
    parser.add_argument('--return-images', action='store_true',
                       help='Return cell images in output')
    
    args = parser.parse_args()
    
    if args.validate_only:
        # Only run validation
        validator = SHGImageValidator(debug=args.debug)
        result = validator.validate_image(args.image)
        return 0 if result['valid'] else 1
    else:
        # Validate then process
        result = process_with_validation(
            args.image,
            debug=args.debug,
            training_mode=args.training_save,
            return_images=args.return_images
        )
        return 0 if result and result.get('success') else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
