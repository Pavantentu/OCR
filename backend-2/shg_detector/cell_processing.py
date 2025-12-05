import cv2
import numpy as np
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed



# ============================================================
# Module: cell_processing
# ============================================================

class CellProcessingMixin:
    """Mixin class for cell_processing functionality"""

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
        # Enhance for OCR (use fast denoising)
        enhanced = self.enhance_cell_for_ocr(shg_img, use_fast_denoising=True)
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
            self.save_debug_image(debug_shg, f"{img_name}_17_shg_id_field.jpg", "SHG ID field")
        # Return enhanced image if return_images is True, else filepath
        return enhanced if self.return_images else filepath

    def get_cells_by_debug_order(self, cells):
        """
        Get cells ordered by their position (top-to-bottom, left-to-right)
        and assign them sequential debug IDs that match the debug visualization.
        OPTIMIZED: Pre-extract keys for faster sorting.
        Returns:
            dict: Mapping of debug_id -> cell
        """
        # OPTIMIZED: Pre-extract coordinates for faster sorting (avoids dict lookups)
        cells_with_keys = [(cell.get('y', 0), cell.get('x', 0), cell) for cell in cells]
        # Sort by position (row, then column)
        cells_with_keys.sort(key=lambda c: (c[0], c[1]))
        # Create mapping (faster dict comprehension)
        debug_id_to_cell = {idx: cell for idx, (_, _, cell) in enumerate(cells_with_keys)}
        # Assign debug_id to cells
        for debug_id, (_, _, cell) in enumerate(cells_with_keys):
            cell['debug_id'] = debug_id
        return debug_id_to_cell

    def filter_and_extract_cells(self, image, cells, img_name):
        """
        Step 7: Filter cells and extract with dynamic padding
        
        OPTIMIZED FOR PERFORMANCE:
        - Fast denoising (bilateral filter ~10-20x faster than NL-means)
        - Skips denoising for very small cells (< 70x70px)
        - Pre-computed image dimensions to avoid repeated lookups
        - Early validation to skip invalid cells quickly
        - Optimized sequential processing (thread overhead was slowing things down)
        
        Key optimization: Replaced slow fastNlMeansDenoising with fast bilateralFilter
        """
        if self.debug:
            print("\n" + "="*70)
            print("STEP 7: FILTERING AND EXTRACTING CELLS")
            print("="*70)
        if not cells:
            if self.debug:
                print("  ✗ No cells available to filter")
            return []
        
        # OPTIMIZED: Use cached dimensions (faster than shape[:2])
        img_h, img_w, _, _ = self.get_image_dimensions(image)
        size_config = self.get_size_config(img_h, img_w)
        
        # Remove table lines using cell edges (smart mask based on image size)
        if self.debug:
            print("  Removing table lines using cell edges...")
        image_clean = self.remove_lines_using_cell_edges(image, cells, img_name)
        
        # Save the cleaned "no lines" image for debugging FIRST
        if self.debug:
            self.save_debug_image(image_clean, f"{img_name}_18_lines_removed.jpg", 
                                "Table with lines removed using cell edges")
        
        # ============================================================
        # STEP 1: Create debug ID mapping (matches debug visualization)
        # ============================================================
        debug_id_map = self.get_cells_by_debug_order(cells)
        if self.debug:
            print(f"  Created debug ID mapping for {len(debug_id_map)} cells")
            # OPTIMIZED: Only create debug image if needed (expensive copy)
            debug_ref = image.copy()
            # OPTIMIZED: Pre-compute font settings (avoid repeated lookups)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            for debug_id, cell in debug_id_map.items():
                x, y, w, h = cell['bbox']
                cv2.rectangle(debug_ref, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # OPTIMIZED: Pre-compute text position
                text_x = x + (w >> 1) - 10  # Bit shift instead of division
                text_y = y + (h >> 1)
                cv2.putText(debug_ref, str(debug_id), (text_x, text_y),
                        font, font_scale, (255, 0, 0), thickness)
            self.save_debug_image(debug_ref, f"{img_name}_18a_debug_ids.jpg", 
                                "Debug IDs (use these for cell_debug_ids)")
        
        # ============================================================
        # CONFIGURE HERE: Which cells to extract by DEBUG ID
        # ============================================================
        def build_cell_debug_ids(num_rows):
            result = [2]  # SHG MBK ID always at front
            # Base row = your Row 2 pattern
            base_A = 25
            base_C = 27
            base_E = 29
            base_F = 36
            # The pattern increases by +18 each row
            step = 17
            for i in range(num_rows):
                offset = i * step
                A = base_A + offset
                C = base_C + offset
                E = base_E + offset
                F = base_F + offset
                result += [A, C, (E, F)]
            return result
        
        cell_debug_ids = build_cell_debug_ids(15)
        
        # OPTIMIZED: Expand ID list more efficiently
        if self.debug:
            print(f"  Filtering by debug cell IDs (preserving order)")
        ordered_ids = []
        # Pre-allocate list size estimate (faster than repeated appends)
        estimated_size = len(cell_debug_ids) * 2  # Rough estimate
        ordered_ids = []
        for item in cell_debug_ids:
            if isinstance(item, tuple) and len(item) == 2:
                start, end = item
                ordered_ids.extend(range(start, end + 1))
            else:
                ordered_ids.append(item)
        
        # OPTIMIZED: Extract cells with pre-allocated lists and reduced lookups
        valid_cells = []
        missing_ids = []
        # Pre-compute corner_ids accessor to avoid repeated dict lookups
        get_corner_tl = lambda c: c.get('corner_ids', {}).get('tl', -1) if self.debug else -1
        for debug_id in ordered_ids:
            cell = debug_id_map.get(debug_id)  # Use .get() for faster missing check
            if cell is not None:
                valid_cells.append(cell)
                if self.debug:
                    tl_id = get_corner_tl(cell)
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
            print(f"  Filtered to {len(valid_cells)} valid cells")
            print("\n  Extracting cell images...")
        
        # OPTIMIZED: Sequential processing with faster algorithms
        # Note: Sequential is faster here due to thread overhead and GIL contention
        # The key optimization is using fast denoising instead of slow NL-means
        if self.debug:
            print(f"  Processing {len(valid_cells)} cells (optimized sequential)...")
        
        extracted_cells = []
        start_time = time.time()
        # Use cached dimensions
        img_h, img_w, _, _ = self.get_image_dimensions(image_clean)
        
        # OPTIMIZED: Pre-compute all crop coordinates with minimal dict lookups
        cell_coords = []
        minimal_padding = 2
        minimal_inset = 2
        min_cell_size = 10
        
        # Pre-extract all bbox values to avoid repeated dict lookups
        for idx, cell in enumerate(valid_cells):
            bbox = cell['bbox']
            x, y, w, h = bbox  # Tuple unpacking is faster than dict access
            col = cell.get('col', -1)
            corner_ids = cell.get('corner_ids')
            tl_id = corner_ids.get('tl', -1) if corner_ids else -1
            debug_id = cell.get('debug_id', -1)
            
            # OPTIMIZED: Calculate crop coordinates with reduced operations
            # Add padding BEFORE cropping by expanding the crop area outward
            corners = cell.get('corners')
            if corners:
                # Use corners if available
                tl = corners.get('top_left', (x, y))
                tr = corners.get('top_right', (x + w, y))
                bl = corners.get('bottom_left', (x, y + h))
                br = corners.get('bottom_right', (x + w, y + h))
                # Expand outward: subtract from left/top, add to right/bottom
                x1 = max(0, min(tl[0], bl[0]) - minimal_padding)
                y1 = max(0, min(tl[1], tr[1]) - minimal_padding)
                x2 = min(img_w, max(tr[0], br[0]) + minimal_padding)
                y2 = min(img_h, max(bl[1], br[1]) + minimal_padding)
            else:
                # Fallback: use bbox with padding added outward
                x1 = max(0, x - minimal_inset)
                y1 = max(0, y - minimal_inset)
                x2 = min(img_w, x + w + minimal_inset)
                y2 = min(img_h, y + h + minimal_inset)
            
            # OPTIMIZED: Early validation with bit operations where possible
            if (x2 - x1) < min_cell_size or (y2 - y1) < min_cell_size:
                if self.debug:
                    print(f"    ⚠ Skipping cell {debug_id}: too small after cropping")
                continue
            
            # OPTIMIZED: Use tuple instead of dict for faster access (if possible)
            # Store coordinates as tuple for faster unpacking
            cell_coords.append((idx, cell, x1, y1, x2, y2, x, y, w, h, col, tl_id, debug_id))
        
        # OPTIMIZED: Process cells with minimal overhead
        num_cells = len(cell_coords)
        batch_size = self._batch_processor.batch_size if num_cells > 20 else num_cells
        
        # OPTIMIZED: Pre-compute string template for cell_info (avoid repeated formatting)
        cell_info_template = "DebugID={}, Cell_ID={}, TL_ID={}, Row={}, Col={}, Image={}" if self.return_images else None
        
        # OPTIMIZED: Parallel cell processing for better performance
        def process_single_cell(cell_data):
            """Process a single cell - designed for parallel execution"""
            idx, cell, x1, y1, x2, y2, x, y, w, h, col, tl_id, debug_id = cell_data
            
            # Extract cell image (make a copy for thread safety)
            cell_img = image_clean[y1:y2, x1:x2].copy()
            
            # Enhance for OCR (already optimized internally)
            enhanced = self.enhance_cell_for_ocr(cell_img, use_fast_denoising=True)
            
            # Pre-compute cell values
            cell_id = cell['id']
            cell_row = cell.get('row', -1)
            
            # Build result dict
            cell_result = {
                'debug_id': debug_id,
                'cell_id': cell_id,
                'tl_id': tl_id,
                'row': cell_row,
                'col': col,
                'y': y,
                'x': x,
                'bbox': (x, y, w, h),
                'image': enhanced,
                '_sort_idx': idx  # Preserve order for sorting
            }
            
            # Only format string if needed
            if self.return_images:
                cell_info = cell_info_template.format(debug_id, cell_id, tl_id, cell_row, col, img_name)
                filepath = self.save_training_cell(enhanced, cell_info)
                cell_result['path'] = str(filepath)
            
            return cell_result
        
        # Use parallel processing if we have multiple workers and enough cells
        # OPTIMIZED: Enable parallel processing even in debug mode for better performance
        # (but with fewer workers to avoid too much concurrent debug output)
        use_parallel = (self.num_workers > 1 and num_cells > 5)
        
        if use_parallel:
            # Parallel processing with ThreadPoolExecutor
            max_workers = min(self.num_workers, num_cells)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all cell processing tasks
                future_to_idx = {
                    executor.submit(process_single_cell, cell_coords[i]): i 
                    for i in range(num_cells)
                }
                
                # Collect results maintaining order
                results = [None] * num_cells
                completed = 0
                for future in as_completed(future_to_idx):
                    try:
                        result = future.result()
                        idx = future_to_idx[future]
                        results[idx] = result
                        completed += 1
                        if self.debug and completed % 10 == 0:
                            elapsed = time.time() - start_time
                            if elapsed > 0:
                                rate = completed / elapsed
                                print(f"    Processed {completed}/{num_cells} cells ({rate:.1f} cells/sec)...")
                    except Exception as e:
                        idx = future_to_idx[future]
                        if self.debug:
                            print(f"    Error processing cell {idx}: {e}")
                        results[idx] = None
                
                # Filter out None results and maintain order
                extracted_cells = [r for r in results if r is not None]
        else:
            # Sequential processing (original code path)
            for batch_start in range(0, num_cells, batch_size):
                batch_end = min(batch_start + batch_size, num_cells)
                
                for i in range(batch_start, batch_end):
                    # OPTIMIZED: Tuple unpacking is faster than dict access
                    idx, cell, x1, y1, x2, y2, x, y, w, h, col, tl_id, debug_id = cell_coords[i]
                    
                    # OPTIMIZED: Direct slice (no copy yet) - enhance_cell_for_ocr will handle copying
                    cell_img = image_clean[y1:y2, x1:x2]
                    # Note: enhance_cell_for_ocr needs contiguous array, but let it handle the copy
                    # This avoids one copy operation here
                    
                    # Enhance for OCR (already optimized internally)
                    enhanced = self.enhance_cell_for_ocr(cell_img, use_fast_denoising=True)
                    
                    # OPTIMIZED: Pre-compute cell values to avoid repeated dict lookups
                    cell_id = cell['id']
                    cell_row = cell.get('row', -1)
                    
                    # OPTIMIZED: Build result dict more efficiently
                    cell_result = {
                        'debug_id': debug_id,
                        'cell_id': cell_id,
                        'tl_id': tl_id,
                        'row': cell_row,
                        'col': col,
                        'y': y,
                        'x': x,
                        'bbox': (x, y, w, h),
                        'image': enhanced
                    }
                    
                    # OPTIMIZED: Only format string if needed
                    if self.return_images:
                        cell_info = cell_info_template.format(debug_id, cell_id, tl_id, cell_row, col, img_name)
                        filepath = self.save_training_cell(enhanced, cell_info)
                        cell_result['path'] = str(filepath)
                    
                    extracted_cells.append(cell_result)
                
                # OPTIMIZED: Progress reporting (every batch) - avoid repeated calculations
                if self.debug:
                    processed = batch_end
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        rate = processed / elapsed
                        print(f"    Processed {processed}/{num_cells} cells ({rate:.1f} cells/sec)...")
        
        if self.debug:
            total_time = time.time() - start_time
            print(f"  ✓ Cell processing completed in {total_time:.2f}s "
                  f"({len(extracted_cells)} cells, {len(extracted_cells)/total_time:.1f} cells/sec)")
        
        if self.debug:
            print(f"\n  ✓ Extracted {len(extracted_cells)} cells")
            # OPTIMIZED: Draw debug image only if needed (expensive copy)
            debug_extracted = image.copy()
            # OPTIMIZED: Pre-compute font settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            for idx, cell in enumerate(valid_cells):
                x, y, w, h = cell['bbox']
                cv2.rectangle(debug_extracted, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # OPTIMIZED: Pre-compute text positions
                debug_id = cell.get('debug_id', -1)
                text_x = x + 5
                cv2.putText(debug_extracted, f"#{idx+1}", (text_x, y + 20),
                        font, 0.5, (255, 0, 0), 2)
                cv2.putText(debug_extracted, f"D{debug_id}", (text_x, y + 40),
                        font, 0.4, (0, 255, 255), 1)
            self.save_debug_image(debug_extracted, f"{img_name}_19_extracted_cells.jpg", 
                                f"Extracted {len(extracted_cells)} cells in order")
        
        return extracted_cells
