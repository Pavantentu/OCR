import cv2
import numpy as np
import os
from pathlib import Path
import json
import argparse
from collections import defaultdict
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial



# ============================================================
# Module: processor
# ============================================================

class ProcessorMixin:
    """Mixin class for processor functionality"""

    def process_image(self, image_path, training_mode=False):
        """Main processing pipeline"""
        # Start overall timer
        overall_start = time.time()
        timing_info = {}
        
        print("\n" + "#"*70)
        print(f"# SHG FORM DETECTOR - Processing: {image_path}")
        print("#"*70)
        
        # Image loading and scaling
        step_start = time.time()
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"\n✗ ERROR: Could not read image: {image_path}")
            return None
        image = self.analyze_image_scale(image)
        img_name = Path(image_path).stem
        print(f"\nImage: {image_path}")
        print(f"Original size: {self.size_profile['original_shape'][1]}x{self.size_profile['original_shape'][0]} pixels")
        print(f"Working size: {image.shape[1]}x{image.shape[0]} pixels ({self.size_profile['category']})")
        timing_info['Image Loading & Scaling'] = time.time() - step_start
        
        # Preprocess
        step_start = time.time()
        image = self.preprocess_scanned(image)
        if self.debug:
            self.save_debug_image(image, f"{img_name}_00_preprocessed.jpg", "Document scanner preprocessing")
        if image is None:
            print(f"\n✗ ERROR: Could not read image: {image_path}")
            return None
        timing_info['Preprocessing'] = time.time() - step_start
        
        # Step 1: Detect table boundary
        step_start = time.time()
        bbox = self.detect_table_boundary(image, img_name)
        if bbox is None:
            print("\n✗ FAILED: Could not detect table boundary")
            return None
        timing_info['Step 1: Detect Table Boundary'] = time.time() - step_start
        
        # Step 2: Crop and deskew
        step_start = time.time()
        cropped = self.crop_and_deskew_table(image, bbox, img_name)
        timing_info['Step 2: Crop and Deskew'] = time.time() - step_start
        
        # Step 3: Verify table shape
        step_start = time.time()
        if not self.verify_table_shape(cropped, img_name):
            print("\n✗ FAILED: Wrong table shape - not an SHG form")
            return None
        timing_info['Step 3: Verify Table Shape'] = time.time() - step_start
        
        # Step 4: Detect lines and intersections
        step_start = time.time()
        h_lines, v_lines, intersections = self.detect_lines_and_intersections(cropped, img_name)
        timing_info['Step 4: Detect Lines & Intersections'] = time.time() - step_start
        
        # Step 5: Trace cells
        step_start = time.time()
        cells = self.trace_cells_from_intersections(h_lines, v_lines, intersections, img_name, cropped)
        timing_info['Step 5: Trace Cells'] = time.time() - step_start
        
        if training_mode:
            step_start = time.time()
            self.save_for_training(cropped, cells, img_name)
            # Save counter ONCE at the very end
            self.save_counter()
            timing_info['Step 6: Save for Training'] = time.time() - step_start
            
            overall_time = time.time() - overall_start
            timing_info['TOTAL TIME'] = overall_time
            
            # Print timing summary
            print("\n" + "="*70)
            print("PROCESSING COMPLETE - TIMING SUMMARY")
            print("="*70)
            for step_name, step_time in timing_info.items():
                if step_name == 'TOTAL TIME':
                    print(f"⏱  {step_name}: {step_time:.2f} seconds ({step_time/60:.2f} minutes)")
                else:
                    percentage = (step_time / overall_time) * 100
                    print(f"   {step_name}: {step_time:.2f}s ({percentage:.1f}%)")
            print("="*70 + "\n")
            
            return {
                'training_mode': True,
                'counter': self.current_counter,
                'timing': timing_info
            }
        
        # Normal mode: extract and return
        # OPTIMIZED: Run Steps 6 & 7 in parallel since they're independent
        step_start = time.time()
        # Enable parallel processing even in debug mode (but with fewer workers)
        if self.num_workers > 1:
            # Use parallel execution for better performance
            # In debug mode, use fewer workers to avoid too much output
            max_workers = 2 if not self.debug else min(2, self.num_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                shg_id_future = executor.submit(self.extract_shg_id_field, cropped, cells, img_name)
                extracted_cells_future = executor.submit(self.filter_and_extract_cells, cropped, cells, img_name)
                
                shg_id_resource = shg_id_future.result()
                extracted_cells = extracted_cells_future.result()
        else:
            # Sequential execution for single-threaded systems
            shg_id_resource = self.extract_shg_id_field(cropped, cells, img_name)
            extracted_cells = self.filter_and_extract_cells(cropped, cells, img_name)
        timing_info['Step 6 & 7: Extract SHG ID & Filter Cells (Parallel)'] = time.time() - step_start
        
        shg_id_path = None
        if isinstance(shg_id_resource, (str, Path)):
            shg_id_path = str(shg_id_resource)
        
        # Save summary
        step_start = time.time()
        summary = {
            'image': str(image_path),
            'table_bbox': bbox,
            'total_cells': len(cells),
            'extracted_cells': len(extracted_cells),
            'shg_id_path': str(shg_id_path) if shg_id_path else None,
            'training_counter': self.current_counter,
            'timing': timing_info
        }
        json_path = self.result_folder / f"{img_name}_summary.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        timing_info['Save Summary'] = time.time() - step_start
        
        overall_time = time.time() - overall_start
        timing_info['TOTAL TIME'] = overall_time
        
        print("\n" + "="*70)
        print("PROCESSING COMPLETE - TIMING SUMMARY")
        print("="*70)
        print(f"✓ Table detected and verified")
        print(f"✓ Total cells found: {len(cells)}")
        print(f"✓ Data cells extracted: {len(extracted_cells)}")
        print(f"✓ Training images: {self.current_counter}")
        print(f"✓ Results saved to: {self.result_folder}")
        print(f"✓ Training cells in: {self.training_folder}")
        print("\n" + "-"*70)
        print("TIMING BREAKDOWN:")
        print("-"*70)
        for step_name, step_time in timing_info.items():
            if step_name == 'TOTAL TIME':
                print(f"⏱  {step_name}: {step_time:.2f} seconds ({step_time/60:.2f} minutes)")
            else:
                percentage = (step_time / overall_time) * 100
                print(f"   {step_name}: {step_time:.2f}s ({percentage:.1f}%)")
        print("="*70 + "\n")
        return summary
