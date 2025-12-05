"""
SHG Form Detector - Main Entry Point
This file assembles the SHGFormDetector class from modular components.
"""

import argparse
import os
import sys

# Import all mixin classes
from shg_detector.core import CoreMixin
from shg_detector.utils import UtilsMixin
from shg_detector.preprocessing import PreprocessingMixin
from shg_detector.table_detection import TableDetectionMixin
from shg_detector.config import ConfigMixin
from shg_detector.line_detection import LineDetectionMixin
from shg_detector.cell_tracing import CellTracingMixin
from shg_detector.cell_processing import CellProcessingMixin
from shg_detector.image_enhancement import ImageEnhancementMixin
from shg_detector.training import TrainingMixin
from shg_detector.processor import ProcessorMixin


class SHGFormDetector(
    CoreMixin,
    UtilsMixin,
    PreprocessingMixin,
    TableDetectionMixin,
    ConfigMixin,
    LineDetectionMixin,
    CellTracingMixin,
    CellProcessingMixin,
    ImageEnhancementMixin,
    TrainingMixin,
    ProcessorMixin
):
    """
    Main SHG Form Detector class assembled from modular mixins.
    
    This class inherits functionality from all module mixins, providing
    a complete table detection and cell extraction system for SHG forms.
    """
    pass


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
    sys.exit(main())

