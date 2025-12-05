# No imports needed - this module only delegates to other methods



# ============================================================
# Module: training
# ============================================================

class TrainingMixin:
    """Mixin class for training functionality"""

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

