import cv2
import numpy as np
import os
from pathlib import Path
import json
import argparse
from collections import defaultdict



# ============================================================
# Module: utils
# ============================================================

class UtilsMixin:
    """Mixin class for utils functionality"""

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

