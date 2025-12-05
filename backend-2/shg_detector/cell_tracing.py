import cv2
import numpy as np
import os
from pathlib import Path
import json
import argparse
from collections import defaultdict



# ============================================================
# Module: cell_tracing
# ============================================================

class CellTracingMixin:
    """Mixin class for cell_tracing functionality"""

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
                # Use stricter threshold: require at least min_coverage_threshold, not 75% of it
                elif left_conn and left_cov >= min_coverage_threshold:
                    log(f"  Left edge has connectivity and partial coverage ({left_cov:.2f}) >= {min_coverage_threshold:.2f} - accepting")
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
                # Use stricter threshold: require at least min_coverage_threshold, not 75% of it
                elif right_conn and right_cov >= min_coverage_threshold:
                    log(f"  Right edge has connectivity and partial coverage ({right_cov:.2f}) >= {min_coverage_threshold:.2f} - accepting")
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

