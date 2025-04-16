# grid_processing.py
import cv2
import numpy as np
import time
import os
from config import USE_PREVIOUS_GRIDLINES, STABILITY_TIME_NEEDED, GRID_TOLERANCE, EDGE_THRESHOLD
from helpers import dprint

# Global variables for grid locking (if desired)
locked_corners = None
grid_locked = False
locked_verticals = None
locked_horizontals = None
locked_intersections = None
grid_stability_start_time = None
grid_history = []  # will store tuples: (verticals, horizontals, intersections)
square_db = None  # Global square database

def is_near_edge(x, y, img_shape, threshold=EDGE_THRESHOLD):
    """Check if a point is too close to the frame edges."""
    height, width = img_shape[:2]
    return (x < threshold or 
            x > width - threshold or 
            y < threshold or 
            y > height - threshold)

def is_line_on_edges(x1, y1, x2, y2, img_shape, threshold=EDGE_THRESHOLD):
    """Check if both endpoints of a line are near edges."""
    return (is_near_edge(x1, y1, img_shape, threshold) and 
            is_near_edge(x2, y2, img_shape, threshold))

def interpolate_missing_lines(lines, expected_count=9, tolerance=0.1):
    """Interpolate missing lines based on average spacing."""
    if not lines or len(lines) < 2:
        return lines
    
    # Sort lines
    lines = sorted(lines)
    
    # Calculate average spacing
    distances = [lines[i+1] - lines[i] for i in range(len(lines)-1)]
    avg_spacing = np.median(distances)
    min_spacing = avg_spacing * (1 - tolerance)
    max_spacing = avg_spacing * (1 + tolerance)
    
    # Interpolate missing lines
    interpolated = [lines[0]]  # Start with first line
    current_pos = lines[0]
    
    while len(interpolated) < expected_count and current_pos < lines[-1]:
        next_expected = current_pos + avg_spacing
        
        # Find closest actual line
        closest_line = None
        min_diff = float('inf')
        for line in lines:
            if line > current_pos:
                diff = abs(line - next_expected)
                if diff < min_diff:
                    min_diff = diff
                    closest_line = line
        
        if closest_line is None or abs(closest_line - next_expected) > avg_spacing * tolerance:
            # No suitable line found, interpolate
            interpolated.append(int(next_expected))
            current_pos = next_expected
        else:
            # Use existing line
            interpolated.append(closest_line)
            current_pos = closest_line
    
    # If we don't have enough lines, add more at the end
    while len(interpolated) < expected_count:
        next_pos = interpolated[-1] + avg_spacing
        interpolated.append(int(next_pos))
    
    # If we have too many lines, trim to expected count
    interpolated = interpolated[:expected_count]
    
    return interpolated

def group_lines(vals, threshold=10):
    if not vals:
        return []
    vals = sorted(vals)
    grouped = []
    group = [vals[0]]
    for v in vals[1:]:
        if abs(v - group[-1]) < threshold:
            group.append(v)
        else:
            grouped.append(int(np.mean(group)))
            group = [v]
    if group:
        grouped.append(int(np.mean(group)))
    return grouped

def enforce_uniform_spacing(lines, max_ratio_deviation=0.2):
    """Enforce uniform spacing between lines and interpolate missing ones."""
    if not lines or len(lines) < 2:
        print("[DEBUG] Not enough lines for spacing enforcement")
        return lines
    
    # First, filter out lines with non-uniform spacing
    distances = [lines[i+1]-lines[i] for i in range(len(lines)-1)]
    median_dist = np.median(distances)
    
    if median_dist <= 0:
        print("[DEBUG] Invalid median distance, returning original lines")
        return lines
    
    filtered = [lines[0]]
    for i in range(len(distances)):
        current_gap = distances[i]
        ratio = abs(current_gap - median_dist) / float(median_dist)
        print(f"[DEBUG] Line spacing ratio: {ratio:.3f} (threshold: {max_ratio_deviation})")
        if ratio <= max_ratio_deviation:
            filtered.append(lines[i+1])
        else:
            print(f"[DEBUG] Rejected line at index {i+1} due to spacing ratio")
    
    # Now interpolate missing lines to ensure exactly 9 lines
    interpolated = interpolate_missing_lines(filtered, expected_count=9, tolerance=0.1)
    print(f"[DEBUG] Line count: original={len(lines)}, filtered={len(filtered)}, interpolated={len(interpolated)}")
    
    return interpolated

def detect_grid(warped):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blurred, 50, 150)
    cv2.imshow("Canny Edge Detection", canny)
    
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=1)
    cv2.imshow("Dilated", dilated)
    
    # Create a copy of dilated image for drawing lines
    lines_image = dilated.copy()
    debug_image = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    
    # Keep original HoughLine parameters
    lines = cv2.HoughLinesP(dilated, 1, np.pi/180, threshold=200, minLineLength=500, maxLineGap=20)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            # Draw all detected lines in green
            cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    cv2.imshow("Detected Lines", lines_image)
    cv2.imshow("Debug Lines", debug_image)
    
    if lines is None:
        return None, None, None

    vertical_lines = []
    horizontal_lines = []
    img_shape = warped.shape

    # Create a new image for classified lines
    classified_lines = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Skip lines only if both endpoints are near edges
        if is_line_on_edges(x1, y1, x2, y2, img_shape):
            cv2.line(classified_lines, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red for filtered
            continue
            
        dx = x2 - x1
        dy = y2 - y1
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        
        # More lenient angle thresholds with better classification
        if angle < 10 or angle > 170:  # Nearly horizontal
            horizontal_lines.append(y1)
            cv2.line(classified_lines, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue for horizontal
        elif 80 < angle < 100:  # Nearly vertical
            vertical_lines.append(x1)
            cv2.line(classified_lines, (x1, y1), (x2, y2), (0, 255, 255), 1)  # Yellow for vertical
        
        # Draw edge points for debugging
        if is_near_edge(x1, y1, img_shape):
            cv2.circle(classified_lines, (x1, y1), 3, (0, 0, 255), -1)  # Red dot for edge point
        if is_near_edge(x2, y2, img_shape):
            cv2.circle(classified_lines, (x2, y2), 3, (0, 0, 255), -1)  # Red dot for edge point
    
    cv2.imshow("Line Classification", classified_lines)
    
    # Print debug information
    print(f"[DEBUG] Found {len(vertical_lines)} vertical and {len(horizontal_lines)} horizontal lines")
    
    grouped_vertical = sorted(group_lines(vertical_lines))
    grouped_horizontal = sorted(group_lines(horizontal_lines))
    
    # Apply uniform spacing and interpolation
    grouped_vertical = enforce_uniform_spacing(grouped_vertical)
    grouped_horizontal = enforce_uniform_spacing(grouped_horizontal)
    
    # Ensure exactly 9 lines in each direction
    if len(grouped_vertical) != 9 or len(grouped_horizontal) != 9:
        print("[DEBUG] Failed to get exactly 9 lines in each direction")
        return None, None, None
    
    intersections = []
    for x in grouped_vertical:
        for y in grouped_horizontal:
            intersections.append((x, y))
    intersections = sorted(intersections, key=lambda p: (p[1], p[0]))
    filtered_points = remove_outlier_points(intersections)
    return filtered_points, grouped_vertical, grouped_horizontal

def remove_outlier_points(intersections):
    return intersections
    if not intersections or len(intersections) < 4:
        return intersections
    min_dists = []
    for i, (x1, y1) in enumerate(intersections):
        best = float('inf')
        for j, (x2, y2) in enumerate(intersections):
            if i == j:
                continue
            dist = np.hypot(x2 - x1, y2 - y1)
            if dist < best:
                best = dist
        min_dists.append(best)
    side = np.median(min_dists)
    threshold = side * 1.3
    print(f"[DEBUG] Median distance: {side:.2f}, Threshold: {threshold:.2f}")
    filtered = []
    for i, (x, y) in enumerate(intersections):
        if min_dists[i] <= threshold:
            filtered.append((x, y))
        else:
            print(f"[DEBUG] Rejected point ({x:.1f}, {y:.1f}) with distance {min_dists[i]:.2f}")
    print(f"[DEBUG] Filtered from {len(intersections)} to {len(filtered)} points")
    return filtered

def compute_square_side(verticals):
    if len(verticals) < 2:
        return None
    diffs = [verticals[i+1] - verticals[i] for i in range(len(verticals)-1)]
    return np.median(diffs)

def grid_is_stable(history, tolerance=GRID_TOLERANCE):
    sides = []
    for (v, h, i) in history:
        side = compute_square_side(v)
        if side is not None:
            sides.append(side)
    if not sides:
        print("Sides False")
        return False
    overall_median = np.median(sides)
    for s in sides:
        if abs(s - overall_median) / overall_median > tolerance:
            return False
    return True

def create_square_database(verticals, horizontals):
    """Create a database mapping square labels to their coordinates."""
    db = {}
    for i in range(8):
        for j in range(8):
            label = f"{chr(ord('A')+i)}{8-j}"
            x_min = verticals[i]
            x_max = verticals[i+1]
            y_min = horizontals[j]
            y_max = horizontals[j+1]
            center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
            db[label] = {"bbox": (x_min, y_min, x_max, y_max), "center": center}
    return db

def get_square_db():
    global square_db
    return square_db

def update_square_database(verticals, horizontals):
    """Update the global square database with new grid coordinates."""
    global square_db
    square_db = create_square_database(verticals, horizontals)
    return square_db

def find_square_by_coordinate(x, y, square_db):
    """Find which square contains the given coordinates."""
    for label, data in square_db.items():
        x_min, y_min, x_max, y_max = data["bbox"]
        if x_min <= x < x_max and y_min <= y < y_max:
            return label
    return None

def detect_and_lock_grid(warped):
    global grid_locked, locked_verticals, locked_horizontals, locked_intersections
    global grid_stability_start_time, grid_history
    
    # Try to load saved positions first only if USE_PREVIOUS_GRIDLINES is True
    if not grid_locked and USE_PREVIOUS_GRIDLINES:
        corners, verticals, horizontals, intersections = load_locked_positions()
        if all(x is not None for x in [corners, verticals, horizontals, intersections]):
            # Update square database when loading previous gridlines
            update_square_database(verticals, horizontals)
            return intersections, verticals, horizontals
    
    if grid_locked:
        return locked_intersections, locked_verticals, locked_horizontals
    
    intersections, verticals, horizontals = detect_grid(warped)
    if intersections is None or verticals is None or horizontals is None:
        return None, None, None
        
    if len(verticals) != 9 or len(horizontals) != 9:
        return None, None, None
        
    if grid_stability_start_time is None:
        grid_stability_start_time = time.time()
        grid_history.clear()
        
    grid_history.append((verticals, horizontals, intersections))
    window_duration = time.time() - grid_stability_start_time
    
    if window_duration >= STABILITY_TIME_NEEDED:
        v_arrays = [np.array(v) for v, _, _ in grid_history]
        h_arrays = [np.array(h) for _, h, _ in grid_history]
        i_arrays = [np.array(i) for _, _, i in grid_history]
        
        if not (all(arr.shape == h_arrays[0].shape for arr in h_arrays) and
                all(arr.shape == v_arrays[0].shape for arr in v_arrays) and
                all(arr.shape == i_arrays[0].shape for arr in i_arrays)):
            return None, None, None
            
        median_verticals = np.median(np.stack(v_arrays, axis=0), axis=0).tolist()
        median_horizontals = np.median(np.stack(h_arrays, axis=0), axis=0).tolist()
        median_intersections = np.median(np.stack(i_arrays, axis=0), axis=0).tolist()
        
        median_side = compute_square_side(median_verticals)
        if median_side is None:
            return None, None, None
            
        stable = grid_is_stable(grid_history, tolerance=GRID_TOLERANCE)
        if stable:
            grid_locked = True
            locked_verticals = median_verticals
            locked_horizontals = median_horizontals
            locked_intersections = median_intersections
            print("[DEBUG] Grid locked using median measurements.")
            grid_history.clear()
            grid_stability_start_time = None
            
            # Update square database when grid is newly locked
            update_square_database(locked_verticals, locked_horizontals)
            
            import board_processing
            if board_processing.locked_corners is not None:
                save_locked_positions(corners=board_processing.locked_corners,
                                    verticals=locked_verticals,
                                    horizontals=locked_horizontals,
                                    intersections=locked_intersections)
            else:
                print("[DEBUG] No corners found; not saving locked board data.")
            return locked_intersections, locked_verticals, locked_horizontals
        else:
            print("[DEBUG] Grid instability detected; continuing accumulation.")
            return None, None, None
            
    return intersections, verticals, horizontals

def save_locked_positions(corners, verticals, horizontals, intersections):
    data = {
        "corners": corners.tolist() if corners is not None else None,
        "verticals": verticals,
        "horizontals": horizontals,
        "intersections": intersections
    }
    np.savez("locked_board.npz", **data)
    print("[DEBUG] Locked board data saved to locked_board.npz")

def load_locked_positions():
    """Load previously saved grid positions and update global variables."""
    global locked_corners, locked_verticals, locked_horizontals, locked_intersections, grid_locked
    
    filename = "locked_board.npz"
    if not os.path.exists(filename):
        print("[DEBUG] No locked_board.npz file found; starting fresh.")
        return None, None, None, None
        
    try:
        data = np.load(filename, allow_pickle=True)
        
        # Convert numpy arrays to lists
        corners = np.array(data["corners"], dtype=np.float32) if "corners" in data and data["corners"] is not None else None
        verticals = data["verticals"].tolist() if "verticals" in data else None
        horizontals = data["horizontals"].tolist() if "horizontals" in data else None
        intersections = data["intersections"].tolist() if "intersections" in data else None
        
        # Update global variables
        locked_corners = corners
        locked_verticals = verticals
        locked_horizontals = horizontals
        locked_intersections = intersections
        grid_locked = True
        
        # Update board_processing's board_locked variable
        import board_processing
        board_processing.board_locked = True
        board_processing.locked_corners = corners
        
        print("[DEBUG] Successfully loaded locked board data from locked_board.npz")
        print(f"[DEBUG] Loaded {len(verticals) if verticals else 0} vertical lines")
        print(f"[DEBUG] Loaded {len(horizontals) if horizontals else 0} horizontal lines")
        print(f"[DEBUG] Loaded corners: {corners}")
        
        return corners, verticals, horizontals, intersections
    except Exception as e:
        print(f"[DEBUG] Error loading locked_board.npz: {e}")
        print(f"[DEBUG] Error details: {str(e)}")
        return None, None, None, None
