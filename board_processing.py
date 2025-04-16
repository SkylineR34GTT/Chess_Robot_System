# board_processing.py
import cv2
import numpy as np
import time
from digital_board import render_digital_board  # if needed for drawing notation
from helpers import dprint

# Global variables
locked_corners = None
prev_corners = None

# The smoothing factor for corner smoothing.
corner_smooth_factor = 0.9

def smooth_corners(new_corners):
    global prev_corners
    try:
        prev = prev_corners
    except NameError:
        prev = None
    if prev is None:
        prev_corners = new_corners
        return new_corners
    smoothed_corners = prev * (1 - corner_smooth_factor) + new_corners * corner_smooth_factor
    prev_corners = smoothed_corners
    return smoothed_corners

def order_corners(corners):
    sorted_by_y = corners[corners[:,1].argsort()]
    top_two = sorted_by_y[:2]
    bottom_two = sorted_by_y[2:]
    top_two = top_two[top_two[:,0].argsort()]
    bottom_two = bottom_two[bottom_two[:,0].argsort()]
    tl, tr = top_two[0], top_two[1]
    bl, br = bottom_two[0], bottom_two[1]
    return np.array([tl, tr, br, bl], dtype="float32")

def corners_are_within_threshold(new_corners, ref_corners, threshold=0.1, fixed_tol=10):
    for i in range(4):
        rx, ry = ref_corners[i]
        nx, ny = new_corners[i]
        diff_x = abs(nx - rx)
        diff_y = abs(ny - ry)
        rel_diff_x = diff_x / (abs(rx) + 1e-5)
        rel_diff_y = diff_y / (abs(ry) + 1e-5)
        dprint(f"[DEBUG] Corner {i}: ref=({rx:.2f},{ry:.2f}), new=({nx:.2f},{ny:.2f}), diff=({diff_x:.2f},{diff_y:.2f}), rel_diff=({rel_diff_x:.2f},{rel_diff_y:.2f})")
        if (diff_x > fixed_tol and rel_diff_x > threshold) or (diff_y > fixed_tol and rel_diff_y > threshold):
            dprint(f"[DEBUG] -> Corner {i} NOT within threshold.")
            return False
        else:
            dprint(f"[DEBUG] -> Corner {i} is within threshold.")
    return True

def median_corners(corner_list):
    stack = np.stack(corner_list, axis=0)
    med = np.median(stack, axis=0)
    return med

def transform_point(pt, M):
    pt_arr = np.array([[pt]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt_arr, M)
    return tuple(transformed[0][0])

def find_board_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blurred, 80, 200)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=1)
    kernel_close = np.ones((7,7), np.uint8)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
    cv2.imshow("Closed", closed)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        dprint("[DEBUG] No contours found after closing.")
        return None
    dprint(f"[DEBUG] Found {len(contours)} contours")
    largest_contour = max(contours, key=cv2.contourArea, default=None)
    if largest_contour is None:
        dprint("[DEBUG] No largest contour found.")
        return None
    area = cv2.contourArea(largest_contour)
    dprint(f"[DEBUG] Largest contour area: {area}")
    if area < 5000:
        dprint("[DEBUG] Chessboard not detected or too small.")
        return None
    epsilon = 0.05 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    cv2.imshow("Debug Contour", frame.copy())
    dprint(f"[DEBUG] Approximated contour has {len(approx)} points")
    if len(approx) != 4:
        dprint("[DEBUG] Chessboard corners not detected correctly (not 4).")
        return None
    new_corners = np.array([p[0] for p in approx], dtype=np.float32)
    smoothed_corners = smooth_corners(new_corners)
    dprint(f"[DEBUG] Found corners: {smoothed_corners}")
    return smoothed_corners

def warp_with_corners(frame, corners, output_size=(800,800),
                      locked_verticals=None, locked_horizontals=None,
                      locked_intersections=None, square_db=None,
                      USE_PREVIOUS_GRIDLINES=False):
    if corners is None:
        dprint("[DEBUG] No valid corners provided, skipping warp process.")
        return frame, None

    from grid_processing import detect_and_lock_grid, update_square_database
    stable_corners = order_corners(corners)
    dst_pts = np.array([
        [0, 0],
        [output_size[0]-1, 0],
        [output_size[0]-1, output_size[1]-1],
        [0, output_size[1]-1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(stable_corners, dst_pts)
    transformation_matrix = M
    warped = cv2.warpPerspective(frame, M, output_size)
    if USE_PREVIOUS_GRIDLINES and locked_verticals is not None and locked_horizontals is not None:
        intersections = locked_intersections
        verticals = locked_verticals
        horizontals = locked_horizontals
    else:
        intersections, verticals, horizontals = detect_and_lock_grid(warped)
    if verticals is not None and horizontals is not None and len(verticals) == 9 and len(horizontals) == 9 and square_db is None:
        update_square_database(verticals, horizontals)
    if intersections is not None and verticals is not None and horizontals is not None:
        for x in verticals:
            cv2.line(warped, (int(x), 0), (int(x), warped.shape[0]), (255, 0, 0), 2)
        for y in horizontals:
            cv2.line(warped, (0, int(y)), (warped.shape[1], int(y)), (255, 0, 0), 2)
        if len(verticals) == 9 and len(horizontals) == 9:
            for i in range(8):
                for j in range(8):
                    center_x = int((verticals[i] + verticals[i+1]) // 2)
                    center_y = int((horizontals[j] + horizontals[j+1]) // 2)
                    notation = f"{chr(ord('A')+i)}{8-j}"
                    cv2.putText(warped, notation, (center_x-20, center_y+10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    else:
        dprint("[DEBUG] Grid detection failed.")
    return warped, transformation_matrix

def detectBoard(frame, corners, board_locked, stability_start_time, corner_history,
                output_size=(800,800), USE_PREVIOUS_BOARD_POSITIONS=False,
                PERMANENT_CORNER_LOCK=False, RECHECK_INTERVAL=3.0, STABLE_TIME_NEEDED=1.0):
    global locked_corners  # Declare global first
    now = time.time()
    
    # If we have locked corners and should use previous positions, use them directly
    if USE_PREVIOUS_BOARD_POSITIONS and locked_corners is not None:
        warped, transformation_matrix = warp_with_corners(frame, locked_corners, output_size)
        return warped, locked_corners, True, stability_start_time, corner_history, transformation_matrix
        
    if board_locked:
        if PERMANENT_CORNER_LOCK:
            warped, transformation_matrix = warp_with_corners(frame, locked_corners, output_size)
            return warped, locked_corners, board_locked, stability_start_time, corner_history, transformation_matrix
        else:
            elapsed_since_lock = now - stability_start_time
            if elapsed_since_lock < RECHECK_INTERVAL:
                warped, transformation_matrix = warp_with_corners(frame, locked_corners, output_size)
                return warped, locked_corners, board_locked, stability_start_time, corner_history, transformation_matrix
            else:
                new_corners = find_board_corners(frame)
                if new_corners is not None and corners_are_within_threshold(new_corners, locked_corners):
                    stability_start_time = now
                    warped, transformation_matrix = warp_with_corners(frame, locked_corners, output_size)
                    return warped, locked_corners, board_locked, stability_start_time, corner_history, transformation_matrix
                else:
                    board_locked = False
                    locked_corners = None
                    stability_start_time = None
                    corner_history.clear()
    new_corners = find_board_corners(frame)
    if new_corners is None:
        dprint("[DEBUG] No corners found in current frame")
        return frame, locked_corners, board_locked, stability_start_time, corner_history, None
    if stability_start_time is None:
        stability_start_time = now
        corner_history.clear()
    corner_history.append(new_corners)
    window_duration = now - stability_start_time
    if window_duration >= STABLE_TIME_NEEDED:
        median_ref = median_corners(corner_history)
        stable = True
        for arr in corner_history:
            if not corners_are_within_threshold(arr, median_ref):
                stable = False
                break
        if stable:
            board_locked = True
            locked_corners = median_ref
            stability_start_time = now
            corner_history.clear()
            dprint("[DEBUG] Corners locked and stable")
            # Save the corners when they're locked
            from grid_processing import save_locked_positions
            save_locked_positions(corners=locked_corners, verticals=None, horizontals=None, intersections=None)
            warped, transformation_matrix = warp_with_corners(frame, locked_corners, output_size)
            return warped, locked_corners, board_locked, stability_start_time, corner_history, transformation_matrix
        else:
            dprint("[DEBUG] Corners not stable enough")
            stability_start_time = now
            corner_history.clear()
            corner_history.append(new_corners)
    warped, transformation_matrix = warp_with_corners(frame, new_corners, output_size)
    return warped, locked_corners, board_locked, stability_start_time, corner_history, transformation_matrix
