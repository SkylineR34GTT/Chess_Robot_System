import numpy as np
import cv2
import time
from ultralytics import YOLO
import os
import logging

# Suppress YOLO/Ultralytics logging output
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Global flag to disable printing of model detection output
SHOW_DETECTION_OUTPUT = False

Prev_positions = True

# If True, skip corner detection and always reuse the last locked corners (if any).
USE_PREVIOUS_BOARD_POSITIONS = Prev_positions
# If True, skip grid detection and always reuse the last locked grid lines (if any).
USE_PREVIOUS_GRIDLINES = Prev_positions

# Toggle for corner lock behavior: if True, once locked the corners remain locked permanently.
PERMANENT_CORNER_LOCK = False

# Toggle for test click mode: when True, debug prints (via dprint) are suppressed (except for square lookup).
TEST_CLICK_MODE = False  # Set to True to suppress debug prints except for square lookup

force_standard_notation = False

# Chessboard settings
output_size = (800, 800)  # Final warped image size

# Stability / locking parameters (for board corners)
STABILITY_THRESHOLD = 0.1  # Allowed relative variation threshold for corners
STABLE_TIME_NEEDED = 1.0  # Must be stable for 1 second (for corners)
RECHECK_INTERVAL = 3.0  # Recheck corners every 3 seconds (if not using permanent lock)

# Tolerance for grid locking (2% variation allowed for square side length)
GRID_TOLERANCE = 0.02

# Exponential moving average factor for corner smoothing
corner_smooth_factor = 0.9

# Globals for smoothing lines and intersections (temporary, before locking)
prev_verticals = None
prev_horizontals = None
prev_intersections = None

# Globals to track locking state for board corners
board_locked = False
locked_corners = None
lock_start_time = 0.0

# Globals for grid locking
grid_locked = False
locked_verticals = None
locked_horizontals = None
locked_intersections = None
grid_stability_start_time = None
grid_history = []  # will store tuples: (verticals, horizontals, intersections)

# Buffer to track recent corner detections for board locking
corner_history = []
stability_start_time = None

prev_corners = None

# Global square database variable (updated only once after grid and corners are locked).
square_db = None

# Global digital board variables.
digital_board = {}
last_digital_board_state = {}  # (Remains from previous implementation)

# Global transformation matrix (from original image to warped view); set once board is locked.
transformation_matrix = None

# Global variable to hold the previous frame for movement detection.
prev_frame = None

# New global variable to track the last board state printed to the terminal.
LAST_PRINTED_BOARD = None

# Global variable to hold the board orientation.
# "normal" means that (in our default labeling) white is on the bottom (ranks 1–2)
# while "reversed" means white is on the top (ranks 7–8).
BOARD_ORIENTATION = "normal"

# -------------------- Load YOLOv11 Model for Chess Pieces --------------------
model = YOLO("C:/Users/blueb/Downloads/yoloChess3.pt", verbose=False)
DETECTION_CONFIDENCE_THRESHOLD = 0.3

# -------------------- Helper Functions --------------------
import os


def get_castling_rights(board):
    """
    Determines castling rights based on the board configuration.
    Returns a string with available rights (e.g., "KQkq") or "-" if none.
    """
    rights = ""
    # White kingside: King on E1 and Rook on H1.
    if board.get("E1", ".") == "K" and board.get("H1", ".") == "R":
        rights += "K"
    # White queenside: King on E1 and Rook on A1.
    if board.get("E1", ".") == "K" and board.get("A1", ".") == "R":
        rights += "Q"
    # Black kingside: King on E8 and Rook on H8.
    if board.get("E8", ".") == "k" and board.get("H8", ".") == "r":
        rights += "k"
    # Black queenside: King on E8 and Rook on A8.
    if board.get("E8", ".") == "k" and board.get("A8", ".") == "r":
        rights += "q"
    if rights == "":
        rights = "-"
    return rights


def generate_fen(board):
    """
    Generates a FEN string for the current digital board.
    Iterates over ranks 8 to 1 and files A-H, compressing consecutive empty squares.
    Then appends the side to move, castling rights, en passant, halfmove clock and fullmove number.
    """
    fen_rows = []
    for rank in range(8, 0, -1):
        row = ""
        empty_count = 0
        for file in "ABCDEFGH":
            square = f"{file}{rank}"
            piece = board[square]
            if piece == ".":
                empty_count += 1
            else:
                if empty_count > 0:
                    row += str(empty_count)
                    empty_count = 0
                row += piece
        if empty_count > 0:
            row += str(empty_count)
        fen_rows.append(row)
    fen_position = "/".join(fen_rows)
    castling_rights = get_castling_rights(board)
    fen_string = fen_position + " w " + castling_rights + " - 0 1"
    return fen_string


def update_board_from_detection(detected_squares):
    """
    Updates the digital board based on YOLO detections.
    """
    global digital_board
    old_binary = {sq: ("P" if digital_board[sq] != "." else ".") for sq in digital_board}
    new_binary = {}
    for sq in digital_board.keys():
        new_binary[sq] = "P" if sq in detected_squares else "."

    lost = [sq for sq in digital_board if old_binary[sq] == "P" and new_binary[sq] == "."]
    gained = [sq for sq in digital_board if old_binary[sq] == "." and new_binary[sq] == "P"]

    if len(lost) == 1 and len(gained) == 1:
        src = lost[0]
        dst = gained[0]
        if SHOW_DETECTION_OUTPUT:
            print(f"[DEBUG] Move detected: {src} -> {dst} (piece: {digital_board[src]})")
        digital_board[dst] = digital_board[src]
        digital_board[src] = "."
    elif len(lost) == 1 and len(gained) == 0:
        src = lost[0]
        candidates = [sq for sq in digital_board if old_binary[sq] == "P" and new_binary[sq] == "P" and sq != src]
        if candidates and square_db is not None:
            src_center = square_db[src]["center"]
            dst = min(candidates, key=lambda s: np.hypot(square_db[s]["center"][0] - src_center[0],
                                                         square_db[s]["center"][1] - src_center[1]))
            if SHOW_DETECTION_OUTPUT:
                print(f"[DEBUG] Capture move detected: {src} -> {dst} (capturing piece: {digital_board[src]})")
            digital_board[dst] = digital_board[src]
            digital_board[src] = "."
        else:
            if SHOW_DETECTION_OUTPUT:
                print("[DEBUG] Capture move detected but no candidate destination found.")
    else:
        if SHOW_DETECTION_OUTPUT:
            print(f"[DEBUG] No unique move detected. Lost: {lost}, Gained: {gained}")

    display_digital_board()


def print_digital_board(board):
    print("Digital Board State:")
    for rank in range(8, 0, -1):
        row = []
        for file in "ABCDEFGH":
            label = f"{file}{rank}"
            row.append(board[label])
        print(" ".join(row))
    print()


def display_digital_board():
    """
    Prints the current global digital_board and its FEN only if there is a change,
    then displays the board image.
    """
    global LAST_PRINTED_BOARD, digital_board
    if LAST_PRINTED_BOARD is None or digital_board != LAST_PRINTED_BOARD:
        print_digital_board(digital_board)
        fen_string = generate_fen(digital_board)
        print("[DEBUG] Current FEN:", fen_string)
        LAST_PRINTED_BOARD = digital_board.copy()
    board_img = render_digital_board(digital_board)
    cv2.imshow("Digital Board", board_img)


def save_locked_positions(corners, verticals, horizontals, intersections):
    import numpy as np
    data = {
        "corners": corners,
        "verticals": verticals,
        "horizontals": horizontals,
        "intersections": intersections
    }
    np.savez("locked_board.npz", **data)
    print("[DEBUG] Locked board data saved to locked_board.npz")


def load_locked_positions():
    import numpy as np
    filename = "locked_board.npz"
    if not os.path.exists(filename):
        print("[DEBUG] No locked_board.npz file found; starting fresh.")
        return None, None, None, None

    try:
        data = np.load(filename)
        corners = data["corners"]
        verticals = data["verticals"]
        horizontals = data["horizontals"]
        intersections = data["intersections"]
        print("[DEBUG] Loaded locked board data from locked_board.npz")
        return corners, verticals, horizontals, intersections
    except Exception as e:
        print(f"[DEBUG] Error loading locked_board.npz: {e}")
        return None, None, None, None


def check_if_starting_position(board):
    for file in "ABCDEFGH":
        if board[f"{file}1"] != "P":
            print(f"[DEBUG] check_if_starting_position failed: {file}1 is {board[f'{file}1']}, expected 'P'")
            return False
    for file in "ABCDEFGH":
        if board[f"{file}2"] != "P":
            print(f"[DEBUG] check_if_starting_position failed: {file}2 is {board[f'{file}2']}, expected 'P'")
            return False
    for file in "ABCDEFGH":
        if board[f"{file}7"] != "P":
            print(f"[DEBUG] check_if_starting_position failed: {file}7 is {board[f'{file}7']}, expected 'P'")
            return False
    for file in "ABCDEFGH":
        if board[f"{file}8"] != "P":
            print(f"[DEBUG] check_if_starting_position failed: {file}8 is {board[f'{file}8']}, expected 'P'")
            return False
    for rank in range(3, 7):
        for file in "ABCDEFGH":
            if board[f"{file}{rank}"] != ".":
                print("[DEBUG] ... expected '.' for empty")
                return False
    return True


def set_standard_notation(board):
    """
    Converts the digital board to full standard FEN notation.
    For normal orientation white is assigned to ranks 1 and 2;
    for reversed orientation white is assigned to ranks 7 and 8.
    """
    if BOARD_ORIENTATION == "normal":
        board["A1"] = "R"
        board["B1"] = "N"
        board["C1"] = "B"
        board["D1"] = "Q"
        board["E1"] = "K"
        board["F1"] = "B"
        board["G1"] = "N"
        board["H1"] = "R"

        for file in "ABCDEFGH":
            board[f"{file}2"] = "P"

        board["A8"] = "r"
        board["B8"] = "n"
        board["C8"] = "b"
        board["D8"] = "q"
        board["E8"] = "k"
        board["F8"] = "b"
        board["G8"] = "n"
        board["H8"] = "r"
    else:
        # In reversed orientation, white will appear on the top.
        board["H1"] = "R"
        board["G1"] = "N"
        board["F1"] = "B"
        board["E1"] = "Q"
        board["D1"] = "K"
        board["C1"] = "B"
        board["B1"] = "N"
        board["A1"] = "R"

        for file in "HGFEDCBA":
            board[f"{file}2"] = "P"

        board["H8"] = "r"
        board["G8"] = "n"
        board["F8"] = "b"
        board["E8"] = "q"
        board["D8"] = "k"
        board["C8"] = "b"
        board["B8"] = "n"
        board["A8"] = "r"
    # Clear the middle squares in either case.
    if BOARD_ORIENTATION == "normal":
        for file in "ABCDEFGH":
            for rank in range(3, 7):
                board[f"{file}{rank}"] = "."
    else:
        for file in "HGFEDCBA":
            for rank in range(3, 7):
                board[f"{file}{rank}"] = "."
    print("[DEBUG] set_standard_notation complete. Current board layout:")
    display_digital_board()


def dprint(*args, **kwargs):
    if not TEST_CLICK_MODE:
        print(*args, **kwargs)


def initialize_digital_board():
    board = {}
    for i in range(8):
        for j in range(8):
            label = f"{chr(ord('A') + i)}{8 - j}"
            board[label] = "."
    return board


def create_square_database(verticals, horizontals):
    db = {}
    if BOARD_ORIENTATION == "normal":
        for i in range(8):
            for j in range(8):
                label = f"{chr(ord('A') + i)}{8 - j}"
                x_min = verticals[i]
                x_max = verticals[i + 1]
                y_min = horizontals[j]
                y_max = horizontals[j + 1]
                center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
                db[label] = {"bbox": (x_min, y_min, x_max, y_max), "center": center}
    else:
        for i in range(8):
            for j in range(8):
                label = f"{chr(ord('H') - i)}{j + 1}"
                x_min = verticals[i]
                x_max = verticals[i + 1]
                y_min = horizontals[j]
                y_max = horizontals[j + 1]
                center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
                db[label] = {"bbox": (x_min, y_min, x_max, y_max), "center": center}
    return db


def render_digital_board(board, square_size=50):
    board_img = np.zeros((square_size * 8, square_size * 8, 3), dtype=np.uint8)
    dark_color = (29, 101, 181)
    light_color = (181, 217, 240)

    if BOARD_ORIENTATION == "normal":
        for i in range(8):
            for j in range(8):
                label = f"{chr(ord('A') + i)}{8 - j}"
                top_left = (i * square_size, j * square_size)
                bottom_right = ((i + 1) * square_size, (j + 1) * square_size)
                base_color = dark_color if (i + j) % 2 == 0 else light_color
                cv2.rectangle(board_img, top_left, bottom_right, base_color, thickness=-1)
                cv2.rectangle(board_img, top_left, bottom_right, (0, 0, 0), thickness=1)

                piece = board[label]
                if piece != ".":
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    thickness = 2
                    text_size, _ = cv2.getTextSize(piece, font, font_scale, thickness)
                    text_x = top_left[0] + (square_size - text_size[0]) // 2
                    text_y = top_left[1] + (square_size + text_size[1]) // 2
                    text_color = (255, 255, 255) if piece.isupper() else (0, 0, 0)
                    cv2.putText(board_img, piece, (text_x, text_y), font, font_scale, text_color, thickness,
                                cv2.LINE_AA)
                else:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.3
                    thickness = 1
                    text_color = (0, 0, 0) if base_color == light_color else (255, 255, 255)
                    cv2.putText(board_img, label, (top_left[0] + 3, bottom_right[1] - 3), font, font_scale, text_color,
                                thickness, cv2.LINE_AA)
    else:
        for i in range(8):
            for j in range(8):
                label = f"{chr(ord('H') - i)}{j + 1}"
                top_left = (i * square_size, j * square_size)
                bottom_right = ((i + 1) * square_size, (j + 1) * square_size)
                base_color = dark_color if (i + j) % 2 == 0 else light_color
                cv2.rectangle(board_img, top_left, bottom_right, base_color, thickness=-1)
                cv2.rectangle(board_img, top_left, bottom_right, (0, 0, 0), thickness=1)

                piece = board[label]
                if piece != ".":
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    thickness = 2
                    text_size, _ = cv2.getTextSize(piece, font, font_scale, thickness)
                    text_x = top_left[0] + (square_size - text_size[0]) // 2
                    text_y = top_left[1] + (square_size + text_size[1]) // 2
                    text_color = (255, 255, 255) if piece.isupper() else (0, 0, 0)
                    cv2.putText(board_img, piece, (text_x, text_y), font, font_scale, text_color, thickness,
                                cv2.LINE_AA)
                else:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.3
                    thickness = 1
                    text_color = (0, 0, 0) if base_color == light_color else (255, 255, 255)
                    cv2.putText(board_img, label, (top_left[0] + 3, bottom_right[1] - 3), font, font_scale, text_color,
                                thickness, cv2.LINE_AA)
    return board_img


def update_digital_board(piece_list):
    """
    Updates the digital board with the given list of square labels where a piece is present.
    """
    global digital_board, force_standard_notation
    if force_standard_notation:
        print("[DEBUG] Board locked; skipping update from YOLO detections.")
        display_digital_board()
        return

    new_board = {label: "." for label in digital_board}
    for sq in piece_list:
        if sq in new_board:
            new_board[sq] = "P"

    digital_board = new_board.copy()
    display_digital_board()


def detect_movement(current_frame, previous_frame, diff_threshold=5, pixel_fraction_threshold=0.005):
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(current_gray, previous_gray)
    _, diff_thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    changed_pixels = np.count_nonzero(diff_thresh)
    total_pixels = diff_thresh.shape[0] * diff_thresh.shape[1]
    if changed_pixels / total_pixels > pixel_fraction_threshold:
        print("Movement detected!")
        return True
    else:
        return False


def detect_chess_pieces(frame):
    """
    Detects chess pieces using the YOLO model.
    Returns a list of tuples: (bottom_center, color) where color is expected to be "white" or "black".
    """
    results = model(frame)
    detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                conf = float(box.conf.item())
                if conf > DETECTION_CONFIDENCE_THRESHOLD:
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = coords
                    bottom_center = ((x1 + x2) / 2, y2)
                    cls_id = int(box.cls.item())
                    color = model.names[cls_id].lower()  # expected "white" or "black"
                    detections.append((bottom_center, color))
    gamma_frame = apply_gamma_correction(frame, gamma=2.0)
    results_bright = model(gamma_frame)
    for result in results_bright:
        if result.boxes is not None:
            for box in result.boxes:
                conf = float(box.conf.item())
                if conf > DETECTION_CONFIDENCE_THRESHOLD:
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = coords
                    bottom_center = ((x1 + x2) / 2, y2)
                    cls_id = int(box.cls.item())
                    color = model.names[cls_id].lower()
                    found_nearby = False
                    for (dx, dy), _ in detections:
                        if abs(dx - bottom_center[0]) < 10 and abs(dy - bottom_center[1]) < 10:
                            found_nearby = True
                            break
                    if not found_nearby:
                        detections.append((bottom_center, color))
    return detections


def apply_gamma_correction(image, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def transform_point(pt, M):
    pt_arr = np.array([[pt]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt_arr, M)
    return tuple(transformed[0][0])


def smooth_corners(new_corners):
    global prev_corners
    if prev_corners is None:
        prev_corners = new_corners
        return new_corners
    smoothed_corners = prev_corners * (1 - corner_smooth_factor) + new_corners * corner_smooth_factor
    prev_corners = smoothed_corners
    return smoothed_corners


def order_corners(corners):
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    diff = np.diff(corners, axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
    return rect


def remove_outlier_points(intersections):
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
    threshold = side * 2.0
    filtered = []
    for i, (x, y) in enumerate(intersections):
        if min_dists[i] <= threshold:
            filtered.append((x, y))
    return filtered


def detect_grid(warped):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((6, 6), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=1)
    lines = cv2.HoughLinesP(dilated, 1, np.pi / 180, threshold=60, minLineLength=40, maxLineGap=70)
    if lines is None:
        return None, None, None
    vertical_lines = []
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 5:
            vertical_lines.append(x1)
        elif abs(dy) < 5:
            horizontal_lines.append(y1)

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

    def enforce_uniform_spacing(lines, max_ratio_deviation=0.35):
        if len(lines) < 3:
            return lines
        distances = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
        median_dist = np.median(distances)
        if median_dist <= 0:
            return lines
        filtered = [lines[0]]
        for i in range(len(distances)):
            current_gap = distances[i]
            ratio = abs(current_gap - median_dist) / float(median_dist)
            if ratio <= max_ratio_deviation:
                filtered.append(lines[i + 1])
        if len(filtered) < 4:
            return lines
        return filtered

    grouped_vertical = sorted(group_lines(vertical_lines))
    grouped_horizontal = sorted(group_lines(horizontal_lines))
    grouped_vertical = enforce_uniform_spacing(grouped_vertical)
    grouped_horizontal = enforce_uniform_spacing(grouped_horizontal)
    intersections = []
    for x in grouped_vertical:
        for y in grouped_horizontal:
            intersections.append((x, y))
    intersections = sorted(intersections, key=lambda p: (p[1], p[0]))
    filtered_points = remove_outlier_points(intersections)
    return filtered_points, grouped_vertical, grouped_horizontal


def mask_hands(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skinMask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    mask_inv = cv2.bitwise_not(skinMask)
    frame_masked = cv2.bitwise_and(frame, frame, mask=mask_inv)
    return frame_masked


def corners_are_within_threshold(new_corners, ref_corners, threshold=STABILITY_THRESHOLD, fixed_tol=10):
    for i in range(4):
        rx, ry = ref_corners[i]
        nx, ny = new_corners[i]
        diff_x = abs(nx - rx)
        diff_y = abs(ny - ry)
        rel_diff_x = diff_x / (abs(rx) + 1e-5)
        rel_diff_y = diff_y / (abs(ry) + 1e-5)
        dprint(
            f"[DEBUG] Corner {i}: ref=({rx:.2f},{ry:.2f}), new=({nx:.2f},{ny:.2f}), diff=({diff_x:.2f},{diff_y:.2f}), rel_diff=({rel_diff_x:.2f},{rel_diff_y:.2f})")
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


def compute_square_side(verticals):
    if len(verticals) < 2:
        return None
    diffs = [verticals[i + 1] - verticals[i] for i in range(len(verticals) - 1)]
    return np.median(diffs)


def grid_is_stable(history, tolerance=GRID_TOLERANCE):
    sides = []
    for (v, h, i) in history:
        side = compute_square_side(v)
        if side is not None:
            sides.append(side)
    if not sides:
        return False
    overall_median = np.median(sides)
    for s in sides:
        if abs(s - overall_median) / overall_median > tolerance:
            return False
    return True


def detect_and_lock_grid(warped):
    global grid_locked, locked_verticals, locked_horizontals, locked_intersections
    global grid_stability_start_time, grid_history
    now = time.time()
    if grid_locked:
        return locked_intersections, locked_verticals, locked_horizontals
    intersections, verticals, horizontals = detect_grid(warped)
    if intersections is None or verticals is None or horizontals is None:
        return None, None, None
    if len(verticals) != 9 or len(horizontals) != 9:
        dprint("[DEBUG] Grid detection did not return expected number of lines (9 each); skipping this frame.")
        return intersections, verticals, horizontals
    if grid_stability_start_time is None:
        grid_stability_start_time = now
        grid_history.clear()
    grid_history.append((verticals, horizontals, intersections))
    window_duration = now - grid_stability_start_time
    if window_duration >= STABLE_TIME_NEEDED:
        v_arrays = [np.array(v) for v, _, _ in grid_history]
        h_arrays = [np.array(h) for _, h, _ in grid_history]
        i_arrays = [np.array(i) for _, _, i in grid_history]
        if not (all(arr.shape == h_arrays[0].shape for arr in h_arrays) and
                all(arr.shape == v_arrays[0].shape for arr in v_arrays) and
                all(arr.shape == i_arrays[0].shape for arr in i_arrays)):
            dprint("[DEBUG] Inconsistent grid detection shapes; continuing accumulation.")
            return intersections, verticals, horizontals
        median_verticals = np.median(np.stack(v_arrays, axis=0), axis=0).tolist()
        median_horizontals = np.median(np.stack(h_arrays, axis=0), axis=0).tolist()
        median_intersections = np.median(np.stack(i_arrays, axis=0), axis=0).tolist()
        median_side = compute_square_side(median_verticals)
        if median_side is None:
            stable = False
        else:
            stable = grid_is_stable(grid_history, tolerance=GRID_TOLERANCE)
        if stable:
            grid_locked = True
            locked_verticals = median_verticals
            locked_horizontals = median_horizontals
            locked_intersections = median_intersections
            dprint("[DEBUG] Grid locked using median measurements.")
            grid_history.clear()
            grid_stability_start_time = None
            save_locked_positions(
                corners=locked_corners,
                verticals=locked_verticals,
                horizontals=locked_horizontals,
                intersections=locked_intersections
            )
            return locked_intersections, locked_verticals, locked_horizontals
        else:
            dprint("[DEBUG] Grid instability detected; continuing accumulation.")
    return intersections, verticals, horizontals


def create_square_database(verticals, horizontals):
    db = {}
    if BOARD_ORIENTATION == "normal":
        for i in range(8):
            for j in range(8):
                label = f"{chr(ord('A') + i)}{8 - j}"
                x_min = verticals[i]
                x_max = verticals[i + 1]
                y_min = horizontals[j]
                y_max = horizontals[j + 1]
                center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
                db[label] = {"bbox": (x_min, y_min, x_max, y_max), "center": center}
    else:
        for i in range(8):
            for j in range(8):
                label = f"{chr(ord('H') - i)}{j + 1}"
                x_min = verticals[i]
                x_max = verticals[i + 1]
                y_min = horizontals[j]
                y_max = horizontals[j + 1]
                center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
                db[label] = {"bbox": (x_min, y_min, x_max, y_max), "center": center}
    return db


def find_square_by_coordinate(x, y, square_db):
    for label, data in square_db.items():
        x_min, y_min, x_max, y_max = data["bbox"]
        if x_min <= x < x_max and y_min <= y < y_max:
            return label
    return None


def update_square_database(verticals, horizontals):
    global square_db
    square_db = create_square_database(verticals, horizontals)
    dprint("[DEBUG] Square database updated.")


def determine_orientation_from_starting_detections(frame):
    """
    Determines board orientation by tallying detection colours on the two starting groups:
    (assumed in the digital board, ranks 1-2 are one side and 7-8 are the other).
    If the bottom group (ranks 1-2) has more white detections and the top group (7-8) more black,
    BOARD_ORIENTATION is set to "normal". Otherwise, if the reverse is true, it is "reversed".
    """
    global BOARD_ORIENTATION
    dets = detect_chess_pieces(frame)
    bottom_white = 0
    bottom_black = 0
    top_white = 0
    top_black = 0
    if transformation_matrix is None or square_db is None:
        BOARD_ORIENTATION = "normal"
        return
    for (pt, color) in dets:
        pt_transformed = transform_point(pt, transformation_matrix)
        sq = find_square_by_coordinate(pt_transformed[0], pt_transformed[1], square_db)
        if sq is not None and len(sq) >= 2 and sq[1].isdigit():
            rank = int(sq[1])
            if rank in [1, 2]:
                if color == "white":
                    bottom_white += 1
                else:
                    bottom_black += 1
            elif rank in [7, 8]:
                if color == "white":
                    top_white += 1
                else:
                    top_black += 1
    if (bottom_white >= bottom_black) and (top_black >= top_white):
        BOARD_ORIENTATION = "normal"
    elif (bottom_black > bottom_white) and (top_white > top_black):
        BOARD_ORIENTATION = "reversed"
    else:
        BOARD_ORIENTATION = "normal"


def detectBoard(frame):
    global board_locked, locked_corners, lock_start_time
    global corner_history, stability_start_time
    now = time.time()
    if USE_PREVIOUS_BOARD_POSITIONS and locked_corners is not None:
        return warp_with_corners(frame, locked_corners)
    if board_locked:
        if PERMANENT_CORNER_LOCK:
            return warp_with_corners(frame, locked_corners)
        else:
            elapsed_since_lock = now - lock_start_time
            if elapsed_since_lock < RECHECK_INTERVAL:
                return warp_with_corners(frame, locked_corners)
            else:
                new_corners = find_board_corners(frame)
                if new_corners is not None and corners_are_within_threshold(new_corners, locked_corners):
                    lock_start_time = now
                    return warp_with_corners(frame, locked_corners)
                else:
                    board_locked = False
                    locked_corners = None
                    stability_start_time = None
                    corner_history.clear()
    new_corners = find_board_corners(frame)
    if new_corners is None:
        return frame
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
            lock_start_time = now
            corner_history.clear()
            stability_start_time = None
            save_locked_positions(
                corners=locked_corners,
                verticals=locked_verticals,
                horizontals=locked_horizontals,
                intersections=locked_intersections
            )
            return warp_with_corners(frame, locked_corners)
        else:
            stability_start_time = now
            corner_history.clear()
            corner_history.append(new_corners)
    return warp_with_corners(frame, new_corners)


def warp_with_corners(frame, corners):
    global prev_verticals, prev_horizontals, square_db, transformation_matrix
    stable_corners = order_corners(corners)
    dst_pts = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
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
    if (verticals is not None and horizontals is not None and
            len(verticals) == 9 and len(horizontals) == 9 and square_db is None):
        update_square_database(verticals, horizontals)
    if intersections is not None and verticals is not None and horizontals is not None:
        for x in verticals:
            cv2.line(warped, (int(x), 0), (int(x), warped.shape[0]), (255, 0, 0), 2)
        for y in horizontals:
            cv2.line(warped, (0, int(y)), (warped.shape[1], int(y)), (255, 0, 0), 2)
        for (x, y) in intersections:
            cv2.circle(warped, (int(x), int(y)), 3, (0, 0, 255), -1)
        if len(verticals) == 9 and len(horizontals) == 9:
            for i in range(8):
                for j in range(8):
                    center_x = int((verticals[i] + verticals[i + 1]) // 2)
                    center_y = int((horizontals[j] + horizontals[j + 1]) // 2)
                    notation = f"{chr(ord('A') + i)}{8 - j}" if BOARD_ORIENTATION == "normal" else f"{chr(ord('H') - i)}{j + 1}"
                    cv2.putText(warped, notation, (center_x - 20, center_y + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        dprint("[DEBUG] Grid detection failed.")
    return warped


def find_board_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 80, 200)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=1)
    kernel_close = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
    cv2.imshow("Closed", closed)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        dprint("[DEBUG] No contours found after closing.")
        return None
    largest_contour = max(contours, key=cv2.contourArea, default=None)
    if largest_contour is None or cv2.contourArea(largest_contour) < 5000:
        dprint("[DEBUG] Chessboard not detected or too small.")
        return None
    epsilon = 0.05 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    cv2.imshow("Debug Contour", frame.copy())
    if len(approx) > 5:
        dprint("[DEBUG] Chessboard corners not detected correctly (not 4).")
        return None
    new_corners = np.array([p[0] for p in approx], dtype=np.float32)
    smoothed_corners = smooth_corners(new_corners)
    return smoothed_corners


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if square_db is not None:
            square_label = find_square_by_coordinate(x, y, square_db)
            print(square_label)
        else:
            print("Square database not available.")


# New function to determine board orientation based on starting detections.
def determine_orientation_from_starting_detections(frame):
    global BOARD_ORIENTATION
    dets = detect_chess_pieces(frame)
    bottom_white = 0
    bottom_black = 0
    top_white = 0
    top_black = 0
    if transformation_matrix is None or square_db is None:
        BOARD_ORIENTATION = "normal"
        return
    for (pt, color) in dets:
        pt_transformed = transform_point(pt, transformation_matrix)
        sq = find_square_by_coordinate(pt_transformed[0], pt_transformed[1], square_db)
        if sq is not None and len(sq) >= 2 and sq[1].isdigit():
            rank = int(sq[1])
            if rank in [1, 2]:
                if color == "white":
                    bottom_white += 1
                else:
                    bottom_black += 1
            elif rank in [7, 8]:
                if color == "white":
                    top_white += 1
                else:
                    top_black += 1
    if (bottom_white >= bottom_black) and (top_black >= top_white):
        BOARD_ORIENTATION = "normal"
    elif (bottom_black > bottom_white) and (top_white > top_black):
        BOARD_ORIENTATION = "reversed"
    else:
        BOARD_ORIENTATION = "normal"


# Initialize the digital board.
digital_board = initialize_digital_board()
last_digital_board_state = digital_board.copy()

if USE_PREVIOUS_BOARD_POSITIONS or USE_PREVIOUS_GRIDLINES:
    c, v, h, i = load_locked_positions()
    if c is not None:
        locked_corners = c
        board_locked = True
    if v is not None and h is not None:
        locked_verticals = v
        locked_horizontals = h
        locked_intersections = i
        grid_locked = True

cv2.namedWindow("Warped Chessboard")
cv2.namedWindow("Digital Board")
if TEST_CLICK_MODE:
    cv2.setMouseCallback("Warped Chessboard", mouse_callback)

cap = cv2.VideoCapture(0)
prev_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_board = detectBoard(frame)
    cv2.imshow("Warped Chessboard", processed_board)

    # Movement detection with increased sensitivity.
    movement_detected = False
    if prev_frame is not None:
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(current_gray, previous_gray)
        _, diff_thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        changed_pixels = np.count_nonzero(diff_thresh)
        total_pixels = diff_thresh.size
        if changed_pixels / total_pixels > 0.015:
            movement_detected = True
            print("Movement detected, board update paused")

    # If movement is detected, wait 1 second and recheck.
    if movement_detected:
        print("Waiting 1 second to recheck movement...")
        time.sleep(1)
        ret_delay, frame_after_delay = cap.read()
        if not ret_delay:
            break
        current_gray = cv2.cvtColor(frame_after_delay, cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(current_gray, previous_gray)
        _, diff_thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        changed_pixels = np.count_nonzero(diff_thresh)
        total_pixels = diff_thresh.size
        if changed_pixels / total_pixels > 0.015:
            print("Movement still detected after delay, board update paused")
            prev_frame = frame_after_delay.copy()
            continue  # Skip board update
        else:
            print("Movement stopped after delay, proceeding with board update")
            frame = frame_after_delay
    prev_frame = frame.copy()

    MIN_PIECES_EXPECTED = 2

    if force_standard_notation and transformation_matrix is not None and square_db is not None and not movement_detected:
        piece_points = detect_chess_pieces(frame)
        detected_squares = []
        for (pt, _) in piece_points:
            pt_transformed = transform_point(pt, transformation_matrix)
            sq = find_square_by_coordinate(pt_transformed[0], pt_transformed[1], square_db)
            if sq is not None:
                detected_squares.append(sq)
        update_board_from_detection(detected_squares)
    elif transformation_matrix is not None and square_db is not None and not movement_detected:
        piece_points = detect_chess_pieces(frame)
        squares_detected = []
        for (pt, _) in piece_points:
            pt_transformed = transform_point(pt, transformation_matrix)
            sq = find_square_by_coordinate(pt_transformed[0], pt_transformed[1], square_db)
            if sq is not None:
                squares_detected.append(sq)
        if len(squares_detected) >= MIN_PIECES_EXPECTED:
            update_digital_board(squares_detected)
        else:
            print("[DEBUG] Detected too few squares/pieces; skipping update to avoid clearing board.")

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        print("[DEBUG] 's' key pressed.")
        print("[DEBUG] Checking if board is in the starting position...")
        display_digital_board()
        if check_if_starting_position(digital_board):
            # Determine orientation based on detections in the starting groups.
            determine_orientation_from_starting_detections(frame)
            print(f"[DEBUG] BOARD_ORIENTATION determined as: {BOARD_ORIENTATION}")
            set_standard_notation(digital_board)
            force_standard_notation = True
            print("[DEBUG] Successfully set standard notation and locked board.")
            squares_with_pieces = [sq for sq, piece in digital_board.items() if piece != "."]
            update_digital_board(squares_with_pieces)
        else:
            print("[DEBUG] Board is NOT recognized as starting position - 's' key press ignored.")

cap.release()
cv2.destroyAllWindows()
