import numpy as np
import cv2
import time
from ultralytics import YOLO
import logging
import robot_move
from robot_move import sleep_joints


# Suppress YOLO/Ultralytics logging output
logging.getLogger("ultralytics").setLevel(logging.ERROR)

#############################################
# NEW: Prompt the user to choose their side.
USER_SIDE = input("Choose your side (w for white, b for black): ").strip().lower()
if USER_SIDE not in ['w', 'b']:
    print("Invalid input. Defaulting to white.")
    USER_SIDE = 'w'
print(f"[DEBUG] User playing as: {'White' if USER_SIDE == 'w' else 'Black'}")
#############################################

# Global flag to disable printing of model detection output
SHOW_DETECTION_OUTPUT = False

Prev_positions = False

# If True, skip corner detection and always reuse the last locked corners (if any).
USE_PREVIOUS_BOARD_POSITIONS = Prev_positions
# If True, skip grid detection and always reuse the last locked grid lines (if any).
USE_PREVIOUS_GRIDLINES = Prev_positions

# Toggle for corner lock behavior: if True, once locked the corners remain locked permanently.
PERMANENT_CORNER_LOCK = False

# Toggle for test click mode: when True, debug prints (via dprint) are suppressed (except for square lookup).
TEST_CLICK_MODE = False  # Set to True to suppress debug prints except for square lookup

# NEW: Toggle for Stockfish test mode (simulated tracking via digital board clicks)
TEST_STOCKFISH_MODE = False

force_standard_notation = False

# Chessboard settings
output_size = (800, 800)  # Final warped image size

# Stability / locking parameters (for board corners)
STABILITY_THRESHOLD = 0.1  # Allowed relative variation threshold for corners
STABLE_TIME_NEEDED = 1.0   # Must be stable for 1 second (for corners)
RECHECK_INTERVAL = 3.0     # Recheck corners every 3 seconds (if not using permanent lock)

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

# New global variable to hold the previous board state printed to the terminal.
LAST_PRINTED_BOARD = None

# NEW: Global variables for Stockfish test mode simulated clicks.
selected_source_square = None
selected_dest_square = None

# NEW: Global flag to control simulation mode for robot moves.
SIMULATE_ROBOT_MOVES = False

# -------------------- Load YOLOv11 Model for Chess Pieces --------------------
model = YOLO("C:/Users/blueb/Downloads/yoloChess3.pt", verbose=False)
DETECTION_CONFIDENCE_THRESHOLD = 0.3

# NEW: Import the Stockfish module and initialize the engine.
from stockfish import Stockfish
# Adjust STOCKFISH_PATH as needed for your environment.
STOCKFISH_PATH = "C:\\Users\\blueb\\Documents\\stockfish\\stockfish-windows-x86-64-avx2"
stockfish = Stockfish(path=STOCKFISH_PATH, parameters={"Threads": 2, "Minimum Thinking Time": 30})


#############################################
# NEW: Robot move integration
# This function is taken from your robot_moves module.
def execute_robot_move(client, move_str, white, is_capture=False):
    """
    Simulate and execute the robot arm move based on the move notation.
    move_str: string in UCI format, e.g., "e2e4", "e1g1", "e1c1", etc.
    white: boolean, True if the moving piece is white.
    is_capture: boolean, True if the move is a capture move.

    Determines if the move is castling, a capture move, or a simple move and calls the
    relevant robot arm function.
    """
    move_str = move_str.upper()
    print(f"[DEBUG] Executing move: {move_str} (White: {white}, Capture: {is_capture})")
    # Check for castling moves.
    if white and move_str in ["E1G1", "E1C1"]:
        if move_str == "E1G1":
            print("[DEBUG] Detected kingside castling for white.")
            robot_move.king_castle(client, True)
        else:
            print("[DEBUG] Detected queenside castling for white.")
            robot_move.queen_castle(client, True)
    elif not white and move_str in ["E8G8", "E8C8"]:
        if move_str == "E8G8":
            print("[DEBUG] Detected kingside castling for black.")
            robot_move.king_castle(client, False)
        else:
            print("[DEBUG] Detected queenside castling for black.")
            robot_move.queen_castle(client, False)
    else:
        start_square = move_str[0:2]
        end_square = move_str[2:4]
        if is_capture:
            print(f"[DEBUG] Detected capture move from {start_square} to {end_square}.")
            robot_move.take_piece(client, start_square, end_square)
        else:
            print(f"[DEBUG] Detected simple move from {start_square} to {end_square}.")
            robot_move.move_piece(client, start_square, end_square)
            robot_move.sleep(client)


# -------------------- Helper Functions --------------------
import os


def get_castling_rights(board):
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

    for file in "ABCDEFGH":
        board[f"{file}7"] = "p"

    for rank in range(3, 7):
        for file in "ABCDEFGH":
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
            board[label] = "."  # dot means empty
    return board


def render_digital_board(board, square_size=50):
    board_img = np.zeros((square_size * 8, square_size * 8, 3), dtype=np.uint8)
    dark_color = (29, 101, 181)
    light_color = (181, 217, 240)

    if USER_SIDE == 'w':
        # White perspective, but invert shading so that A1 is dark.
        for i in range(8):
            for j in range(8):
                label = f"{chr(ord('A') + i)}{8 - j}"
                top_left = (i * square_size, j * square_size)
                bottom_right = ((i + 1) * square_size, (j + 1) * square_size)
                # Inverted shading: A1 is dark if (i + j) is odd => so base_color = light if (i+j) is even
                base_color = light_color if (i + j) % 2 == 0 else dark_color
                cv2.rectangle(board_img, top_left, bottom_right, base_color, thickness=-1)
                cv2.rectangle(board_img, top_left, bottom_right, (0, 0, 0), thickness=1)

                # Square label
                font = cv2.FONT_HERSHEY_SIMPLEX
                small_scale = 0.3
                small_thickness = 1
                label_color = (0, 0, 0) if (i + j) % 2 == 0 else (255, 255, 255)
                cv2.putText(board_img, label, (top_left[0] + 3, bottom_right[1] - 3),
                            font, small_scale, label_color, small_thickness, cv2.LINE_AA)

                # Piece if present
                piece = board[label]
                if piece != ".":
                    font_scale = 1.0
                    thickness = 2
                    text_size, _ = cv2.getTextSize(piece, font, font_scale, thickness)
                    text_x = top_left[0] + (square_size - text_size[0]) // 2
                    text_y = top_left[1] + (square_size + text_size[1]) // 2

                    # Now pick piece color based on whether it's uppercase (white) or lowercase (black).
                    if piece.isupper():
                        piece_color = (255, 255, 255)  # white pieces
                    else:
                        piece_color = (0, 0, 0)        # black pieces

                    cv2.putText(board_img, piece, (text_x, text_y), font, font_scale, piece_color, thickness, cv2.LINE_AA)
    else:
        # Black perspective, also invert shading so that from black's view, A8 is dark, etc.
        for i in range(8):
            for j in range(8):
                col = 7 - i
                row = 7 - j
                label = f"{chr(ord('A') + col)}{row + 1}"
                top_left = (i * square_size, j * square_size)
                bottom_right = ((i + 1) * square_size, (j + 1) * square_size)
                # (col + row) is even => use light_color; otherwise dark_color
                base_color = light_color if (col + row) % 2 == 0 else dark_color
                cv2.rectangle(board_img, top_left, bottom_right, base_color, thickness=-1)
                cv2.rectangle(board_img, top_left, bottom_right, (0, 0, 0), thickness=1)

                font = cv2.FONT_HERSHEY_SIMPLEX
                small_scale = 0.3
                small_thickness = 1
                label_color = (0, 0, 0) if (col + row) % 2 == 0 else (255, 255, 255)
                cv2.putText(board_img, label, (top_left[0] + 3, bottom_right[1] - 3),
                            font, small_scale, label_color, small_thickness, cv2.LINE_AA)

                piece = board[label]
                if piece != ".":
                    font_scale = 1.0
                    thickness = 2
                    text_size, _ = cv2.getTextSize(piece, font, font_scale, thickness)
                    text_x = top_left[0] + (square_size - text_size[0]) // 2
                    text_y = top_left[1] + (square_size + text_size[1]) // 2

                    # Use uppercase check for white, lowercase for black.
                    if piece.isupper():
                        piece_color = (255, 255, 255)
                    else:
                        piece_color = (0, 0, 0)

                    cv2.putText(board_img, piece, (text_x, text_y), font, font_scale, piece_color, thickness, cv2.LINE_AA)

    return board_img


def update_digital_board(piece_list):
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
    total_pixels = diff_thresh.shape[0]*diff_thresh.shape[1]
    if changed_pixels / total_pixels > pixel_fraction_threshold:
        print("Movement detected!")
        return True
    else:
        return False


def detect_chess_pieces(frame):
    results = model(frame)
    detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                conf = float(box.conf.item())
                if conf > DETECTION_CONFIDENCE_THRESHOLD:
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = coords
                    bottom_center = ((x1 + x2)/2, y2)
                    detections.append(bottom_center)
    gamma_frame = apply_gamma_correction(frame, gamma=2.0)
    results_bright = model(gamma_frame)
    for result in results_bright:
        if result.boxes is not None:
            for box in result.boxes:
                conf = float(box.conf.item())
                if conf > DETECTION_CONFIDENCE_THRESHOLD:
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = coords
                    bottom_center = ((x1 + x2)/2, y2)
                    found_nearby = False
                    for (dx, dy) in detections:
                        if abs(dx - bottom_center[0])<10 and abs(dy - bottom_center[1])<10:
                            found_nearby = True
                            break
                    if not found_nearby:
                        detections.append(bottom_center)
    return detections


def apply_gamma_correction(image, gamma=1.2):
    invGamma = 1.0/gamma
    table = np.array([((i/255.0)**invGamma)*255 for i in np.arange(0,256)]).astype("uint8")
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
    smoothed_corners = prev_corners*(1-corner_smooth_factor) + new_corners*corner_smooth_factor
    prev_corners = smoothed_corners
    return smoothed_corners


def order_corners(corners):
    # sort by y coordinate
    sorted_by_y = corners[corners[:,1].argsort()]
    top_two = sorted_by_y[:2]
    bottom_two = sorted_by_y[2:]
    # sort top two by x coordinate
    top_two = top_two[top_two[:,0].argsort()]
    # sort bottom two by x coordinate
    bottom_two = bottom_two[bottom_two[:,0].argsort()]
    tl, tr = top_two[0], top_two[1]
    bl, br = bottom_two[0], bottom_two[1]
    return np.array([tl, tr, br, bl], dtype="float32")


def remove_outlier_points(intersections):
    if not intersections or len(intersections)<4:
        return intersections
    min_dists = []
    for i,(x1,y1) in enumerate(intersections):
        best = float('inf')
        for j,(x2,y2) in enumerate(intersections):
            if i==j:
                continue
            dist = np.hypot(x2-x1,y2-y1)
            if dist<best:
                best = dist
        min_dists.append(best)
    side = np.median(min_dists)
    threshold = side*2.0
    filtered = []
    for i,(x,y) in enumerate(intersections):
        if min_dists[i]<=threshold:
            filtered.append((x,y))
    return filtered


def detect_grid(warped):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blurred,50,150)
    kernel = np.ones((6,6), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=1)
    lines = cv2.HoughLinesP(dilated,1, np.pi/180, threshold=60, minLineLength=40, maxLineGap=70)
    if lines is None:
        return None, None, None
    vertical_lines = []
    horizontal_lines = []
    for line in lines:
        x1,y1,x2,y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx)<5:
            vertical_lines.append(x1)
        elif abs(dy)<5:
            horizontal_lines.append(y1)

    def group_lines(vals, threshold=10):
        if not vals:
            return []
        vals = sorted(vals)
        grouped = []
        group = [vals[0]]
        for v in vals[1:]:
            if abs(v - group[-1])<threshold:
                group.append(v)
            else:
                grouped.append(int(np.mean(group)))
                group = [v]
        if group:
            grouped.append(int(np.mean(group)))
        return grouped

    def enforce_uniform_spacing(lines, max_ratio_deviation=0.35):
        if len(lines)<3:
            return lines
        distances = [lines[i+1]-lines[i] for i in range(len(lines)-1)]
        median_dist = np.median(distances)
        if median_dist<=0:
            return lines
        filtered = [lines[0]]
        for i in range(len(distances)):
            current_gap = distances[i]
            ratio = abs(current_gap-median_dist)/float(median_dist)
            if ratio<=max_ratio_deviation:
                filtered.append(lines[i+1])
        if len(filtered)<4:
            return lines
        return filtered

    grouped_vertical = sorted(group_lines(vertical_lines))
    grouped_horizontal = sorted(group_lines(horizontal_lines))
    grouped_vertical = enforce_uniform_spacing(grouped_vertical)
    grouped_horizontal = enforce_uniform_spacing(grouped_horizontal)
    intersections = []
    for x in grouped_vertical:
        for y in grouped_horizontal:
            intersections.append((x,y))
    intersections = sorted(intersections, key=lambda p:(p[1],p[0]))
    filtered_points = remove_outlier_points(intersections)
    return filtered_points, grouped_vertical, grouped_horizontal


def mask_hands(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0,48,80], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)
    skinMask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)
    skinMask = cv2.GaussianBlur(skinMask,(3,3),0)
    mask_inv = cv2.bitwise_not(skinMask)
    frame_masked = cv2.bitwise_and(frame, frame, mask=mask_inv)
    return frame_masked


def corners_are_within_threshold(new_corners, ref_corners, threshold=STABILITY_THRESHOLD, fixed_tol=10):
    for i in range(4):
        rx, ry = ref_corners[i]
        nx, ny = new_corners[i]
        diff_x = abs(nx-rx)
        diff_y = abs(ny-ry)
        rel_diff_x = diff_x/(abs(rx)+1e-5)
        rel_diff_y = diff_y/(abs(ry)+1e-5)
        dprint(f"[DEBUG] Corner {i}: ref=({rx:.2f},{ry:.2f}), new=({nx:.2f},{ny:.2f}), diff=({diff_x:.2f},{diff_y:.2f}), rel_diff=({rel_diff_x:.2f},{rel_diff_y:.2f})")
        if (diff_x>fixed_tol and rel_diff_x>threshold) or (diff_y>fixed_tol and rel_diff_y>threshold):
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
    if len(verticals)<2:
        return None
    diffs = [verticals[i+1]-verticals[i] for i in range(len(verticals)-1)]
    return np.median(diffs)


def grid_is_stable(history, tolerance=GRID_TOLERANCE):
    sides = []
    for (v,h,i) in history:
        side = compute_square_side(v)
        if side is not None:
            sides.append(side)
    if not sides:
        return False
    overall_median = np.median(sides)
    for s in sides:
        if abs(s-overall_median)/overall_median>tolerance:
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
    if len(verticals)!=9 or len(horizontals)!=9:
        dprint("[DEBUG] Grid detection did not return expected number of lines (9 each); skipping this frame.")
        return intersections, verticals, horizontals
    if grid_stability_start_time is None:
        grid_stability_start_time = now
        grid_history.clear()
    grid_history.append((verticals, horizontals, intersections))
    window_duration = now - grid_stability_start_time
    if window_duration>=STABILITY_TIME_NEEDED:
        v_arrays = [np.array(v) for v,_,_ in grid_history]
        h_arrays = [np.array(h) for _,h,_ in grid_history]
        i_arrays = [np.array(i) for _,_,i in grid_history]
        if not (all(arr.shape==h_arrays[0].shape for arr in h_arrays) and
                all(arr.shape==v_arrays[0].shape for arr in v_arrays) and
                all(arr.shape==i_arrays[0].shape for arr in i_arrays)):
            dprint("[DEBUG] Inconsistent grid detection shapes; continuing accumulation.")
            return intersections, verticals, horizontals
        median_verticals = np.median(np.stack(v_arrays,axis=0), axis=0).tolist()
        median_horizontals = np.median(np.stack(h_arrays,axis=0), axis=0).tolist()
        median_intersections = np.median(np.stack(i_arrays,axis=0), axis=0).tolist()
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
    for i in range(8):
        for j in range(8):
            label = f"{chr(ord('A')+i)}{8-j}"
            x_min = verticals[i]
            x_max = verticals[i+1]
            y_min = horizontals[j]
            y_max = horizontals[j+1]
            center = ((x_min+x_max)/2, (y_min+y_max)/2)
            db[label] = {"bbox": (x_min,y_min,x_max,y_max), "center": center}
    return db


def find_square_by_coordinate(x, y, square_db):
    for label, data in square_db.items():
        x_min, y_min, x_max, y_max = data["bbox"]
        if x_min<=x<x_max and y_min<=y<y_max:
            return label
    return None


def update_square_database(verticals, horizontals):
    global square_db
    square_db = create_square_database(verticals, horizontals)
    dprint("[DEBUG] Square database updated.")


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
            if elapsed_since_lock<RECHECK_INTERVAL:
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
    if window_duration>=STABLE_TIME_NEEDED:
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
        [0,0],
        [output_size[0]-1,0],
        [output_size[0]-1,output_size[1]-1],
        [0,output_size[1]-1]
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
            len(verticals)==9 and len(horizontals)==9 and square_db is None):
        update_square_database(verticals, horizontals)
    if intersections is not None and verticals is not None and horizontals is not None:
        for x in verticals:
            cv2.line(warped, (int(x),0), (int(x),warped.shape[0]), (255,0,0),2)
        for y in horizontals:
            cv2.line(warped, (0,int(y)), (warped.shape[1],int(y)), (255,0,0),2)
        for (x,y) in intersections:
            cv2.circle(warped, (int(x),int(y)),3, (0,0,255),-1)
        if len(verticals)==9 and len(horizontals)==9:
            for i in range(8):
                for j in range(8):
                    center_x = int((verticals[i]+verticals[i+1])//2)
                    center_y = int((horizontals[j]+horizontals[j+1])//2)
                    notation = f"{chr(ord('A')+i)}{8-j}"
                    cv2.putText(warped, notation, (center_x-20, center_y+10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
    else:
        dprint("[DEBUG] Grid detection failed.")
    return warped


def find_board_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blurred,80,200)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=1)
    kernel_close = np.ones((7,7), np.uint8)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
    cv2.imshow("Closed", closed)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        dprint("[DEBUG] No contours found after closing.")
        return None
    largest_contour = max(contours, key=cv2.contourArea, default=None)
    if largest_contour is None or cv2.contourArea(largest_contour)<5000:
        dprint("[DEBUG] Chessboard not detected or too small.")
        return None
    epsilon = 0.05 * cv2.arcLength(largest_contour,True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    cv2.imshow("Debug Contour", frame.copy())
    if len(approx)>5:
        dprint("[DEBUG] Chessboard corners not detected correctly (not 4).")
        return None
    new_corners = np.array([p[0] for p in approx], dtype=np.float32)
    smoothed_corners = smooth_corners(new_corners)
    return smoothed_corners


def mouse_callback(event, x, y, flags, param):
    if event==cv2.EVENT_LBUTTONDOWN:
        if square_db is not None:
            square_label = find_square_by_coordinate(x,y,square_db)
            print(square_label)
        else:
            print("Square database not available.")


# Line 1
def digital_board_mouse_callback(event, x, y, flags, param):
    # Line 1
    global selected_source_square, selected_dest_square, digital_board, stockfish, robot_client
    # Line 2
    square_size = 50
    # Line 3
    if event == cv2.EVENT_LBUTTONDOWN:
        # Line 4
        if USER_SIDE == 'w':
            col = x // square_size
            row = y // square_size
            clicked_square = f"{chr(ord('A') + int(col))}{8 - int(row)}"
        else:  # USER_SIDE == 'b'
            col = 7 - (x // square_size)
            row = 7 - (y // square_size)
            clicked_square = f"{chr(ord('A') + int(col))}{int(row) + 1}"
        # Line 12
        print(f"[TEST] Clicked square: {clicked_square}")
        # Line 13
        if selected_source_square is None:
            if digital_board.get(clicked_square, ".") != ".":
                selected_source_square = clicked_square
                print(f"[TEST] Selected source square: {selected_source_square}")
            else:
                print(f"[TEST] No piece on {clicked_square} to move.")
        else:
            # Line 22
            selected_dest_square = clicked_square
            print(f"[TEST] Selected destination square: {selected_dest_square}")
            # Line 24: Simulate the move on the digital board.
            piece = digital_board.get(selected_source_square, ".")
            digital_board[selected_dest_square] = piece
            digital_board[selected_source_square] = "."
            print(f"[TEST] Simulated move: {selected_source_square} -> {selected_dest_square}")
            display_digital_board()
            # Line 30: Generate FEN and force active color based on engine side.
            fen = generate_fen(digital_board)
            fen_parts = fen.split(" ")
            if USER_SIDE == 'w':
                # If user is white, engine (black) should move.
                fen_parts[1] = 'b'
            else:
                # If user is black, engine (white) should move.
                fen_parts[1] = 'w'
            fen_for_engine = " ".join(fen_parts)
            print(f"[TEST] Sending corrected FEN to Stockfish: {fen_for_engine}")
            stockfish.set_fen_position(fen_for_engine)
            # Line 40: Get engine move.
            best_move = stockfish.get_best_move()
            print(f"[TEST] Stockfish best move: {best_move}")
            # Line 42: Apply engine move on digital board.
            if best_move and len(best_move) >= 4:
                src_sq = best_move[0:2].upper()
                dst_sq = best_move[2:4].upper()
                is_capture = (digital_board.get(dst_sq, ".") != ".")
                digital_board[dst_sq] = digital_board.get(src_sq, ".")
                digital_board[src_sq] = "."
                print(f"[TEST] Applied Stockfish move: {src_sq} -> {dst_sq}")
                # Line 49: Wait until the system move is "completed"
                time.sleep(1)  # Simulate waiting for the robot move to complete
                display_digital_board()
                # Line 52: Set engine color flag (engine is white if user is black)
                engine_white = (USER_SIDE == 'b')
                execute_robot_move(robot_client, best_move, engine_white, is_capture)
            else:
                print("[TEST] No best move returned.")
            # Line 57
            selected_source_square = None
            selected_dest_square = None


digital_board = initialize_digital_board()
last_digital_board_state = digital_board.copy()

if USE_PREVIOUS_BOARD_POSITIONS or USE_PREVIOUS_GRIDLINES:
    c,v,h,i = load_locked_positions()
    if c is not None:
        locked_corners = c
        board_locked = True
    if v is not None and h is not None:
        locked_verticals = v
        locked_horizontals = h
        locked_intersections = i
        grid_locked = True

# NEW: Auto-populate the digital board with the standard chess starting position when in test mode.
if TEST_STOCKFISH_MODE:
    set_standard_notation(digital_board)
    force_standard_notation = True

if USE_PREVIOUS_BOARD_POSITIONS or USE_PREVIOUS_GRIDLINES:
    c,v,h,i = load_locked_positions()
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
if TEST_STOCKFISH_MODE:
    cv2.setMouseCallback("Digital Board", digital_board_mouse_callback)

if not TEST_STOCKFISH_MODE:
    cap = cv2.VideoCapture(0)
else:
    cap = None

prev_frame = None
opening_move_done = False

if SIMULATE_ROBOT_MOVES:
    from niryo_one_tcp_client import RobotTool

    class DummyClient:
        def move_pose(self, x, y, z, roll, pitch, yaw):
            print(f"[SIMULATION] move_pose(x={x}, y={y}, z={z})")
        def open_gripper(self, tool, speed):
            print(f"[SIMULATION] open_gripper(tool={tool}, speed={speed})")
        def close_gripper(self, tool, speed):
            print(f"[SIMULATION] close_gripper(tool={tool}, speed={speed})")
        def move_joints(self, *joints):
            print(f"[SIMULATION] move_joints({joints})")
        def calibrate(self, mode):
            print(f"[SIMULATION] calibrate(mode={mode})")
        def change_tool(self, tool):
            print(f"[SIMULATION] change_tool(tool={tool})")
        def set_arm_max_velocity(self, speed):
            print(f"[SIMULATION] set_arm_max_velocity(speed={speed})")
        def set_learning_mode(self, mode):
            print(f"[SIMULATION] set_learning_mode(mode={mode})")
        def quit(self):
            print("DummyClient: quit()")
        def connect(self, ip):
            print(f"DummyClient: connect(ip={ip})")
    robot_client = DummyClient()
else:
    from niryo_one_tcp_client import NiryoOneClient, CalibrateMode, RobotTool
    robot_client = NiryoOneClient()
    robot_client.connect("192.168.1.104")
    robot_client.set_arm_max_velocity(15)
    robot_client.change_tool(RobotTool.GRIPPER_1)
    robot_client.calibrate(CalibrateMode.AUTO)

while True:
    if TEST_STOCKFISH_MODE:
        frame = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        processed_board = frame.copy()
        cv2.imshow("Warped Chessboard", processed_board)

        if USER_SIDE=='b' and not opening_move_done:
            fen = generate_fen(digital_board)
            fen_parts = fen.split(" ")
            fen_parts[1] = "w"
            fen_for_engine = " ".join(fen_parts)
            print(f"[TEST] Engine (white) opening move. Adjusted FEN: {fen_for_engine}")
            stockfish.set_fen_position(fen_for_engine)
            best_move = stockfish.get_best_move()
            print(f"[TEST] Stockfish opening move: {best_move}")
            if best_move and len(best_move)>=4:
                src_sq = best_move[0:2].upper()
                dst_sq = best_move[2:4].upper()
                is_capture = (digital_board.get(dst_sq, ".")!=".")
                digital_board[dst_sq] = digital_board.get(src_sq, ".")
                digital_board[src_sq] = "."
                display_digital_board()
                opening_move_done = True
                engine_white = True
                execute_robot_move(robot_client, best_move, engine_white, is_capture)
    else:
        ret, frame = cap.read()
        if not ret:
            break
        processed_board = detectBoard(frame)
        cv2.imshow("Warped Chessboard", processed_board)

    if not TEST_STOCKFISH_MODE:
        movement_detected = False
        if prev_frame is not None:
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            previous_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(current_gray, previous_gray)
            _, diff_thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
            changed_pixels = np.count_nonzero(diff_thresh)
            total_pixels = diff_thresh.size
            if changed_pixels/total_pixels>0.015:
                movement_detected = True
                print("Movement detected, board update paused")
        if movement_detected:
            print("Waiting 1 second to recheck movement...")
            time.sleep(1)
            ret_delay, frame_after_delay = cap.read()
            if not ret_delay:
                break
            current_gray = cv2.cvtColor(frame_after_delay, cv2.COLOR_BGR2GRAY)
            previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(current_gray, previous_gray)
            _, diff_thresh = cv2.threshold(diff,15,255,cv2.THRESH_BINARY)
            changed_pixels = np.count_nonzero(diff_thresh)
            total_pixels = diff_thresh.size
            if changed_pixels/total_pixels>0.015:
                print("Movement still detected after delay, board update paused")
                prev_frame = frame_after_delay.copy()
                continue
            else:
                print("Movement stopped after delay, proceeding with board update")
                frame = frame_after_delay
        prev_frame = frame.copy()

    MIN_PIECES_EXPECTED = 2

    if not TEST_STOCKFISH_MODE:
        if force_standard_notation and transformation_matrix is not None and square_db is not None and not movement_detected:
            piece_points = detect_chess_pieces(frame)
            detected_squares = []
            for pt in piece_points:
                pt_transformed = transform_point(pt, transformation_matrix)
                sq = find_square_by_coordinate(pt_transformed[0], pt_transformed[1], square_db)
                if sq is not None:
                    detected_squares.append(sq)
            update_board_from_detection(detected_squares)
        elif transformation_matrix is not None and square_db is not None and not movement_detected:
            piece_points = detect_chess_pieces(frame)
            squares_detected = []
            for pt in piece_points:
                pt_transformed = transform_point(pt, transformation_matrix)
                sq = find_square_by_coordinate(pt_transformed[0], pt_transformed[1], square_db)
                if sq is not None:
                    squares_detected.append(sq)
            if len(squares_detected)>=MIN_PIECES_EXPECTED:
                update_digital_board(squares_detected)
            else:
                print("[DEBUG] Detected too few squares/pieces; skipping update to avoid clearing board.")
    else:
        pass

    key = cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
    elif key==ord('s'):
        print("[DEBUG] 's' key pressed.")
        print("[DEBUG] Checking if board is in the starting position...")
        display_digital_board()
        if check_if_starting_position(digital_board):
            print("[DEBUG] Board is recognized as the 'starting position'.")
            set_standard_notation(digital_board)
            force_standard_notation = True
            print("[DEBUG] Successfully set standard notation and locked board.")
            squares_with_pieces = [sq for sq,piece in digital_board.items() if piece!="."]
            update_digital_board(squares_with_pieces)
        else:
            print("[DEBUG] Board is NOT recognized as starting position - 's' key press ignored.")

if not TEST_STOCKFISH_MODE and cap is not None:
    cap.release()
cv2.destroyAllWindows()

if not SIMULATE_ROBOT_MOVES:
    robot_client.move_joints(*sleep_joints)
    robot_client.set_learning_mode(True)
    robot_client.quit()
