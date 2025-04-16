# main.py
import sys
import os
import time
# Add the ApiTCP_Python_NiryoOne directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ApiTCP_Python_NiryoOne'))

import cv2
import numpy as np
import logging
from ultralytics import YOLO
from stockfish import Stockfish

# Import our custom modules
import digital_board
from robot_intergration import execute_robot_move
from digital_board import (initialize_digital_board, display_digital_board,
                           set_standard_notation, update_digital_board,
                           update_board_from_detection, generate_fen, check_if_starting_position, 
                           update_active_color, update_fen_after_move)
from board_processing import detectBoard, transform_point
from grid_processing import (load_locked_positions, update_square_database, 
                           find_square_by_coordinate, get_square_db)
from detection import detect_chess_pieces, detect_movement, detect_piece_type_and_color
import mouse_callbacks
from mouse_callbacks import mouse_callback, digital_board_mouse_callback
from helpers import dprint
from config import (USE_PREVIOUS_BOARD_POSITIONS, USE_PREVIOUS_GRIDLINES,
                   PERMANENT_CORNER_LOCK, TEST_CLICK_MODE, TEST_STOCKFISH_MODE,
                   TEST_PIECE_DETECTION, output_size, STABILITY_THRESHOLD,
                   STABLE_TIME_NEEDED, RECHECK_INTERVAL, GRID_TOLERANCE,
                   MOVEMENT_SETTLE_TIME)

import robot_move
from robot_move import sleep_joints
from captures import (check_capture_square, find_capture_move, get_legal_capture_moves, model)
from tracking import find_differences, process_move

# Suppress YOLO/Ultralytics logging output
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Global flags and variables
USER_SIDE = input("Choose your side (w for white, b for black): ").strip().lower()
if USER_SIDE not in ['w', 'b']:
    print("Invalid input. Defaulting to white.")
    USER_SIDE = 'w'
print(f"[DEBUG] User playing as: {'White' if USER_SIDE=='w' else 'Black'}")

SHOW_DETECTION_OUTPUT = False
force_standard_notation = False  # Initialize as False, will be set to True after board setup

# Global variables for board detection and grid locking
prev_corners = None
board_locked = False
locked_corners = None
lock_start_time = 0.0
prev_verticals = None
prev_horizontals = None
locked_verticals = None
locked_horizontals = None
locked_intersections = None
grid_stability_start_time = None
grid_history = []
corner_history = []
prev_frame = None
square_db = None
transformation_matrix = None
LAST_PRINTED_BOARD = None

# Global digital board
digital_board = initialize_digital_board()
last_digital_board_state = digital_board.copy()

# Global variables for Stockfish test mode simulated clicks
selected_source_square = None
selected_dest_square = None

SIMULATE_ROBOT_MOVES = False

# Initialize Stockfish engine
STOCKFISH_PATH = "C:\\Users\\blueb\\Documents\\stockfish\\stockfish-windows-x86-64-avx2"
stockfish = Stockfish(path=STOCKFISH_PATH, parameters={"Threads": 2, "Minimum Thinking Time": 5})

# Load locked positions if available
if USE_PREVIOUS_BOARD_POSITIONS or USE_PREVIOUS_GRIDLINES:
    c, v, h, i = load_locked_positions()
    if c is not None:
        locked_corners = c
        board_locked = True
    if v is not None and h is not None:
        locked_verticals = v
        locked_horizontals = h
        locked_intersections = i

# Auto-populate digital board in test mode
if TEST_STOCKFISH_MODE:
    set_standard_notation(digital_board)
    force_standard_notation = True

if USE_PREVIOUS_BOARD_POSITIONS or USE_PREVIOUS_GRIDLINES:
    c, v, h, i = load_locked_positions()
    if c is not None:
        try:
            c = np.array(c, dtype="float32")
        except Exception as e:
            print("[DEBUG] Error converting loaded corners to numpy array:", e)
            c = None

    if c is not None:
        locked_corners = c
        board_locked = True
    if v is not None and h is not None:
        locked_verticals = v
        locked_horizontals = h
        locked_intersections = i

cv2.namedWindow("Warped Chessboard")
cv2.namedWindow("Digital Board")
if TEST_CLICK_MODE:
    cv2.setMouseCallback("Warped Chessboard", mouse_callback)

if not TEST_STOCKFISH_MODE:
    cap = cv2.VideoCapture(0)
else:
    cap = None

opening_move_done = False

# Add active color tracking at the top with other global variables
active_color = 'w'  # Start with white's turn

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

# Set up mouse callbacks after robot client is initialized
if TEST_STOCKFISH_MODE:
    # Set up global variables for mouse callbacks
    mouse_callbacks.selected_source_square = selected_source_square
    mouse_callbacks.selected_dest_square = selected_dest_square
    mouse_callbacks.digital_board = digital_board
    mouse_callbacks.stockfish = stockfish
    mouse_callbacks.robot_client = robot_client
    # Pass USER_SIDE to the digital_board_mouse_callback via the param dictionary.
    cv2.setMouseCallback("Digital Board", digital_board_mouse_callback, {"USER_SIDE": USER_SIDE})

# Initialize movement tracking variables
movement_detected = False
movement_start_time = None

while True:
    if TEST_STOCKFISH_MODE:
        frame = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        processed_board = frame.copy()
        cv2.imshow("Warped Chessboard", processed_board)

        if USER_SIDE == 'b' and not opening_move_done:
            fen = generate_fen(digital_board)
            fen_parts = fen.split(" ")
            fen_parts[1] = "w"
            fen_for_engine = " ".join(fen_parts)
            print(f"[TEST] Engine (white) opening move. Adjusted FEN: {fen_for_engine}")
            stockfish.set_fen_position(fen_for_engine)
            best_move = stockfish.get_best_move()
            print(f"[TEST] Stockfish opening move: {best_move}")
            if best_move and len(best_move) >= 4:
                src_sq = best_move[0:2].upper()
                dst_sq = best_move[2:4].upper()
                is_capture = (digital_board.get(dst_sq, ".") != ".")
                digital_board[dst_sq] = digital_board.get(src_sq, ".")
                digital_board[src_sq] = "."
                display_digital_board(digital_board)
                opening_move_done = True
                engine_white = True
                execute_robot_move(robot_client, best_move, engine_white, is_capture)
    elif TEST_PIECE_DETECTION:
        test_piece_detection()
        break
    else:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect board and get transformation matrix
        warped, corners, board_locked, stability_start_time, corner_history, transformation_matrix = detectBoard(
            frame, locked_corners, board_locked, 0, corner_history, output_size,
            USE_PREVIOUS_BOARD_POSITIONS, PERMANENT_CORNER_LOCK, RECHECK_INTERVAL, STABLE_TIME_NEEDED)
            
        if warped is not None:
            cv2.imshow("Warped Chessboard", warped)
            
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):  # Press 't' to enter test mode
            if board_locked and transformation_matrix is not None and square_db is not None:
                from captures import test_piece_detection
                test_piece_detection(frame, square_db, transformation_matrix)
            else:
                print("Board not detected yet. Please wait for board detection to complete.")
        elif key == ord('s'):
            if (transformation_matrix is None):
                print("DEBUG: TRANSFORMATION MATRIX IS NONE")
            print("[DEBUG] 's' key pressed.")
            print("[DEBUG] Checking if board is in the starting position...")
            display_digital_board(digital_board)
            if check_if_starting_position(digital_board):
                print("[DEBUG] Board is recognized as the 'starting position'.")
                set_standard_notation(digital_board)
                force_standard_notation = True
                active_color = 'w'  # Set active color to white
                print("[DEBUG] Successfully set standard notation and locked board.")
                print("[DEBUG] Active color set to white.")
                squares_with_pieces = [sq for sq, piece in digital_board.items() if piece != "."]
                digital_board = update_digital_board(squares_with_pieces, digital_board, force_standard_notation)
                # Output the initial FEN
                initial_fen = generate_fen(digital_board)
                print(f"[DEBUG] Initial FEN: {initial_fen}")
            else:
                print("[DEBUG] Board is NOT recognized as starting position - 's' key press ignored.")

        # Movement detection
        if prev_frame is not None:
            # Check for movement
            current_movement_detected, frame, prev_frame = detect_movement(frame, prev_frame)
            
            if current_movement_detected and not movement_detected:
                # Movement just started
                movement_detected = True
                movement_start_time = time.time()
                print("[DEBUG] Movement detected, waiting for settle time...")
                
            elif movement_detected and not current_movement_detected:
                # Movement has stopped, check if we've waited long enough
                if time.time() - movement_start_time >= MOVEMENT_SETTLE_TIME:
                    print("[DEBUG] Movement settled, processing board state...")
                    
                    # Store current board state for comparison
                    old_board_state = digital_board.copy()
                    
                    # Run detection on current frame
                    current_pieces = detect_chess_pieces(frame)
                    detected_squares = []
                    for pt in current_pieces:
                        pt_transformed = transform_point(pt, transformation_matrix)
                        sq = find_square_by_coordinate(pt_transformed[0], pt_transformed[1], square_db)
                        if sq is not None:
                            detected_squares.append(sq)
                    
                    # Create new board state preserving piece types
                    new_board_state = old_board_state.copy()  # Start with current board state
                    
                    # For each detected square, check if it matches the current board state
                    for square in detected_squares:
                        # If the square was empty before but now has a piece, mark it as gained
                        if old_board_state[square] == ".":
                            # We don't know the piece type yet, so we'll let process_move handle it
                            new_board_state[square] = "P"  # Temporary placeholder
                    
                    # Find squares that were occupied but are now empty
                    for square in old_board_state:
                        if old_board_state[square] != "." and square not in detected_squares:
                            new_board_state[square] = "."
                    
                    # Find differences between old and new board states
                    lost = [sq for sq in old_board_state if old_board_state[sq] != "." and new_board_state[sq] == "."]
                    gained = [sq for sq in old_board_state if old_board_state[sq] == "." and new_board_state[sq] != "."]
                    
                    print(f"[DEBUG] Board state changes - Lost: {lost}, Gained: {gained}")
                    print(f"[DEBUG] Old board state: {old_board_state}")
                    print(f"[DEBUG] New board state: {new_board_state}")
                    
                    # Process the move
                    if process_move(digital_board, lost, gained, frame, transformation_matrix, square_db, model, USER_SIDE, SHOW_DETECTION_OUTPUT):
                        display_digital_board(digital_board)
                    
                    # Reset movement tracking
                    movement_detected = False
                    movement_start_time = None
                    
            # Update prev_frame for next iteration
            prev_frame = frame.copy()
        else:
            # Initialize prev_frame if it's None
            prev_frame = frame.copy()

    if not TEST_STOCKFISH_MODE:
        MIN_PIECES_EXPECTED = 2
        square_db = get_square_db()
        if transformation_matrix is not None and square_db is not None and not movement_detected:
            piece_points = detect_chess_pieces(frame)
            detected_squares = []
            for pt in piece_points:
                pt_transformed = transform_point(pt, transformation_matrix)
                sq = find_square_by_coordinate(pt_transformed[0], pt_transformed[1], square_db)
                if sq is not None:
                    detected_squares.append(sq)
            if len(detected_squares) >= MIN_PIECES_EXPECTED:
                digital_board = update_board_from_detection(detected_squares, digital_board, SHOW_DETECTION_OUTPUT, square_db, force_standard_notation)
            else:
                print("[DEBUG] Detected too few squares/pieces; skipping update to avoid clearing board.")
    else:
        pass

    # Update the FEN display after each move
    if SHOW_DETECTION_OUTPUT:
        print(f"[DEBUG] Current FEN: {generate_fen(digital_board)}")
        print("Digital Board State:")
        display_digital_board(digital_board)

if not TEST_STOCKFISH_MODE and cap is not None:
    cap.release()
cv2.destroyAllWindows()

if not SIMULATE_ROBOT_MOVES:
    robot_client.move_joints(*sleep_joints)
    robot_client.set_learning_mode(True)
    robot_client.quit()


