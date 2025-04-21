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
                   MOVEMENT_SETTLE_TIME, ELO)

import robot_move
from robot_move import sleep_joints, king_castle, queen_castle
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

# Initialize system_turn based on USER_SIDE
# If user is white, system (robot) starts as black (False)
# If user is black, system (robot) starts as white (True)
system_turn = (USER_SIDE == 'b')

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
stockfish = Stockfish(path=STOCKFISH_PATH, parameters={"Threads": 1, "Minimum Thinking Time": 1})

# Set Stockfish to a more manageable ELO level (1800 is approximately club player level)
stockfish.set_elo_rating(ELO)
print(f"[DEBUG] Stockfish engine set to ELO rating:", ELO)

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

# Import the active_color from digital_board module
from digital_board import active_color as digital_board_active_color

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
                print(f"[DEBUG] Successfully set standard notation and locked board.")
                print(f"[DEBUG] Active color set to white.")
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
                    
                    # Find differences between old and new board states
                    lost = []
                    gained = []
                    
                    # First find lost pieces (pieces that were there but are now gone)
                    # Only consider pieces of the active color as potentially lost
                    for square in old_board_state:
                        piece = old_board_state[square]
                        if piece != "." and square not in detected_squares:
                            # Check if the piece is of the active color
                            is_white_piece = piece.isupper()
                            is_black_piece = piece.islower()
                            is_active_color_piece = (is_white_piece and active_color == 'w') or (is_black_piece and active_color == 'b')
                            
                            if is_active_color_piece:
                                lost.append(square)
                                print(f"[DEBUG] Lost piece detected at {square}: {piece} (active color: {active_color})")
                            else:
                                print(f"[DEBUG] Ignoring non-active color piece at {square}: {piece} (active color: {active_color})")
                    
                    # Only look for gained pieces if we found lost pieces
                    if lost:
                        for square in detected_squares:
                            if old_board_state[square] == ".":
                                gained.append(square)
                                print(f"[DEBUG] Gained piece detected at {square}")
                    
                    print(f"[DEBUG] Board state changes - Lost: {lost}, Gained: {gained}")
                    print(f"[DEBUG] Old board state: {old_board_state}")
                    
                    # Process the move
                    if process_move(digital_board, lost, gained, frame, transformation_matrix, square_db, model, USER_SIDE, SHOW_DETECTION_OUTPUT):
                        print("[DEBUG] Move detected and processed")
                        # Update the previous frame only after successful move processing
                        prev_frame = frame.copy()
                        print("[DEBUG] Updated previous frame after move processing")
                    else:
                        print("[DEBUG] No move detected or move processing failed")
                        # Initialize prev_frame if it's None
                        if prev_frame is None:
                            prev_frame = frame.copy()
                            print("[DEBUG] Initialized previous frame")
                    
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
            # Skip detection and board updates if standard notation is forced
            if force_standard_notation:
                # If standard notation is forced, don't update the board from detections
                pass
            else:
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

    # Check if it's the robot's turn to play based on FEN active color
    if force_standard_notation:
        fen = generate_fen(digital_board)
        fen_parts = fen.split(" ")
        active_color = fen_parts[1]  # Get active color from FEN
        
        # Update the digital_board module's active_color only if it's different
        from digital_board import update_active_color, active_color as digital_board_active_color
        if active_color != digital_board_active_color:
            update_active_color(active_color)
        
        # If active color doesn't match user's side, it's the robot's turn
        if active_color != USER_SIDE:
            print(f"[DEBUG] Robot's turn to play (active_color: {active_color}, USER_SIDE: {USER_SIDE})")
            print(f"[DEBUG] Current FEN for engine: {fen}")
            stockfish.set_fen_position(fen)
            best_move = stockfish.get_best_move()
            print(f"[DEBUG] Stockfish move: {best_move}")
            
            if best_move and len(best_move) >= 4:
                src_sq = best_move[0:2].upper()
                dst_sq = best_move[2:4].upper()
                is_capture = (digital_board.get(dst_sq, ".") != ".")
                
                # Validate the move before executing
                piece_type = digital_board.get(src_sq, ".").lower()
                if piece_type != ".":
                    from tracking import is_legal_move
                    if is_legal_move(src_sq, dst_sq, piece_type, digital_board):
                        print(f"[DEBUG] Executing robot move: {src_sq} -> {dst_sq}")
                        
                        # Check if this is a castling move
                        is_castling = False
                        is_kingside = False
                        
                        # Check for kingside castling
                        if (src_sq == "E1" and dst_sq == "G1") or (src_sq == "E8" and dst_sq == "G8"):
                            is_castling = True
                            is_kingside = True
                            print(f"[DEBUG] Detected kingside castling: {src_sq}->{dst_sq}")
                        
                        # Check for queenside castling
                        elif (src_sq == "E1" and dst_sq == "C1") or (src_sq == "E8" and dst_sq == "C8"):
                            is_castling = True
                            is_kingside = False
                            print(f"[DEBUG] Detected queenside castling: {src_sq}->{dst_sq}")
                        
                        # Move the king
                        digital_board[dst_sq] = digital_board.get(src_sq, ".")
                        digital_board[src_sq] = "."
                        
                        # If it's a castling move, also move the rook
                        if is_castling:
                            if is_kingside:
                                if src_sq == "E1":
                                    digital_board["F1"] = digital_board.get("H1", ".")
                                    digital_board["H1"] = "."
                                else:  # E8
                                    digital_board["F8"] = digital_board.get("H8", ".")
                                    digital_board["H8"] = "."
                            else:  # queenside
                                if src_sq == "E1":
                                    digital_board["D1"] = digital_board.get("A1", ".")
                                    digital_board["A1"] = "."
                                else:  # E8
                                    digital_board["D8"] = digital_board.get("A8", ".")
                                    digital_board["A8"] = "."
                            print(f"[DEBUG] Moved rook for castling")
                        
                        display_digital_board(digital_board)
                        
                        # Use the appropriate castling function if it's a castling move
                        if is_castling:
                            # Determine if we're castling white or black pieces
                            # If USER_SIDE is 'w', then the robot is playing black
                            # If USER_SIDE is 'b', then the robot is playing white
                            # For castling, we need to pass True for white castling, False for black castling
                            is_white_castling = (USER_SIDE == 'b')  # Robot is playing white if USER_SIDE is 'b'
                            
                            # Check if the castling move is for white or black
                            # If the source square is E1, it's white castling
                            # If the source square is E8, it's black castling
                            is_white_castling = (src_sq == "E1")
                            
                            print(f"[DEBUG] Castling {'white' if is_white_castling else 'black'} pieces")
                            
                            if is_kingside:
                                king_castle(robot_client, is_white_castling)
                            else:
                                queen_castle(robot_client, is_white_castling)
                        else:
                            execute_robot_move(robot_client, best_move, USER_SIDE == 'w', is_capture)
                            
                        # Update active color after robot move
                        new_active_color = 'w' if active_color == 'b' else 'b'
                        if new_active_color != digital_board_active_color:
                            update_active_color(new_active_color)
                        active_color = new_active_color
                        fen = update_fen_after_move(digital_board)
                        print(f"[DEBUG] FEN after robot move: {fen}")
                    else:
                        print(f"[DEBUG] Illegal move detected by engine: {src_sq} -> {dst_sq}")
                else:
                    print(f"[DEBUG] No piece found at source square: {src_sq}")
            else:
                print("[DEBUG] No valid move returned by engine")
        

if not TEST_STOCKFISH_MODE and cap is not None:
    cap.release()
cv2.destroyAllWindows()

if not SIMULATE_ROBOT_MOVES:
    robot_client.move_joints(*sleep_joints)
    robot_client.set_learning_mode(True)
    robot_client.quit()


