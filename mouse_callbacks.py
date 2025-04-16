# mouse_callbacks.py
import cv2
import time
import numpy as np
from digital_board import display_digital_board, generate_fen
from grid_processing import find_square_by_coordinate

# Initialize global variables
selected_source_square = None
selected_dest_square = None
digital_board = None
stockfish = None
robot_client = None

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if 'square_db' in param and param['square_db'] is not None:
            square_label = find_square_by_coordinate(x, y, param['square_db'])
            print(square_label)
        else:
            print("Square database not available.")

def digital_board_mouse_callback(event, x, y, flags, param):
    # Retrieve USER_SIDE from param (default to 'w' if not provided)
    user_side = param.get("USER_SIDE", "w")
    global selected_source_square, selected_dest_square, digital_board, stockfish, robot_client
    
    # Check if global variables are properly initialized
    if digital_board is None:
        print("[ERROR] digital_board not initialized")
        return
    if stockfish is None:
        print("[ERROR] stockfish not initialized")
        return
    if robot_client is None:
        print("[ERROR] robot_client not initialized")
        return
        
    square_size = 50
    if event == cv2.EVENT_LBUTTONDOWN:
        # Calculate column and row indices (0-based)
        col = x // square_size
        row = y // square_size
        
        # Convert to chess coordinates (A-H for files, 1-8 for ranks)
        file = chr(ord('A') + col)  # A through H from left to right
        rank = 8 - row  # 8 through 1 from top to bottom
            
        clicked_square = f"{file}{rank}"
        print(f"[TEST] Clicked square: {clicked_square}")
        
        if selected_source_square is None:
            if digital_board.get(clicked_square, ".") != ".":
                selected_source_square = clicked_square
                print(f"[TEST] Selected source square: {selected_source_square}")
            else:
                print(f"[TEST] No piece on {clicked_square} to move.")
        else:
            selected_dest_square = clicked_square
            print(f"[TEST] Selected destination square: {selected_dest_square}")
            piece = digital_board.get(selected_source_square, ".")
            digital_board[selected_dest_square] = piece
            digital_board[selected_source_square] = "."
            print(f"[TEST] Simulated move: {selected_source_square} -> {selected_dest_square}")
            display_digital_board(digital_board)
            fen = generate_fen(digital_board)
            fen_parts = fen.split(" ")
            if user_side == 'w':
                fen_parts[1] = 'b'
            else:
                fen_parts[1] = 'w'
            fen_for_engine = " ".join(fen_parts)
            print(f"[TEST] Sending corrected FEN to Stockfish: {fen_for_engine}")
            stockfish.set_fen_position(fen_for_engine)
            best_move = stockfish.get_best_move()
            print(f"[TEST] Stockfish best move: {best_move}")
            if best_move and len(best_move) >= 4:
                src_sq = best_move[0:2].upper()
                dst_sq = best_move[2:4].upper()
                is_capture = (digital_board.get(dst_sq, ".") != ".")
                digital_board[dst_sq] = digital_board.get(src_sq, ".")
                digital_board[src_sq] = "."
                print(f"[TEST] Applied Stockfish move: {src_sq} -> {dst_sq}")
                time.sleep(1)
                display_digital_board(digital_board)
                engine_white = (user_side == 'b')
                from robot_intergration import execute_robot_move
                execute_robot_move(robot_client, best_move, engine_white, is_capture)
            else:
                print("[TEST] No best move returned.")
            selected_source_square = None
            selected_dest_square = None
