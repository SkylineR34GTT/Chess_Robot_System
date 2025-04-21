# digital_board.py
import cv2
import numpy as np

# Add this at the top of the file with other imports
last_printed_board = None
active_color = 'w'  # Initialize active color to white

def initialize_digital_board():
    board = {}
    for i in range(8):
        for j in range(8):
            label = f"{chr(ord('A') + i)}{8 - j}"
            board[label] = "."  # dot means empty
    return board

def render_digital_board(board, square_size=50, USER_SIDE='w'):
    board_img = np.zeros((square_size * 8, square_size * 8, 3), dtype=np.uint8)
    dark_color = (29, 101, 181)
    light_color = (181, 217, 240)

    # Draw the base board with alternating colors
    for i in range(8):
        for j in range(8):
            # Calculate square coordinates
            file = chr(ord('A') + i)  # A through H
            rank = 8 - j  # 8 through 1
            square = f"{file}{rank}"
            
            # Draw square
            top_left = (i * square_size, j * square_size)
            bottom_right = ((i + 1) * square_size, (j + 1) * square_size)
            base_color = light_color if (i + j) % 2 == 0 else dark_color
            cv2.rectangle(board_img, top_left, bottom_right, base_color, thickness=-1)
            cv2.rectangle(board_img, top_left, bottom_right, (0, 0, 0), thickness=1)

            # Draw square label
            font = cv2.FONT_HERSHEY_SIMPLEX
            small_scale = 0.3
            small_thickness = 1
            label_color = (0, 0, 0) if (i + j) % 2 == 0 else (255, 255, 255)
            cv2.putText(board_img, square, (top_left[0] + 3, bottom_right[1] - 3),
                       font, small_scale, label_color, small_thickness, cv2.LINE_AA)

            # Draw the piece if present
            piece = board[square]
            if piece != ".":
                font_scale = 1.0
                thickness = 2
                text_size, _ = cv2.getTextSize(piece, font, font_scale, thickness)
                text_x = top_left[0] + (square_size - text_size[0]) // 2
                text_y = top_left[1] + (square_size + text_size[1]) // 2
                piece_color = (255, 255, 255) if piece.isupper() else (0, 0, 0)
                cv2.putText(board_img, piece, (text_x, text_y), font, font_scale, piece_color, thickness, cv2.LINE_AA)

    return board_img

def print_digital_board(board):
    print("Digital Board State:")
    for rank in range(8, 0, -1):
        row = []
        for file in "ABCDEFGH":
            label = f"{file}{rank}"
            row.append(board[label])
        print(" ".join(row))
    print()

def update_fen_after_move(digital_board):
    """Update the FEN string after a move is made."""
    global active_color
    # Generate FEN with current color (already toggled by update_active_color)
    fen = generate_fen(digital_board)
    print(f"[DEBUG] Current FEN after move: {fen}")
    return fen

def update_active_color(new_color=None):
    """Update the active color. If new_color is provided, set it directly.
    Otherwise, toggle between 'w' and 'b'."""
    global active_color
    old_color = active_color
    
    if new_color is not None:
        active_color = new_color
    else:
        active_color = 'b' if active_color == 'w' else 'w'
    
    # Only print if the color actually changed
    if old_color != active_color:
        print(f"[DEBUG] Active color changed from {old_color} to {active_color}")
    
    return active_color

def generate_fen(board):
    """Generate FEN string from the current board state."""
    fen_parts = []
    
    # Convert board to FEN piece placement
    for rank in range(8, 0, -1):
        empty = 0
        rank_fen = ""
        for file in range(8):
            square = f"{chr(ord('A') + file)}{rank}"
            piece = board.get(square, ".")
            if piece == ".":
                empty += 1
            else:
                if empty > 0:
                    rank_fen += str(empty)
                    empty = 0
                rank_fen += piece
        if empty > 0:
            rank_fen += str(empty)
        fen_parts.append(rank_fen)
    
    # Join ranks with '/'
    piece_placement = "/".join(fen_parts)
    
    # Active color
    color = active_color
    
    # Castling rights (simplified for now)
    castling = "KQkq"
    
    # En passant (simplified for now)
    en_passant = "-"
    
    # Halfmove clock and fullmove number (simplified for now)
    halfmove = "0"
    fullmove = "1"
    
    return f"{piece_placement} {color} {castling} {en_passant} {halfmove} {fullmove}"

def get_castling_rights(board):
    rights = ""
    if board.get("E1", ".") == "K" and board.get("H1", ".") == "R":
        rights += "K"
    if board.get("E1", ".") == "K" and board.get("A1", ".") == "R":
        rights += "Q"
    if board.get("E8", ".") == "k" and board.get("H8", ".") == "r":
        rights += "k"
    if board.get("E8", ".") == "k" and board.get("A8", ".") == "r":
        rights += "q"
    if rights == "":
        rights = "-"
    return rights

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
    display_digital_board(board)

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

def update_digital_board(piece_list, digital_board, force_standard_notation=False):
    if force_standard_notation:
        print("[DEBUG] Board locked; skipping update from YOLO detections.")
        display_digital_board(digital_board)
        return digital_board
    new_board = {label: "." for label in digital_board}
    for sq in piece_list:
        if sq in new_board:
            new_board[sq] = "P"
    digital_board = new_board.copy()
    display_digital_board(digital_board)
    return digital_board

def update_board_from_detection(detected_squares, digital_board, SHOW_DETECTION_OUTPUT=False, square_db=None, force_standard_notation=False):
    """Updates the digital board based on detection results."""
    # If standard notation is forced, don't update board based on detections
    if force_standard_notation:
        print("[DEBUG] Standard notation is forced; preserving current board state.")
        # Return the board unchanged to preserve all pieces
        return digital_board
        
    old_binary = {sq: ("P" if digital_board[sq] != "." else ".") for sq in digital_board}
    new_binary = {}
    for sq in digital_board.keys():
        new_binary[sq] = "P" if sq in detected_squares else "."
    
    # First, update the board with all detected pieces if not forcing standard notation
    if not force_standard_notation:
        for square in detected_squares:
            if square in digital_board:
                digital_board[square] = "P"  # Mark as occupied
    
    # Then handle move detection (always track moves)
    lost = [sq for sq in digital_board if old_binary[sq] == "P" and new_binary[sq] == "."]
    gained = [sq for sq in digital_board if old_binary[sq] == "." and new_binary[sq] == "P"]
    
    if len(lost) == 1 and len(gained) == 1:
        src = lost[0]
        dst = gained[0]
        if SHOW_DETECTION_OUTPUT:
            print(f"[DEBUG] Move detected: {src} -> {dst} (piece: {digital_board[src]})")
        # Always update moves, but preserve piece type if forcing standard notation
        if force_standard_notation:
            piece_type = digital_board[src]
            digital_board[dst] = piece_type
            digital_board[src] = "."
        else:
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
            # Always update moves, but preserve piece type if forcing standard notation
            if force_standard_notation:
                piece_type = digital_board[src]
                digital_board[dst] = piece_type
                digital_board[src] = "."
            else:
                digital_board[dst] = digital_board[src]
                digital_board[src] = "."
        else:
            if SHOW_DETECTION_OUTPUT:
                print("[DEBUG] Capture move detected but no candidate destination found.")
    else:
        if SHOW_DETECTION_OUTPUT:
            print(f"[DEBUG] No unique move detected. Lost: {lost}, Gained: {gained}")
    
    display_digital_board(digital_board)
    return digital_board

def display_digital_board(board):
    global last_printed_board
    
    # Only print if the board state has changed
    if board != last_printed_board:
        print_digital_board(board)
        last_printed_board = board.copy()
    
    board_img = render_digital_board(board)
    cv2.imshow("Digital Board", board_img)
