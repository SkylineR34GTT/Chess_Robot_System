import chess
import chess.engine
import time
import os
from digital_board import initialize_digital_board, generate_fen, update_digital_board, display_digital_board
from mouse_callbacks import digital_board_mouse_callback

# Initialize the chess engine (ensure Stockfish binary path is correct)
engine_path = r"C:\Users\blueb\Documents\stockfish\stockfish-windows-x86-64-avx2.exe"  # Replace with the actual Stockfish executable path

# Check if the given path exists and is valid
if not os.path.exists(engine_path):
    raise FileNotFoundError(f"Stockfish executable not found at: {engine_path}")

try:
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
except Exception as e:
    raise RuntimeError(f"Failed to initialize the chess engine: {e}")

# Initialize the digital board
digital_board = initialize_digital_board()

# Global variables to handle user interactions
selected_source_square = None
selected_dest_square = None


def update_game_board(chessboard, from_square, to_square):
    """
    Updates the chessboard and digital board based on a move.
    """
    # Fetch the piece being moved
    piece = digital_board.get(from_square, ".")

    # Update digital board for the move
    digital_board[to_square] = piece
    digital_board[from_square] = "."

    # Call the update_digital_board function with the necessary arguments
    update_digital_board(digital_board, from_square, to_square)

def display_robot_move_command(move, is_capture, white_side):
    """
    Prints the robot move string command.
    """
    move_str = f"execute_robot_move(client, '{move}', {'True' if white_side else 'False'}, {'True' if is_capture else 'False'})"
    print(move_str)


def user_move_input_with_click(digital_board, stockfish, chessboard, user_side, robot_client):
    """
    Enables users to make a move by clicking on squares on the digital board.
    """
    global selected_source_square, selected_dest_square

    waiting_for_move = True
    while waiting_for_move:
        print("[INFO] Click on a source or destination square.")
        time.sleep(1)  # Dummy delay to simulate waiting for a click

        event_data = {
            "USER_SIDE": user_side,
            "selected_source_square": selected_source_square,
            "selected_dest_square": selected_dest_square,
            "digital_board": digital_board,
            "stockfish": stockfish,
        }
        digital_board_mouse_callback(None, 0, 0, None, event_data)

        if selected_source_square and selected_dest_square:
            try:
                move = chess.Move.from_uci(f"{selected_source_square.lower()}{selected_dest_square.lower()}")
                if move not in chessboard.legal_moves:
                    raise ValueError("Illegal move")
                chessboard.push(move)
                update_game_board(chessboard, selected_source_square, selected_dest_square)
                display_digital_board(digital_board)
                waiting_for_move = False

            except ValueError as ve:
                print(f"[ERROR] Invalid move: {ve}")
                selected_source_square = None
                selected_dest_square = None

    selected_source_square = None
    selected_dest_square = None


def stockfish_move_response(chessboard):
    """
    Handles Stockfish's move, updating the board and generating robot move command strings.
    """
    print("Stockfish is thinking...")
    result = engine.play(chessboard, chess.engine.Limit(time=1.0))
    move = result.move
    chessboard.push(move)

    is_capture = digital_board.get(move.uci()[2:].upper(), ".") != "."

    update_game_board(chessboard, move.uci()[:2].upper(), move.uci()[2:].upper())
    display_digital_board(digital_board)

    display_robot_move_command(move.uci().upper(), is_capture, chessboard.turn)


def simulate_game():
    """
    Simulates playing a game between the user and Stockfish.
    """
    board = chess.Board()

    print("Welcome to Digital Chess!")
    user_color = input("Do you want to play as White or Black? (W/B): ").strip().upper()
    if user_color not in ("W", "B"):
        print("Invalid choice. Defaulting to White.")
        user_color = "W"

    user_is_white = user_color == "W"
    print("Game started!")
    print("You are playing as White" if user_is_white else "You are playing as Black")
    display_digital_board(digital_board)

    while not board.is_game_over():
        if (board.turn and user_is_white) or (not board.turn and not user_is_white):
            user_move_input_with_click(digital_board, stockfish=engine, chessboard=board, user_side=user_color.lower(),
                                       robot_client=None)
            fen = generate_fen(digital_board)
            board.set_fen(fen)
        else:
            stockfish_move_response(board)

    print("Game over!")
    if board.is_checkmate():
        print("Checkmate!")
    elif board.is_stalemate():
        print("Stalemate!")
    else:
        print("Draw!")
    engine.quit()


simulate_game()