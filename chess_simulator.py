import chess
import chess.engine
import time
import os

# Initialize the chess engine (ensure Stockfish binary path is correct)
engine_path = r"C:\Users\blueb\Documents\stockfish\stockfish-windows-x86-64-avx2.exe"  # Replace with the actual Stockfish executable path

# Check if the given path exists and is valid
if not os.path.exists(engine_path):
    raise FileNotFoundError(f"Stockfish executable not found at: {engine_path}")

try:
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
except Exception as e:
    raise RuntimeError(f"Failed to initialize the chess engine: {e}")


def move_piece_command(source, destination):
    """
    Generates the robot arm command for moving a piece.
    """
    return f"move_piece {source.upper()} - {destination.upper()}"


def take_piece_command(source, destination):
    """
    Generates the robot arm command for taking a piece.
    """
    return f"take_piece {source.upper()} - {destination.upper()}"


def display_robot_arm_command(move, board_before_move):
    """
    Based on the chessboard state, generate and display the appropriate command
    for the robot arm to execute the move.
    """
    source = chess.square_name(move.from_square)
    destination = chess.square_name(move.to_square)

    # Check if a piece was captured
    if board_before_move.is_capture(move):
        print(take_piece_command(source, destination))
    else:
        print(move_piece_command(source, destination))


def simulate_game(simulate_robot_moves=False):
    """
    Simulates the chess game with user vs Stockfish with robot arm move outputs.
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
    print(board)

    while not board.is_game_over():
        if (board.turn and user_is_white) or (not board.turn and not user_is_white):
            # User's turn
            print("\nYour move (e.g., E2E4): ", end="")
            move_input = input().strip().lower()

            try:
                user_move = chess.Move.from_uci(move_input)
                if user_move not in board.legal_moves:
                    raise ValueError("Illegal move")
                board.push(user_move)
                if simulate_robot_moves:
                    display_robot_arm_command(user_move, board.copy(stack=False))
            except ValueError as e:
                print(f"Invalid move: {e}")
                continue
        else:
            # Stockfish's turn
            print("Stockfish is thinking...")

            # Stockfish generates the best move
            result = engine.play(board, chess.engine.Limit(time=1.0))  # 1-second max thinking time
            stockfish_move = result.move
            board.push(stockfish_move)
            if simulate_robot_moves:
                display_robot_arm_command(stockfish_move, board.copy(stack=False))

        # Display the board
        print("\n", board)

    # Game over
    print("Game over!")
    if board.is_checkmate():
        print("Checkmate!")
    elif board.is_stalemate():
        print("Stalemate!")
    else:
        print("Draw!")

    engine.quit()


# Enable simulation with stockfish and robot arm move generation
simulate_game(simulate_robot_moves=True)
