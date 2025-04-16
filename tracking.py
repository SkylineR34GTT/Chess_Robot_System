# tracking.py
import numpy as np
from detection import detect_chess_pieces, detect_piece_type_and_color
from board_processing import transform_point
from grid_processing import find_square_by_coordinate
from digital_board import update_active_color, update_fen_after_move, display_digital_board
from captures import get_legal_capture_moves

def find_differences(prev_pieces, current_pieces, transformation_matrix, square_db):
    """Find lost and gained pieces between two frames."""
    print("\n[DEBUG] Finding differences between frames:")
    print(f"[DEBUG] Previous pieces count: {len(prev_pieces)}")
    print(f"[DEBUG] Current pieces count: {len(current_pieces)}")
    
    prev_squares = set()
    current_squares = set()
    
    print("\n[DEBUG] Previous pieces:")
    for pt in prev_pieces:
        pt_transformed = transform_point(pt, transformation_matrix)
        sq = find_square_by_coordinate(pt_transformed[0], pt_transformed[1], square_db)
        if sq is not None:
            prev_squares.add(sq)
            print(f"[DEBUG] Found piece at {sq}")
    
    print("\n[DEBUG] Current pieces:")
    for pt in current_pieces:
        pt_transformed = transform_point(pt, transformation_matrix)
        sq = find_square_by_coordinate(pt_transformed[0], pt_transformed[1], square_db)
        if sq is not None:
            current_squares.add(sq)
            print(f"[DEBUG] Found piece at {sq}")
    
    # Calculate lost squares (squares that had pieces but no longer do)
    lost = list(prev_squares - current_squares)
    
    # Calculate gained squares (squares that now have pieces but didn't before)
    # But exclude squares that were previously occupied
    gained = [sq for sq in current_squares if sq not in prev_squares]
    
    print(f"\n[DEBUG] Lost squares: {lost}")
    print(f"[DEBUG] Gained squares: {gained}")
    
    return lost, gained

def is_legal_move(src, dst, piece_type, digital_board):
    """Check if a move is legal for the given piece type."""
    # Get all legal moves for the piece
    legal_moves = get_legal_capture_moves(src, piece_type, digital_board)
    
    # For regular moves (not captures), we need to check if the destination is empty
    if digital_board[dst] == ".":
        # For pawns, check if it's a valid forward move
        if piece_type == 'p':
            file_src = ord(src[0]) - ord('A')
            rank_src = int(src[1])
            file_dst = ord(dst[0]) - ord('A')
            rank_dst = int(dst[1])
            
            # Pawns can only move forward one square (or two from starting position)
            direction = 1 if digital_board[src].isupper() else -1
            if file_src == file_dst:  # Must be same file for non-capture
                if rank_dst == rank_src + direction:
                    return True
                # Allow two-square move from starting position
                if (rank_src == 2 and direction == 1) or (rank_src == 7 and direction == -1):
                    if rank_dst == rank_src + 2*direction:
                        return True
            return False
            
        # For other pieces, we'd need to implement their movement rules
        # For now, we'll only validate captures
        return False
    
    # For captures, check if the destination is in legal moves
    return dst in legal_moves

def process_move(digital_board, lost, gained, frame, transformation_matrix, square_db, model, USER_SIDE, SHOW_DETECTION_OUTPUT=False):
    """Process a detected move and update the board state."""
    print("\n[DEBUG] Processing move:")
    print(f"[DEBUG] Lost pieces: {lost}")
    print(f"[DEBUG] Gained pieces: {gained}")
    
    if len(lost) == 1 and len(gained) == 1:
        src = lost[0]
        dst = gained[0]
        piece_type = digital_board[src].lower()
        
        print(f"[DEBUG] Regular move detected: {src} -> {dst}")
        print(f"[DEBUG] Moving piece: {digital_board[src]} (type: {piece_type})")
        
        # Validate the move
        if not is_legal_move(src, dst, piece_type, digital_board):
            print(f"[DEBUG] Illegal move detected: {src} -> {dst}")
            return False
            
        digital_board[dst] = digital_board[src]
        digital_board[src] = "."
        update_active_color()
        fen = update_fen_after_move(digital_board)
        print(f"[DEBUG] FEN after move: {fen}")
        return True
        
    elif len(lost) == 1 and len(gained) == 0:
        from detection import detect_piece_type_and_color
        
        src = lost[0]
        piece_type = digital_board[src].lower()
        print(f"\n[DEBUG] Capture detection started:")
        print(f"[DEBUG] Lost piece at {src}: {digital_board[src]} (type: {piece_type})")
        
        possible_captures = get_legal_capture_moves(src, piece_type, digital_board)
        print(f"[DEBUG] Legal capture moves for {piece_type} at {src}: {possible_captures}")
        
        if possible_captures:
            # For each possible capture square, check the color of the piece
            for dst in possible_captures:
                print(f"\n[DEBUG] Checking square {dst}:")
                print(f"[DEBUG] Current piece on {dst}: {digital_board[dst]}")
                
                detected_piece = detect_piece_type_and_color(frame, dst, transformation_matrix, square_db)
                print(f"[DEBUG] Detected piece on {dst}: {detected_piece}")
                
                if detected_piece != ".":  # If a piece is detected
                    # Check if the detected piece's color matches the moving piece's color
                    src_piece_color = 'w' if digital_board[src].isupper() else 'b'
                    dst_piece_color = 'w' if detected_piece.isupper() else 'b'
                    
                    print(f"[DEBUG] Source piece color: {src_piece_color}")
                    print(f"[DEBUG] Detected piece color: {dst_piece_color}")
                    
                    # If the colors match, this is likely the capture destination
                    if src_piece_color == dst_piece_color:
                        # Validate the capture move
                        if not is_legal_move(src, dst, piece_type, digital_board):
                            print(f"[DEBUG] Illegal capture detected: {src} -> {dst}")
                            continue
                            
                        print(f"[DEBUG] Color match found! Processing capture: {src} -> {dst}")
                        print(f"[DEBUG] Before move - {dst}: {digital_board[dst]}")
                        
                        digital_board[dst] = digital_board[src]
                        digital_board[src] = "."
                        
                        print(f"[DEBUG] After move - {dst}: {digital_board[dst]}")
                        print(f"[DEBUG] After move - {src}: {digital_board[src]}")
                        
                        update_active_color()
                        fen = update_fen_after_move(digital_board)
                        print(f"[DEBUG] FEN after capture: {fen}")
                        return True
                    else:
                        print(f"[DEBUG] Color mismatch - skipping {dst}")
                else:
                    print(f"[DEBUG] No piece detected on {dst}")
            
            print("[DEBUG] Could not determine capture destination - no matching piece colors found")
        else:
            print("[DEBUG] No legal capture moves found")
    return False 