# tracking.py
import numpy as np
from detection import detect_chess_pieces, detect_piece_type_and_color
from board_processing import transform_point
from grid_processing import find_square_by_coordinate
from digital_board import update_active_color, update_fen_after_move, display_digital_board, active_color
from captures import get_legal_capture_moves, find_capture_move

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
    
    # Calculate potential gained squares (squares that now have pieces but didn't before)
    potential_gained = list(current_squares - prev_squares)
    
    print(f"\n[DEBUG] Potential gained squares before filtering: {potential_gained}")
    
    # Initialize gained as empty list
    gained = []
    
    # IMPORTANT: If there are no lost squares, don't allow any gained squares
    # This prevents false detections from affecting future moves
    if len(lost) == 0:
        print(f"[DEBUG] No lost squares detected, ignoring all potential gained squares")
        return lost, gained
    
    # Check for castling moves first
    if len(lost) == 2 and len(potential_gained) == 2:
        # Sort lost and gained squares to ensure consistent ordering
        lost.sort()
        potential_gained.sort()
        
        # Define all possible castling square combinations
        castling_moves = {
            # White kingside castling
            'white_kingside': {
                'lost': ['E1', 'H1'],
                'gained': ['G1', 'F1']
            },
            # White queenside castling
            'white_queenside': {
                'lost': ['E1', 'A1'],
                'gained': ['C1', 'D1']
            },
            # Black kingside castling
            'black_kingside': {
                'lost': ['E8', 'H8'],
                'gained': ['G8', 'F8']
            },
            # Black queenside castling
            'black_queenside': {
                'lost': ['E8', 'A8'],
                'gained': ['C8', 'D8']
            }
        }
        
        # Check if the lost and gained squares match any castling move
        for castling_type, squares in castling_moves.items():
            if (set(lost) == set(squares['lost']) and 
                set(potential_gained) == set(squares['gained'])):
                print(f"[DEBUG] Valid {castling_type} castling move detected")
                print(f"[DEBUG] Lost squares: {lost}")
                print(f"[DEBUG] Gained squares: {potential_gained}")
                gained = potential_gained
                return lost, gained
    
    # For regular moves (1 lost piece), we need to filter the gained squares 
    # to only include squares that the lost piece could legally move to
    if len(lost) == 1:
        src = lost[0]
        piece_type = digital_board[src].lower()
        print(f"[DEBUG] Lost piece at {src} is type {piece_type}")
        
        # Track already added squares to prevent duplicates
        added_squares = set()
        
        # For each potential gained square, check if it's a legal move from the lost square
        for dst in potential_gained:
            # Skip if we've already added this square
            if dst in added_squares:
                print(f"[DEBUG] {dst} is already in gained squares, skipping duplicate")
                continue
                
            # Check if the move is legal for this piece type
            is_legal = is_legal_move(src, dst, piece_type, digital_board)
            print(f"[DEBUG] Checking if {src} -> {dst} is legal for {piece_type}: {is_legal}")
            
            if is_legal:
                gained.append(dst)
                added_squares.add(dst)
                print(f"[DEBUG] {dst} is a legal move for {piece_type} from {src}, adding to gained squares")
            else:
                print(f"[DEBUG] {dst} is not a legal move for {piece_type} from {src}, excluding from gained squares")
    # Multiple lost pieces (but not castling)
    else:
        # If we have multiple lost pieces but it's not a castling move,
        # we can't easily determine which gained square corresponds to which lost piece
        # For now, we'll just use the potential gained squares, but ensure no duplicates
        added_squares = set()
        for sq in potential_gained:
            if sq not in added_squares:
                gained.append(sq)
                added_squares.add(sq)
            else:
                print(f"[DEBUG] {sq} is already in gained squares, skipping duplicate")
        
        print(f"[DEBUG] Multiple lost pieces (non-castling): Using potential gained squares: {gained}")
    
    print(f"\n[DEBUG] Final lost squares: {lost}")
    print(f"[DEBUG] Final gained squares: {gained}")
    
    return lost, gained

def is_legal_move(src, dst, piece_type, digital_board):
    """Check if a move is legal for the given piece type."""
    # Get file and rank coordinates
    file_src = ord(src[0]) - ord('A')
    rank_src = int(src[1]) - 1
    file_dst = ord(dst[0]) - ord('A')
    rank_dst = int(dst[1]) - 1
    
    print(f"[DEBUG] is_legal_move: Checking if {src} -> {dst} is legal for {piece_type}")
    print(f"[DEBUG] is_legal_move: Source coordinates: ({file_src}, {rank_src})")
    print(f"[DEBUG] is_legal_move: Destination coordinates: ({file_dst}, {rank_dst})")
    
    # Check for castling
    if piece_type == 'k':
        # Check if this is a castling move (king moving 2 squares horizontally)
        if abs(file_dst - file_src) == 2 and rank_src == rank_dst:
            print(f"[DEBUG] is_legal_move: Potential castling move detected")
            
            # Determine if it's kingside or queenside castling
            is_kingside = file_dst > file_src
            
            # Check if the king has moved (this would be tracked in the game state)
            # For now, we'll assume the king hasn't moved if it's on its starting square
            king_start_square = "E1" if digital_board[src].isupper() else "E8"
            if src != king_start_square:
                print(f"[DEBUG] is_legal_move: King has moved from its starting square, castling not allowed")
                return False
                
            # Check if the rook has moved
            rook_file = "H" if is_kingside else "A"
            rook_rank = "1" if digital_board[src].isupper() else "8"
            rook_square = f"{rook_file}{rook_rank}"
            rook_piece = "R" if digital_board[src].isupper() else "r"
            
            if digital_board[rook_square] != rook_piece:
                print(f"[DEBUG] is_legal_move: Rook not in expected position for castling")
                return False
                
            # Check if the path is clear between king and rook
            start_file = min(file_src, file_dst) + 1
            end_file = max(file_src, file_dst) - 1
            
            for file in range(start_file, end_file + 1):
                square = f"{chr(ord('A') + file)}{rank_src + 1}"
                if digital_board[square] != ".":
                    print(f"[DEBUG] is_legal_move: Path not clear for castling at {square}")
                    return False
                    
            # Check if the king is in check or would pass through check
            # This is a simplified check - in a real implementation, you'd need to check
            # if the king is in check and if the squares it passes through are under attack
            # For now, we'll just allow castling if the basic conditions are met
            print(f"[DEBUG] is_legal_move: Castling move is legal")
            return True
    
    # Check if destination is empty (non-capture move)
    if digital_board[dst] == ".":
        print(f"[DEBUG] is_legal_move: Destination {dst} is empty")
        
        # For pawns, check if it's a valid forward move
        if piece_type == 'p':
            # Pawns can only move forward one square (or two from starting position)
            direction = 1 if digital_board[src].isupper() else -1
            print(f"[DEBUG] is_legal_move: Pawn direction: {direction}")
            
            if file_src == file_dst:  # Must be same file for non-capture
                print(f"[DEBUG] is_legal_move: Pawn moving in same file")
                if rank_dst == rank_src + direction:
                    print(f"[DEBUG] is_legal_move: Pawn moving one square forward")
                    return True
                # Allow two-square move from starting position
                if (rank_src == 1 and direction == 1) or (rank_src == 6 and direction == -1):
                    if rank_dst == rank_src + 2*direction:
                        print(f"[DEBUG] is_legal_move: Pawn moving two squares from starting position")
                        return True
            print(f"[DEBUG] is_legal_move: Pawn move is not legal")
            return False
        
        # For knights
        elif piece_type == 'n':
            # Knight moves in L-shape: 2 squares in one direction and 1 in the other
            dx = abs(file_dst - file_src)
            dy = abs(rank_dst - rank_src)
            is_legal = (dx == 2 and dy == 1) or (dx == 1 and dy == 2)
            print(f"[DEBUG] is_legal_move: Knight move is {'legal' if is_legal else 'not legal'}")
            return is_legal
        
        # For bishops
        elif piece_type == 'b':
            # Bishop moves diagonally
            dx = abs(file_dst - file_src)
            dy = abs(rank_dst - rank_src)
            if dx == dy and dx > 0:  # Must move diagonally
                # Check if path is clear
                is_legal = is_path_clear(src, dst, digital_board)
                print(f"[DEBUG] is_legal_move: Bishop move is {'legal' if is_legal else 'not legal'}")
                return is_legal
            print(f"[DEBUG] is_legal_move: Bishop move is not diagonal")
            return False
        
        # For rooks
        elif piece_type == 'r':
            # Rook moves horizontally or vertically
            if (file_src == file_dst or rank_src == rank_dst) and (file_src != file_dst or rank_src != rank_dst):
                # Check if path is clear
                is_legal = is_path_clear(src, dst, digital_board)
                print(f"[DEBUG] is_legal_move: Rook move is {'legal' if is_legal else 'not legal'}")
                return is_legal
            print(f"[DEBUG] is_legal_move: Rook move is not horizontal or vertical")
            return False
        
        # For queens
        elif piece_type == 'q':
            # Queen moves like bishop or rook
            dx = abs(file_dst - file_src)
            dy = abs(rank_dst - rank_src)
            if ((file_src == file_dst or rank_src == rank_dst) or (dx == dy)) and (file_src != file_dst or rank_src != rank_dst):
                # Check if path is clear
                is_legal = is_path_clear(src, dst, digital_board)
                print(f"[DEBUG] is_legal_move: Queen move is {'legal' if is_legal else 'not legal'}")
                return is_legal
            print(f"[DEBUG] is_legal_move: Queen move is not horizontal, vertical, or diagonal")
            return False
        
        # For kings
        elif piece_type == 'k':
            # King moves one square in any direction
            dx = abs(file_dst - file_src)
            dy = abs(rank_dst - rank_src)
            is_legal = dx <= 1 and dy <= 1 and (dx > 0 or dy > 0)
            print(f"[DEBUG] is_legal_move: King move is {'legal' if is_legal else 'not legal'}")
            return is_legal
    
    # For captures, check if the destination is in legal moves
    else:
        print(f"[DEBUG] is_legal_move: Destination {dst} is not empty, checking for capture")
        # Get all legal capture moves for the piece
        legal_moves = get_legal_capture_moves(src, piece_type, digital_board)
        is_legal = dst in legal_moves
        print(f"[DEBUG] is_legal_move: Capture move is {'legal' if is_legal else 'not legal'}")
        return is_legal
    
    return False

def is_path_clear(src, dst, digital_board):
    """Check if the path between src and dst is clear of pieces."""
    file_src = ord(src[0]) - ord('A')
    rank_src = int(src[1]) - 1
    file_dst = ord(dst[0]) - ord('A')
    rank_dst = int(dst[1]) - 1
    
    print(f"[DEBUG] is_path_clear: Checking path from {src} to {dst}")
    print(f"[DEBUG] is_path_clear: Source coordinates: ({file_src}, {rank_src})")
    print(f"[DEBUG] is_path_clear: Destination coordinates: ({file_dst}, {rank_dst})")
    
    # Determine direction of movement
    dx = 0 if file_src == file_dst else (file_dst - file_src) // abs(file_dst - file_src)
    dy = 0 if rank_src == rank_dst else (rank_dst - rank_src) // abs(rank_dst - rank_src)
    
    print(f"[DEBUG] is_path_clear: Movement direction: dx={dx}, dy={dy}")
    
    # Check each square along the path
    x, y = file_src + dx, rank_src + dy
    while x != file_dst or y != rank_dst:
        square = f"{chr(ord('A') + x)}{y + 1}"
        print(f"[DEBUG] is_path_clear: Checking square {square}")
        if digital_board[square] != ".":
            print(f"[DEBUG] is_path_clear: Path blocked at {square} by {digital_board[square]}")
            return False
        x += dx
        y += dy
    
    print(f"[DEBUG] is_path_clear: Path is clear")
    return True

def process_move(digital_board, lost, gained, frame, transformation_matrix, square_db, model, USER_SIDE, SHOW_DETECTION_OUTPUT=False):
    """Process a move based on lost and gained squares."""
    print("\n[DEBUG] Processing move:")
    print(f"[DEBUG] Lost pieces: {lost}")
    print(f"[DEBUG] Unfiltered gained pieces: {gained}")
    
    # Store the original gained squares before any further filtering
    original_gained = gained.copy()
    
    # Filter gained squares for legal moves
    filtered_gained = []
    if len(lost) == 1 and len(gained) > 0:
        src = lost[0]
        piece_type = digital_board[src].lower()
        print(f"[DEBUG] Filtering gained squares based on legality of moves for {piece_type} from {src}")
        
        # Use a set to track already added squares to prevent duplicates
        added_squares = set()
        
        for dst in gained:
            # Skip if we've already processed this square
            if dst in added_squares:
                print(f"[DEBUG] {dst} is already in gained squares, skipping duplicate")
                continue
                
            if is_legal_move(src, dst, piece_type, digital_board):
                filtered_gained.append(dst)
                added_squares.add(dst)
                print(f"[DEBUG] {dst} is a legal move for {piece_type} from {src}")
            else:
                print(f"[DEBUG] {dst} is not a legal move for {piece_type} from {src}, excluding")
    else:
        # Remove duplicates from gained squares
        filtered_gained = list(dict.fromkeys(gained))
        if len(filtered_gained) != len(gained):
            print(f"[DEBUG] Removed duplicate gained squares. Original: {gained}, Filtered: {filtered_gained}")
        
    print(f"[DEBUG] Filtered gained pieces: {filtered_gained}")
    print(f"[DEBUG] Current active_color: {active_color}")
    print(f"[DEBUG] USER_SIDE: {USER_SIDE}")
    
    # Check for castling moves first (using original gained squares)
    if len(lost) == 2 and len(gained) == 2:
        # Sort lost and gained squares to ensure consistent ordering
        lost.sort()
        gained.sort()
        
        # Define all possible castling square combinations
        castling_moves = {
            # White kingside castling
            'white_kingside': {
                'lost': ['E1', 'H1'],
                'gained': ['G1', 'F1']
            },
            # White queenside castling
            'white_queenside': {
                'lost': ['E1', 'A1'],
                'gained': ['C1', 'D1']
            },
            # Black kingside castling
            'black_kingside': {
                'lost': ['E8', 'H8'],
                'gained': ['G8', 'F8']
            },
            # Black queenside castling
            'black_queenside': {
                'lost': ['E8', 'A8'],
                'gained': ['C8', 'D8']
            }
        }
        
        # Check if the lost and gained squares match any castling move
        for castling_type, squares in castling_moves.items():
            if (set(lost) == set(squares['lost']) and 
                set(gained) == set(squares['gained'])):
                print(f"[DEBUG] Processing {castling_type} castling move")
                
                # Move the king
                king_src = squares['lost'][0]  # E1 or E8
                king_dst = squares['gained'][0]  # G1/G8 or C1/C8
                digital_board[king_dst] = digital_board[king_src]
                digital_board[king_src] = "."
         
                # Move the rook
                rook_src = squares['lost'][1]  # H1/A1 or H8/A8
                rook_dst = squares['gained'][1]  # F1/D1 or F8/D8
                digital_board[rook_dst] = digital_board[rook_src]
                digital_board[rook_src] = "."
                
                # Update active color after castling
                new_color = 'b' if active_color == 'w' else 'w'
                update_active_color(new_color)
                
                print(f"[DEBUG] Castling move completed: {king_src}->{king_dst}, {rook_src}->{rook_dst}")
                return True
    
    # Now work with the filtered gained squares for regular moves
    # Use the filtered gained squares for move processing
    gained = filtered_gained
    
    # Handle regular moves (one piece lost, one piece gained)
    if len(lost) == 1 and len(gained) == 1:
        src = lost[0]
        dst = gained[0]
        piece_type = digital_board[src].lower()
        
        print(f"[DEBUG] Regular move detected: {src} -> {dst}")
        print(f"[DEBUG] Moving piece: {digital_board[src]} (type: {piece_type})")
        
        # We've already validated the move during filtering, but double-check
        if not is_legal_move(src, dst, piece_type, digital_board):
            print(f"[DEBUG] Illegal move detected: {src} -> {dst}")
            return False
            
        digital_board[dst] = digital_board[src]
        digital_board[src] = "."
        
        # Update active color after move - only toggle if it's a valid move
        new_color = 'b' if active_color == 'w' else 'w'
        update_active_color(new_color)
        fen = update_fen_after_move(digital_board)
        print(f"[DEBUG] FEN after move: {fen}")
        print(f"[DEBUG] Active color after move: {active_color}")
        print(f"[DEBUG] Move successfully processed: {src} -> {dst}")
        return True
        
    # Handle captures (one piece lost, no pieces gained)
    elif len(lost) == 1 and len(gained) == 0:
        from captures import find_capture_move
        
        src = lost[0]
        piece_type = digital_board[src].lower()
        print(f"\n[DEBUG] Capture detection started:")
        print(f"[DEBUG] Lost piece at {src}: {digital_board[src]} (type: {piece_type})")
        
        # Verify that the lost piece is of the active color
        is_white_piece = digital_board[src].isupper()
        is_black_piece = digital_board[src].islower()
        is_active_color_piece = (is_white_piece and active_color == 'w') or (is_black_piece and active_color == 'b')
        
        if not is_active_color_piece:
            print(f"[DEBUG] Lost piece at {src} is not of the active color, ignoring capture")
            return False
        
        possible_captures = get_legal_capture_moves(src, piece_type, digital_board)
        print(f"[DEBUG] Legal capture moves for {piece_type} at {src}: {possible_captures}")
        
        if possible_captures:
            # Use find_capture_move to determine the capture destination
            dst = find_capture_move(frame, src, possible_captures, transformation_matrix, square_db, model, USER_SIDE)
            
            if dst is not None:
                print(f"[DEBUG] Capture destination found: {dst}")
                
                # Verify that the captured piece is of the opposite color
                captured_piece = digital_board.get(dst, ".")
                if captured_piece != ".":
                    is_captured_white = captured_piece.isupper()
                    is_captured_black = captured_piece.islower()
                    is_opposite_color = (is_white_piece and is_captured_black) or (is_black_piece and is_captured_white)
                    
                    if not is_opposite_color:
                        print(f"[DEBUG] Cannot capture piece of the same color at {dst}: {captured_piece}")
                        return False
                
                # Validate the capture move
                if not is_legal_move(src, dst, piece_type, digital_board):
                    print(f"[DEBUG] Illegal capture detected: {src} -> {dst}")
                    return False
                    
                print(f"[DEBUG] Processing capture: {src} -> {dst}")
                print(f"[DEBUG] Before move - {dst}: {digital_board[dst]}")
                
                digital_board[dst] = digital_board[src]
                digital_board[src] = "."
                
                print(f"[DEBUG] After move - {dst}: {digital_board[dst]}")
                print(f"[DEBUG] After move - {src}: {digital_board[src]}")
                
                # Update active color after capture - only toggle if it's a valid capture
                new_color = 'b' if active_color == 'w' else 'w'
                update_active_color(new_color)
                fen = update_fen_after_move(digital_board)
                print(f"[DEBUG] FEN after capture: {fen}")
                print(f"[DEBUG] Capture successfully processed: {src} -> {dst}")
                return True
            else:
                print("[DEBUG] Could not determine capture destination")
        else:
            print("[DEBUG] No legal capture moves found")
    
    print("[DEBUG] No move detected or move processing failed")
    return False 