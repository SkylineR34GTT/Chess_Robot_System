# captures.py
import cv2
import numpy as np
from ultralytics import YOLO
from config import DETECTION_CONFIDENCE_THRESHOLD

# Load YOLO model for chess pieces
model = YOLO("C:/Users/blueb/Downloads/yoloChess3.pt", verbose=False)


def check_capture_square(frame, square, transformation_matrix, square_db, model):
    """Check a specific square for piece detection using full-frame detection."""
    if square not in square_db:
        return None, 0.0
        
    # Get the square's bounding box
    x_min, y_min, x_max, y_max = square_db[square]["bbox"]
    
    # Transform the coordinates back to camera view
    inv_matrix = np.linalg.inv(transformation_matrix)
    camera_coords = []
    for x, y in [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]:
        pt = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, inv_matrix)
        camera_coords.append(transformed[0][0])
    
    # Get the bounding box in camera coordinates
    camera_coords = np.array(camera_coords, dtype=np.int32)
    x_min_cam = min(camera_coords[:, 0])
    x_max_cam = max(camera_coords[:, 0])
    y_min_cam = min(camera_coords[:, 1])
    y_max_cam = max(camera_coords[:, 1])
    
    # Calculate center point
    center_x = (x_min_cam + x_max_cam) // 2
    center_y = (y_min_cam + y_max_cam) // 2
    
    # Run YOLO detection on the full frame
    results = model(frame)
    
    best_class = None
    best_confidence = 0.0
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                conf = float(box.conf.item())
                if conf > DETECTION_CONFIDENCE_THRESHOLD:
                    # Get the box coordinates
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = coords
                    box_center = ((x1 + x2) / 2, y2)  # Using bottom center point
                    
                    # Check if this detection is within our square's bounds
                    if (x_min_cam <= box_center[0] <= x_max_cam and 
                        y_min_cam <= box_center[1] <= y_max_cam):
                        if conf > best_confidence:
                            best_confidence = conf
                            best_class = int(box.cls.item())
    
    return best_class, best_confidence

def find_capture_move(frame, src_square, possible_dst_squares, transformation_matrix, square_db, model, user_color):
    """Find which square a piece moved to during a capture using full-frame detection."""
    print("\n[DEBUG] Starting capture move detection")
    print(f"[DEBUG] Source square: {src_square}")
    print(f"[DEBUG] Possible destination squares: {possible_dst_squares}")
    
    best_square = None
    best_confidence = 0.0
    
    # Create a copy of the frame for visualization
    debug_frame = frame.copy()
    
    # Run YOLO detection on the full frame once
    results = model(frame)
    print(f"[DEBUG] YOLO detection complete, processing results...")
    
    # Process all detections
    detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                conf = float(box.conf.item())
                if conf > DETECTION_CONFIDENCE_THRESHOLD:
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = coords
                    bottom_center = ((x1 + x2) / 2, y2)
                    cls = int(box.cls.item())
                    detections.append((bottom_center, cls, conf))
                    # Draw all detections on debug frame
                    cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(debug_frame, (int(bottom_center[0]), int(bottom_center[1])), 5, (0, 0, 255), -1)
    
    print(f"[DEBUG] Found {len(detections)} detections above confidence threshold")
    
    # For each possible capture square
    for dst_square in possible_dst_squares:
        if dst_square not in square_db:
            print(f"[DEBUG] Square {dst_square} not found in square database")
            continue
            
        # Get the square's bounding box
        x_min, y_min, x_max, y_max = square_db[dst_square]["bbox"]
        print(f"\n[DEBUG] Processing square {dst_square}")
        print(f"[DEBUG] Warped coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
        
        # Transform the coordinates back to camera view
        inv_matrix = np.linalg.inv(transformation_matrix)
        camera_coords = []
        for x, y in [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]:
            pt = np.array([[[x, y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, inv_matrix)
            camera_coords.append(transformed[0][0])
        
        # Get the bounding box in camera coordinates
        camera_coords = np.array(camera_coords, dtype=np.int32)
        x_min_cam = min(camera_coords[:, 0])
        x_max_cam = max(camera_coords[:, 0])
        y_min_cam = min(camera_coords[:, 1])
        y_max_cam = max(camera_coords[:, 1])
        
        print(f"[DEBUG] Camera coordinates: x_min={x_min_cam}, y_min={y_min_cam}, x_max={x_max_cam}, y_max={y_max_cam}")
        
        # Draw the transformed square on debug frame
        cv2.rectangle(debug_frame, (x_min_cam, y_min_cam), (x_max_cam, y_max_cam), (255, 0, 0), 2)
        cv2.putText(debug_frame, dst_square, (x_min_cam, y_min_cam - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Find detections within this square
        square_detections = []
        for detection in detections:
            bottom_center, cls, conf = detection
            if (x_min_cam <= bottom_center[0] <= x_max_cam and 
                y_min_cam <= bottom_center[1] <= y_max_cam):
                square_detections.append((cls, conf))
                print(f"[DEBUG] Detection in square {dst_square}: class={cls}, confidence={conf:.2f}")
                
                # Check if the detected piece matches the user's color
                is_white_piece = 6 <= cls <= 11
                is_user_color = (user_color == 'w' and is_white_piece) or (user_color == 'b' and not is_white_piece)
                
                if is_user_color and conf > best_confidence:
                    best_confidence = conf
                    best_square = dst_square
                    print(f"[DEBUG] New best match: {dst_square} with confidence {conf:.2f}")
    
    # Show the debug visualization
    cv2.imshow("Capture Detection Debug", debug_frame)
    cv2.waitKey(1)  # Update the display
    
    if best_square:
        print(f"\n[DEBUG] Best capture destination: {best_square} with confidence {best_confidence:.2f}")
    else:
        print("\n[DEBUG] No valid capture destination found")
    
    return best_square

def get_legal_capture_moves(src_square, piece_type, board):
    """Get all possible legal capture moves for a piece."""
    file = ord(src_square[0]) - ord('A')
    rank = int(src_square[1]) - 1
    moves = []
    
    # Helper function to check if a square contains an opponent's piece
    def is_opponent_piece(square):
        if square not in board or board[square] == '.':
            return False
        piece = board[square]
        src_piece = board[src_square]
        return (piece.isupper() and src_piece.islower()) or (piece.islower() and src_piece.isupper())
    
    # Helper function to add a move if it's a valid capture
    def add_capture_move(new_file, new_rank):
        if 0 <= new_file < 8 and 0 <= new_rank < 8:
            dst_square = f"{chr(ord('A') + new_file)}{new_rank + 1}"
            if is_opponent_piece(dst_square):
                moves.append(dst_square)
    
    piece_type = piece_type.lower()
    
    if piece_type == 'p':  # Pawn
        # Pawns capture diagonally
        direction = 1 if board[src_square].isupper() else -1  # White moves up, black moves down
        # Left capture
        add_capture_move(file - 1, rank + direction)
        # Right capture
        add_capture_move(file + 1, rank + direction)
        
    elif piece_type == 'n':  # Knight
        # Knight moves in L-shape
        knight_moves = [
            (file + 2, rank + 1), (file + 2, rank - 1),
            (file - 2, rank + 1), (file - 2, rank - 1),
            (file + 1, rank + 2), (file + 1, rank - 2),
            (file - 1, rank + 2), (file - 1, rank - 2)
        ]
        for new_file, new_rank in knight_moves:
            add_capture_move(new_file, new_rank)
            
    elif piece_type == 'b':  # Bishop
        # Bishop moves diagonally
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dx, dy in directions:
            new_file, new_rank = file + dx, rank + dy
            while 0 <= new_file < 8 and 0 <= new_rank < 8:
                dst_square = f"{chr(ord('A') + new_file)}{new_rank + 1}"
                if board[dst_square] != '.':
                    if is_opponent_piece(dst_square):
                        moves.append(dst_square)
                    break
                new_file += dx
                new_rank += dy
                
    elif piece_type == 'r':  # Rook
        # Rook moves horizontally and vertically
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dx, dy in directions:
            new_file, new_rank = file + dx, rank + dy
            while 0 <= new_file < 8 and 0 <= new_rank < 8:
                dst_square = f"{chr(ord('A') + new_file)}{new_rank + 1}"
                if board[dst_square] != '.':
                    if is_opponent_piece(dst_square):
                        moves.append(dst_square)
                    break
                new_file += dx
                new_rank += dy
                
    elif piece_type == 'q':  # Queen
        # Queen moves like both bishop and rook
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dx, dy in directions:
            new_file, new_rank = file + dx, rank + dy
            while 0 <= new_file < 8 and 0 <= new_rank < 8:
                dst_square = f"{chr(ord('A') + new_file)}{new_rank + 1}"
                if board[dst_square] != '.':
                    if is_opponent_piece(dst_square):
                        moves.append(dst_square)
                    break
                new_file += dx
                new_rank += dy
                
    elif piece_type == 'k':  # King
        # King moves one square in any direction
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dx, dy in directions:
            add_capture_move(file + dx, rank + dy)
    
    return moves 

def test_piece_detection(frame, square_db, transformation_matrix):
    """Test mode for detecting piece type and color on a specific square using full-frame detection."""
    print("\nPiece Detection Test Mode")
    print("Enter square coordinates (e.g., A1, C3) or 'q' to quit")
    print("Note: This will show the square region and full-frame YOLO detection results")
    
    while True:
        square = input("Enter square: ").strip().upper()
        if square == 'Q':
            break
            
        if len(square) != 2 or not square[0].isalpha() or not square[1].isdigit():
            print("Invalid square format. Use format like A1, C3, etc.")
            continue
            
        if not (ord('A') <= ord(square[0]) <= ord('H') and 1 <= int(square[1]) <= 8):
            print("Square out of bounds. Use A1-H8.")
            continue
            
        # Get the square's coordinates from the square database
        if square not in square_db:
            print(f"Square {square} not found in square database")
            continue
            
        # Get the square's bounding box
        x_min, y_min, x_max, y_max = square_db[square]["bbox"]
        print(f"\n[DEBUG] Processing square {square}")
        print(f"[DEBUG] Warped coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
        
        # Transform the coordinates back to camera view
        inv_matrix = np.linalg.inv(transformation_matrix)
        camera_coords = []
        for x, y in [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]:
            pt = np.array([[[x, y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, inv_matrix)
            camera_coords.append(transformed[0][0])
        
        # Get the bounding box in camera coordinates
        camera_coords = np.array(camera_coords, dtype=np.int32)
        x_min_cam = min(camera_coords[:, 0])
        x_max_cam = max(camera_coords[:, 0])
        y_min_cam = min(camera_coords[:, 1])
        y_max_cam = max(camera_coords[:, 1])
        
        print(f"[DEBUG] Camera coordinates: x_min={x_min_cam}, y_min={y_min_cam}, x_max={x_max_cam}, y_max={y_max_cam}")
        
        # Create a debug frame for visualization
        debug_frame = frame.copy()
        
        # Draw the transformed square on debug frame
        cv2.rectangle(debug_frame, (x_min_cam, y_min_cam), (x_max_cam, y_max_cam), (255, 0, 0), 2)
        cv2.putText(debug_frame, square, (x_min_cam, y_min_cam - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Run YOLO detection on the full frame
        results = model(frame)
        print(f"[DEBUG] YOLO detection complete, processing results...")
        
        # Process detections
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf.item())
                    if conf > DETECTION_CONFIDENCE_THRESHOLD:
                        coords = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = coords
                        bottom_center = ((x1 + x2) / 2, y2)
                        cls = int(box.cls.item())
                        detections.append((bottom_center, cls, conf))
                        # Draw all detections on debug frame
                        cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.circle(debug_frame, (int(bottom_center[0]), int(bottom_center[1])), 5, (0, 0, 255), -1)
        
        print(f"[DEBUG] Found {len(detections)} detections above confidence threshold")
        
        # Find detections within this square
        square_detections = []
        for detection in detections:
            bottom_center, cls, conf = detection
            if (x_min_cam <= bottom_center[0] <= x_max_cam and 
                y_min_cam <= bottom_center[1] <= y_max_cam):
                square_detections.append((cls, conf))
                print(f"[DEBUG] Detection in square {square}: class={cls}, confidence={conf:.2f}")
                
                # Draw the piece type on the debug frame
                piece_mapping = {
                    # Black pieces
                    0: "b", 1: "k", 2: "n", 3: "p", 4: "q", 5: "r",
                    # White pieces
                    6: "B", 7: "K", 8: "N", 9: "P", 10: "Q", 11: "R"
                }
                piece_type = piece_mapping[cls]
                piece_color = "white" if piece_type.isupper() else "black"
                cv2.putText(debug_frame, f"{piece_type} ({piece_color})", 
                           (int(bottom_center[0]) + 10, int(bottom_center[1])),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Show the debug visualization
        cv2.imshow("Piece Detection Debug", debug_frame)
        
        if square_detections:
            best_piece = max(square_detections, key=lambda x: x[1])
            cls, conf = best_piece
            piece_mapping = {
                # Black pieces
                0: "b", 1: "k", 2: "n", 3: "p", 4: "q", 5: "r",
                # White pieces
                6: "B", 7: "K", 8: "N", 9: "P", 10: "Q", 11: "R"
            }
            piece_type = piece_mapping[cls]
            piece_color = "white" if piece_type.isupper() else "black"
            print(f"\nBest detection: {piece_type} ({piece_color}) with confidence {conf:.2f}")
        else:
            print("\nNo detections above confidence threshold in this square")
            
        cv2.waitKey(0)  # Wait for key press before continuing 