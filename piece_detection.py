# piece_detection.py
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model for chess pieces
model = YOLO("C:/Users/blueb/Downloads/yoloChess3.pt", verbose=False)
DETECTION_CONFIDENCE_THRESHOLD = 0.3

def detect_chess_pieces(frame):
    """Detect chess pieces in a frame using YOLO."""
    results = model(frame)
    detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                conf = float(box.conf.item())
                if conf > DETECTION_CONFIDENCE_THRESHOLD:
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = coords
                    bottom_center = ((x1 + x2) / 2, y2)
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
                    bottom_center = ((x1 + x2) / 2, y2)
                    found_nearby = False
                    for (dx, dy) in detections:
                        if abs(dx - bottom_center[0]) < 10 and abs(dy - bottom_center[1]) < 10:
                            found_nearby = True
                            break
                    if not found_nearby:
                        detections.append(bottom_center)
    return detections

def apply_gamma_correction(image, gamma=1.2):
    """Apply gamma correction to an image."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def detect_piece_type_and_color(frame, square_coords, transformation_matrix, square_db):
    """Detect the type and color of a piece on a specific square."""
    if square_coords not in square_db:
        return "."
        
    # Get the square's center coordinates
    square_center = square_db[square_coords]["center"]
    
    # Transform the center point back to camera coordinates
    inv_matrix = np.linalg.inv(transformation_matrix)
    camera_point = cv2.perspectiveTransform(
        np.array([[[square_center[0], square_center[1]]]], dtype=np.float32),
        inv_matrix
    )[0][0]
    
    # Create a small ROI around the piece
    x, y = int(camera_point[0]), int(camera_point[1])
    roi_size = 50  # Size of the region to analyze
    x1 = max(0, x - roi_size)
    y1 = max(0, y - roi_size)
    x2 = min(frame.shape[1], x + roi_size)
    y2 = min(frame.shape[0], y + roi_size)
    
    if x1 >= x2 or y1 >= y2:
        return "."
        
    roi = frame[y1:y2, x1:x2]
    
    # Run detection on the ROI
    results = model(roi)
    
    # Process detections
    piece_detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                conf = float(box.conf.item())
                if conf > DETECTION_CONFIDENCE_THRESHOLD:
                    cls = int(box.cls.item())
                    # Class order: black pieces first (bishop, king, knight, pawn, queen, rook)
                    # then white pieces in same order
                    piece_mapping = {
                        # Black pieces
                        0: "b", 1: "k", 2: "n", 3: "p", 4: "q", 5: "r",
                        # White pieces
                        6: "B", 7: "K", 8: "N", 9: "P", 10: "Q", 11: "R"
                    }
                    piece_detections.append((conf, piece_mapping[cls]))
    
    if not piece_detections:
        return "."
        
    # Return the piece with highest confidence
    return max(piece_detections, key=lambda x: x[0])[1] 