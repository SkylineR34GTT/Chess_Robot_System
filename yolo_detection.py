import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("C:/Users/blueb/Downloads/yoloChess3.pt", verbose=False)
DETECTION_CONFIDENCE_THRESHOLD = 0.3

# Class mapping for chess pieces
piece_mapping = {
    # Black pieces
    0: "b", 1: "k", 2: "n", 3: "p", 4: "q", 5: "r",
    # White pieces
    6: "B", 7: "K", 8: "N", 9: "P", 10: "Q", 11: "R"
}

def apply_gamma_correction(image, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def draw_detections(frame, results):
    """Draw bounding boxes and labels on the frame."""
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                conf = float(box.conf.item())
                if conf > DETECTION_CONFIDENCE_THRESHOLD:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get class and confidence
                    cls = int(box.cls.item())
                    piece = piece_mapping[cls]
                    
                    # Draw bounding box
                    color = (0, 255, 0) if piece.isupper() else (0, 0, 255)  # Green for white, Red for black
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{piece} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Press 'q' to quit")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Run YOLO detection
        results = model(frame)
        
        # Draw detections
        draw_detections(frame, results)
        
        # Show frame
        cv2.imshow("YOLO Chess Detection", frame)
        
        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 