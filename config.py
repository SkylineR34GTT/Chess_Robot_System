# Configuration flags for the chess robot

# Board and grid position settings
Prev_positions = True
USE_PREVIOUS_BOARD_POSITIONS = Prev_positions
USE_PREVIOUS_GRIDLINES = Prev_positions

# Detection settings
DETECTION_CONFIDENCE_THRESHOLD = 0.1

# Corner lock settings
PERMANENT_CORNER_LOCK = True    #keep using saved positions

# Test mode settings
TEST_CLICK_MODE = False
TEST_STOCKFISH_MODE = False
TEST_PIECE_DETECTION = False

# Output and processing settings
output_size = (800, 800)
STABILITY_THRESHOLD = 0.2
STABLE_TIME_NEEDED = 1.0 #how long for corner lock to be considered stable
RECHECK_INTERVAL = 3.0
GRID_TOLERANCE = 0.02 

# Movement detection settings
MOVEMENT_SETTLE_TIME = 2.0  # Time to wait after movement is detected before processing

# Constants for grid processing
STABILITY_TIME_NEEDED = 1.0
GRID_TOLERANCE = 0.02
EDGE_THRESHOLD = 30  # increased to allow more valid lines