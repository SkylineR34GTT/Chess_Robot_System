import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ApiTCP_Python_NiryoOne'))

import time
import math
import threading
import keyboard  # pip install keyboard
import mouse     # pip install mouse
import sqlite3
from niryo_one_tcp_client import *
import robot_move

# --- Configuration parameters ---
robot_ip_address = "192.168.1.104"  # Change to your robot's IP
gripper_used = RobotTool.GRIPPER_1
gripper_speed = 400
arm_speed = 20
# Initial movement increment (in meters)
step = 0.02
sleep_joints = [0.0, 0.55, -1.2, 0.0, 0.0, 0.0]

# Set to True if using previously stored calibration coordinates (non-calibration mode).
# In this mode, the system will always compute the grid from the stored corner coordinates
# and allow square commands. Manual key-based movement is disabled.
use_existing_coordinates = True



# Database filenames
CALIB_DB = "robot_positions.db"   # Contains the 4 corner positions
GRID_DB = "chessboard_grid.db"      # Will contain all 64 chessboard square centers

# The known square side length in meters (27.5 mm = 0.0275 m)
SQUARE_SIZE = 0.0275

# Starting pose for manual control
current_pose = PoseObject(
    x=0.3,
    y=0.0,
    z=0.12,
    roll=0.0,
    pitch=1.57,
    yaw=-1.57
)

# --- Calibration Database Functions ---
def create_calib_table():
    with sqlite3.connect(CALIB_DB) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS chessboard_positions (
                square TEXT PRIMARY KEY,
                x REAL,
                y REAL,
                z REAL,
                roll REAL,
                pitch REAL,
                yaw REAL
            )
        """)
        conn.commit()

def store_calibration_position(square, pose):
    with sqlite3.connect(CALIB_DB) as conn:
        c = conn.cursor()
        c.execute("""
            REPLACE INTO chessboard_positions (square, x, y, z, roll, pitch, yaw)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (square, pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw))
        conn.commit()

def load_calibration_positions():
    positions = {}
    with sqlite3.connect(CALIB_DB) as conn:
        c = conn.cursor()
        c.execute("SELECT square, x, y, z, roll, pitch, yaw FROM chessboard_positions")
        rows = c.fetchall()
        for row in rows:
            square, x, y, z, roll, pitch, yaw = row
            positions[square] = PoseObject(x, y, z, roll, pitch, yaw)
    return positions

# --- Mouse Speed Adjustment ---
def mouse_event_callback(event):
    global step
    if isinstance(event, mouse.WheelEvent):
        if event.delta > 0:
            step += 0.001
        elif event.delta < 0:
            step = max(0.001, step - 0.001)
        print(f"Movement step size updated: {step:.3f}")

# --- Instruction Printout ---
def print_instructions():
    print("\nRobot Control Menu:")
    print("=========================================")
    if not use_existing_coordinates:
        print("Movement: w (X+), s (X-), d (Y+), a (Y-), e (Z+), q (Z-)")
        print("Gripper: o (open), c (close)")
        print("Calibration: p to store current position for next chessboard corner")
        print("Compute Grid: g to compute & store all square centers")
    else:
        print("Square Command: Type a square notation (e.g., E4) to move there")
        print("Grid Update: Grid positions are recalculated on startup from the corner coordinates")
        print("Swap Board: Type 'swap' to swap the board orientation (A1 becomes H8, etc.)")
    print("Manual Update: Type a command starting with 'm' to update a corner coordinate manually")
    print("Print Corner: Type 'printcorner' or 'printcorner <corner>' to display corner coordinates")
    print("Print Grid: Type 'printgrid' or 'printgrid <square>' to display grid coordinates")
    print("Exit: x")
    print("Use mouse scroll wheel to adjust movement step size.")
    print("=========================================\n")

# --- Revised Chessboard Grid Calculation using Bilinear Interpolation ---
def compute_chessboard_grid(square_size=SQUARE_SIZE):
    # Load calibration positions for the 4 corners.
    calib_positions = {}
    with sqlite3.connect(CALIB_DB) as conn:
        c = conn.cursor()
        c.execute("SELECT square, x, y FROM chessboard_positions WHERE square IN ('A1','A8','H8','H1')")
        rows = c.fetchall()
        for row in rows:
            square, x, y = row
            calib_positions[square] = (x, y)
    required = ['A1', 'A8', 'H8', 'H1']
    for r in required:
        if r not in calib_positions:
            print(f"Calibration point {r} not found. Cannot compute grid.")
            return

    # Assign corner positions.
    pA1 = calib_positions['A1']  # Bottom-left (A1)
    pA8 = calib_positions['A8']  # Top-left (A8)
    pH8 = calib_positions['H8']  # Top-right (H8)
    pH1 = calib_positions['H1']  # Bottom-right (H1)

    # Expected total x span along left and right edges:
    expected_dx = 7 * square_size

    grid_positions = {}

    # Loop through each row (j) and column (i)
    for j in range(8):  # Rows 1 to 8 (j=0 to 7)
        v = j / 7.0

        # For the left edge: raw x interpolated from A1 to A8.
        raw_left_x = (1 - v) * pA1[0] + v * pA8[0]
        # Compute scaling factor so that the left edge spans exactly expected_dx.
        if (pA8[0] - pA1[0]) != 0:
            scale_left = expected_dx / (pA8[0] - pA1[0])
        else:
            scale_left = 1.0
        adjusted_left_x = pA1[0] + (raw_left_x - pA1[0]) * scale_left

        # Similarly, for the right edge: raw x interpolated from H1 to H8.
        raw_right_x = (1 - v) * pH1[0] + v * pH8[0]
        if (pH8[0] - pH1[0]) != 0:
            scale_right = expected_dx / (pH8[0] - pH1[0])
        else:
            scale_right = 1.0
        adjusted_right_x = pH1[0] + (raw_right_x - pH1[0]) * scale_right

        # For the y coordinate, we use standard bilinear interpolation along each edge.
        left_y = (1 - v) * pA1[1] + v * pA8[1]
        right_y = (1 - v) * pH1[1] + v * pH8[1]

        for i in range(8):  # Columns A to H (i=0 to 7)
            u = i / 7.0
            # Interpolate x between the adjusted left and right boundaries.
            x = (1 - u) * adjusted_left_x + u * adjusted_right_x
            # Standard interpolation for y.
            y = (1 - u) * left_y + u * right_y
            square_name = f"{chr(ord('A') + i)}{j + 1}"
            grid_positions[square_name] = (x, y)

    # Store computed grid in GRID_DB.
    with sqlite3.connect(GRID_DB) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS squares (
                square TEXT PRIMARY KEY,
                col INTEGER,
                row INTEGER,
                x REAL,
                y REAL
            )
        """)
        for square, (x, y) in grid_positions.items():
            col = ord(square[0]) - ord('A')
            row = int(square[1:]) - 1
            c.execute("""
                REPLACE INTO squares (square, col, row, x, y)
                VALUES (?, ?, ?, ?, ?)
            """, (square, col, row, x, y))
        conn.commit()
    print("Chessboard grid positions computed and stored in", GRID_DB)

# --- Swap Board Orientation Function ---
def swap_board_orientation():
    """
    Swaps the board orientation in the grid database.
    This makes A1 become H8, H1 become A8, etc.
    Useful for switching between playing as white and black.
    """
    print("Swapping board orientation in the grid database...")
    
    # Create a mapping for the square swaps
    swap_map = {}
    for i in range(8):
        for j in range(8):
            original_square = f"{chr(ord('A') + i)}{j + 1}"
            swapped_square = f"{chr(ord('H') - i)}{8 - j}"
            swap_map[original_square] = swapped_square
    
    # Connect to the database
    with sqlite3.connect(GRID_DB) as conn:
        c = conn.cursor()
        
        # Create a temporary table to store the swapped data
        c.execute("""
            CREATE TABLE IF NOT EXISTS squares_temp (
                square TEXT PRIMARY KEY,
                col INTEGER,
                row INTEGER,
                x REAL,
                y REAL
            )
        """)
        
        # Get all squares from the original table
        c.execute("SELECT square, col, row, x, y FROM squares")
        rows = c.fetchall()
        
        # Insert swapped data into the temporary table
        for row in rows:
            original_square, col, row, x, y = row
            swapped_square = swap_map[original_square]
            swapped_col = ord(swapped_square[0]) - ord('A')
            swapped_row = int(swapped_square[1:]) - 1
            
            c.execute("""
                REPLACE INTO squares_temp (square, col, row, x, y)
                VALUES (?, ?, ?, ?, ?)
            """, (swapped_square, swapped_col, swapped_row, x, y))
        
        # Drop the original table and rename the temporary table
        c.execute("DROP TABLE squares")
        c.execute("ALTER TABLE squares_temp RENAME TO squares")
        
        conn.commit()
    
    print("Board orientation has been swapped successfully.")
    print("A1 is now H8, H1 is now A8, etc.")
    print("You can now play from either side of the board.")

# --- Manual Update Corner Function ---
def manual_update_corner():
    corner = input("Enter corner square to update (A1, A8, H8, H1): ").strip().upper()
    if corner not in ["A1", "A8", "H8", "H1"]:
        print("Invalid corner. Must be one of A1, A8, H8, H1.")
        return
    coords_str = input("Enter new x,y coordinates as comma-separated values: ").strip()
    try:
        parts = coords_str.split(',')
        if len(parts) != 2:
            print("Invalid number of values. Please enter only x and y.")
            return
        new_x, new_y = [float(x) for x in parts]
        # Retrieve the current stored calibration if it exists, otherwise use current_pose as fallback.
        calib_positions = load_calibration_positions()
        if corner in calib_positions:
            old_pose = calib_positions[corner]
        else:
            old_pose = current_pose
        # Create a new pose with the updated x, y while keeping z, roll, pitch, yaw the same.
        new_pose = PoseObject(new_x, new_y, old_pose.z, old_pose.roll, old_pose.pitch, old_pose.yaw)
        store_calibration_position(corner, new_pose)
        print(f"Updated {corner} with new coordinates: x={new_pose.x:.3f}, y={new_pose.y:.3f} (z, roll, pitch, yaw unchanged)")
    except Exception as e:
        print("Error parsing coordinates:", e)


# --- Command Input Thread for Square and Print Commands ---
def command_input(client):
    global current_pose
    while True:
        cmd = input("Enter command (e.g., E4, m to update corner, printcorner, printgrid, swap, or 'exit'): ").strip()
        if cmd.lower() == 'exit':
            break
        if not cmd:
            continue

        # Manual update command: starts with "m"
        if cmd.lower().startswith("m"):
            manual_update_corner()
            compute_chessboard_grid()  # Recompute grid after update.
            continue

        # Print corner coordinates command.
        if cmd.lower().startswith("printcorner"):
            parts = cmd.split()
            corners = load_calibration_positions()
            if len(parts) == 1:
                # Print all corners.
                for corner, pose in corners.items():
                    print(f"{corner}: x={pose.x:.3f}, y={pose.y:.3f}, z={pose.z:.3f}, roll={pose.roll:.3f}, pitch={pose.pitch:.3f}, yaw={pose.yaw:.3f}")
            elif len(parts) == 2:
                corner = parts[1].upper()
                if corner in corners:
                    pose = corners[corner]
                    print(f"{corner}: x={pose.x:.3f}, y={pose.y:.3f}, z={pose.z:.3f}, roll={pose.roll:.3f}, pitch={pose.pitch:.3f}, yaw={pose.yaw:.3f}")
                else:
                    print(f"Corner {corner} not found.")
            continue

        # Print grid coordinates command.
        if cmd.lower().startswith("printgrid"):
            parts = cmd.split()
            with sqlite3.connect(GRID_DB) as conn:
                c = conn.cursor()
                if len(parts) == 1:
                    c.execute("SELECT square, x, y FROM squares ORDER BY square")
                    rows = c.fetchall()
                    for row in rows:
                        print(f"{row[0]}: x={row[1]:.3f}, y={row[2]:.3f}")
                elif len(parts) == 2:
                    square = parts[1].upper()
                    c.execute("SELECT square, x, y FROM squares WHERE square = ?", (square,))
                    row = c.fetchone()
                    if row:
                        print(f"{row[0]}: x={row[1]:.3f}, y={row[2]:.3f}")
                    else:
                        print(f"Square {square} not found.")
            continue

        # Swap board orientation command
        if cmd.lower() == 'swap':
            swap_board_orientation()
            continue

        # Otherwise, treat the command as a square command.
        if len(cmd) >= 2 and cmd[0].upper() in "ABCDEFGH" and cmd[1] in "12345678":
            square = cmd[0].upper() + cmd[1]
            with sqlite3.connect(GRID_DB) as conn:
                c = conn.cursor()
                c.execute("SELECT x, y FROM squares WHERE square = ?", (square,))
                result = c.fetchone()
                if result:
                    target_x, target_y = result
                    print(f"Moving to {square}: x={target_x:.3f}, y={target_y:.3f}")
                    client.move_pose(target_x, target_y, current_pose.z,
                                     current_pose.roll, current_pose.pitch, current_pose.yaw)
                    current_pose.x = target_x
                    current_pose.y = target_y
                else:
                    print(f"Square {square} not found in grid database.")
        else:
            print("Invalid command. Please enter a valid square notation, 'm', 'printcorner', 'printgrid', or 'swap'.")

# --- Main Program ---
def main():
    global step, current_pose
    create_calib_table()
    mouse.hook(mouse_event_callback)

    client = NiryoOneClient()
    client.connect(robot_ip_address)
    client.change_tool(gripper_used)
    client.calibrate(CalibrateMode.AUTO)
    client.move_pose(current_pose.x, current_pose.y, current_pose.z,
                     current_pose.roll, current_pose.pitch, current_pose.yaw)

    print_instructions()
    client.set_arm_max_velocity(arm_speed)
    # In non-calibration mode, always compute/update grid positions from the corner coordinates.
    if use_existing_coordinates:
        compute_chessboard_grid()

        #robot_moves.move_piece(client, "E2", "E4")
        #robot_moves.take_piece(client, "E2", "E4")
        white = False
        #robot_move.king_castle(client, white)


        thread = threading.Thread(target=command_input, args=(client,), daemon=True)
        thread.start()

    # Calibration mode: allow manual key-based movement and corner calibration.
    calibration_mode = not use_existing_coordinates
    squares_order = ["A1", "A8", "H8", "H1"]
    calib_count = 0

    try:
        while True:
            if keyboard.is_pressed('x'):
                print("Returning to sleep joints position...")
                client.move_joints(*sleep_joints)
                time.sleep(2)  # Allow time for movement to complete.
                print("Exiting program...")
                break

            if calibration_mode:
                # Allow storing of the 4 corner positions.
                if keyboard.is_pressed('p'):
                    if calib_count < len(squares_order):
                        square = squares_order[calib_count]
                        store_calibration_position(square, current_pose)
                        print(f"Stored calibration for {square}: x={current_pose.x:.3f}, y={current_pose.y:.3f}")
                        calib_count += 1
                        time.sleep(0.3)
                # Allow manual movement in calibration mode.
                if keyboard.is_pressed('w'):
                    current_pose.x += step
                    client.move_pose(current_pose.x, current_pose.y, current_pose.z,
                                     current_pose.roll, current_pose.pitch, current_pose.yaw)
                    print(f"New Pose: x={current_pose.x:.3f}, y={current_pose.y:.3f}")
                    time.sleep(0.2)
                elif keyboard.is_pressed('s'):
                    current_pose.x -= step
                    client.move_pose(current_pose.x, current_pose.y, current_pose.z,
                                     current_pose.roll, current_pose.pitch, current_pose.yaw)
                    print(f"New Pose: x={current_pose.x:.3f}, y={current_pose.y:.3f}")
                    time.sleep(0.2)
                elif keyboard.is_pressed('d'):
                    current_pose.y += step
                    client.move_pose(current_pose.x, current_pose.y, current_pose.z,
                                     current_pose.roll, current_pose.pitch, current_pose.yaw)
                    print(f"New Pose: x={current_pose.x:.3f}, y={current_pose.y:.3f}")
                    time.sleep(0.2)
                elif keyboard.is_pressed('a'):
                    current_pose.y -= step
                    client.move_pose(current_pose.x, current_pose.y, current_pose.z,
                                     current_pose.roll, current_pose.pitch, current_pose.yaw)
                    print(f"New Pose: x={current_pose.x:.3f}, y={current_pose.y:.3f}")
                    time.sleep(0.2)
                elif keyboard.is_pressed('e'):
                    current_pose.z += step
                    client.move_pose(current_pose.x, current_pose.y, current_pose.z,
                                     current_pose.roll, current_pose.pitch, current_pose.yaw)
                    print(f"New Pose: z={current_pose.z:.3f}")
                    time.sleep(0.2)
                elif keyboard.is_pressed('q'):
                    current_pose.z -= step
                    client.move_pose(current_pose.x, current_pose.y, current_pose.z,
                                     current_pose.roll, current_pose.pitch, current_pose.yaw)
                    print(f"New Pose: z={current_pose.z:.3f}")
                    time.sleep(0.2)
                elif keyboard.is_pressed('o'):
                    client.open_gripper(gripper_used, gripper_speed)
                    print("Gripper opened.")
                    time.sleep(0.2)
                elif keyboard.is_pressed('c'):
                    client.close_gripper(gripper_used, gripper_speed)
                    print("Gripper closed.")
                    time.sleep(0.2)
                elif keyboard.is_pressed('g'):
                    compute_chessboard_grid()
                    time.sleep(0.3)
            else:
                # In non-calibration mode, manual key movement is disabled.
                pass

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        client.set_learning_mode(True)
        client.quit()
        print("Robot connection closed.")

if __name__ == '__main__':
    main()
