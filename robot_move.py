import sys
import os
# Add the ApiTCP_Python_NiryoOne directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ApiTCP_Python_NiryoOne'))
from niryo_one_tcp_client import *
import sqlite3

GRID_DB = "chessboard_grid.db"

# -- MUST Change these variables
robot_ip_address = "192.168.1.104"  # IP address of the Niryo One
gripper_used = RobotTool.GRIPPER_1
arm_speed = 35
gripper_speed = 400

sleep_joints = [0.0, 0.55, -1.2, 0.0, 0.0, 0.0]
dead_pieces = PoseObject(0.110, -0.255, 0.405, -0., 1.57, -1.57)

basic_height = 0.124
height_offset = 0.09  # Offset according to Z-Axis to go over pick & place poses
place_height = 0.01

pick_pose = PoseObject(
    x=0.22, y=0, z=basic_height,
    roll=-0., pitch=1.57, yaw=-1.57)

place_pose = PoseObject(
    x=0.42, y=0, z=basic_height,
    roll=-0., pitch=1.57, yaw=-1.57)
# a board square is ruffly 27mm or 0.027m

# Global flag to control simulation mode for robot moves.
SIMULATE_ROBOT_MOVES = False


def sleep(client):
    client.move_joints(*sleep_joints)
    return



def get_square_coordinates(square):
    with sqlite3.connect(GRID_DB) as conn:
        c = conn.cursor()
        c.execute("SELECT x, y FROM squares WHERE square = ?", (square,))
        result = c.fetchone()
        if result:
            return result  # returns a tuple (x, y)
        else:
            print(f"Square {square} not found in the grid.")
            return None


def pick_piece(client, x, y):
    client.open_gripper(gripper_used, gripper_speed)
    client.move_pose(x, y, pick_pose.z,
                     pick_pose.roll, pick_pose.pitch, pick_pose.yaw)  # lowering
    client.close_gripper(gripper_used, gripper_speed)  # picking
    client.move_pose(x, y, pick_pose.z + height_offset,
                     pick_pose.roll, pick_pose.pitch, pick_pose.yaw)  # raising
    return


def place_piece(client, x, y):
    client.move_pose(x, y, pick_pose.z + place_height,
                     pick_pose.roll, pick_pose.pitch, pick_pose.yaw)  # lowering
    client.open_gripper(gripper_used, gripper_speed)  # placing
    client.move_pose(x, y, place_pose.z + height_offset,
                     place_pose.roll, place_pose.pitch, place_pose.yaw)  # raising
    return


def move_piece(client, square1, square2):
    if SIMULATE_ROBOT_MOVES:
        print(f"[SIMULATION] move_piece from {square1} to {square2}")
        return
    start_coords = get_square_coordinates(square1)
    end_coords = get_square_coordinates(square2)
    if start_coords is None or end_coords is None:
        return
    start_x, start_y = start_coords
    end_x, end_y = end_coords
    client.move_pose(start_x, start_y, pick_pose.z + height_offset,
                     pick_pose.roll, pick_pose.pitch, pick_pose.yaw)
    pick_piece(client, start_x, start_y)
    client.move_pose(end_x, end_y, pick_pose.z + height_offset,
                     pick_pose.roll, pick_pose.pitch, pick_pose.yaw)
    place_piece(client, end_x, end_y)
    return


def take_piece(client, square1, square2):
    if SIMULATE_ROBOT_MOVES:
        print(f"[SIMULATION] take_piece (capture) from {square1} to {square2}")
        return
    start_coords = get_square_coordinates(square1)
    end_coords = get_square_coordinates(square2)
    if start_coords is None or end_coords is None:
        return
    end_x, end_y = end_coords
    client.move_pose(end_x, end_y, pick_pose.z + height_offset,
                     pick_pose.roll, pick_pose.pitch, pick_pose.yaw)
    pick_piece(client, end_x, end_y)
    client.move_pose(dead_pieces.x, dead_pieces.y, dead_pieces.z,
                     dead_pieces.roll, dead_pieces.pitch, dead_pieces.yaw)
    client.open_gripper(gripper_used, gripper_speed)
    move_piece(client, square1, square2)
    client.move_joints(*sleep_joints)
    return


def king_castle(client, white):
    if SIMULATE_ROBOT_MOVES:
        side = "white" if white else "black"
        print(f"[SIMULATION] king_castle for {side}")
        return
    if white == True:
        k_square1 = "E1"
        k_square2 = "G1"
        r_square1 = "H1"
        r_square2 = "F1"
    else:
        k_square1 = "E8"
        k_square2 = "G8"
        r_square1 = "H8"
        r_square2 = "F8"

    move_piece(client, k_square1, k_square2)
    move_piece(client, r_square1, r_square2)
    client.move_joints(*sleep_joints)
    return


def queen_castle(client, white):
    if SIMULATE_ROBOT_MOVES:
        side = "white" if white else "black"
        print(f"[SIMULATION] queen_castle for {side}")
        return
    if white == True:
        k_square1 = "E1"
        k_square2 = "C1"
        r_square1 = "A1"
        r_square2 = "D1"
    else:
        k_square1 = "E8"
        k_square2 = "C8"
        r_square1 = "A8"
        r_square2 = "D8"

    move_piece(client, k_square1, k_square2)
    move_piece(client, r_square1, r_square2)
    client.move_joints(*sleep_joints)
    return


# Function to simulate and execute the robot arm move based on the move notation.
def execute_robot_move(client, move_str, white, is_capture=False):
    """
    Simulate and execute the robot arm move based on the move notation.
    move_str: string in UCI format, e.g., "e2e4", "e1g1", "e1c1", etc.
    white: boolean, True if white is moving.
    is_capture: boolean, True if the move is a capture move.

    Determines if the move is castling, a capture move, or a simple move and calls the
    relevant robot arm function.
    """
    move_str = move_str.upper()
    print(f"[DEBUG] Executing move: {move_str} (White: {white}, Capture: {is_capture})")
    # Check for castling moves.
    if white and move_str in ["E1G1", "E1C1"]:
        if move_str == "E1G1":
            print("[DEBUG] Detected kingside castling for white.")
            king_castle(client, True)
        else:
            print("[DEBUG] Detected queenside castling for white.")
            queen_castle(client, True)
    elif not white and move_str in ["E8G8", "E8C8"]:
        if move_str == "E8G8":
            print("[DEBUG] Detected kingside castling for black.")
            king_castle(client, False)
        else:
            print("[DEBUG] Detected queenside castling for black.")
            queen_castle(client, False)
    else:
        start_square = move_str[0:2]
        end_square = move_str[2:4]
        if is_capture:
            print(f"[DEBUG] Detected capture move from {start_square} to {end_square}.")
            take_piece(client, start_square, end_square)
        else:
            print(f"[DEBUG] Detected simple move from {start_square} to {end_square}.")
            move_piece(client, start_square, end_square)


if __name__ == '__main__':
    # The SIMULATE_ROBOT_MOVES flag is controlled by the global variable.
    if SIMULATE_ROBOT_MOVES:
        print("Simulating robot moves (text output only):")


        # Define a dummy client that does nothing.
        class DummyClient:
            def move_pose(self, x, y, z, roll, pitch, yaw):
                pass

            def open_gripper(self, tool, speed):
                pass

            def close_gripper(self, tool, speed):
                pass

            def move_joints(self, *joints):
                pass

            def calibrate(self, mode):
                pass

            def change_tool(self, tool):
                pass

            def set_arm_max_velocity(self, speed):
                pass

            def set_learning_mode(self, mode):
                pass

            def quit(self):
                pass

            def connect(self, ip):
                pass


        dummy_client = DummyClient()
        # Simulate a simple move: white moves from e2 to e4.
        execute_robot_move(dummy_client, "e2e4", white=True, is_capture=False)
        # Simulate a capture move: white moves from d5 to e4 (capturing).
        execute_robot_move(dummy_client, "d5e4", white=True, is_capture=True)
        # Simulate kingside castling for white.
        execute_robot_move(dummy_client, "e1g1", white=True, is_capture=False)
        # Simulate queenside castling for black.
        execute_robot_move(dummy_client, "e8c8", white=False, is_capture=False)
    else:
        # Connect to robot
        client = NiryoOneClient()
        client.connect(robot_ip_address)
        client.set_arm_max_velocity(arm_speed)
        # Changing tool
        client.change_tool(gripper_used)
        # Calibrate robot if robot needs calibration
        client.calibrate(CalibrateMode.AUTO)



        # Ending
        client.move_joints(*sleep_joints)
        client.set_learning_mode(True)
        # Releasing connection
        client.quit()
