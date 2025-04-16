import sys
import os
# Add the ApiTCP_Python_NiryoOne directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ApiTCP_Python_NiryoOne'))

import time
import robot_move

def execute_robot_move(client, move_str, white, is_capture=False):
    """
    Simulate and execute the robot arm move based on the move notation.
    move_str: string in UCI format, e.g., "e2e4", "e1g1", "e1c1", etc.
    white: boolean, True if the moving piece is white.
    is_capture: boolean, True if the move is a capture move.
    """
    move_str = move_str.upper()
    print(f"[DEBUG] Executing move: {move_str} (White: {white}, Capture: {is_capture})")
    # Check for castling moves.
    if white and move_str in ["E1G1", "E1C1"]:
        if move_str == "E1G1":
            print("[DEBUG] Detected kingside castling for white.")
            robot_move.king_castle(client, True)
        else:
            print("[DEBUG] Detected queenside castling for white.")
            robot_move.queen_castle(client, True)
    elif not white and move_str in ["E8G8", "E8C8"]:
        if move_str == "E8G8":
            print("[DEBUG] Detected kingside castling for black.")
            robot_move.king_castle(client, False)
        else:
            print("[DEBUG] Detected queenside castling for black.")
            robot_move.queen_castle(client, False)
    else:
        start_square = move_str[0:2]
        end_square = move_str[2:4]
        if is_capture:
            print(f"[DEBUG] Detected capture move from {start_square} to {end_square}.")
            robot_move.take_piece(client, start_square, end_square)
        else:
            print(f"[DEBUG] Detected simple move from {start_square} to {end_square}.")
            robot_move.move_piece(client, start_square, end_square)
            robot_move.sleep(client)

# When running this module directly, you can test the dummy integration:
if __name__ == "__main__":
    # Dummy client for testing
    class DummyClient:
        def move_pose(self, x, y, z, roll, pitch, yaw):
            print(f"[SIMULATION] move_pose(x={x}, y={y}, z={z})")
        def open_gripper(self, tool, speed):
            print(f"[SIMULATION] open_gripper(tool={tool}, speed={speed})")
        def close_gripper(self, tool, speed):
            print(f"[SIMULATION] close_gripper(tool={tool}, speed={speed})")
        def move_joints(self, *joints):
            print(f"[SIMULATION] move_joints({joints})")
        def calibrate(self, mode):
            print(f"[SIMULATION] calibrate(mode={mode})")
        def change_tool(self, tool):
            print(f"[SIMULATION] change_tool(tool={tool})")
        def set_arm_max_velocity(self, speed):
            print(f"[SIMULATION] set_arm_max_velocity(speed={speed})")
        def set_learning_mode(self, mode):
            print(f"[SIMULATION] set_learning_mode(mode={mode})")
        def quit(self):
            print("DummyClient: quit()")
        def connect(self, ip):
            print(f"DummyClient: connect(ip={ip})")
    dummy = DummyClient()
    execute_robot_move(dummy, "e2e4", True, False)
