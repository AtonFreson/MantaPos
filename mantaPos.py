# https://medium.com/@nflorent7/a-comprehensive-guide-to-camera-calibration-using-charuco-boards-and-opencv-for-perspective-9a0fa71ada5f

import cv2
import numpy as np
import os
from math import atan2, sqrt
import mantaPosLib as manta  # Ensure this module is correctly implemented
import genMarker
from datetime import datetime
import socket
import json
import threading

# Initialize parameters
CAMERA_RTSP_ADDR = "rtsp://admin:@169.254.178.11:554/" # Overwrites CAMERA_INPUT if 4K selected
camera_calibration_file = 'camera_calibration_4K_6s.npz'

MPU_UNIT = 4  # MPU unit number for recording the camera position/rotation data

MARKER_TYPE = ["ChArUco", "Single"]  # Select the marker type to use
# Options are "ChArUco" or "ArUco", and "Single" or "Quad" respectively

# Set to True to visualise the frame distortion based on the camera calibration. High computational cost (~110ms).
visualise_calib_dist = True
new_camera_matrix = None

if MARKER_TYPE[0] == "ChArUco" and MARKER_TYPE[1] == "Single":
    # ChArUco board settings
    squares_vertically = 6
    squares_horizontally = squares_vertically
    square_pixels = int(140*7/squares_horizontally) # Pixel size of the chessboard squares
    grid_edge = 20 # Pixel margin outside the ChArUco grid
    marker_ratio = 0.75 # Marker ratio of square_length to fit within white squares; acceptable maximum 0.85, recommended 0.7. Rounds marker size to int.
    square_length = 0.2975/6 * square_pixels/200 # Real world length of square in meters
    #square_length = 1.110/7 * square_pixels/280 # Lyftkranen - large conference room screen

    print(f"Reference length of {squares_horizontally} squares:", square_length*squares_horizontally, "meters")
    
elif MARKER_TYPE[0] == "ArUco" and MARKER_TYPE[1] == "Single":
    # Define the real-world size of the ArUco markers (same as during calibration)
    screen_dpm = 200/0.0495 # Length of marker's side in pixels / meters

    marker_size = 200
    marker_border_size = 25
    num_markers = 0 # Set to 0 if all

    # Set the selected camera: axis, axis_low or gopro.
    CAMERA_TYPE = "4K"
    CAMERA_INPUT = 2 # OBS Virtual Camera
    CAMERA_RTSP_ADDR = "rtsp://admin:@169.254.178.12:554/" # Overwrites CAMERA_INPUT if 4K selected

    marker_length = marker_size/screen_dpm # in meters

    object_points = np.array([[0, 0, 0], 
                            [marker_length, 0, 0], 
                            [marker_length, marker_length, 0], 
                            [0, marker_length, 0]], dtype=np.float32)
    
    # The marker ID of the first marker in the order. OBS: Markers in folder need to be sequential!
    grid_rows, marker_base = manta.display_marker_grid(num_markers, marker_size, marker_border_size)
    grid_cols = grid_rows # Assume square
    grid_marker_spacing = marker_length * (marker_border_size*2 / marker_size + 1) # Distance between markers in the grid (meters)


# Setup UDP socket for sending camera position/rotation data
UDP_IP = "127.0.0.1"   # Localhost
UDP_PORT = 13235       # Port for sending data
UDP_IP_RECV = ""       # Receive data from any IP
UDP_PORT_RECV = 13233  # Port for receiving data
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Create a UDP listener for receiving depth data for displaying in the GUI
depth_main = None
depth_sec = None
stop_thread = False
data_lock = threading.Lock()

def depth_listener():
    global depth_main, depth_sec
    depth_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    depth_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # allow multiple binds
    depth_sock.bind((UDP_IP_RECV, UDP_PORT_RECV))  # Depth data listening port
    depth_sock.settimeout(0.5)  # Set timeout for recvfrom

    while not stop_thread:
        try:
            data, _ = depth_sock.recvfrom(4096)
            with data_lock:
                data_dict = json.loads(data.decode())
                unit_num = int(data_dict.get("mpu_unit"))
                if unit_num == 1:
                    enc = data_dict.get("encoder")
                    depth_main = float(enc["distance"])
                elif unit_num == 2:
                    enc = data_dict.get("encoder")
                    depth_sec = float(enc["distance"])
        except socket.timeout:
            continue
        except Exception as e:
            print(f"Error in UDP listener: {e}")
            break

# Start depth UDP listener thread
depth_thread = threading.Thread(target=depth_listener)
depth_thread.start()

# Generate and display the marker grid
board, dictionary = genMarker.create_and_save_ChArUco_board(square_length, square_pixels, grid_edge, marker_ratio, squares_vertically, squares_horizontally)
manta.display_marker_grid(board_type="ChArUco")

# Define the detector and parameters
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, params)

# Initialize camera
cap = manta.RealtimeCapture(CAMERA_RTSP_ADDR)
cv2.namedWindow("Camera Preview with Position", cv2.WINDOW_NORMAL)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Load previously saved camera calibration data
calibration_dir = './calibrations/camera_calibrations'
calibration_data = np.load(os.path.join(calibration_dir, camera_calibration_file))
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# Precompute board center offset to center the coordinate system
board_width = (squares_horizontally - 0) * square_length
board_height = (squares_vertically - 0) * square_length
board_center_offset = [-board_width / 2, -board_height / 2, 0]

board_pos = board_center_offset#[0,0,0]  # Position of the marker in meters
board_rot = [0,0,0]  # Euler rotation of the marker in degrees, origin is normal around z


# Main loop
while True:
    success = False

    # Capture camera frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera.")
        break
    
    #frame = manta.frame_corner_cutout(frame, 0.3)  # Cut out the corners of the frame 
    #frame = manta.frame_crop(frame, 0.7)  # Crop the frame to remove fisheye edges
    
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejected_img_points = detector.detectMarkers(gray_frame)

    if ids is not None and len(ids) > 0:
        # Refine detected markers for better accuracy
        detector.refineDetectedMarkers(
            image=gray_frame,
            board=board,
            detectedCorners=corners,
            detectedIds=ids,
            rejectedCorners=rejected_img_points
        )

        # Interpolate ChArUco corners
        num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray_frame,
            board=board
        )

        # Modify the displayed markers to make them unreadable. Use: blur, cross, fill, diamond
        try:
            frame = manta.censor_marker(frame, corners, "diamond")
            #frame = manta.censor_charuco_board(frame, charuco_corners, corners, 0.5)
        except:
            pass

        # Draw detected markers and ChArUco corners
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        if charuco_ids is not None and num_corners > 0:
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, cornerColor=(255, 0, 0))
            
            for i, corner in enumerate(charuco_corners):
                cv2.putText(frame, str(charuco_ids[i][0]), (int(corner[0][0]), int(corner[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        if charuco_ids is not None and num_corners >= 4:
            # Estimate pose of the ChArUco board
            success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids,
                board=board,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs,
                rvec = np.zeros((3, 1), dtype=np.float64),
                tvec = np.zeros((3, 1), dtype=np.float64)
            )

            if success:
                # Collect object points and image points
                object_points_all = board.getChessboardCorners()[charuco_ids.flatten()]
                image_points_all = charuco_corners.reshape(-1, 2)

                # Store translation and rotation vectors
                tvec_list = []
                rvec_list = []
                markers_pos_rot = []
                # Store the position and rotation of the board
                tvec_list.append(tvec.flatten())
                rvec_list.append(rvec.flatten())
                markers_pos_rot.append([board_pos, board_rot])

                # Draw axes of the board
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, square_length*squares_vertically/2, round(square_length*square_pixels/2))

    # Display the undistorted camera feed if selected, based on the calibration data
    if visualise_calib_dist:
        if new_camera_matrix is None:
            h, w = frame.shape[:2]
            new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1)
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    if success:
        # Display position and rotation
        position, position_std, rotation, rotation_std = manta.display_position_ChArUco(frame, tvec_list, rvec_list, markers_pos_rot, camera_matrix, dist_coeffs, object_points_all, image_points_all, font_scale=2.5, thickness=3, rect_padding=(10,10,1900,400))
    
        # Send data via UDP        
        json_data = {
            "mpu_unit": MPU_UNIT,
            "camera": {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "position": position.tolist(),
                "position_std": position_std.tolist(),
                "rotation": rotation.tolist(),
                "rotation_std": rotation_std.tolist()
            }
        }
        sock.sendto(json.dumps(json_data).encode(), (UDP_IP, UDP_PORT))

    # Display the winch depth balancing reference
    manta.display_balance_bar(frame, depth_main, depth_sec, font_scale=8, thickness=20, bar_height=500)

	# Display the camera preview with overlays
    manta.resize_window_with_aspect_ratio("Camera Preview with Position", frame) # Ensure this function exists
    cv2.imshow("Camera Preview with Position", frame)

    # Check if the user wants to quit
    key = cv2.waitKey(1)
    if key == 27:
        break

# Stop the depth listener thread
stop_thread = True
depth_thread.join()

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
