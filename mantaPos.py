# https://medium.com/@nflorent7/a-comprehensive-guide-to-camera-calibration-using-charuco-boards-and-opencv-for-perspective-9a0fa71ada5f

import cv2
import numpy as np
import os
from math import atan2, sqrt
import mantaPosLib as manta  # Ensure this module is correctly implemented
import genMarker
from datetime import datetime
import json

# Initialize parameters
CAMERA_RTSP_ADDR = "rtsp://admin:@169.254.178.11:554/" # Overwrites CAMERA_INPUT if 4K selected
camera_calibration_file = 'camera_calibration_4K-38_20-picked.npz'

MPU_UNIT = 4  # MPU unit number for recording the camera position/rotation data

MARKER_TYPE = ["ChArUco", "Single"]  # Select the marker type to use
# Options are "ChArUco" or "ArUco", and "Single" or "Quad" respectively

# Quad ChArUco positions in meters and order: 36, 37, 38, 39
QUAD_CHARUCO_POS = [[1.5, 0.0], [0.0, 1.5], [-1.5, 0.0], [0.0, -1.5]]

# Quad ArUco positions in meters and order: b1, b2, b3, b4
QUAD_ARUCO_POS = [[1.5, 0.0], [0.0, 1.5], [-1.5, 0.0], [0.0, -1.5]]

# Set to True to visualise the frame distortion based on the camera calibration. High computational cost (~110ms).
visualise_calib_dist = True
new_camera_matrix = None

# Set to True to enable position zeroing of markers based on camera position.
enable_encoder_zeroing = False


if MARKER_TYPE[0] == "ChArUco" and MARKER_TYPE[1] == "Single":
    # ChArUco board settings
    squares_vertically = 6
    squares_horizontally = squares_vertically
    square_pixels = int(140*7/squares_horizontally) # Pixel size of the chessboard squares
    grid_edge = 20 # Pixel margin outside the ChArUco grid
    marker_ratio = 0.75 # Marker ratio of square_length to fit within white squares; acceptable maximum 0.85, recommended 0.7. Rounds marker size to int.
    square_length = 1.000/squares_vertically # Real world length of square in meters
    
    # Generate the marker grid
    board, dictionary = genMarker.create_and_save_ChArUco_board(square_length, square_pixels, grid_edge, marker_ratio, squares_vertically, squares_horizontally)

    # Precompute board center offset to center the coordinate system
    single_board_width = (squares_horizontally - 0) * square_length
    single_board_height = (squares_vertically - 0) * square_length
    single_board_center_offset = [-single_board_width / 2, -single_board_height / 2, 0]

    single_board_pos = single_board_center_offset#[0,0,0]  # Position of the marker in meters
    single_board_rot = [0,0,0]  # Euler rotation of the marker in degrees, origin is normal around z
    
elif MARKER_TYPE[0] == "ArUco" and MARKER_TYPE[1] == "Single":
    # Large ArUco marker settings
    marker_length = 1.000 # in meters
    marker_number = 0  # Marker number to use
    single_marker_pos = [0.1, 0.1]  # Offset position of the marker in meters
    single_marker_rot = [0,0,0]  # Euler rotation of the marker in degrees, origin is normal around z

    object_points = np.array([[0, 0, 0], 
                            [marker_length, 0, 0], 
                            [marker_length, marker_length, 0], 
                            [0, marker_length, 0]], dtype=np.float32)
    
    # Create a custom dictionary with only the specified marker
    base_dict = cv2.aruco.getPredefinedDictionary(genMarker.ARUCO_DICT)
    dictionary = cv2.aruco.Dictionary(1, base_dict.markerSize, base_dict.maxCorrectionBits)
    dictionary.bytesList = base_dict.bytesList[marker_number:marker_number+1]
    
# Define the detector and parameters
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, params)

# Initialize camera
cap = manta.RealtimeCapture(CAMERA_RTSP_ADDR)
cv2.namedWindow("Camera Preview with Position", cv2.WINDOW_NORMAL)
frame_number = 0

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

# Initialize shared memory for depth (consumer) and for camera position/rotation (consumer)
depth_shared = manta.DepthSharedMemory(create=False)
position_shared = manta.PositionSharedMemory(create=False)

# Test frame
#test_frame = cv2.imread("ChArUco_Marker_test.png")
test_frame = cv2.imread("./snapshots/snapshot_0793.png")
snapnr = 1

# Main loop
try:
    while True:
        success = False

        # Capture camera frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from camera.")
            break
        
        #frame = manta.frame_corner_cutout(frame, 0.3)  # Cut out the corners of the frame 
        #frame = manta.frame_crop(frame, 0.7)  # Crop the frame to remove fisheye edges
        #test_frame = cv2.imread(f"../cam_captures/snapshot_{snapnr:04d}.png")
        #frame = test_frame.copy()
        #cv2.imwrite('ArUco_Marker_test.png', frame)

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, rejected_img_points = detector.detectMarkers(gray_frame)
        
        # Clear translation and rotation vectors
        tvec_list = []
        rvec_list = []
        markers_pos_rot = []

        if ids is not None and len(ids) > 0:
            # Refine detected markers for better accuracy
            detector.refineDetectedMarkers(
                image=gray_frame,
                board=board,
                detectedCorners=corners,
                detectedIds=ids,
                rejectedCorners=rejected_img_points
            )

            # Modify the displayed markers to make them unreadable. Use: blur, cross, fill, diamond
            try:
                frame = manta.censor_marker(frame, corners, "diamond")
                #frame = manta.censor_charuco_board(frame, charuco_corners, corners, 0.5)
            except:
                pass
            
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Single ChArUco board detection
            if MARKER_TYPE[0] == "ChArUco" and MARKER_TYPE[1] == "Single":
                # Interpolate ChArUco corners
                num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    markerCorners=corners,
                    markerIds=ids,
                    image=gray_frame,
                    board=board
                )

                # Draw ChArUco corners
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

                        # Store the position and rotation of the board
                        tvec_list.append(tvec.flatten())
                        rvec_list.append(rvec.flatten())
                        markers_pos_rot.append([single_board_pos, single_board_rot])

                        # Draw axes of the board
                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, square_length*squares_vertically/2, round(square_length*square_pixels/2))

            # Single ArUco marker detection
            elif MARKER_TYPE[0] == "ArUco" and MARKER_TYPE[1] == "Single":
                # Loop through all detected markers
                for i in range(len(ids)):
                    # Get image points for the detected marker corners
                    image_points = corners[i][0]

                    # Solve PnP for each detected marker
                    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

                    if success:
                        # Store translation and rotation vectors
                        tvec_list.append(tvec.flatten())
                        rvec_list.append(rvec.flatten())
                        markers_pos_rot.append([single_marker_pos, single_marker_rot])

        # Display the undistorted camera feed if selected, based on the calibration data
        if visualise_calib_dist:
            if new_camera_matrix is None:
                h, w = frame.shape[:2]
                new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1)
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

        if success:
            # Calculate and display position and rotation
            if MARKER_TYPE[1] == "Single":
                #position, position_std, rotation, rotation_std = manta.display_position_ChArUco(frame, tvec_list, rvec_list, markers_pos_rot, camera_matrix, 
                # dist_coeffs, object_points_all, image_points_all, font_scale=2.5, thickness=3, rect_padding=(10,10,1900,400))
                position, rotation = manta.calculate_camera_position(tvec_list[0], rvec_list[0], markers_pos_rot[0])
                manta.display_camera_position(frame, position, rotation, font_scale=2.5, thickness=3, rect_padding=(10,10,1900,200))

            json_data = {
                "mpu_unit": MPU_UNIT,
                "packet_number": frame_number,
                "camera": {
                    "timestamp": int(datetime.now().timestamp() * 1000),
                    "position": position.tolist(),
                    "rotation": rotation.tolist()
                }
            }
            frame_number += 1
            position_shared.write_position(json.dumps(json_data))

        # Get the depth values from receiver.py
        depth_main, depth_sec, frame_pos = depth_shared.get_depth()

        # Display the winch depth balancing reference
        manta.display_balance_bar(frame, depth_main, depth_sec, font_scale=3, thickness=8, bar_height=200)

        # Display the camera preview with overlays
        manta.resize_window_with_aspect_ratio("Camera Preview with Position", frame) # Ensure this function exists
        cv2.imshow("Camera Preview with Position", frame)

        # Check if the user wants to quit(esc) or take a snapshot(spacebar)
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 32:
            # Check what the highest number in the snapshot folder is, and enumerate from there
            snapshot_number = 0
            for file in os.listdir("./cam_captures"):
                if file.endswith(".png"):
                    number = int(file.split("_")[1].split(".")[0])
                    print(number)
                    if number > snapshot_number:
                        snapshot_number = number
            snapshot_number = snapshot_number+1
            cv2.imwrite(f"./cam_captures/snapshot_{snapshot_number:04d}.png", frame)
            print(f"Snapshot saved in cam_captures as snapshot_{snapshot_number:04d}.png")
        elif key == ord('c'):
            snapnr = snapnr + 1
        elif key == ord('o'):
            if enable_encoder_zeroing: 
                if MARKER_TYPE[0] == "ChArUco" and MARKER_TYPE[1] == "Single":
                    # Zero the position of the markers based on the camera position
                    single_board_pos, single_board_rot = manta.zero_marker_position(tvec_list[0], rvec_list[0], rotation, depth_main, depth_sec, frame_pos)
                    print(f"Zeroed position: {single_board_pos}, rotation: {single_board_rot}")

except KeyboardInterrupt:
    print("User interrupted...")
except Exception as e:
    print(f"Error: {e}")
finally:
    print("Cleaning up shared memory...")
    depth_shared.close()
    position_shared.close()

    print("Exiting...")
    cv2.destroyAllWindows()