# https://medium.com/@nflorent7/a-comprehensive-guide-to-camera-calibration-using-charuco-boards-and-opencv-for-perspective-9a0fa71ada5f

import cv2
import numpy as np
import os
import sys
from math import atan2, sqrt
import mantaPosLib as manta  # Ensure this module is correctly implemented
import genMarker
from datetime import datetime
import json

# Initialize parameters
MPU_UNIT = 4  # MPU unit number for recording the camera position/rotation data
CAMERA_RTSP_ADDR = "rtsp://admin:@169.254.178.11:554/" # Overwrites CAMERA_INPUT if 4K selected
camera_calibration_file = 'camera_calibration_4K-38_20-picked.npz'
disable_camera = True  # Set to True to disable the camera and use test frames instead
enable_ocr_timestamp = False  # Set to True to enable OCR timestamp reading for precise camera timestamping

# Set to True to visualise the frame distortion based on the camera calibration. High computational cost (~110ms).
visualise_calib_dist = False

MARKER_TYPE = ["ChArUco", "Single"]  # Select the marker type to use
# Options are "ChArUco" or "ArUco", and "Single" or "Quad" respectively

# Quad marker positions in meters and order: 36, 37, 38, 39 & ArUco: b1, b2, b3, b4
QUAD_MARKER_POS = [[1.5, 0.0], [0.0, 1.5], [-1.5, 0.0], [0.0, -1.5]]
MARKERS_Z_LEVEL = -0.3+0.1124  # Height of the markers in meters relative to origo

new_camera_matrix = None

# Set to True to enable position zeroing of markers based on camera position.
enable_encoder_zeroing = False

# Small helper function to flatten the data
def reformat(data, type="corners"):
    if type == "corners":
        return [point for sublist in data if sublist is not None for point in sublist]
    elif type == "ids" or type == "charuco_corners":
        flattened_points = [item for item in data if item is not None]
        return np.vstack(flattened_points) if flattened_points else None

base_dict = cv2.aruco.getPredefinedDictionary(genMarker.ARUCO_DICT)
dictionary = cv2.aruco.Dictionary(1, base_dict.markerSize, base_dict.maxCorrectionBits)
if MARKER_TYPE[0] == "ChArUco" and MARKER_TYPE[1] == "Single":
    # ChArUco board settings
    squares_vertically = 6
    squares_horizontally = squares_vertically
    marker_ratio = 0.75 # Marker ratio of square_length to fit within white squares; acceptable maximum 0.85, recommended 0.7. Rounds marker size to int.
    square_length = 1.000/squares_vertically # Real world length of square in meters
    marker_number = 17  # Maximum marker number to use, starts at 0

    # Create a custom dictionary with only the specified marker
    dictionary.bytesList = base_dict.bytesList[0:marker_number+1]
    # Generate the marker grid
    board = genMarker.create_and_save_ChArUco_board(square_length, 400, 20, marker_ratio, squares_vertically, squares_horizontally, dictionary=dictionary)

    # Precompute board center offset to center the coordinate system
    width = (squares_horizontally - 0) * square_length
    height = (squares_vertically - 0) * square_length
    center_offset = square_length / 4 # Offset center due to hole placement

    single_board_pos = [-width/2  +center_offset, -height/2 +center_offset, MARKERS_Z_LEVEL]#[0,0,0]  # Position of the marker in meters
    single_board_rot = [0,0,0]  # Euler rotation of the marker in degrees, origin is normal around z
    
elif MARKER_TYPE[0] == "ArUco" and MARKER_TYPE[1] == "Single":
    # Large ArUco marker settings
    marker_length = 1.000 # in meters
    marker_number = 3  # Marker number to use

    # Precompute board center offset to center the coordinate system
    single_marker_pos = [-marker_length/2, -marker_length/2, MARKERS_Z_LEVEL]  # Offset position of the marker in meters
    single_marker_rot = [0,0,0]  # Euler rotation of the marker in degrees, origin is normal around z

    object_points = np.array([[0, 0, 0], 
                            [marker_length, 0, 0], 
                            [marker_length, marker_length, 0], 
                            [0, marker_length, 0]], dtype=np.float32)
    
    # Create a custom dictionary with only the specified marker
    dictionary.bytesList = base_dict.bytesList[marker_number:marker_number+1]

elif MARKER_TYPE[0] == "ChArUco" and MARKER_TYPE[1] == "Quad":
    # Quad ChArUco board settings
    squares_vertically = 3
    squares_horizontally = squares_vertically
    marker_ratio = 0.75
    square_length = 0.500/squares_vertically # Real world length of each square in meters
    marker_numbers = [18, 22, 26, 30] # Marker numbers to use in the quad ChArUco boards, number then 3 next

    corner_offset = square_length*squares_vertically/2
    quad_marker_pos = [[QUAD_MARKER_POS[0][0] +corner_offset, QUAD_MARKER_POS[0][1] +corner_offset, MARKERS_Z_LEVEL],
                       [QUAD_MARKER_POS[1][0] -corner_offset, QUAD_MARKER_POS[1][1] +corner_offset, MARKERS_Z_LEVEL], 
                       [QUAD_MARKER_POS[2][0] +corner_offset, QUAD_MARKER_POS[2][1] -corner_offset, MARKERS_Z_LEVEL],
                       [QUAD_MARKER_POS[3][0] +corner_offset, QUAD_MARKER_POS[3][1] +corner_offset, MARKERS_Z_LEVEL]]
    quad_marker_rot = [[0,0,180], [0,0,270], [0,0,90], [0,0,180]]
    
    # Set the bytesList directly
    dictionary.bytesList = base_dict.bytesList[marker_numbers[0]:marker_numbers[3]+4]
    
    # Create separate boards with the same unified dictionary
    boards = []
    for i in range(4):
        board = cv2.aruco.CharucoBoard((squares_horizontally, squares_vertically), 
                                       square_length, 
                                       square_length*marker_ratio, 
                                       dictionary)
        boards.append(board)
    
    # Create a mapping from marker ID to board index
    marker_to_board = {}
    for i, marker_number in enumerate(marker_numbers):
        for j in range(4):
            marker_to_board[marker_number + j] = i

# Define the detector
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

# Initialize camera
cap = manta.RealtimeCapture(CAMERA_RTSP_ADDR, disable_camera, enable_ocr_timestamp)
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



#depth_main_list = [3.29482, 3.29482, 3.29482, 3.29482, 3.29482, 4.56092, 4.56092, 4.56092, 4.56092, 7.01261, 7.01261, 7.01261]
#depth_sec_list = [3.28974, 3.28974, 3.28974, 3.28974, 3.28974, 4.55399, 4.55399, 4.55399, 4.55399, 7.00989, 7.00989, 7.00989]
#frame_pos_list = [2.70515, 2.70515, 2.70515, 2.70515/2, 0.0, 0.0, 0.0, 2.70515/2, 2.70515, 0.0, 2.70515/2, 2.70515]
depth_main_list = [0, 4.00778, 4.00778, 4.00781]
depth_sec_list = [0, 4.00824, 4.00824, 4.00824]
frame_pos_list = [0, 1.42882+0.00188, 2.70383+0.00188, 0]
# Main loop
try:
    while True:
        success = False
        
        # Capture camera frame
        ret, frame, camera_timestamp, camera_fps = cap.read()
        if not ret:
            print("Error: Could not read from camera.")
            break
        
        #frame = manta.frame_corner_cutout(frame, 0.3)  # Cut out the corners of the frame 
        #frame = manta.frame_crop(frame, 0.7)  # Crop the frame to remove fisheye edges
        if disable_camera:
            frame = test_frame.copy()
        #cv2.imwrite('ArUco_Marker_test.png', frame)

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, rejected_img_points = detector.detectMarkers(gray_frame)
        if MARKER_TYPE[1] == "Quad":
            # Sort detected markers into their respective boards
            if ids is not None and len(ids) > 0:
                corners_list = [[] for _ in range(4)]
                ids_list = [[] for _ in range(4)]
                
                for i in range(len(ids)):
                    marker_id = ids[i][0] + marker_numbers[0]
                    if marker_id in marker_to_board:
                        board_idx = marker_to_board[marker_id]
                        corners_list[board_idx].append(corners[i])
                        ids_list[board_idx].append([marker_id])
                
                # Convert to numpy arrays or None if empty
                for i in range(4):
                    if corners_list[i]:
                        corners_list[i] = np.array(corners_list[i])
                        ids_list[i] = np.array(ids_list[i])
                    else:
                        corners_list[i] = None
                        ids_list[i] = None
            else:
                corners_list = [None, None, None, None]
                ids_list = [None, None, None, None]
                
        # Clear translation and rotation vectors
        tvec_list = [None, None, None, None]
        rvec_list = [None, None, None, None]
        markers_pos_rot = [None, None, None, None]

        if ids is not None and len(ids) > 0:
            # Refine detected markers for better accuracy
            if MARKER_TYPE[0] == "ChArUco":
                if MARKER_TYPE[1] == "Single":
                    # Refine detected markers for ChArUco board
                    detector.refineDetectedMarkers(
                        image=gray_frame,
                        board=board,
                        detectedCorners=corners,
                        detectedIds=ids,
                        rejectedCorners=rejected_img_points
                    )
                else:
                    for i in range(4):
                        if ids_list[i] is not None:
                            # Use the unified detector for refinement
                            detector.refineDetectedMarkers(
                                image=gray_frame,
                                board=boards[i],
                                detectedCorners=corners_list[i],
                                detectedIds=ids_list[i],
                                rejectedCorners=rejected_img_points
                            )

            # Modify the displayed markers to make them unreadable. Use: blur, cross, fill, diamond
            try:
                if MARKER_TYPE[1] == "Single":
                    frame = manta.censor_marker(frame, corners, "diamond")
                else:
                    frame = manta.censor_marker(frame, reformat(corners_list, "corners"), "diamond")
                #frame = manta.censor_charuco_board(frame, charuco_corners, corners, 0.5)
            except:
                pass

            # Draw detected markers
            if MARKER_TYPE[1] == "Single":
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            else:
                cv2.aruco.drawDetectedMarkers(frame, reformat(corners_list, "corners"), reformat(ids_list, "ids"))

            # Single ChArUco board detection
            if MARKER_TYPE[0] == "ChArUco":
                # Interpolate ChArUco corners
                if MARKER_TYPE[1] == "Single":
                    num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                        markerCorners=corners,
                        markerIds=ids,
                        image=gray_frame,
                        board=board
                    )
                else:
                    num_corners_list = [None, None, None, None]
                    charuco_corners_list = [None, None, None, None]
                    charuco_ids_list = [None, None, None, None]
                    for i in range(4):
                        if ids_list[i] is not None:
                            ids_list[i] = ids_list[i] - marker_numbers[i]
                            num_corners_list[i], charuco_corners_list[i], charuco_ids_list[i] = cv2.aruco.interpolateCornersCharuco(
                                markerCorners=corners_list[i],
                                markerIds=ids_list[i],
                                image=gray_frame,
                                board=boards[i]
                            )
                            ids_list[i] = ids_list[i] + marker_numbers[i]

                # Draw ChArUco corners
                if MARKER_TYPE[1] == "Single":
                    if charuco_ids is not None and num_corners > 0:
                        cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, cornerColor=(255, 0, 0))
                        
                        for i, corner in enumerate(charuco_corners):
                            cv2.putText(frame, str(charuco_ids[i][0]), (int(corner[0][0]), int(corner[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                else:
                    if reformat(charuco_ids_list, "ids") is not None and sum(num for num in num_corners_list if num is not None) > 0:
                        cv2.aruco.drawDetectedCornersCharuco(frame, reformat(charuco_corners_list, "charuco_corners"), reformat(charuco_ids_list, "ids"), cornerColor=(255, 0, 0))

                        for i, corner in enumerate(reformat(charuco_corners_list, "charuco_corners")):
                            cv2.putText(frame, str(reformat(charuco_ids_list, "ids")[i][0]), (int(corner[0][0]), int(corner[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                
                if MARKER_TYPE[1] == "Single":
                    if charuco_ids is not None and num_corners >= 6:
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
                            #object_points_all = board.getChessboardCorners()[charuco_ids.flatten()]
                            #image_points_all = charuco_corners.reshape(-1, 2)

                            # Store the position and rotation of the board
                            tvec_list[0] = tvec.flatten()
                            rvec_list[0] = rvec.flatten()
                            markers_pos_rot[0] = [single_board_pos, single_board_rot]

                            # Draw axes of the board
                            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, square_length*squares_vertically/2, 10)
                else:
                    for i in range(4):
                        if charuco_ids_list[i] is not None and num_corners_list[i] >= 2:
                            # Estimate pose of the ChArUco board
                            success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                                charucoCorners=charuco_corners_list[i],
                                charucoIds=charuco_ids_list[i],
                                board=boards[i],
                                cameraMatrix=camera_matrix,
                                distCoeffs=dist_coeffs,
                                rvec = np.zeros((3, 1), dtype=np.float64),
                                tvec = np.zeros((3, 1), dtype=np.float64)
                            )

                            if success:
                                # Store the position and rotation of the board
                                tvec_list[i] = tvec.flatten()
                                rvec_list[i] = rvec.flatten()
                                markers_pos_rot[i] = [quad_marker_pos[i], quad_marker_rot[i]]

                                # Draw axes of the board
                                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, square_length*squares_vertically/2, 10)

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

        # Get the depth values from receiver.py
        depth_main, depth_sec, frame_pos, depth_timestamp = depth_shared.get_depth()
        if disable_camera:
            depth_main = depth_main_list[snapnr]
            depth_sec = depth_sec_list[snapnr]
            frame_pos = frame_pos_list[snapnr]
            depth_timestamp = int(datetime.now().timestamp()*1000)
        
        ref_pos, ref_rot = manta.global_reference_pos(depth_main, depth_sec, frame_pos)

        json_data = {
            "mpu_unit": MPU_UNIT,
            "packet_number": frame_number,
            "camera": {
                "timestamp": camera_timestamp,
                "fps": camera_fps
            }
        }
        if ref_pos is not None and ref_rot is not None:
            json_data["global_pos"] = {
                "timestamp": depth_timestamp,
                "position": ref_pos.tolist(),
                "rotation": ref_rot.tolist()
            }
        
        if success or MARKER_TYPE[1] == "Quad":
            # Calculate and display position and rotation
            if MARKER_TYPE[1] == "Single":
                #position, position_std, rotation, rotation_std = manta.display_position_ChArUco(frame, tvec_list, rvec_list, markers_pos_rot, camera_matrix, 
                # dist_coeffs, object_points_all, image_points_all, font_scale=2.5, thickness=3, rect_padding=(10,10,1900,400))
                position, rotation = manta.calculate_camera_position(tvec_list[0], rvec_list[0], markers_pos_rot[0])
                manta.display_camera_position(frame, position, rotation, ref_pos, ref_rot, font_scale=2.5, thickness=6, rect_padding=(10,10,1900,350))

                json_data["camera_pos_0"] = {
                    "position": position.tolist(),
                    "rotation": rotation.tolist()
                }
            else:
                position = []
                rotation = []
                for i in range(4):
                    if tvec_list[i] is not None and rvec_list[i] is not None:
                        temp_position, temp_rotation = manta.calculate_camera_position(tvec_list[i], rvec_list[i], markers_pos_rot[i])
                        position.append(temp_position), rotation.append(temp_rotation)
                        
                        json_data[f"camera_pos_{i}"] = {
                            "position": temp_position.tolist(),
                            "rotation": temp_rotation.tolist()
                        }
                # Display averaged position and rotation
                if position and rotation:
                    manta.display_camera_position(frame, np.mean(position, axis=0), np.mean(rotation, axis=0), ref_pos, ref_rot, font_scale=2.5, thickness=6, rect_padding=(10,10,1900,350))

        position_shared.write_position(json.dumps(json_data))
        frame_number += 1

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
            snapshot_number = -1
            if not os.path.exists("./cam_captures"):
                os.makedirs("./cam_captures")
            for file in os.listdir("./cam_captures"):
                if file.endswith(".png"):
                    number = int(file.split("_")[1].split(".")[0])
                    if number > snapshot_number:
                        snapshot_number = number
            snapshot_number = snapshot_number+1
            cv2.imwrite(f"./cam_captures/snapshot_{snapshot_number:04d}.png", gray_frame)
            print(f"Snapshot saved in cam_captures as snapshot_{snapshot_number:04d}.png")
        elif key == ord('d'):
            snapnr = snapnr + 1
            if snapnr > len(depth_main_list)-1:
                snapnr = 1
            try:
                test_frame = cv2.imread(f"./cam_captures/snapshot_{snapnr:04d}.png")
                #test_frame = cv2.imread(f"../cam_captures-full_test/snapshot_{snapnr:04d}.png")
            except:
                print(f"Error: Could not load snapshot_{snapnr:04d}.png")
        elif key == ord('a'):
            snapnr = snapnr - 1
            if snapnr < 1:
                snapnr = len(depth_main_list)-1
            try:
                test_frame = cv2.imread(f"./cam_captures/snapshot_{snapnr:04d}.png")
                #test_frame = cv2.imread(f"../cam_captures-full_test/snapshot_{snapnr:04d}.png")
            except:
                print(f"Error: Could not load snapshot_{snapnr:04d}.png")
        elif key == ord('o'):
            if enable_encoder_zeroing: 
                if MARKER_TYPE[0] == "ChArUco" and MARKER_TYPE[1] == "Single":
                    # Zero the position of the markers based on the camera position
                    single_board_pos, single_board_rot = manta.zero_marker_position(tvec_list[0], rvec_list[0], ref_pos, ref_rot)
                    print(f"Zeroed position: {single_board_pos}, rotation: {single_board_rot}")
        
except KeyboardInterrupt:
    print("User interrupted...")
except Exception as e:
    print(f"Main Loop Error: {e}")
    print(f"Given error: {sys.exc_info()} at location: {sys.exc_info()[2].tb_lineno}")
finally:
    print("Cleaning up shared memory...")
    depth_shared.close()
    position_shared.close()

    print("Exiting...")
    cv2.destroyAllWindows()