# https://medium.com/@nflorent7/a-comprehensive-guide-to-camera-calibration-using-charuco-boards-and-opencv-for-perspective-9a0fa71ada5f

import cv2
import numpy as np
import os
from math import atan2, sqrt
import mantaPosLib as manta  # Ensure this module is correctly implemented or adjust accordingly
import genMarker

# Initialize parameters
# Set the selected camera: '4K', 'gopro' or 'axis'.
CAMERA_TYPE = "4K"
CAMERA_INPUT = 3  # Camera index if the source is webcam-based
CAMERA_RTSP_ADDR = "rtsp://admin:@169.254.56.12:554/" # Overwrites CAMERA_INPUT if 4K selected

# ChArUco board settings
squares_vertically = 6
squares_horizontally = squares_vertically
square_pixels = int(250*7/squares_horizontally) # Pixel size of the chessboard squares
grid_edge = 60 # Pixel margin outside the ChArUco grid
marker_ratio = 0.75 # Marker ratio of square_length to fit within white squares; acceptable maximum 0.85, recommended 0.7. Rounds marker size to int.
#square_length = 0.2975/6 * square_pixels/200 # Real worl-d length of square in meters
square_length = 1.110/7 * square_pixels/280 # Lyftkranen - large conference room screen

print(f"Reference length of {squares_horizontally} squares:", square_length*squares_horizontally, "meters")

# Generate and display the marker grid
board, dictionary = genMarker.create_and_save_ChArUco_board(square_length, square_pixels, grid_edge, marker_ratio, squares_vertically, squares_horizontally)
manta.display_marker_grid(board_type="ChArUco")

# Define the detector and parameters
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, params)

# Initialize camera
if CAMERA_TYPE == "4K":
    #cap = cv2.VideoCapture(CAMERA_RTSP_ADDR)
    cap = manta.RealtimeCapture(CAMERA_RTSP_ADDR)
    cv2.waitKey(500)
else:
    cap = cv2.VideoCapture(CAMERA_INPUT)
cv2.namedWindow("Camera Preview with Position", cv2.WINDOW_NORMAL)

# Set camera resolution based on the camera type
match CAMERA_TYPE:
    case "axis":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    case "gopro":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    case "4K":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
if not cap.isOpened():
    print("Error: Could not open camera",CAMERA_INPUT)
    exit()

# Load previously saved camera calibration data
calibration_dir = './camera_calibrations'
match CAMERA_TYPE:
    case "axis":
        calibration_data = np.load(os.path.join(calibration_dir,'camera_calibration_axis.npz'))
    case "axis_low":
        calibration_data = np.load(os.path.join(calibration_dir,'camera_calibration_axis_low.npz'))
    case "gopro":
        calibration_data = np.load(os.path.join(calibration_dir,'camera_calibration_gopro.npz'))
    case "4K":
        calibration_data = np.load(os.path.join(calibration_dir,'camera_calibration_4K_6s.npz'))
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# Precompute board center offset to center the coordinate system
board_width = (squares_horizontally - 0) * square_length
board_height = (squares_vertically - 0) * square_length
board_center_offset = [
    -board_width / 2,
    -board_height / 2,
    0
]
board_pos = board_center_offset#[0,0,0]  # Position of the marker in meters
board_rot = [0,0,0]  # Euler rotation of the marker in degrees, origin is normal around z

while True:
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
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, cornerColor=(0, 255, 0))
            if CAMERA_TYPE == "4K":
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

                # Display position and rotation
                match CAMERA_TYPE:
                    case "axis":
                        manta.display_position_ChArUco(frame, tvec_list, rvec_list, markers_pos_rot, camera_matrix, dist_coeffs, object_points_all, image_points_all, font_scale=1.3, thickness=2, rect_padding=(10,10,1000,200))
                    case "axis_low":
                        manta.display_position_ChArUco(frame, tvec_list, rvec_list, markers_pos_rot, camera_matrix, dist_coeffs, object_points_all, image_points_all)
                    case "gopro":
                        manta.display_position_ChArUco(frame, tvec_list, rvec_list, markers_pos_rot, camera_matrix, dist_coeffs, object_points_all, image_points_all, font_scale=1.5, thickness=2, rect_padding=(10,10,1100,280))
                    case "4K":
                        manta.display_position_ChArUco(frame, tvec_list, rvec_list, markers_pos_rot, camera_matrix, dist_coeffs, object_points_all, image_points_all, font_scale=2.5, thickness=3, rect_padding=(10,10,1900,400))
                # Draw axis on the board
                #cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, square_length)
    else:
        # Handle cases where no markers are detected
        pass

    # Undistort the camera feed
    #h, w = frame.shape[:2]
    #new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1)
    #frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Display the camera preview with position overlay
    manta.resize_window_with_aspect_ratio("Camera Preview with Position", frame)  # Ensure this function exists
    cv2.imshow("Camera Preview with Position", frame)

    # Check if the user wants to quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
