# https://medium.com/@nflorent7/a-comprehensive-guide-to-camera-calibration-using-charuco-boards-and-opencv-for-perspective-9a0fa71ada5f

import cv2
import numpy as np
import os
from math import atan2, sqrt
import mantaPosLib as manta  # Ensure this module is correctly implemented or adjust accordingly
import genMarker

# Initialize parameters
# Set the selected camera: 'gopro' or 'axis'.
CAMERA_TYPE = "axis"
CAMERA_INPUT = 1  # Camera index or video file

# ChArUco board settings
squares_vertically = 5
squares_horizontally = 7
square_pixels = 200 # Pixel size of the chessboard squares
grid_edge = 30 # Pixel margin outside the ChArUco grid
marker_ratio = 0.7 # Marker ratio of square_length to fit within white squares; acceptable maximum 0.85, recommended 0.7 
square_length = 0.2975/6 # Real world length of square in meters

# Define the aruco dictionary, charuco board and detector
marker_length = square_length*marker_ratio
dictionary = cv2.aruco.getPredefinedDictionary(genMarker.ARUCO_DICT)
board = cv2.aruco.CharucoBoard((squares_horizontally, squares_vertically), square_length, marker_length, dictionary)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, params)

# Initialize camera
cap = cv2.VideoCapture(CAMERA_INPUT)
cv2.namedWindow("Camera Preview with Position", cv2.WINDOW_NORMAL)

match CAMERA_TYPE:
    case "axis":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    case "gopro":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
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
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

""" def rotation_matrix_to_euler_angles(rotation_matrix):
    #Convert a rotation matrix to Euler angles (roll, pitch, yaw) in degrees.
    sy = sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = atan2(-rotation_matrix[2, 0], sy)
        z = atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = atan2(-rotation_matrix[2, 0], sy)
        z = 0

    # Convert from radians to degrees
    return np.degrees(np.array([x, y, z]))

def display_position_single(frame, position, euler_angles, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8,
                            text_color=(0, 255, 0), thickness=2, alpha=0.5, rect_padding=(10, 10, 600, 150)):
    position_text = (f"Pos: X={position[0]: >+6.3f}m, Y={position[1]: >+6.3f}m, Z={position[2]: >+6.3f}m")
    rotation_text = (f"Rot: R={euler_angles[0]: >+6.3f}°, P={euler_angles[1]: >+6.3f}°, Y={euler_angles[2]: >+6.3f}°")

    # Unpack rectangle bounds
    x, y, w, h = rect_padding

    # Create a copy of the frame for overlay
    overlay = frame.copy()

    # Draw the rectangle
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)

    # Put text on the overlay
    cv2.putText(overlay, position_text, (x + 20, y + int(h / 3)), font, font_scale, text_color, thickness)
    cv2.putText(overlay, rotation_text, (x + 20, y + int(h / 1.5)), font, font_scale, text_color, thickness)

    # Apply the overlay
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame) """

# Precompute board center offset to center the coordinate system
board_width = (squares_horizontally - 1) * square_length
board_height = (squares_vertically - 1) * square_length
board_center_offset = [
    -board_width / 2,
    -board_height / 2,
    0
]
board_rot = [0,0,0]  # Euler rotation of the marker in degrees, origin is normal around z

manta.display_marker_grid(board_type="ChArUco")

while True:
    # Capture camera frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera.")
        break

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

        if charuco_ids is not None and num_corners > 0:
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
                # Store translation and rotation vectors
                tvec_list = []
                rvec_list = []
                markers_pos_rot = []
                
                tvec_list.append(tvec.flatten())
                rvec_list.append(rvec.flatten())
                markers_pos_rot.append([board_center_offset, board_rot])

                # Display position and rotation
                match CAMERA_TYPE:
                    case "axis":
                        manta.display_position(frame, tvec_list, rvec_list, markers_pos_rot, font_scale=1.3, thickness=2, rect_padding=(10,10,950,200))
                    case "axis_low":
                        manta.display_position(frame, tvec_list, rvec_list, markers_pos_rot)
                    case "gopro":
                        manta.display_position(frame, tvec_list, rvec_list, markers_pos_rot, font_scale=1.5, thickness=2, rect_padding=(10,10,1100,280))

                # Draw axis on the board
                #cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, square_length)

        # Modify the displayed markers to make them unreadable. Use: blur, cross, fill, diamond
        frame = manta.censor_marker(frame, corners, "diamond")

        # Draw detected markers and ChArUco corners
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        if charuco_ids is not None and num_corners > 0:
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
    else:
        # Handle cases where no markers are detected
        pass

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
