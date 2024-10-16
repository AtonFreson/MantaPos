# https://medium.com/@nflorent7/a-comprehensive-guide-to-camera-calibration-using-charuco-boards-and-opencv-for-perspective-9a0fa71ada5f

import cv2
import numpy as np
import os
from math import atan2, sqrt
import mantaPosLib as manta  # Ensure this module is correctly implemented or adjust accordingly

# Initialize parameters
# Set the selected camera: 'gopro' or 'axis'.
CAMERA_TYPE = "axis"
CAMERA_INPUT = 2  # Camera index or video file

# Load camera calibration data
if CAMERA_TYPE == "axis":
    calibration_file = 'camera_calibration_axis_low.npz'
elif CAMERA_TYPE == "gopro":
    calibration_file = 'camera_calibration_gopro.npz'
else:
    raise ValueError("Invalid CAMERA_TYPE. Choose 'gopro' or 'axis'.")

if not os.path.exists(calibration_file):
    raise FileNotFoundError(f"Calibration file '{calibration_file}' not found.")

calibration_data = np.load(calibration_file)
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# ChArUco board settings
square_length = 0.04  # Square length in meters
marker_length = 0.02  # Marker side length in meters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Create ChArUco board using the updated class method
grid_cols = 5  # Number of squares in X direction
grid_rows = 7  # Number of squares in Y direction
charuco_board = cv2.aruco.CharucoBoard.create(
    squaresX=grid_cols,
    squaresY=grid_rows,
    squareLength=square_length,
    markerLength=marker_length,
    dictionary=aruco_dict
)

# Detector parameters
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

# Generate and save ChArUco board image (optional)
if not os.path.exists('charuco_board.png'):
    board_image = charuco_board.draw((600, 900))
    cv2.imwrite('charuco_board.png', board_image)
    print("ChArUco board image saved as 'charuco_board.png'.")

# Initialize camera
cap = cv2.VideoCapture(CAMERA_INPUT)
cv2.namedWindow("Camera Preview with Position", cv2.WINDOW_NORMAL)

if CAMERA_TYPE == "axis":
    # Example settings; adjust as needed
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    pass
elif CAMERA_TYPE == "gopro":
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

def rotation_matrix_to_euler_angles(rotation_matrix):
    """Convert a rotation matrix to Euler angles (roll, pitch, yaw) in degrees."""
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
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

# Precompute board center offset to center the coordinate system
board_width = (grid_cols - 1) * square_length
board_height = (grid_rows - 1) * square_length
board_center_offset = np.array([
    board_width / 2,
    board_height / 2,
    0
])

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
            board=charuco_board,
            detectedCorners=corners,
            detectedIds=ids,
            rejectedCorners=rejected_img_points
        )

        # Interpolate ChArUco corners
        num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray_frame,
            board=charuco_board
        )

        if charuco_ids is not None and num_corners > 0:
            # Estimate pose of the ChArUco board
            success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids,
                board=charuco_board,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs
            )

            if success:
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rvec)

                # Adjust tvec to center the coordinate system at the board center
                tvec = tvec - rotation_matrix @ board_center_offset.reshape((3, 1))

                # Convert rotation matrix to Euler angles
                euler_angles = rotation_matrix_to_euler_angles(rotation_matrix)

                # Display position and rotation
                position = tvec.flatten()
                if CAMERA_TYPE == "gopro":
                    display_position_single(frame, position, euler_angles, font_scale=1.5, thickness=2, rect_padding=(10, 10, 1100, 280))
                else:
                    display_position_single(frame, position, euler_angles)

                # Draw axis on the board
                cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, square_length)

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
