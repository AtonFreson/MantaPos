import cv2
import numpy as np
import os
from math import atan2, sqrt
import mantaPosLib as manta



# ArUco marker settings
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

# Define the real-world size of the ArUco markers (same as during calibration)
screen_dpm = 200/0.0495 # Length of marker's side in pixels / meters

marker_size = 200
marker_border_size = 25
num_markers = 0 # Set to 0 if all

# Set the selected camera: axis, axis_low or gopro.
CAMERA_TYPE = "axis_low"
CAMERA_INPUT = 2 # OBS Virtual Camera

marker_length = marker_size/screen_dpm # in meters

object_points = np.array([[0, 0, 0], 
                          [marker_length, 0, 0], 
                          [marker_length, marker_length, 0], 
                          [0, marker_length, 0]], dtype=np.float32)

# Load previously saved camera calibration data
match CAMERA_TYPE:
    case "axis":
        calibration_data = np.load('camera_calibration_axis.npz')
    case "axis_low":
        calibration_data = np.load('camera_calibration_axis_low.npz')
    case "gopro":
        calibration_data = np.load('camera_calibration_gopro.npz')
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# Number of markers in the grid (example: 3x3 grid)
# The marker ID of the first marker in the order. OBS: Markers in folder need to be sequential!
grid_rows, marker_base = manta.display_marker_grid(num_markers, marker_size, marker_border_size)
grid_cols = grid_rows # Assume square
grid_marker_spacing = marker_length * (marker_border_size*2 / marker_size + 1) # Distance between markers in the grid (meters)

# Initialize camera
cap = cv2.VideoCapture(CAMERA_INPUT) 
cv2.namedWindow("Camera Preview with Position", cv2.WINDOW_NORMAL)
match CAMERA_TYPE:
    case "axis":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    case "axis_low":
        None
    case "gopro":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

def rotation_matrix_to_euler_angles(rotation_matrix):
    """ Convert a rotation matrix to Euler angles (roll, pitch, yaw). """
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

    return np.degrees(np.array([x, y, z]))

def display_position(frame, tvec_list, rvec_list, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8,
                     text_color=(0, 255, 0), thickness=1, alpha=0.5, rect_padding=(10, 10, 600, 150)):
    
    # Display only if tags are visible
    if len(tvec_list) == 0:
        return
    
    tvec_array = np.array(tvec_list)
    rvec_array = np.array(rvec_list)

    # Calculate average position and rotation vectors
    position = np.mean(tvec_array, axis=0)
    avg_rvec = np.mean(rvec_array, axis=0)

    # Calculate standard deviations for position and rotation
    position_std = np.std(tvec_array, axis=0)
    rvec_std = np.std(rvec_array, axis=0)

    # Convert average rotation vector to Euler angles
    avg_rotation_matrix, _ = cv2.Rodrigues(avg_rvec)
    euler_angles = rotation_matrix_to_euler_angles(avg_rotation_matrix)

    # Convert rotation std from rvec to Euler std
    rvec_std_matrix, _ = cv2.Rodrigues(rvec_std)
    rotation_std = rotation_matrix_to_euler_angles(rvec_std_matrix)

    # Create a position text with fixed-width formatting to prevent text shifting
    
    position_text = (f"Pos: X={position[0]: >+6.3f}m, Y={position[1]: >+6.3f}m, Z={position[2]: >+6.3f}m")
    rotation_text = (f"Rot: R={euler_angles[0]: >+6.3f}', P={euler_angles[1]: >+6.3f}', Y={euler_angles[2]: >+6.3f}'")
    position_std_text = (f"-Std: X={position_std[0]: >6.3f}m, Y={position_std[1]: >6.3f}m, Z={position_std[2]: >6.3f}m")
    rotation_std_text = (f"-Std: R={rotation_std[0]: >6.3f}', P={rotation_std[1]: >6.3f}', Y={rotation_std[2]: >6.3f}'")

    # Unpack rectangle bounds
    x, y, w, h = rect_padding

    # Create a copy of the frame for overlay
    overlay = frame.copy()

    # Draw the rectangle
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)

    # Put text on the overlay
    cv2.putText(overlay, position_text, (x+20, y + int(h / 4.5)), font, font_scale, text_color, thickness)
    cv2.putText(overlay, position_std_text, (x+20, y + int(h / 2.5)), font, font_scale, text_color, thickness)
    cv2.putText(overlay, rotation_text, (x+20, y + int(h / 1.5)), font, font_scale, text_color, thickness)
    cv2.putText(overlay, rotation_std_text, (x+20, y + int(h / 1.2)), font, font_scale, text_color, thickness)

    # Apply the overlay
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)



while True:
    # Capture camera frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera.")
        break

    #print(frame.shape)
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers in the frame using the ArUcoDetector class
    corners, ids, rejected_img_points = detector.detectMarkers(gray_frame)

    if ids is not None:
        # Modify the displayed markers to make them unreadable. Use: blur, cross, fill, diamond
        frame = manta.censor_marker(frame, corners, "diamond")

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        tvec_list = []  # List to store all translation vectors
        rvec_list = []  # List to store all rotation vectors

        # Loop through all detected markers
        for i in range(len(ids)):
            # Get image points for the detected marker corners
            image_points = corners[i][0]

            # Solve PnP for each detected marker
            success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

            if success:
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rvec)

                # Apply marker position offset for each marker in the grid. Assume sequential, starting at marker_base.
                marker_row = (ids[i][0] - marker_base) // grid_cols
                marker_col = (ids[i][0] - marker_base) % grid_cols
                marker_offset = np.array([
                    (marker_col - (grid_cols - 1) / 2) * grid_marker_spacing,
                    (marker_row - (grid_rows - 1) / 2) * grid_marker_spacing,
                    0
                ])

                tvec -= rotation_matrix @ marker_offset.reshape((3, 1))  # Adjust for marker grid offset

                # Store translation and rotation vectors
                tvec_list.append(tvec.flatten())
                rvec_list.append(rvec.flatten())

        # Calculate & Display the average translation and rotation vectors if markers were detected
        match CAMERA_TYPE:
            case "axis":
                display_position(frame, tvec_list, rvec_list, font_scale=1.3, thickness=2, rect_padding=(10,10,950,200))
            case "axis_low":
                display_position(frame, tvec_list, rvec_list)
            case "gopro":
                display_position(frame, tvec_list, rvec_list, font_scale=1.5, thickness=2, rect_padding=(10,10,1100,280))

    # Display the camera preview with position overlay
    manta.resize_window_with_aspect_ratio("Camera Preview with Position", frame)
    cv2.imshow("Camera Preview with Position", frame)
    
    # Check if the user wants to quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
