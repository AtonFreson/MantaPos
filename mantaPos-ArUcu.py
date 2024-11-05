import cv2
import numpy as np
import os
from math import atan2, sqrt
import mantaPosLib as manta
import genMarker



# ArUco marker settings
aruco_dict = cv2.aruco.getPredefinedDictionary(genMarker.ARUCO_DICT)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

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
        calibration_data = np.load(os.path.join(calibration_dir,'camera_calibration_4K.npz'))
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# Number of markers in the grid (example: 3x3 grid)
# The marker ID of the first marker in the order. OBS: Markers in folder need to be sequential!
grid_rows, marker_base = manta.display_marker_grid(num_markers, marker_size, marker_border_size)
grid_cols = grid_rows # Assume square
grid_marker_spacing = marker_length * (marker_border_size*2 / marker_size + 1) # Distance between markers in the grid (meters)

# Initialize camera
if CAMERA_TYPE == "4K":
    #cap = cv2.VideoCapture(CAMERA_RTSP_ADDR)
    cap = manta.RealtimeCapture(CAMERA_RTSP_ADDR)
else:
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
    case "4K":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


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

        tvec_list = []  # List to store all marker translation vectors
        rvec_list = []  # List to store all marker rotation vectors
        markers_pos_rot = []  # List to store all global marker positions and rotations

        # Loop through all detected markers
        for i in range(len(ids)):
            # Get image points for the detected marker corners
            image_points = corners[i][0]

            # Solve PnP for each detected marker
            success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

            if success:
                marker_pos = [0,0,0]  # Position of the marker in meters
                marker_rot = [0,0,0]  # Euler rotation of the marker in degrees, origin is normal around z

                # Apply marker position offset for each marker in the grid. Assume sequential, starting at marker_base.
                marker_row = (ids[i][0] - marker_base) // grid_cols
                marker_col = (ids[i][0] - marker_base) % grid_cols
                marker_offset = [
                    (marker_col - (grid_cols - 1) / 2) * grid_marker_spacing,
                    (marker_row - (grid_rows - 1) / 2) * grid_marker_spacing,
                    0
                ]
                marker_pos = marker_offset  # Adjust for marker grid offset

                # Store translation and rotation vectors
                tvec_list.append(tvec.flatten())
                rvec_list.append(rvec.flatten())
                markers_pos_rot.append([marker_pos, marker_rot])

        # Calculate & Display the average translation and rotation vectors if markers were detected
        match CAMERA_TYPE:
            case "axis":
                manta.display_position(frame, tvec_list, rvec_list, markers_pos_rot, font_scale=1.3, thickness=2, rect_padding=(10,10,950,200))
            case "axis_low":
                manta.display_position(frame, tvec_list, rvec_list, markers_pos_rot)
            case "gopro":
                manta.display_position(frame, tvec_list, rvec_list, markers_pos_rot, font_scale=1.5, thickness=2, rect_padding=(10,10,1100,280))
            case "4K":
                manta.display_position(frame, tvec_list, rvec_list, markers_pos_rot, font_scale=2.5, thickness=3, rect_padding=(10,10,1900,400))

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
