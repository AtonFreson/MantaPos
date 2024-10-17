import cv2
import numpy as np
import os
import time
import mantaPosLib as manta
import shutil
import genMarker

# Set the selected camera: gopro or axis.
CAMERA_TYPE = "axis"
CAMERA_INPUT = 2 # OBS Virtual Camera

delay_time = 0.5 # 500ms delay between capture

squares_vertically = 5
squares_horizontally = 7
square_pixels = 200 # Pixel size of the square
grid_edge = 30 # Pixel margin of ChArUco grid
marker_ratio = 0.7 # Marker ratio of square_length to fit within white squares, recommended maximum 0.85
square_lenght = 5 # Real world length of square

# Define the aruco dictionary, charuco board and detector
marker_length = square_lenght*marker_ratio
dictionary = cv2.aruco.getPredefinedDictionary(genMarker.ARUCO_DICT)
board = cv2.aruco.CharucoBoard((squares_vertically, squares_horizontally), square_lenght, marker_length, dictionary)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, params)

# Initialize camera
cap = cv2.VideoCapture(CAMERA_INPUT)
match CAMERA_TYPE:
    case "axis":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    case "gopro":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL) 
if not cap.isOpened():
    print("Error: Could not open camera",CAMERA_INPUT)
    exit()

# Prepare to save captured snapshots, clear previous directory
snapshot_dir = './snapshots'
if os.path.exists(snapshot_dir):
    shutil.rmtree(snapshot_dir)
os.makedirs(snapshot_dir)

# Generate and display the marker grid
genMarker.create_and_save_ChArUco_board(square_pixels, grid_edge, marker_ratio, squares_vertically, squares_horizontally)
manta.display_marker_grid(board_type="ChArUco")

# Lists to store object points (3D) and image points (2D)
all_charuco_ids = []
all_charuco_corners = []

# Start capturing camera frames
next_snapshot_time = time.time() + 0.5  # First snapshot in 500ms
snapshot_counter = 0

print("Press 'q' to quit the program.")


while True:
    # Capture camera frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera.")
        break

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers in the frame using the ArUcoDetector class
    marker_corners, marker_ids, rejectedCandidates = detector.detectMarkers(gray_frame)

    if marker_ids is not None:
        # Censor the markers
        frame = manta.censor_marker(frame, marker_corners, "diamond")
        # Draw the markers on the frame for visual feedback
        cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

        # If at least 2 markers are detected, save the frame
        if len(marker_ids) >= 2 and time.time() >= next_snapshot_time:
            # Save the frame to the snapshots folder
            snapshot_filename = os.path.join(snapshot_dir, f'snapshot_{snapshot_counter:04d}.png')
            cv2.imwrite(snapshot_filename, gray_frame)
            print(f"Saved snapshot: {snapshot_filename}")
            snapshot_counter += 1

            # Save detected marker corners (for calibration)
            ret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray_frame, board)

            if charucoIds is not None and len(charucoCorners) > 3:
                all_charuco_corners.append(charucoCorners)
                all_charuco_ids.append(charucoIds)

            next_snapshot_time = time.time() + delay_time  # Wait another X ms before the next snapshot

    # Display the camera preview
    manta.resize_window_with_aspect_ratio("Camera Preview", frame)
    cv2.imshow("Camera Preview", frame)

    # Check if the user wants to quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

# Calibration using the saved snapshots
if all_charuco_corners and all_charuco_ids:
    # Perform camera calibration
    print("Calibrating...")
    result, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners, all_charuco_ids, board, gray_frame.shape[:2], None, None)#, flags = cv2.CALIB_USE_LU)
    
    # Save calibration data
    match CAMERA_TYPE:
        case "axis":
            np.savez('camera_calibration_axis.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        case "gopro":
            np.savez('camera_calibration_gopro.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

    print("Calibration complete.\n")
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)
else:
    print("Insufficient data for calibration.")
