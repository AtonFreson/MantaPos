import cv2
import numpy as np
import os
import time
import mantaPosLib as manta
import shutil
import genMarker

# Set the selected camera: 4K, gopro or axis.
CAMERA_TYPE = "4K"
CAMERA_INPUT = 2 # Select OBS Virtual Camera
CAMERA_RTSP_ADDR = "rtsp://admin:@169.254.178.12:554/" # Overwrites CAMERA_INPUT if 4K selected

use_existing_images = False # Use existing images for calibration, found in snapshot_dir
delay_time = 1 # 1s delay between capture

squares_vertically = 5
squares_horizontally = 7
square_pixels = 200 # Pixel size of the chessboard squares
grid_edge = 30 # Pixel margin outside the ChArUco grid
marker_ratio = 0.7 # Marker ratio of square_length to fit within white squares; acceptable maximum 0.85, recommended 0.7 
square_length = 0.2975/6 # Real world length of square in meters

# Generate and display the marker grid
board, dictionary = genMarker.create_and_save_ChArUco_board(square_length, square_pixels, grid_edge, marker_ratio, squares_vertically, squares_horizontally)
if not use_existing_images: manta.display_marker_grid(board_type="ChArUco")

# Define the detector and parameters
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, params)

if not use_existing_images: 
    # Initialize camera
    if CAMERA_TYPE == "4K":
        #cap = cv2.VideoCapture(CAMERA_RTSP_ADDR)
        cap = manta.RealtimeCapture(CAMERA_RTSP_ADDR)
    else:
        cap = cv2.VideoCapture(CAMERA_INPUT)
    cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL) 

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
        print("Error: Could not open camera", "over RTSP" if CAMERA_TYPE == "4K" else CAMERA_INPUT)
        exit()

# Prepare to save captured snapshots, clear previous directory
snapshot_dir = './snapshots'
if not use_existing_images:
    if os.path.exists(snapshot_dir):
        shutil.rmtree(snapshot_dir)
    os.makedirs(snapshot_dir)

# Prepare to save the created calibration file
calibration_dir = './camera_calibrations'
if not os.path.exists(calibration_dir):
    os.makedirs(calibration_dir)

# Lists to store object points (3D) and image points (2D)
all_charuco_ids = []
all_charuco_corners = []

def load_images_and_detect_ChArUco(directory_path):
    all_charuco_ids = []
    all_charuco_corners = []
    image_files = [f for f in os.listdir(directory_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)
    for idx, filename in enumerate(image_files):
        image_path = os.path.join(directory_path, filename)
        gray_frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray_frame is None:
            continue

        # Detect markers in the frame using the ArUcoDetector class
        marker_corners, marker_ids, rejectedCandidates = detector.detectMarkers(gray_frame)

        if marker_ids is not None and len(marker_ids) > 0:
        # Refine detected markers for better accuracy
            detector.refineDetectedMarkers(
                image=gray_frame,
                board=board,
                detectedCorners=marker_corners,
                detectedIds=marker_ids,
                rejectedCorners=rejectedCandidates
            )

            # Interpolate ChArUco corners
            num_corners, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
                markerCorners=marker_corners,
                markerIds=marker_ids,
                image=gray_frame,
                board=board
            )

            # If at least 2 markers and 6 corners are detected, save the frame
            if len(marker_ids) >= 2:
                if charucoIds is not None and len(charucoCorners) >= 6:
                    all_charuco_corners.append(charucoCorners)
                    all_charuco_ids.append(charucoIds)

        # Print progress
        print(f"Processed image {idx + 1} of {total_images}. Detected {len(charucoIds)} ChArUco corners.")
    return all_charuco_corners, all_charuco_ids, gray_frame

if use_existing_images:
    print("Using existing images for calibration.")
    all_charuco_corners, all_charuco_ids, gray_frame = load_images_and_detect_ChArUco(snapshot_dir)
else:
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

        if marker_ids is not None and len(marker_ids) >= 2:
        # Refine detected markers for better accuracy
            detector.refineDetectedMarkers(
                image=gray_frame,
                board=board,
                detectedCorners=marker_corners,
                detectedIds=marker_ids,
                rejectedCorners=rejectedCandidates
            )

            # Interpolate ChArUco corners
            num_corners, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
                markerCorners=marker_corners,
                markerIds=marker_ids,
                image=gray_frame,
                board=board
            )
            
            # Censor the markers
            frame = manta.censor_marker(frame, marker_corners, "diamond")
            # Draw the markers on the frame for visual feedback
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

            # If at least 6 corners are detected, save the frame
            if charucoIds is not None and len(charucoCorners) >= 6 and time.time() >= next_snapshot_time:
                # Save the frame to the snapshots folder
                snapshot_filename = os.path.join(snapshot_dir, f'snapshot_{snapshot_counter:04d}.png')
                cv2.imwrite(snapshot_filename, gray_frame)
                print(f"Saved snapshot: {snapshot_filename}")
                snapshot_counter += 1

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

# Calibration using the collected snapshots
if all_charuco_corners and all_charuco_ids:
    # Perform camera calibration
    print("Calibrating...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=gray_frame.shape,
        cameraMatrix=None,
        distCoeffs=None)
    
    # Save calibration data
    if CAMERA_TYPE == "axis":
        camera_calibration_name = 'camera_calibration_axis.npz'
    elif CAMERA_TYPE == "gopro":
        camera_calibration_name = 'camera_calibration_gopro.npz'
    elif CAMERA_TYPE == "4K":
        camera_calibration_name = 'camera_calibration_4K.npz'
    else:
        camera_calibration_name = 'camera_calibration.npz'
    np.savez(os.path.join(calibration_dir, camera_calibration_name), camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

    print("Calibration complete.\n")
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)
    print("Saved calibration data to", camera_calibration_name)
else:
    print("Insufficient data for calibration.")
