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

use_existing_images = True # Use existing images for calibration, found in snapshot_dir
delay_time = 1 # 1s delay between capture

squares_vertically = 7
squares_horizontally = 12
square_pixels = 140 # Pixel size of the chessboard squares
grid_edge = 30 # Pixel margin outside the ChArUco grid
marker_ratio = 0.7 # Marker ratio of square_length to fit within white squares; acceptable maximum 0.85, recommended 0.7 
square_length = 0.2975/6 * square_pixels/200 # Real world length of square in meters

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

        if idx%3 != 0:
            continue

        image_path = os.path.join(directory_path, filename)
        gray_frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if gray_frame is None:
            continue
        
        #gray_frame = manta.frame_corner_cutout(gray_frame, 0.3)  # Cut out the corners of the frame        
        #gray_frame = manta.frame_crop(gray_frame, 0.7)  # Crop the frame to remove fisheye edges

        # Detect markers in the frame using the ArUcoDetector class
        marker_corners, marker_ids, rejectedCandidates = detector.detectMarkers(gray_frame)
        charucoIds = []

        if marker_ids is not None and len(marker_ids) >= 2:
            # Refine detected markers for better accuracy
            detector.refineDetectedMarkers(gray_frame, board, marker_corners, marker_ids, rejectedCandidates)

            # Save detected marker corners (for calibration)
            num_corners, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray_frame, board)
            
            # If at least 6 corners are detected, save the frame
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

        #frame = manta.frame_corner_cutout(frame, 0.3)  # Cut out the corners of the frame 
        #frame = manta.frame_crop(frame, 0.7)  # Crop the frame to remove fisheye edges   

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers in the frame using the ArUcoDetector class
        marker_corners, marker_ids, rejectedCandidates = detector.detectMarkers(gray_frame)

        if marker_ids is not None and len(marker_ids) >= 2:
            # Refine detected markers for better accuracy
            detector.refineDetectedMarkers(gray_frame, board, marker_corners, marker_ids, rejectedCandidates)

            # Interpolate ChArUco corners from detected markers and image frame
            num_corners, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray_frame, board)
            
            # Censor the markers
            frame = manta.censor_marker(frame, marker_corners, "diamond")
            frame = manta.censor_charuco_board(frame, charucoCorners, marker_corners, 0.5)

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

    result, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board, gray_frame.shape[:2], None, None)
    
    # Load previously saved camera calibration data
    '''calibration_dir = './camera_calibrations'
    match CAMERA_TYPE:
        case "axis":
            calibration_data = np.load(os.path.join(calibration_dir,'camera_calibration_axis.npz'))
        case "axis_low":
            calibration_data = np.load(os.path.join(calibration_dir,'camera_calibration_axis_low.npz'))
        case "gopro":
            calibration_data = np.load(os.path.join(calibration_dir,'camera_calibration_gopro.npz'))
        case "4K":
            calibration_data = np.load(os.path.join(calibration_dir,'camera_calibration_4K.npz'))
    cameraMatrixInit = calibration_data['camera_matrix']
    distCoeffsInit = calibration_data['dist_coeffs']

    #cameraMatrixInit = np.array([[ 2000.,    0., gray_frame.shape[0]/2.], [    0., 2000., gray_frame.shape[1]/2.], [    0.,    0.,           1.]])
    #distCoeffsInit = np.zeros((5,1))
    #distCoeffsInit = np.array([[ 2.62894047e-01],[-8.51860457e-02],[ 4.99831259e-04],[-3.46592485e-03],[ 6.44091108e-01]])
    
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    #flags = (cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    #flags = (cv2.CALIB_RATIONAL_MODEL)
    
    (ret, camera_matrix, dist_coeffs,
     _, _,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=all_charuco_corners,
                      charucoIds=all_charuco_ids,
                      board=board,
                      imageSize=gray_frame.shape,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))'''

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
    print("Distortion coefficients:\n", dist_coeffs.T)
    print("Saved calibration data to", camera_calibration_name)
else:
    print("Insufficient data for calibration.")
