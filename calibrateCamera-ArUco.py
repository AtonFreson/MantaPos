import cv2
import numpy as np
import os
import time
import mantaPosLib as manta
import shutil
import genMarker

# Set the selected camera: gopro, axis, or 4K.
CAMERA_TYPE = "4K"
CAMERA_INPUT = 2  # OBS Virtual Camera
CAMERA_RTSP_ADDR = "rtsp://admin:@169.254.178.12:554/"  # Overwrites CAMERA_INPUT if 4K selected

delay_time = 1  # 1s delay between capture

# ArUco marker settings
aruco_dict = cv2.aruco.getPredefinedDictionary(genMarker.ARUCO_DICT)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

# Initialize camera
if CAMERA_TYPE == "4K":
    cap = manta.RealtimeCapture(CAMERA_RTSP_ADDR)
else:
    cap = cv2.VideoCapture(CAMERA_INPUT)

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

cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)
if not cap.isOpened():
    print("Error: Could not open camera", "over RTSP" if CAMERA_TYPE == "4K" else CAMERA_INPUT)
    exit()

# Prepare to save captured snapshots, clear previous directory
snapshot_dir = './snapshots'
if os.path.exists(snapshot_dir):
    shutil.rmtree(snapshot_dir)
os.makedirs(snapshot_dir)

# Prepare to save the created calibration file
calibration_dir = './calibrations/camera_calibrations'
if not os.path.exists(calibration_dir):
    os.makedirs(calibration_dir)

# Display the marker grid
manta.display_marker_grid()

# Prepare object points for calibration (assuming a square ArUco marker layout)
marker_length = 0.0495  # The real-world size of each marker in meters
object_points = np.array([[0, 0, 0],
                          [marker_length, 0, 0],
                          [marker_length, marker_length, 0],
                          [0, marker_length, 0]], dtype=np.float32)

# Lists to store object points (3D) and image points (2D)
all_object_points = []  # 3D points in real world
all_image_points = []   # 2D points in image plane

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
    corners, ids, rejected_img_points = detector.detectMarkers(gray_frame)

    if ids is not None:
        # Censor the markers
        frame = manta.censor_marker(frame, corners, "diamond")
        # Draw the markers on the frame for visual feedback
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # If at least 2 markers are detected, save the frame
        if len(ids) >= 2 and time.time() >= next_snapshot_time:
            # Save the frame to the snapshots folder
            snapshot_filename = os.path.join(snapshot_dir, f'snapshot_{snapshot_counter:04d}.png')
            cv2.imwrite(snapshot_filename, gray_frame)
            print(f"Saved snapshot: {snapshot_filename}")
            snapshot_counter += 1

            # Save detected marker corners (for calibration)
            for corner in corners:
                all_image_points.append(corner)
                all_object_points.append(object_points)

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
if all_object_points and all_image_points:
    # Perform camera calibration
    print("Calibrating...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_object_points, all_image_points, gray_frame.shape, None, None, flags=cv2.CALIB_USE_LU
    )

    # Save calibration data
    if CAMERA_TYPE == "axis":
        camera_calibration_name = 'camera_calibration_axis.npz'
    elif CAMERA_TYPE == "gopro":
        camera_calibration_name = 'camera_calibration_gopro.npz'
    elif CAMERA_TYPE == "4K":
        camera_calibration_name = 'camera_calibration_4K.npz'
    else:
        camera_calibration_name = 'camera_calibration.npz'

    calibration_filename = os.path.join(calibration_dir, camera_calibration_name)
    np.savez(calibration_filename, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

    print("Calibration complete.\n")
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)
    print("Saved calibration data to", calibration_filename)
else:
    print("Insufficient data for calibration.")
