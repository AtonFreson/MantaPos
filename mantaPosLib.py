import cv2
from contextlib import contextmanager
import numpy as np
from threading import Thread, Lock
import time
from datetime import datetime
import os
import sys
import re
from scipy.spatial.transform import Rotation as R
import pickle
import struct
from multiprocessing import shared_memory
from pressureSensorCalibration import PiecewiseLinearAlternate

# Function to extract the numerical index from the filename
def extract_index(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

# Function to display ArUco markers in a grid layout / ChArUco-board
def display_marker_grid(num_markers=0, marker_size=200, border_size=25, board_type="ArUco"):
    # Folder containing marker images
    calibration_image_dir = './markers'

    match board_type:
        case "ArUco":
            # Get the list of marker image files and sort them numerically by index
            marker_image_files = [os.path.join(calibration_image_dir, f) for f in os.listdir(calibration_image_dir) if f.endswith('.png') and f.find('aruco_marker') != -1]
            marker_image_files.sort(key=extract_index)

            # Extract the first marker (smallest index) for reference
            first_marker = extract_index(os.path.basename(marker_image_files[0])) if marker_image_files else None
            
            if num_markers:
                marker_image_files = marker_image_files[0:num_markers]

            # Check for gaps in the sequence
            sorted_indices = [extract_index(os.path.basename(f)) for f in marker_image_files]
            sequential_marker_files = []

            for i in range(len(sorted_indices) - 1):
                sequential_marker_files.append(marker_image_files[i])
                if sorted_indices[i + 1] != sorted_indices[i] + 1:
                    print(f"Gap detected after marker {sorted_indices[i]}. Displaying up to this point.")
                    break
            else:
                # Add the last marker if no gap was found till the end
                sequential_marker_files.append(marker_image_files[-1])

            marker_images = [cv2.imread(f) for f in sequential_marker_files]
            grid_size = int(np.ceil(np.sqrt(len(marker_images))))  # Grid size

            # Resize all markers to the same size
            marker_images = [cv2.resize(img, (marker_size, marker_size)) for img in marker_images]

            # Add white border to each image
            padded_marker_images = [
                cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, 
                                cv2.BORDER_CONSTANT, value=[255, 255, 255])
                for img in marker_images
            ]

            # Create a grid
            rows = []
            for i in range(0, len(padded_marker_images), grid_size):
                row = padded_marker_images[i:i + grid_size]

                if len(row) < grid_size:
                    # Calculate the missing number of images
                    missing_images = grid_size - len(row)

                    # Create white padding images of the same size as the marker images
                    white_padding = np.full(
                        (marker_size + 2 * border_size, marker_size + 2 * border_size, 3), 255, dtype=np.uint8)

                    # Add the white padding to the row
                    row += [white_padding] * missing_images

                rows.append(np.hstack(row))

            # Vertically stack all rows
            grid_image = np.vstack(rows)

            # Show the grid image in a resizable window
            cv2.imshow('ArUco Marker Grid', grid_image)
            cv2.resizeWindow('ArUco Marker Grid', grid_image.shape[1], grid_image.shape[0])

            # Return grid size for reference
            return grid_size, first_marker

        case "ChArUco":
            calibration_image_file = 'ChArUco_Marker.png'
            grid_image = cv2.imread(os.path.join(calibration_image_dir, calibration_image_file))
            cv2.imshow('ChArUco Marker Grid', grid_image)
            cv2.resizeWindow('ChArUco Marker Grid', grid_image.shape[1], grid_image.shape[0])

            return None, None

# Function to make sure the displayed window follows the original aspect ratio when resized
def resize_window_with_aspect_ratio(window_name, frame):
    # Get current window size
    frame_height, frame_width = frame.shape[:2]
    window_width, window_height = cv2.getWindowImageRect(window_name)[2:4]
    if window_height == 0: window_height=1
    if window_width == 0: window_width=1
    
    # Calculate the aspect ratio of the frame
    aspect_ratio = frame_width / float(frame_height)

    # Adjust the window size to maintain the aspect ratio
    if (window_width+1) / float(window_height) > aspect_ratio:
        # If the window is too wide, adjust the width
        new_width = int(window_height * aspect_ratio)
        cv2.resizeWindow(window_name, new_width, window_height)
    else:
        # If the window is too tall, adjust the height
        new_height = int(window_width / aspect_ratio)
        cv2.resizeWindow(window_name, window_width, new_height)

# Censor the markers so that the camera doesn't pick them up a second time. blur, cross, fill or diamond
def censor_marker(frame, corners, type = "blur"):
    for corner in corners:
        match type:
            case "blur":
                # Get corner points and convert to integer
                pts = corner.reshape(4, 2).astype(int)

                # Calculate the width and height of the marker
                width = int(np.linalg.norm(pts[0] - pts[1]))  # Distance between top-left and top-right
                height = int(np.linalg.norm(pts[0] - pts[3]))  # Distance between top-left and bottom-left
                
                # Determine the kernel size based on the size of the marker
                # Set a base kernel size factor and a maximum kernel size
                kernel_size_factor = 5  # Adjust this factor to change the strength of the blur
                max_kernel_size = 31  # Maximum size of the kernel
                kernel_size = min(max(3, width * kernel_size_factor), max_kernel_size)  # Base size

                # Ensure the kernel size is odd
                if kernel_size % 2 == 0:
                    kernel_size += 1

                # Create a mask for the marker
                mask = np.zeros(frame.shape, dtype=np.uint8)
                cv2.fillConvexPoly(mask, pts, (255, 255, 255))

                # Apply the blur to the entire frame
                blurred_frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)  # Dynamic kernel size

                # Combine the original frame and the blurred frame using the mask
                frame = np.where(mask == np.array([255, 255, 255]), blurred_frame, frame)

            case "cross":
                # Modify the markers by drawing a red diagonal line
                # Get corner points
                pts = corner.reshape(4, 2).astype(int)

                # Get min and max coordinates of the marker
                x_min, y_min = np.min(pts, axis=0)
                x_max, y_max = np.max(pts, axis=0)

                # Draw a diagonal red line across the marker
                cv2.line(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 10)  # Color: Red, Thickness: 10
                cv2.line(frame, (x_min, y_max), (x_max, y_min), (0, 0, 255), 10)  # Color: Red, Thickness: 10

            case "fill":
                # Get corner points and convert to integer
                pts = corner.reshape(4, 2).astype(int)
                # Create a mask for the marker
                cv2.fillConvexPoly(frame, pts, (0, 0, 0))  # Fill with black

            case "diamond":
                # Get corner points and convert to integer
                pts = corner.reshape(4, 2).astype(int)

                # Create a mask for the marker
                mask = np.zeros(frame.shape, dtype=np.uint8)
                cv2.fillConvexPoly(mask, pts, (255, 255, 255))  # Fill the detected marker area

                # Create a smaller inner polygon by scaling down the corner points
                inner_pts = np.array([
                    (int((pts[0][0] + pts[1][0]) / 2), int((pts[0][1] + pts[1][1]) / 2)),
                    (int((pts[1][0] + pts[2][0]) / 2), int((pts[1][1] + pts[2][1]) / 2)),
                    (int((pts[2][0] + pts[3][0]) / 2), int((pts[2][1] + pts[3][1]) / 2)),
                    (int((pts[3][0] + pts[0][0]) / 2), int((pts[3][1] + pts[0][1]) / 2))
                ], dtype=int)

                # Fill the center polygon with black
                cv2.fillConvexPoly(frame, inner_pts, (0, 0, 0))  # Fill the inner area with black

    return frame

# Censor the ChArUco board by drawing a black rectangle over it
def censor_charuco_board(frame, charucoCorners, marker_corners, margin_percentage=0.1):
    # Check if charucoCorners is valid
    if (charucoCorners is None or len(charucoCorners) == 0) and (marker_corners is None or len(marker_corners) == 0):
        return frame

    # Collect all corner points from charucoCorners and marker_corners
    all_corners = []

    if charucoCorners is not None and len(charucoCorners) > 0:
        all_corners.extend(charucoCorners.reshape(-1, 2))

    if marker_corners is not None and len(marker_corners) > 0:
        for marker in marker_corners:
            all_corners.extend(marker.reshape(-1, 2))

    # Convert to numpy array of float coordinates
    corners = np.array(all_corners, dtype=np.float32)

    # Compute the convex hull of all the corner points
    hull = cv2.convexHull(corners)

    # Compute the centroid of the convex hull
    M = cv2.moments(hull)
    if M['m00'] == 0:
        # Avoid division by zero
        cX = np.mean(hull[:, 0, 0])
        cY = np.mean(hull[:, 0, 1])
    else:
        cX = M['m10'] / M['m00']
        cY = M['m01'] / M['m00']
    centroid = np.array([cX, cY])

    # Expand the hull points by the margin_percentage
    expanded_hull = []
    for point in hull[:, 0, :]:
        vector = point - centroid
        expanded_point = centroid + vector * (1 + margin_percentage)
        expanded_hull.append(expanded_point)
    expanded_hull = np.array(expanded_hull, dtype=np.float32)

    # Clip the points to be within image boundaries
    height, width = frame.shape[:2]
    expanded_hull[:, 0] = np.clip(expanded_hull[:, 0], 0, width - 1)
    expanded_hull[:, 1] = np.clip(expanded_hull[:, 1], 0, height - 1)

    # Convert to integer coordinates for drawing
    expanded_hull = expanded_hull.astype(np.int32)

    # Fill the expanded convex hull with black color to censor the board area
    cv2.fillConvexPoly(frame, expanded_hull, color=(0, 0, 0))

    return frame

# Function to calculate the global position and rotation of the camera based on multiple markers, 
# and display it overlaid on the video feed, together with the standard deviation of the estimates  
def display_position(frame, tvec_list, rvec_list, marker_pos_rot, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8,
                     text_color=(0, 255, 0), thickness=1, alpha=0.5, rect_padding=(10, 10, 600, 150)):
    
    # Display only if tags are visible
    if len(tvec_list) == 0:
        return
    
    camera_positions = []
    euler_angles = []

    for i in range(len(rvec_list)):
        # From solvePnP: Pose of marker in camera coordinate system
        rvec = rvec_list[i]
        tvec = tvec_list[i]

        # Convert rotation vector to rotation matrix
        R_marker_camera, _ = cv2.Rodrigues(rvec)
        t_marker_camera = tvec.reshape((3, 1))

        # Invert the transformation to get pose of camera in marker coordinate system
        R_camera_marker = R_marker_camera.T
        t_camera_marker = -R_marker_camera.T @ t_marker_camera

        # Get marker pose in global coordinates
        x, y, z = marker_pos_rot[i][0]
        roll_deg, pitch_deg, yaw_deg = marker_pos_rot[i][1]

        # Convert roll, pitch, yaw from degrees to radians
        roll = np.deg2rad(roll_deg)
        pitch = np.deg2rad(pitch_deg)
        yaw = np.deg2rad(yaw_deg)

        # Compute rotation matrix from global to marker coordinate system
        R_marker_global = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        t_marker_global = np.array([x, y, z]).reshape((3, 1))

        # Compute camera pose in global coordinates
        R_camera_global = R_marker_global @ R_camera_marker
        t_camera_global = R_marker_global @ t_camera_marker + t_marker_global

        # Convert rotation matrix to Euler angles (in degrees)
        euler_angles_global = R.from_matrix(R_camera_global).as_euler('xyz', degrees=True)

        # Store camera position and orientation
        camera_positions.append(t_camera_global.flatten())
        euler_angles.append(euler_angles_global)

    # Now, compute mean and standard deviation
    camera_positions = np.array(camera_positions)
    position = np.mean(camera_positions, axis=0)
    position_std = np.std(camera_positions, axis=0)

    euler_angles = np.array(euler_angles)
    rotation = np.mean(euler_angles, axis=0)
    rotation_std = np.std(euler_angles, axis=0)

    # Create a position text with fixed-width formatting to prevent text shifting
    position_text = (f"Pos: X={position[0]: >+6.3f}m, Y={-position[1]: >+6.3f}m, Z={-position[2]: >+6.3f}m")
    rotation_text = (f"Rot: R={rotation[0]: >+6.3f}', P={rotation[1]: >+6.3f}', Y={rotation[2]: >+6.3f}'")
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

# Function to display a horizontal balance bar between the depth value of the left and right sides of the assembly
def display_balance_bar(frame, depth_left, depth_right, dev_max=120, bar_height=80, font=cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale=1.0, text_color=(0, 0, 0), thickness=2, alpha=0.7):
    if depth_left is None or depth_right is None:
        return

    frame_h, frame_w = frame.shape[:2]
    bar_width = int(frame_w * 0.8)
    bar_x = (frame_w - bar_width) // 2
    bar_y = frame_h - bar_height - 10

    diff = depth_left - depth_right 
    diff_mm = abs(diff) * 1000

    overlay = frame.copy()
    # Draw full bar area as red (outer zone)
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 255), -1)
    
    # Define zones dimensions and draw middle zones: yellow on sides, green at center.
    green_half = round(bar_width / 2 * 0.2 / 10) * 10
    yellow_half = round(bar_width / 2 * 0.4 / 10) * 10
    cv2.rectangle(overlay, (bar_x + (bar_width//2 - yellow_half), bar_y),
                  (bar_x + (bar_width//2 - green_half), bar_y + bar_height), (0, 255, 255), -1)
    cv2.rectangle(overlay, (bar_x + (bar_width//2 + green_half), bar_y),
                  (bar_x + (bar_width//2 + yellow_half), bar_y + bar_height), (0, 255, 255), -1)
    cv2.rectangle(overlay, (bar_x + (bar_width//2 - green_half), bar_y),
                  (bar_x + (bar_width//2 + green_half), bar_y + bar_height), (0, 255, 0), -1)

    # Calculate indicator horizontal position with offset based on depth difference.
    center_x = bar_x + bar_width // 2
    normalized = max(-1, min(diff / (dev_max/1000), 1))
    pixel_offset = int(normalized * (bar_width // 2))
    indicator_x = center_x + pixel_offset

    # Define indicator rectangle dimensions.
    indicator_center = (indicator_x, bar_y + bar_height // 2)
    rect_height = int(bar_height * 1.5)  # 50% taller than the bar
    rect_width = int(min(bar_width * 0.01, bar_height * 0.5))   # width is 50% of bar height or 1% of bar width
    top_left = (indicator_center[0] - rect_width//2, indicator_center[1] - rect_height//2)
    bottom_right = (indicator_center[0] + rect_width//2, indicator_center[1] + rect_height//2)
    radius = rect_width // 2  # round corners based on width

    # Draw the filled black rounded rectangle.
    x1, y1 = top_left
    x2, y2 = bottom_right
    color = (230, 230, 230)
    cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, -1)
    cv2.circle(overlay, (x1+radius, y1+radius), radius, color, -1)
    cv2.circle(overlay, (x2-radius, y1+radius), radius, color, -1)
    cv2.circle(overlay, (x1+radius, y2-radius), radius, color, -1)
    cv2.circle(overlay, (x2-radius, y2-radius), radius, color, -1)

    # Draw white outline around the rectangle.
    color = (255, 255, 255)
    cv2.line(overlay, (x1+radius, y1), (x2-radius, y1), color, thickness, cv2.LINE_AA)
    cv2.line(overlay, (x1+radius, y2), (x2-radius, y2), color, thickness, cv2.LINE_AA)
    cv2.line(overlay, (x1, y1+radius), (x1, y2-radius), color, thickness, cv2.LINE_AA)
    cv2.line(overlay, (x2, y1+radius), (x2, y2-radius), color, thickness, cv2.LINE_AA)
    cv2.ellipse(overlay, (x1+radius, y1+radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(overlay, (x2-radius, y1+radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(overlay, (x1+radius, y2-radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(overlay, (x2-radius, y2-radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)

    # Format texts and place them in the bar.
    left_text = f"{depth_left:.3f}m"
    right_text = f"{depth_right:.3f}m"
    diff_text = f"{diff_mm:.1f}mm"
    (_, th_left), _ = cv2.getTextSize(left_text, font, font_scale, thickness)
    (tw_right, _), _ = cv2.getTextSize(right_text, font, font_scale, thickness)
    (tw_diff, _), _ = cv2.getTextSize(diff_text, font, font_scale, thickness)
    text_y = bar_y + (bar_height + th_left) // 2
    cv2.putText(overlay, left_text, (bar_x + 5, text_y), font, font_scale, text_color, thickness)
    cv2.putText(overlay, right_text, (bar_x + bar_width - tw_right - 5, text_y), font, font_scale, text_color, thickness)
    cv2.putText(overlay, diff_text, (bar_x + (bar_width - tw_diff) // 2, text_y), font, font_scale, text_color, thickness)
    
    # Add centered text above the balance bar.
    balancing_text = "Winch depth balancing"
    (text_width, _), _ = cv2.getTextSize(balancing_text, font, font_scale, thickness)
    text_x = bar_x + (bar_width - text_width) // 2
    text_y_above = bar_y - 25
    cv2.putText(overlay, balancing_text, (text_x, text_y_above), font, font_scale, (0, 255, 0), thickness)

    # Add text on the left above the bar
    left_text_label = "Lower Main"
    (left_text_width, _), _ = cv2.getTextSize(left_text_label, font, font_scale, thickness)
    left_text_x = bar_x + 10
    cv2.putText(overlay, left_text_label, (left_text_x, text_y_above), font, font_scale, (0, 255, 0), thickness)

    # Add text on the right above the bar
    right_text_label = "Lower Sec"
    (right_text_width, _), _ = cv2.getTextSize(right_text_label, font, font_scale, thickness)
    right_text_x = bar_x + bar_width - right_text_width - 10
    cv2.putText(overlay, right_text_label, (right_text_x, text_y_above), font, font_scale, (0, 255, 0), thickness)
    
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

# Computes the standard deviations (errors) for each pose parameter in the global coordinate system.
def compute_pose_errors_global(rvec, tvec, object_points, image_points, camera_matrix, dist_coeffs,
                               marker_pos, marker_rot):
    """
    Parameters:
        rvec (np.ndarray): Rotation vector (3x1 or 1x3).
        tvec (np.ndarray): Translation vector (3x1 or 1x3).
        object_points (np.ndarray): 3D points in the object coordinate space (Nx3).
        image_points (np.ndarray): Corresponding 2D points in the image plane (Nx2).
        camera_matrix (np.ndarray): Camera intrinsic matrix.
        dist_coeffs (np.ndarray): Distortion coefficients.
        marker_pos (list or np.ndarray): Marker position in the global coordinate system (3 elements).
        marker_rot (list or np.ndarray): Marker rotation in degrees (roll, pitch, yaw).

    Returns:
        position_std_global (np.ndarray): Standard deviations for X, Y, Z in global coordinates (in meters).
        rotation_std_deg_global (np.ndarray): Standard deviations for roll, pitch, yaw in global coordinates (in degrees).
    """
    import numpy as np
    import cv2
    from scipy.spatial.transform import Rotation as R

    # Flatten rvec and tvec
    rvec = rvec.flatten()
    tvec = tvec.flatten()

    # Prepare pose parameters
    pose_params = np.hstack((rvec, tvec))

    # Define function to project points
    def project_points(pose_params, object_points):
        rvec = pose_params[0:3].reshape(3, 1)
        tvec = pose_params[3:6].reshape(3, 1)
        projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        return projected_points.reshape(-1, 2)

    # Compute base projected points
    base_projected_points = project_points(pose_params, object_points)

    # Number of parameters and points
    num_params = 6
    num_points = object_points.shape[0]

    # Initialize Jacobian matrix
    J = np.zeros((num_points * 2, num_params))

    delta = 1e-6  # Small perturbation

    # Compute Jacobian numerically
    for i in range(num_params):
        pose_params_perturbed = pose_params.copy()
        pose_params_perturbed[i] += delta
        projected_points_perturbed = project_points(pose_params_perturbed, object_points)
        diff = (projected_points_perturbed - base_projected_points) / delta  # Numerical derivative
        # Flatten the differences
        diff = diff.flatten()
        J[:, i] = diff

    # Compute residuals
    residuals = (image_points - base_projected_points).flatten()

    # Degrees of freedom: number of observations minus number of parameters
    dof = max(0, len(residuals) - num_params)

    # Compute variance of the measurement errors
    if dof > 0:
        sigma2 = np.sum(residuals ** 2) / dof
    else:
        sigma2 = np.var(residuals)

    # Compute covariance matrix
    try:
        Cov_local = np.linalg.inv(J.T @ J) * sigma2
        # Extract standard deviations
        param_std = np.sqrt(np.diag(Cov_local))
        rotation_std_rad_local = param_std[0:3]
        position_std_local = param_std[3:6]
    except np.linalg.LinAlgError:
        # Handle singular matrix
        rotation_std_rad_local = np.zeros(3)
        position_std_local = np.zeros(3)

    # Now, transform the covariance matrix to global coordinate system

    # First, compute the transformation matrices
    # Rotation and translation from marker to global
    roll_deg, pitch_deg, yaw_deg = marker_rot
    roll_rad, pitch_rad, yaw_rad = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    R_marker_global = R.from_euler('xyz', [roll_rad, pitch_rad, yaw_rad]).as_matrix()
    t_marker_global = np.array(marker_pos).reshape(3, 1)

    # Rotation and translation from camera to marker (from solvePnP)
    R_camera_marker, _ = cv2.Rodrigues(rvec)
    t_camera_marker = tvec.reshape(3, 1)

    # Invert to get transformation from marker to camera
    R_marker_camera = R_camera_marker.T
    t_marker_camera = -R_camera_marker.T @ t_camera_marker

    # Total transformation from camera to global
    R_camera_global = R_marker_global @ R_camera_marker
    t_camera_global = R_marker_global @ t_camera_marker + t_marker_global

    # Jacobian of the transformation with respect to local pose parameters
    # Since the transformation is linear for translation and rotational parameters are small, we can approximate
    # the errors in global coordinates as:

    # For position:
    position_std_global = np.sqrt(np.sum((R_marker_global @ np.diag(position_std_local))**2, axis=1))

    # For rotation:
    # Compute the rotation error propagation
    # For small angles, rotation errors can be transformed using the rotation matrix
    rotation_std_rad_global = np.sqrt(np.sum((R_marker_global @ np.diag(rotation_std_rad_local))**2, axis=1))

    # Convert rotation errors to degrees
    rotation_std_deg_global = np.rad2deg(rotation_std_rad_global)

    return position_std_global, rotation_std_deg_global

# Function to display the position and orientation of the camera in the global coordinate system
def display_position_ChArUco(frame, tvec_list, rvec_list, marker_pos_rot, camera_matrix, dist_coeffs, object_points_all, image_points_all,
                     font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, text_color=(0, 255, 0), thickness=1, alpha=0.5,
                     rect_padding=(10, 10, 600, 150)):
    
    if len(tvec_list) == 0:
        return
    
    camera_positions = []
    euler_angles = []
    position_stds = []
    rotation_stds = []

    for i in range(len(rvec_list)):
        # From solvePnP: Pose of marker in camera coordinate system
        rvec = rvec_list[i]
        tvec = tvec_list[i]

        # Marker position and rotation in global coordinates
        marker_pos, marker_rot = marker_pos_rot[i]
        x, y, z = marker_pos
        roll_deg, pitch_deg, yaw_deg = marker_rot

        # Convert rotation vector to rotation matrix
        R_marker_camera, _ = cv2.Rodrigues(rvec)
        t_marker_camera = tvec.reshape((3, 1))

        # Invert the transformation to get pose of camera in marker coordinate system
        R_camera_marker = R_marker_camera.T
        t_camera_marker = -R_marker_camera.T @ t_marker_camera

        # Compute rotation matrix from global to marker coordinate system
        roll_rad, pitch_rad, yaw_rad = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
        R_marker_global = R.from_euler('xyz', [roll_rad, pitch_rad, yaw_rad]).as_matrix()
        t_marker_global = np.array([x, y, z]).reshape((3, 1))

        # Compute camera pose in global coordinates
        R_camera_global = R_marker_global @ R_camera_marker
        t_camera_global = R_marker_global @ t_camera_marker + t_marker_global

        # Convert rotation matrix to Euler angles (in degrees)
        euler_angles_global = R.from_matrix(R_camera_global).as_euler('xyz', degrees=True)

        # Store camera position and orientation
        camera_positions.append(t_camera_global.flatten())
        euler_angles.append(euler_angles_global)

        # Compute pose errors in global coordinates
        position_std_global, rotation_std_deg_global = compute_pose_errors_global(
            rvec, tvec, object_points_all, image_points_all, camera_matrix, dist_coeffs, marker_pos, marker_rot
        )

        position_stds.append(position_std_global)
        rotation_stds.append(rotation_std_deg_global)

    # Now, compute mean and standard deviation
    camera_positions = np.array(camera_positions)
    position = np.mean(camera_positions, axis=0)
    position_std = np.mean(position_stds, axis=0)

    euler_angles = np.array(euler_angles)
    rotation = np.mean(euler_angles, axis=0)
    rotation_std = np.mean(rotation_stds, axis=0)

    # Create a position text with fixed-width formatting to prevent text shifting
    position_text = (f"Pos: X={position[0]: >+6.3f}m, Y={-position[1]: >+6.3f}m, Z={-position[2]: >+6.3f}m")
    rotation_text = (f"Rot: R={rotation[0]: >+6.3f}', P={rotation[1]: >+6.3f}', Y={rotation[2]: >+6.3f}'")
    position_std_text = (f"-Err:X={position_std[0]*1000: >6.3f}mm,Y={position_std[1]*1000: >6.3f}mm,Z={position_std[2]*1000: >6.3f}mm")
    rotation_std_text = (f"-Err: R={rotation_std[0]: >6.3f}', P={rotation_std[1]: >6.3f}', Y={rotation_std[2]: >6.3f}'")

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
    
    return position, position_std, rotation, rotation_std

# Function to calculate the camera position and orientation in the global coordinate system
def calculate_camera_position(tvec, rvec, marker_pos_rot):
    # Marker position and rotation in global coordinates
    marker_pos, marker_rot = marker_pos_rot
    x, y, z = marker_pos
    roll_deg, pitch_deg, yaw_deg = marker_rot
    
    # Convert rotation vector to rotation matrix
    R_marker_camera, _ = cv2.Rodrigues(rvec)
    t_marker_camera = tvec.reshape((3, 1))

    # Invert the transformation to get pose of camera in marker coordinate system
    R_camera_marker = R_marker_camera.T
    t_camera_marker = -R_marker_camera.T @ t_marker_camera

    # Compute rotation matrix from global to marker coordinate system
    roll_rad, pitch_rad, yaw_rad = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    R_marker_global = R.from_euler('xyz', [roll_rad, pitch_rad, yaw_rad]).as_matrix()
    t_marker_global = np.array([x, y, z]).reshape((3, 1))

    # Compute camera pose in global coordinates
    R_camera_global = R_marker_global @ R_camera_marker
    t_camera_global = R_marker_global @ t_camera_marker + t_marker_global

    # Convert rotation matrix to Euler angles (in degrees)
    euler_angles_global = R.from_matrix(R_camera_global).as_euler('xyz', degrees=True)

    return t_camera_global.flatten(), euler_angles_global

# Function to display the position and orientation of the camera in the global coordinate system
def display_camera_position(frame, position, rotation, ref_pos, ref_rot, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, 
                            text_color=(0, 255, 0), thickness=1, alpha=0.65, rect_padding=(10, 10, 600, 150)):
    
    # Create a position text with fixed-width formatting to prevent text shifting
    position_text = (f"Pos: X={position[0]: >+6.3f}m, Y={position[1]: >+6.3f}m, Z={position[2]: >+6.3f}m")
    if ref_pos is not None:
        position_err_text = (f"Err.: X={ref_pos[0]-position[0]: >+6.3f}m, Y={ref_pos[1]-position[1]: >+6.3f}m, Z={ref_pos[2]-position[2]: >+6.3f}m")
    else:
        position_err_text = "Err.: N/A"
        
    rotation_text = (f"Rot: R={rotation[0]: >+6.3f}', P={rotation[1]: >+6.3f}', Y={rotation[2]: >+6.3f}'")
    if ref_rot is not None:
        rotation_err_text = (f"Err.: R={ref_rot[0]-rotation[0]: >+6.3f}', P={ref_rot[1]-rotation[1]: >+6.3f}', Y={ref_rot[2]-rotation[2]: >+6.3f}'")
    else:
        rotation_err_text = "Err.: N/A"

    # Unpack rectangle bounds
    x, y, w, h = rect_padding

    # Create a copy of the frame for overlay
    overlay = frame.copy()

    # Draw the rectangle
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)

    # Put text on the overlay
    cv2.putText(overlay, position_text, (x+20, y + int(h / 4.5)), font, font_scale, text_color, thickness)
    cv2.putText(overlay, position_err_text, (x+20, y + int(h / 2.4)), font, font_scale, text_color, thickness)
    cv2.putText(overlay, rotation_text, (x+20, y + int(h / 1.4)), font, font_scale, text_color, thickness)
    cv2.putText(overlay, rotation_err_text, (x+20, y + int(h / 1.1)), font, font_scale, text_color, thickness)
    
    # Apply the overlay
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

# Class to read frames from an RTSP camera stream, with minimal buffering for real-time applications. Wrapper around OpenCV's VideoCapture class.
# Set ocr_timestamp=True to read the timestamp along with the frame.
class RealtimeCapture:
    def __init__(self, rtsp_url, disable=False, ocr_timestamp = False):
        self.ocr_timestamp = ocr_timestamp
        self.disable = disable
        if self.disable:
            return
        
        self.rtsp_url = rtsp_url
        self._cap = cv2.VideoCapture(rtsp_url)
        if not self._cap.isOpened():
            raise Exception(f"Failed to open RTSP stream: {self.rtsp_url}")
        self._current_frame = None
        self._current_timestamp = None
        self._frame_lock = Lock()
        self._running = False
        self._capture_thread = None
        # Configure camera buffer size
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Instance variables for millisecond offset refinement
        self._frame_ms_offset = 999
        self._prev_ocr_second = None
        
        # Variables for FPS calculation
        self._fps_window = [10]
        self._current_fps = 0.0
        self._last_frame_time = None

        # Start automatically
        self.start()

    def get(self, prop_id):
        if self.disable:
            return None
        return self._cap.get(prop_id)

    def set(self, prop_id, value):
        if self.disable:
            return None
        return self._cap.set(prop_id, value)

    def isOpened(self):
        if self.disable:
            return True
        return self._cap.isOpened()

    def release(self):
        if self.disable:
            return None
        self.stop()

    def start(self):
        if self.disable:
            return None
        if not self._running:
            self._running = True
            self._capture_thread = Thread(target=self._capture_frames, daemon=True)
            self._capture_thread.start()
            time.sleep(0.5)  # Short wait for first frame

    def stop(self):
        if self.disable:
            return None
        self._running = False
        if self._cap:
            self._cap.release()
        if self._capture_thread:
            self._capture_thread.join()

    def _capture_frames(self):
        if self.disable:
            return None
        
        roi_coords = [
            (6,3211,        48,74//2),   # ROI for the first digit, pos of top right corner then width/height
            (6,3211+74//2,  48,74//2)       # ROI for the second digit, pos of top right corner then width/height
        ]
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                # calculate FPS from frame interval
                frame_time = datetime.now().timestamp()
                if self._last_frame_time is not None:
                    interval = frame_time - self._last_frame_time
                    if interval > 0:
                        fps = 1.0 / interval
                        self._fps_window.append(fps)
                        if len(self._fps_window) > 20:
                            self._fps_window.pop(0)
                self._last_frame_time = frame_time

                # OCR timestamp reading
                if self.ocr_timestamp:
                    # Get current computer time
                    comp_time = datetime.now()
                    comp_sec = comp_time.second
                    comp_ms = comp_time.microsecond // 1000
                    
                    #time_test = datetime.now().timestamp()
                    try:
                        ocr = TemplateOCR(template_dir="ref_digits")
                        ocr_sec, matching_scores, roi = ocr.read_number(frame, roi_coords)
                        for score in matching_scores:
                            if score < 0.7:
                                print("Warning: Low matching score detected.")
                                print("Detected number:", ocr_sec)
                                print("Matching scores for each digit:", matching_scores)
                                break
                        cv2.imwrite("roi.png", roi)
                    except Exception:
                        ocr_sec = comp_sec
                        self._frame_ms_offset = 999
                        print(f"Error: OCR failed to read seconds, saving image to 'roi.png'. Using computer time.")
                        print(f"Given error: {sys.exc_info()} at location: {sys.exc_info()[2].tb_lineno}")
                        cv2.imwrite("roi.png", roi)
                    #print(f"OCR time: {datetime.now().timestamp() - time_test}")

                    # Adjust minute if computer seconds is near 0 but OCR seconds is high
                    if comp_sec < 10 and ocr_sec >= 50:
                        comp_sec += 60
                        adjusted_hour = comp_time.hour if comp_time.minute > 0 else (comp_time.hour - 1) % 24
                        comp_time = comp_time.replace(hour=adjusted_hour, minute=(comp_time.minute - 1) % 60)
                    
                    # Set millisecond offset:
                    if self._prev_ocr_second is not None and self._prev_ocr_second != ocr_sec:
                        self._frame_ms_offset = comp_ms
                    self._prev_ocr_second = ocr_sec

                    # Compose the final timestamp with refined ms offset and OCR seconds
                    try:
                        #print(f"{comp_ms} and {self._frame_ms_offset} gives {comp_ms-self._frame_ms_offset}")
                        final_timestamp = comp_time.replace(second=ocr_sec, microsecond=((comp_ms-self._frame_ms_offset)%1000 * 1000))
                    except Exception:
                        print(f"Error: Could not compose final timestamp with OCR seconds '{ocr_sec}' and ms offset '{self._frame_ms_offset}'. Saving image to 'roi.png'.")
                        print(f"Given error: {sys.exc_info()} at location: {sys.exc_info()[2].tb_lineno}")
                        cv2.imwrite("roi.png", roi)
                    timestamp = int(final_timestamp.timestamp()*1000)

                # Store the frame etc
                with self._frame_lock:
                    self._current_frame = frame
                    self._current_fps = sum(self._fps_window) / len(self._fps_window)
                    if self.ocr_timestamp:
                        self._current_timestamp = timestamp
            else:
                print("Error: Could not read from camera.")
                self.stop()
            time.sleep(0.001)

    def read(self):
        if self.disable:
            return (True, None, datetime.now().timestamp() * 1000, None)
        with self._frame_lock:
            if self._current_frame is None:
                return (False, None, datetime.now().timestamp() * 1000, None)
            if self.ocr_timestamp:
                return (True, self._current_frame.copy(), self._current_timestamp, self._current_fps)
            return (True, self._current_frame.copy(), datetime.now().timestamp() * 1000, self._current_fps)

    def __enter__(self):
        if self.disable:
            return None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disable:
            return None
        self.stop()

# Object to perform digit matching from an image using template matching
class TemplateOCR:
    def __init__(self, template_dir, threshold=100):
        self.threshold = threshold
        self.templates = {}
        for d in range(10):
            tmpl = cv2.imread(f"{template_dir}/roi{d}.png", cv2.IMREAD_GRAYSCALE)
            if tmpl is None:
                raise ValueError(f"Template for digit {d} not found in {template_dir}.")
            # Binarize the template
            _, tmpl_bin = cv2.threshold(tmpl, self.threshold, 255, cv2.THRESH_BINARY)
            self.templates[str(d)] = tmpl_bin

    def preprocess_image(self, img):
        # Convert to grayscale if not already
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_bin = cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY)
        return img_bin

    def match_digit(self, roi):
        """
        Match a region of interest (ROI) against all digit templates.
        
        :param roi: The ROI containing the digit.
        :return: A tuple of the best matching digit and its score.
        """
        best_digit = None
        best_score = -1
        for digit, tmpl in self.templates.items():
            # Ensure the template fits within the ROI.
            if roi.shape[0] < tmpl.shape[0] or roi.shape[1] < tmpl.shape[1]:
                continue

            # Slide the template over the ROI using normalized correlation.
            res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
                best_digit = digit
        return best_digit, best_score

    def read_number(self, img, roi_coords):
        """
        Read a multi-digit number from an input image using specified ROIs.
        
        :param image: input image.
        :param roi_coords: List of tuples, each defining the ROI for a digit in the format (top, left, height, width).
        :return: A tuple containing the detected number (as a string), a list of matching scores, and a combined ROI.
        """
        detected_digits = []
        scores = []
        roi_images = []
    
        for roi in roi_coords:
            top, left, height, width = roi
            roi_img = img[top:top+height, left:left+width]
            roi_img = self.preprocess_image(roi_img)
            roi_images.append(roi_img)
    
            digit, score = self.match_digit(roi_img)
            detected_digits.append(digit)
            scores.append(score)
    
        if roi_images:
            combined_roi = np.hstack(roi_images)
        else:
            combined_roi = np.array([])
        detected_number = int(''.join(detected_digits))
        return detected_number, scores, combined_roi

def frame_corner_cutout(frame, cutout_size=0.2, absolute=True):
    if cutout_size <= 0 or cutout_size >= 0.5:
        raise ValueError("Cutout size must be between 0 and 0.5.")

    # Get frame dimensions
    height, width = frame.shape[:2]
    height_offset = int(height * cutout_size)
    if absolute:
        width_offset = height_offset
    else:
        width_offset = int(width * cutout_size)

    # Define the points of the octagon
    pts = np.array([
        [width_offset, 0],
        [width - width_offset, 0],
        [width - 1, height_offset],
        [width - 1, height - height_offset],
        [width - width_offset, height - 1],
        [width_offset, height - 1],
        [0, height - height_offset],
        [0, height_offset]
    ], np.int32)

    # Create a mask
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # Apply the mask to the frame
    frame = cv2.bitwise_and(frame, frame, mask=mask)

    return frame

def frame_crop(frame, crop_size=0.7):

    # Get original dimensions
    height, width = frame.shape[:2]
    
    # Calculate new dimensions
    new_width = int(width * crop_size)
    new_height = int(height * crop_size)
    
    # Calculate crop coordinates to keep center
    start_x = (width - new_width) // 2
    start_y = (height - new_height) // 2
    
    # Crop the frame
    frame = frame[start_y:start_y + new_height, start_x:start_x + new_width]

    return frame

# Function to zero the marker position based on the relative position of the marker from the camera, and the global position of the camera.
def zero_marker_position(tvec, rvec, camera_position, camera_rotation):
    ## Calculate the positon of the marker in the global coordinate system. ##
    R_marker_camera, _ = cv2.Rodrigues(rvec)
	# Invert to get camera pose in marker coordinates
    R_camera_marker = R_marker_camera.T
    t_camera_marker = -R_camera_marker @ tvec.reshape((3, 1))
	
    R_camera_global = R.from_euler('xyz', camera_rotation, degrees=True).as_matrix()
	
	# Since camera_global = R_marker_global @ t_camera_marker + marker_global, then:
    marker_global = camera_position.reshape((3, 1)) - R_camera_global @ t_camera_marker
	
	# Also compute marker's global rotation
    marker_R_global = R_camera_global @ R_marker_camera
    marker_euler = R.from_matrix(marker_R_global).as_euler('xyz', degrees=True)

    return marker_global.flatten(), marker_euler

# Function to provide the global reference coordinate position of the camera system based on the depth and frame_pos encoder values.
def global_reference_pos(z0, z1, frame_pos):
    if None in (z0, z1, frame_pos):
        return None, None
    
    # Invert z0 and z1, since input is from the depth sensors
    z0 = -z0
    z1 = -z1

    # Constants for the frame pool setup, in meters.
    adj = 3.1305 # Horizontal distance between the depth sensors.
    frame_x_pos_offset = 0.196 # Minimum frame offset from the main depth sensor to the camera.
    #frame_z_pos_offset = 0.069 # Vertical frame offset from the main depth sensor to the camera.

    camera_x_offset = -1.369 - 0.170 # Maximum zeroing offset of the camera from the center of the pool in the x-direction.
    camera_y_offset = 1.45 + 0.07 # Offset of the camera from the center of the pool in the y-direction.
    camera_z_offset = -0.1186 + 0.225-0.187 # Offset of the camera at the zeroing position at the top of the pool in the z-direction.
    
    frame_pos = frame_pos + frame_x_pos_offset
    # Determine y position based on the frame position, where frame_pose makes up the hypotenuse of a right triangle.
    opp = z0 - z1
    hyp = np.sqrt((opp**2) + (adj**2))

    x = frame_pos/hyp * adj + camera_x_offset
    y = camera_y_offset
    z = z0 - opp * frame_pos/hyp + camera_z_offset

    # Camera rotation around y-axis based on right triangle. Assume the camera is level otherwise.
    camera_rot_y = np.arctan(opp/adj)

    camera_position = np.array([x, y, z])
    camera_rotation = np.array([0, camera_rot_y, 0])

    return camera_position, camera_rotation

class PressureSensorSystem:
    def __init__(self):
        self.sensor_models = {"bottom": None, "surface": None}
        self.sensor_values = {"bottom": None, "surface": None}
    
    # Unpickle a calibration file and load the model
    def load_calibration(self, sensor="bottom", sensor_file='calibrations/pressure_calibrations/pressure1_calibration.pkl'):
        try:
            with open(sensor_file, 'rb') as f:
                calibration_data = pickle.load(f)
            self.sensor_models[sensor] = calibration_data['model']
            return True
                
        except Exception as e:
            print(f"Error loading calibration: {str(e)}")
            print(f"Given error: {sys.exc_info()} at location: {sys.exc_info()[2].tb_lineno}")
            return False
    
    def convert_raw(self, sensor, sensor_value):
        model = self.sensor_models[sensor]
        if model is None:
            return None
        
        if callable(self.sensor_models[sensor]):
            depth = model(sensor_value)
        else: #  Assume a sklearn-like model
            depth = float(model.predict([[sensor_value]]).item())
            #depth = model.predict([[sensor_value]])[0]
        return depth
    
    def set_sensor_value(self, sensor, value):
        try:
            self.sensor_values[sensor] = self.convert_raw(sensor, value)
        
        except Exception as e:
            print(f"Error converting raw pressure sensor value: {str(e)}")
            print(f"Given error: {sys.exc_info()} at location: {sys.exc_info()[2].tb_lineno}")

    def get_depth(self):
        if self.sensor_values["bottom"] is None or self.sensor_values["surface"] is None:
            return None
        return self.sensor_values["bottom"] - self.sensor_values["surface"]
        
# Class to handle shared memory for depth values, used to send values from receiver to mantaPos
class DepthSharedMemory:
    SHM_NAME = "mantaPos_depth"
    SIZE = 32  # 3 x 8-byte floats (depth_main, depth_sec, frame_pos) + 1 x 8-byte integer for timestamp
    
    def __init__(self, create=False):
        self.create = create
        self.connect()
    
    def connect(self):
        """Attempt to connect to or create shared memory"""
        try:
            if self.create:
                try:
                    # Try to cleanup any existing shared memory with same name
                    existing = shared_memory.SharedMemory(name=self.SHM_NAME)
                    existing.close()
                    existing.unlink()
                except FileNotFoundError:
                    pass
                # Create new shared memory block and initialize with default values
                self.shm = shared_memory.SharedMemory(name=self.SHM_NAME, create=True, size=self.SIZE)
                self.shm.buf[:self.SIZE] = struct.pack('dddq', 99.999999, 99.999999, 99.999999, 0)
            else:
                # Try to attach to existing shared memory
                self.shm = shared_memory.SharedMemory(name=self.SHM_NAME)
        except Exception as e:
            #print(f"Shared memory error: {e}")
            self.shm = None

    def write_depths(self, depth_main, depth_sec, frame_pos, timestamp=0):
        """Write depths, frame_pos and timestamp with reconnection attempt on failure"""
        if self.shm is None and not self.create:
            # Try to reconnect if we're the consumer
            self.connect()
        if self.shm is None:
            return False
        try:
            packed = struct.pack(
                'dddq', 
                depth_main if depth_main is not None else 99.999999,
                depth_sec  if depth_sec  is not None else 99.999999,
                frame_pos  if frame_pos  is not None else 99.999999,
                int(timestamp) if timestamp is not None else 0
            )
            self.shm.buf[:self.SIZE] = packed
            return True
        except Exception as e:
            print(f"Error writing to shared memory: {e}")
            print(f"Given error: {sys.exc_info()} at location: {sys.exc_info()[2].tb_lineno}")
            self.shm = None  # Mark for reconnection attempt
            return False

    def get_depth(self):
        """Read depths, frame_pos and timestamp with reconnection attempt on failure"""
        if self.shm is None and not self.create:
            # Try to reconnect if we're the consumer
            self.connect()
        if self.shm is None:
            return None, None, None, None
        try:
            data = self.shm.buf[:self.SIZE]
            depth_main, depth_sec, frame_pos, timestamp = struct.unpack('dddq', data)
            if depth_main == 99.999999: depth_main = None
            if depth_sec  == 99.999999: depth_sec  = None
            if frame_pos  == 99.999999: frame_pos  = None
            if timestamp  == 0: timestamp = None
            return depth_main, depth_sec, frame_pos, timestamp
        except Exception as e:
            print(f"Error reading from shared memory: {e}")
            print(f"Given error: {sys.exc_info()} at location: {sys.exc_info()[2].tb_lineno}")
            self.shm = None  # Mark for reconnection attempt
            return None, None, None, None

    def close(self):
        if self.shm is not None:
            try:
                self.shm.close()
            except Exception as e:
                print(f"Error closing shared memory: {e}")
                print(f"Given error: {sys.exc_info()} at location: {sys.exc_info()[2].tb_lineno}")

    def unlink(self):
        if self.shm is not None:
            try:
                self.shm.unlink()
            except Exception as e:
                print(f"Error unlinking shared memory: {e}")
                print(f"Given error: {sys.exc_info()} at location: {sys.exc_info()[2].tb_lineno}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Class to handle shared memory for camera position/rotation data, to send json strings.
class PositionSharedMemory:
    SHM_NAME = "mantaPos_position"
    SIZE = 512  # bytes

    def __init__(self, create=False):
        self.create = create
        self.connect()
        self.new_content = bytearray(self.SIZE)

    def connect(self):
        try:
            if self.create:
                try:
                    existing = shared_memory.SharedMemory(name=self.SHM_NAME)
                    existing.close()
                    existing.unlink()
                except FileNotFoundError:
                    pass
                self.shm = shared_memory.SharedMemory(name=self.SHM_NAME, create=True, size=self.SIZE)
                # Initialize shared memory: flag=0 and zeroed rest
                self.shm.buf[0] = 0
                self.shm.buf[1:self.SIZE] = bytes(self.SIZE - 1)
            else:
                self.shm = shared_memory.SharedMemory(name=self.SHM_NAME)
        except Exception as e:
            self.shm = None

    def write_position(self, json_string):
        if self.shm is None and not self.create:
            self.connect()
        if self.shm is None:
            return False
        try:
            encoded = json_string.encode('utf-8')
            if len(encoded) > self.SIZE - 1:
                raise ValueError(f"Data too large for shared memory ({len(encoded)} bytes). Please increase SIZE in the PositionSharedMemory class.")
            # Set the flag to indicate a new unread packet
            self.new_content[0] = 1  
            # Insert the encoded data
            self.new_content[1:1+len(encoded)] = encoded
            # Fill the remaining unused space with null bytes
            self.new_content[1+len(encoded):] = b'\x00' * (self.SIZE - 1 - len(encoded))
            self.shm.buf[:self.SIZE] = self.new_content
            return True
        except Exception as e:
            print(f"Error writing to position shared memory: {e}")
            print(f"Given error: {sys.exc_info()} at location: {sys.exc_info()[2].tb_lineno}")
            self.shm = None
            return False

    def get_position(self):
        if self.shm is None and not self.create:
            self.connect()
        if self.shm is None:
            return None
        try:
            # Check if a new packet exists
            if self.shm.buf[0] != 1:
                return None
            # Read data from buffer (starting at byte 1)
            data = self.shm.buf[1:self.SIZE].tobytes().decode('utf-8').rstrip('\x00')
            # Reset flag after reading
            self.shm.buf[0] = 0
            return data
        except Exception as e:
            print(f"Error reading from position shared memory: {e}")
            print(f"Given error: {sys.exc_info()} at location: {sys.exc_info()[2].tb_lineno}")
            self.shm = None
            return None

    def close(self):
        if self.shm is not None:
            try:
                self.shm.close()
            except Exception as e:
                print(f"Error closing position shared memory: {e}")
                print(f"Given error: {sys.exc_info()} at location: {sys.exc_info()[2].tb_lineno}")

    def unlink(self):
        if self.shm is not None:
            try:
                self.shm.unlink()
            except Exception as e:
                print(f"Error unlinking position shared memory: {e}")
                print(f"Given error: {sys.exc_info()} at location: {sys.exc_info()[2].tb_lineno}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()