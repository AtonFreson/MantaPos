import cv2
import numpy as np
import os
import re
from scipy.spatial.transform import Rotation as R

# Function to extract the numerical index from the filename
def extract_index(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

# Function to display ArUco markers in a grid layout / ChArUco-board
def display_marker_grid(num_markers=0, marker_size=200, border_size=25, board_type="ArUco"):
    
    match board_type:
        case "ArUco":
            # Folder containing ArUco marker images
            calibration_image_dir = './markers'

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
            calibration_image_dir = './markers/ChArUco_Marker.png'
            grid_image = cv2.imread(calibration_image_dir)
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
    position_std_text = (f"-Err: X={position_std[0]: >6.6f}m, Y={position_std[1]: >6.6f}m, Z={position_std[2]: >6.6f}m")
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
