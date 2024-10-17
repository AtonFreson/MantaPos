import cv2
import numpy as np
import os
import re

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
