import cv2
import cv2.aruco as aruco
import os
import numpy as np  # Added for drawing circles

calib_dir = './markers'
file_path = 'ChArUco_Marker.png'

ARUCO_DICT = cv2.aruco.DICT_5X5_50  # Dictionary ID

def generate_aruco_marker(id, dictionary=ARUCO_DICT, size=2000):
    # Get the predefined dictionary based on the specified type
    aruco_dict = aruco.getPredefinedDictionary(dictionary)
    
    # Generate the marker with the given id and size using generateImageMarker()
    marker_image = aruco.generateImageMarker(aruco_dict, id, size)

    # Add circular dots to the marker image
    #marker_image = add_circular_dots(marker_image, 5, 5)

    # Save the generated marker as an image file
    file_name = os.path.join(calib_dir, f'aruco_marker_{id}.png')
    cv2.imwrite(file_name, marker_image)
    print(f"Marker {id} saved as {file_name}")

def draw_dot(marker_image, center, dot_radius):
    # Get the inverse color of the current pixel at the center position
    current_color = int(marker_image[center[1], center[0]])
    dot_color = 255 - current_color
    # Draw the circle with the calculated color
    cv2.circle(marker_image, center, dot_radius, dot_color, -1)

def add_circular_dots(marker_image, dot_offset, dot_radius, dict_size=5):
    # Get image size
    size = marker_image.shape[0]  # Assuming square image

    # Calculate border width
    border_width = size / (dict_size + 2)

    # Define all dot positions
    dot_positions = [
        # Left and right
        (int(border_width / 2), int(size / 2) - dot_offset),
        (int(size - border_width / 2), int(size / 2) - dot_offset),
        # Top and bottom
        (int(size / 2) - dot_offset, int(border_width / 2)),
        (int(size / 2) - dot_offset, int(size - border_width / 2)),
        # Corners
        (int(border_width / 2), int(border_width / 2)),
        (int(size - border_width / 2), int(border_width / 2)),
        (int(border_width / 2), int(size - border_width / 2)),
        (int(size - border_width / 2), int(size - border_width / 2)),
        # Center
        (int(size / 2) - dot_offset, int(size / 2) - dot_offset)
    ]

    # Draw all dots using the helper function
    for position in dot_positions:
        draw_dot(marker_image, position, dot_radius)
    
    return marker_image

def create_and_save_ChArUco_board(sq_len_meters, sq_pixels, margin_pixels, mrkr_ratio=0.9, sqs_vert=5, sqs_hor=7, file_path='ChArUco_Marker.png', img_size=None, marker_ids=None, dot_offset=0):        
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    int_marker_len_in_image = sq_len_meters*(round(sq_pixels*mrkr_ratio)/sq_pixels)
    
    if marker_ids is not None:
    # Use only the specified marker IDs
        board = aruco.CharucoBoard((sqs_hor, sqs_vert), sq_len_meters, int_marker_len_in_image, dictionary, marker_ids)
    else:
        board = aruco.CharucoBoard((sqs_hor, sqs_vert), sq_len_meters, int_marker_len_in_image, dictionary)
    
    if img_size is None:
        img_size = tuple((sqs_hor*sq_pixels + margin_pixels*2, sqs_vert*sq_pixels + margin_pixels*2))

    img = aruco.CharucoBoard.generateImage(board, img_size, marginSize=margin_pixels)

    # save img to A3 paper size pdf
    cv2.imwrite('ChArUco_Marker.png', img)

    # Add circular dots to the marker image
    #img = add_circular_dots(img, dot_offset, 5, 8)
    #img = add_circular_dots(img, dot_offset, 5, 20)

    file_name = os.path.join(calib_dir, file_path)
    cv2.imwrite(file_name, img)

    return board, dictionary

# Generate Aruco marker starting with ID 18, and forward
if __name__ == "__main__":
    if not os.path.exists(calib_dir):
        os.makedirs(calib_dir)

    #for i in range(1,2):
    #    generate_aruco_marker(i, ARUCO_DICT, 11811)


    img_size = (12993, 12993) # Image size in pixels
    #img_size = (7088, 7088) # Image size in pixels
    squares_vertically = 6
    squares_horizontally = squares_vertically
    square_pixels = 200 # Pixel size of the chessboard squares
    grid_edge = 591 # Pixel margin outside the ChArUco grid
    marker_ratio = 0.75 # Marker ratio of square_length to fit within white squares; acceptable maximum 0.85, recommended 0.7. Rounds marker size to int.
    square_length = 0.2975/6 * square_pixels/200 # Real world length of square in meters
    offset = 18 # Offset for the marker IDs, to start after the large ChArUco board
    dot_offset = int(11811/6 / 4) # Pixel offset for the circular dots, shifts to top left

    for i in range(1, 2):
        #marker_ids = np.array(range(i*4-8+offset, i*4+4-8+offset))
        #print(f"Creating board {i} using marker IDs: {marker_ids}")
        #create_and_save_ChArUco_board(square_length, square_pixels, grid_edge, marker_ratio, squares_vertically, squares_horizontally, f'ChArUco_Marker_{i}.png', img_size, marker_ids)
        
        create_and_save_ChArUco_board(square_length, square_pixels, grid_edge, marker_ratio, squares_vertically, squares_horizontally, f'ChArUco_Marker_{i}.png', img_size, None, dot_offset)
