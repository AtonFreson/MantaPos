import cv2
import cv2.aruco as aruco
import os

calib_dir = './markers'

ARUCO_DICT = cv2.aruco.DICT_6X6_250  # Dictionary ID
SQUARES_VERTICALLY = 7               # Number of squares vertically
SQUARES_HORIZONTALLY = 5             # Number of squares horizontally
SQUARE_LENGTH = 300                   # Square side length (in pixels)
MARKER_LENGTH = 200                   # ArUco marker side length (in pixels)
MARGIN_PX = 100                       # Margins size (in pixels)

IMG_SIZE = tuple(i * SQUARE_LENGTH + 2 * MARGIN_PX for i in (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY))
file_path = 'ChArUco_Marker.png'

def generate_aruco_marker(id, dictionary=aruco.DICT_6X6_250, size=2000):
    # Get the predefined dictionary based on the specified type
    aruco_dict = aruco.getPredefinedDictionary(dictionary)
    
    # Generate the marker with the given id and size using generateImageMarker()
    marker_image = aruco.generateImageMarker(aruco_dict, id, size)

    # Save the generated marker as an image file
    file_name = os.path.join(calib_dir, f'aruco_marker_{id}.png')
    cv2.imwrite(file_name, marker_image)
    print(f"Marker {id} saved as {file_name}")

def create_and_save_ChArUco_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    img = cv2.aruco.CharucoBoard.generateImage(board, IMG_SIZE, marginSize=MARGIN_PX)
    
    #file_path = os.path.join(calib_dir, file_path)
    cv2.imwrite(file_path, img)

# Generate Aruco marker with ID 42, and forward
if __name__ == "__main__":
    #generate_aruco_marker(42)
    if not os.path.exists(calib_dir):
        os.makedirs(calib_dir)

    #for i in range(0,16):
    #    generate_aruco_marker(42+i)
    create_and_save_ChArUco_board()
