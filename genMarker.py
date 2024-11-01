import cv2
import cv2.aruco as aruco
import os

calib_dir = './markers'
file_path = 'ChArUco_Marker.png'

ARUCO_DICT = cv2.aruco.DICT_5X5_50  # Dictionary ID

def generate_aruco_marker(id, dictionary=ARUCO_DICT, size=2000):
    # Get the predefined dictionary based on the specified type
    aruco_dict = aruco.getPredefinedDictionary(dictionary)
    
    # Generate the marker with the given id and size using generateImageMarker()
    marker_image = aruco.generateImageMarker(aruco_dict, id, size)

    # Save the generated marker as an image file
    file_name = os.path.join(calib_dir, f'aruco_marker_{id}.png')
    cv2.imwrite(file_name, marker_image)
    print(f"Marker {id} saved as {file_name}")

def create_and_save_ChArUco_board(sq_len, margin_px, mrkr_ratio=0.9, sqs_vert=5, sqs_hor=7):
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    board = aruco.CharucoBoard((sqs_hor, sqs_vert), sq_len, round(sq_len*mrkr_ratio), dictionary)
    img_size = tuple((sqs_hor*sq_len + margin_px*2, sqs_vert*sq_len + margin_px*2))
    img = aruco.CharucoBoard.generateImage(board, img_size, marginSize=margin_px)
    
    file_name = os.path.join(calib_dir, file_path)
    cv2.imwrite(file_name, img)

    return board, dictionary

# Generate Aruco marker with ID 18, and forward
if __name__ == "__main__":
    if not os.path.exists(calib_dir):
        os.makedirs(calib_dir)

    for i in range(0,16):
        generate_aruco_marker(18+i)
