import cv2
import cv2.aruco as aruco

def main():
    # Open the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(1)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Define the dictionary of Aruco markers you are using
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

    # Initialize detector parameters using default values
    parameters = aruco.DetectorParameters()
    
    # Create the detector using detectMarkers with the appropriate call for OpenCV 4.7+
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the markers in the image using the updated API
        corners, ids, rejected = detector.detectMarkers(gray)

        # Draw the detected markers on the frame
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            for i in range(len(ids)):
                print(f"Detected marker ID: {ids[i][0]}")

        # Display the frame with detected markers
        cv2.imshow('Aruco Marker Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
