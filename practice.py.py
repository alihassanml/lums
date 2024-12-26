import cv2
import numpy as np

def detect_green_screen():
    # Start the video capture (0 for default webcam)
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the HSV range for detecting green color
        lower_green = np.array([35, 55, 55])  # Lower bound of green
        upper_green = np.array([85, 255, 255])  # Upper bound of green

        # Create a mask for the green color
        green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

        # Apply morphological operations to reduce noise
        kernel = np.ones((3, 3), np.uint8)  # Kernel size (adjust if needed)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)  # Remove small noise
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)  # Close gaps

        # Apply the mask to the original frame
        green_detected = cv2.bitwise_and(frame, frame, mask=green_mask)

        # Display the original frame and the green screen detection
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Green Screen Detected", green_detected)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) == 27:
            break

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the green screen detection
detect_green_screen()
