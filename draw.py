import cv2
import numpy as np
import time

def display_video_in_hsv(video_path):
    # Open the video file or webcam
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    # Get the dimensions of the video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a blank image to draw the path
    path_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Variables to store the previous position of the green object
    prev_x, prev_y = None, None

    # Wait for 's' key to start
    print("Press 's' to start...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame during wait.")
            break
        frame = cv2.flip(frame, 1)
        cv2.imshow('Press S to Start', frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            print("Starting video processing...")
            break

    # Variables for FPS calculation
    prev_time = time.time()

    while True:
        # Read each frame from the video
        ret, frame = cap.read()

        # Break the loop if no frame is returned (end of video)
        if not ret:
            print("End of video or cannot retrieve frame.")
            break

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Convert the frame to HSV format
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define HSV range for green color
        lower_green = np.array([30, 25, 25])  # Adjust as needed
        upper_green = np.array([90, 255, 255])  # Adjust as needed

        # Create a mask for green color
        green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

        # Apply morphological operations for noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        green_open = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)  # Remove small noise
        green_close = cv2.morphologyEx(green_open, cv2.MORPH_CLOSE, kernel)  # Fill gaps

        # Find contours of the green object
        contours, _ = cv2.findContours(green_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if any valid contours are found
        valid_contour_found = False
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:  # Ignore small areas (e.g., noise)
                valid_contour_found = True
                # Get the center of the green object
                moments = cv2.moments(largest_contour)
                if moments['m00'] != 0:
                    x = int(moments['m10'] / moments['m00'])
                    y = int(moments['m01'] / moments['m00'])

                    # Draw the path if a previous point exists
                    if prev_x is not None and prev_y is not None:
                        cv2.line(path_image, (prev_x, prev_y), (x, y), (0, 255, 0), 2)

                    # Update the previous position
                    prev_x, prev_y = x, y

        if not valid_contour_found:
            # Reset previous positions if no valid green object is detected
            prev_x, prev_y = None, None

        # Display FPS on the original frame
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the original frame
        cv2.imshow('Original Frame', frame)

        # Display the processed mask
        cv2.imshow("Green Processed", green_close)

        # Display the path image
        cv2.imshow("Path Image", path_image)

        # Wait for 1 ms and check if 'Esc' key is pressed to exit
        if cv2.waitKey(1) == 27:
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function with a video source (0 for webcam)
display_video_in_hsv(0)
