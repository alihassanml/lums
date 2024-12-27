def display_drawing_with_opencv(csv_file):
    import cv2
    import csv
    import numpy as np

    # Canvas dimensions (adjust as needed)
    canvas_width = 640
    canvas_height = 480

    # Create a blank image to draw the path
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    x_coords, y_coords = [], []

    # Read the coordinates from the CSV file
    try:
        with open(csv_file, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                x_coords.append(int(row['X']))
                y_coords.append(int(row['Y']))
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        return
    except ValueError:
        print("Error: Invalid data in the CSV file.")
        return

    # Check if there are coordinates to plot
    if not x_coords or not y_coords:
        print("No data found in the CSV file to display.")
        return

    # Draw the path with arrows on the canvas
    for i in range(1, len(x_coords)):
        start_point = (x_coords[i - 1], y_coords[i - 1])
        end_point = (x_coords[i], y_coords[i])
        cv2.arrowedLine(canvas, start_point, end_point, (0, 255, 0), 2, tipLength=0.3)  # Green arrow

    # Highlight the start and end points
    cv2.circle(canvas, (x_coords[0], y_coords[0]), 5, (255, 0, 0), -1)  # Blue start point
    cv2.circle(canvas, (x_coords[-1], y_coords[-1]), 5, (0, 0, 255), -1)  # Red end point

    # Display the canvas using OpenCV
    cv2.imshow("Drawing with Directions", canvas)

    print("Press any key to close the window...")
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()

# Display video and then draw with directions
display_drawing_with_opencv('coordinates.csv')
