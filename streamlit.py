import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
# Path or URL to the CSV file
CSV_FILE_PATH = "coordinates.csv"

def plot_and_save_direction(df, output_image):
    # Extract the X and Y coordinates from the DataFrame
    x_coords = df['X'].tolist()
    y_coords = df['Y'].tolist()

    # Check if there are coordinates to plot
    if not x_coords or not y_coords:
        return None, "No data found in the CSV file to plot."

    # Plot the coordinates
    plt.figure(figsize=(8, 6))
    plt.plot(x_coords, y_coords, color='green', linestyle='-', marker='o', label='Path')

    # Draw direction arrows
    for i in range(1, len(x_coords)):
        # Plot an arrow indicating the direction between consecutive points
        plt.arrow(x_coords[i-1], y_coords[i-1],
                  x_coords[i] - x_coords[i-1], y_coords[i] - y_coords[i-1],
                  head_width=5, head_length=10, fc='green', ec='green')

    # Highlight the start and end points
    plt.scatter(x_coords[0], y_coords[0], color='blue', s=100, label='Start Point')  # Start point
    plt.scatter(x_coords[-1], y_coords[-1], color='red', s=100, label='End Point')  # End point

    # Add annotations for clarity
    plt.text(x_coords[0], y_coords[0], 'Start', fontsize=10, color='blue', ha='right')
    plt.text(x_coords[-1], y_coords[-1], 'End', fontsize=10, color='red', ha='right')

    plt.gca().invert_yaxis()  # Invert the Y-axis to match the OpenCV coordinate system
    plt.title('Drawing from Saved Coordinates')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(False)
    plt.legend()

    # Save the plot as an image
    plt.savefig(output_image)
    plt.close()
    return output_image, None

# Streamlit app
def main():
    st.title("Coordinate Plotter & Image & Data")

    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        st.subheader("Data Overview")
        st.dataframe(df.head())  # Display the first 5 rows of the DataFrame
        st.write(f"**Number of Rows:** {len(df)}")  # Show the number of rows in the DataFrame

        # Define output image path
        output_image = "coordinate_plot.png"

        # Plot and save the direction
        image_path, error = plot_and_save_direction(df, output_image)

        if error:
            st.error(error)
        else:
            # Show the saved image
            st.image(image_path, caption="Plotted Path", use_column_width=True)

            # Optionally remove the temporary file
            if os.path.exists(output_image):
                os.remove(output_image)

    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")

if __name__ == "__main__":
    main()
