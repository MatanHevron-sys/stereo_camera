# gui.py

import cv2
import numpy as np

class GUI:
    def __init__(self):
        """
        Initializes the GUI by setting up window names and creating a combined display window.
        """
        # Define window name
        self.combined_window = 'Real-Time Monitoring'
    
        # Initialize window
        cv2.namedWindow(self.combined_window, cv2.WINDOW_NORMAL)
    
    def combine_images(self, color_image, bounding_box_image, depth_image):
        """
        Combines multiple images into a single image arranged in a grid.
    
        Parameters:
            color_image (numpy.ndarray): Color image with annotations.
            bounding_box_image (numpy.ndarray): Image showing bounding boxes.
            depth_image (numpy.ndarray): Color-mapped depth image.
    
        Returns:
            combined_image (numpy.ndarray): Combined image for display.
        """
        # Resize images to ensure consistency
        height, width, _ = color_image.shape
        bounding_box_resized = cv2.resize(bounding_box_image, (width, height))
        depth_resized = cv2.resize(depth_image, (width, height))
    
        # Stack images horizontally
        top_row = np.hstack((color_image, bounding_box_resized))
        bottom_row = np.hstack((depth_resized, np.zeros_like(depth_resized)))  # Empty space or future use
    
        # Stack rows vertically
        combined_image = np.vstack((top_row, bottom_row))
    
        return combined_image
    
    def annotate_combined_image(self, combined_image, total_objects):
        """
        Adds annotations and statistics to the combined image.
    
        Parameters:
            combined_image (numpy.ndarray): The image to annotate.
            total_objects (int): Total number of detected objects within 3 meters.
    
        Returns:
            annotated_image (numpy.ndarray): Annotated combined image.
        """
        # Define text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (0, 255, 0)  # Green color
        thickness = 2
        line_type = cv2.LINE_AA
    
        # Prepare annotation text
        text = f"Total Objects within 3m: {total_objects}"
    
        # Position for the text (top-left corner)
        position = (10, 30)
    
        # Add text to the image
        cv2.putText(combined_image, text, position, font, font_scale, color, thickness, line_type)
    
        return combined_image
    
    def display(self, color_image, bounding_box_image, depth_image, total_objects):
        """
        Displays the combined image with annotations.
    
        Parameters:
            color_image (numpy.ndarray): Color image with annotations.
            bounding_box_image (numpy.ndarray): Image showing bounding boxes.
            depth_image (numpy.ndarray): Color-mapped depth image.
            total_objects (int): Total number of detected objects within 3 meters.
        """
        # Combine images into a single frame
        combined_image = self.combine_images(color_image, bounding_box_image, depth_image)
    
        # Annotate the combined image with statistics
        annotated_image = self.annotate_combined_image(combined_image, total_objects)
    
        # Display the annotated combined image
        cv2.imshow(self.combined_window, annotated_image)
    
    def close(self):
        """
        Closes all GUI windows.
        """
        cv2.destroyAllWindows()
