# main.py

from camera import Camera
from detection import Detector
from gui import GUI
import cv2
import numpy as np

def main():
    """
    Main function to run the object detection and distance measurement application.
    """
    # ------------------ Initialization ------------------
    # Initialize Camera
    camera = Camera(width=640, height=480, fps=30)

    # Initialize Detector
    detector = Detector(yolo_directory="yolo", confidence_threshold=0.5, nms_threshold=0.4)

    # Initialize GUI
    gui = GUI()
    # ------------------ End Initialization ------------------

    try:
        print("Application started. Press 'Esc' to exit.")
        while True:
            # Capture frames from camera
            depth_frame, color_frame = camera.get_frames()

            if depth_frame is None or color_frame is None:
                print("No frames received. Skipping iteration.")
                continue

            # Convert frames to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Detect objects in the color image
            detections = detector.detect_objects(color_image)

            # Filter detections within 3 meters
            filtered_detections = []
            for det in detections:
                distance_mm = camera.calculate_distance(depth_frame, det)
                if distance_mm <= 3000 and distance_mm > 0:
                    det['distance_mm'] = distance_mm
                    filtered_detections.append(det)

            # Sort detections from left to right based on x-coordinate
            sorted_detections = sorted(filtered_detections, key=lambda d: d['x'])

            # Assign labels: object1, object2, etc.
            for idx, det in enumerate(sorted_detections):
                det['label'] = f"object{idx + 1}"

            # Create a copy for bounding boxes window
            bounding_box_image = color_image.copy()

            # Iterate over sorted detections to annotate images
            for det in sorted_detections:
                x, y, w, h = det['x'], det['y'], det['w'], det['h']
                label = det['label']
                distance_mm = det['distance_mm']

                # Draw bounding box on the main color image
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Calculate position for distance annotation inside the bounding box
                # Positioning it at the bottom-left corner inside the box
                text_position = (x + 5, y + h - 5)  # 5 pixels padding from bottom-left

                # Annotate label and distance inside the bounding box
                annotation_text = f"{label}: {distance_mm:.1f}mm"
                cv2.putText(color_image, annotation_text, text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # Draw bounding box on the separate bounding box image
                cv2.rectangle(bounding_box_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Annotate label and distance inside the bounding box image
                cv2.putText(bounding_box_image, annotation_text, text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Calculate total number of objects within 3 meters
            total_objects = len(sorted_detections)

            # Normalize depth image for visualization
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),
                                              cv2.COLORMAP_JET)

            # Optionally, annotate center distance on the depth image
            center_x_depth = color_image.shape[1] // 2
            center_y_depth = color_image.shape[0] // 2
            # Create a small bounding box around the center for accurate distance measurement
            center_bbox = {
                'x': center_x_depth - 10,  # 10 pixels to the left
                'y': center_y_depth - 10,  # 10 pixels above
                'w': 20,                    # 20 pixels width
                'h': 20                     # 20 pixels height
            }
            center_distance_mm = camera.calculate_distance(depth_frame, center_bbox)
            if center_distance_mm > 0:
                cv2.putText(depth_colormap, f"Center Distance: {center_distance_mm:.1f}mm",
                            (center_x_depth - 150, center_y_depth - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.circle(depth_colormap, (center_x_depth, center_y_depth), 5, (255, 255, 255), -1)

            # Display the images using GUI with annotations
            gui.display(color_image, bounding_box_image, depth_colormap, total_objects)

            # Exit mechanism
            key = cv2.waitKey(1)
            if key == 27:  # Esc key
                print("Escape key pressed. Exiting...")
                break

    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")

    finally:
        # Cleanup
        camera.stop()
        gui.close()
        print("Application closed successfully.")

if __name__ == "__main__":
    main()
