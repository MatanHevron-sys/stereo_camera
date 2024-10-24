# camera.py

import pyrealsense2 as rs
import numpy as np

class Camera:
    def __init__(self, width=640, height=480, fps=30):
        """
        Initializes the Intel RealSense camera pipeline and configures the streams.
        
        Parameters:
            width (int): Width of the video stream.
            height (int): Height of the video stream.
            fps (int): Frames per second.
        """
        # Create a RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable depth and color streams
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # Start streaming
        self.pipeline_profile = self.pipeline.start(self.config)

        # Retrieve depth scale
        depth_sensor = self.pipeline_profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth Scale: {self.depth_scale}")

        # Create an align object to align depth frames to color frames
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

    def get_frames(self):
        """
        Captures and aligns depth and color frames.

        Returns:
            depth_frame (rs.frame): Aligned depth frame.
            color_frame (rs.frame): Aligned color frame.
        """
        # Wait for a new set of frames
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        return depth_frame, color_frame

    def calculate_distance(self, depth_frame, bbox):
        """
        Calculates the average distance from the camera to the object within the bounding box.

        Parameters:
            depth_frame (rs.frame): Aligned depth frame.
            bbox (dict): Bounding box coordinates with keys 'x', 'y', 'w', 'h'.

        Returns:
            distance_mm (float): Average distance in millimeters. Returns 0 if invalid.
        """
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']

        # Define the region within the bounding box to sample depth
        # To avoid edges which might have noise, we can sample a central region
        # For simplicity, we'll sample the entire bounding box here
        # You can modify this to sample a smaller region if needed

        # Ensure coordinates are within frame boundaries
        x_start = max(x, 0)
        y_start = max(y, 0)
        x_end = min(x + w, depth_frame.get_width())
        y_end = min(y + h, depth_frame.get_height())

        # Collect depth values within the bounding box
        depth_values = []
        for i in range(x_start, x_end):
            for j in range(y_start, y_end):
                depth = depth_frame.get_distance(i, j)
                if depth > 0:  # Valid depth
                    depth_values.append(depth)

        if depth_values:
            average_depth = np.mean(depth_values)
            distance_mm = average_depth * 1000  # Convert to millimeters
            return distance_mm
        else:
            return 0

    def stop(self):
        """
        Stops the RealSense pipeline.
        """
        self.pipeline.stop()
