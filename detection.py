# detection.py

import cv2
import numpy as np
import os
import sys

class Detector:
    def __init__(self, yolo_directory="yolo", confidence_threshold=0.5, nms_threshold=0.4):
        """
        Initializes the YOLOv3 object detector.

        Parameters:
            yolo_directory (str): Directory containing YOLOv3 model files.
            confidence_threshold (float): Minimum confidence to filter weak detections.
            nms_threshold (float): Non-Maximum Suppression threshold.
        """
        self.yolo_path = yolo_directory

        # Paths to YOLO files
        self.weights_path = os.path.join(self.yolo_path, "yolov3.weights")
        self.config_path = os.path.join(self.yolo_path, "yolov3.cfg")
        self.names_path = os.path.join(self.yolo_path, "coco.names")

        # Check if all YOLO files exist
        missing_files = []
        for path, desc in zip([self.weights_path, self.config_path, self.names_path],
                              ["Weights", "Config", "Names"]):
            if not os.path.isfile(path):
                missing_files.append((desc, path))

        if missing_files:
            for desc, path in missing_files:
                print(f"Error: {desc} file not found at {path}")
            print("Please ensure all YOLO files are downloaded and placed correctly in the 'yolo' directory.")
            sys.exit(1)  # Exit the script with an error code

        print("All YOLO files found. Loading the network...")

        # Load YOLO
        self.net = cv2.dnn.readNet(self.weights_path, self.config_path)

        # For CPU usage (optional, for GPU use 'cuda')
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Load class names
        with open(self.names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # Detection thresholds
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

    def detect_objects(self, image):
        """
        Detects objects in the given image.

        Parameters:
            image (numpy.ndarray): The input image in which objects are to be detected.

        Returns:
            detections (list): List of detected objects with their bounding boxes.
                               Each detection is a dictionary with keys: 'x', 'y', 'w', 'h'.
        """
        height, width, channels = image.shape

        # ------------------ Preprocessing for YOLO ------------------
        blob = cv2.dnn.blobFromImage(image, 
                                     scalefactor=1/255.0, 
                                     size=(416, 416), 
                                     swapRB=True, 
                                     crop=False)
        self.net.setInput(blob)
        # ------------------ End Preprocessing ------------------

        # ------------------ Forward Pass ------------------
        outs = self.net.forward(self.output_layers)
        # ------------------ End Forward Pass ------------------

        # Initialize lists for detections
        confidences = []
        boxes = []

        # Iterate over each detection from each output layer
        for out in outs:
            for detection in out:
                scores = detection[5:]
                # Remove class_id and confidence since we're not classifying
                # Instead, use the objectness score
                objectness = detection[4]
                confidence = objectness
                if confidence > self.confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))

        # ------------------ Non-Max Suppression ------------------
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        # ------------------ End Non-Max Suppression ------------------

        detections = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                detection = {
                    'x': boxes[i][0],
                    'y': boxes[i][1],
                    'w': boxes[i][2],
                    'h': boxes[i][3]
                }
                detections.append(detection)

        return detections
