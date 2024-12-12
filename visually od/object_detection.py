import cv2
import pyttsx3
import numpy as np
from ultralytics import YOLO
from collections import deque, defaultdict
import time
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class ObjectDetection:
    def __init__(self):
        # Initialize the text-to-speech engine
        self.engine = pyttsx3.init()
        
        # Load the YOLOv8 model
        self.model = YOLO('yolov8n.pt')
        
        # Initialize the video capture
        self.cap = cv2.VideoCapture(0)
        
        # Set to store the last spoken objects
        self.last_spoken = set()
        
        # Dictionary of average heights of common objects (in meters)
        self.object_heights = {
            'person': 1.7,
            'car': 1.5,
            'chair': 1.0,
            'bottle': 0.2,
            'cup': 0.1
        }
        
        # Add confidence threshold
        self.CONFIDENCE_THRESHOLD = 0.5
        self.MIN_DETECTION_CONFIDENCE = 0.6
        self.MIN_DETECTION_COUNT = 3
        self.DETECTION_PERSISTENCE = 10
        
        # Replace the recent_detections deque with a more sophisticated tracking system
        self.object_tracking = defaultdict(list)
        
        # Time of last speech
        self.last_speech_time = 0
        
        # Voice output toggle
        self.voice_output_enabled = False

    def estimate_distance(self, box_height, real_height, focal_length=500):
        return (real_height * focal_length) / box_height

    def process_detections(self, detections):
        object_counts = {}
        for obj in detections:
            name, distance = obj.split(' at ')
            if name in object_counts:
                object_counts[name] += 1
            else:
                object_counts[name] = 1
        
        description = []
        for obj, count in object_counts.items():
            if count > 1:
                description.append(f"{count} {obj}s")
            else:
                description.append(f"one {obj}")
        
        if len(description) == 0:
            return ""
        elif len(description) == 1:
            return f"There is {description[0]} in front of you."
        elif len(description) == 2:
            return f"There are {description[0]} and {description[1]} in front of you."
        else:
            return f"There are {', '.join(description[:-1])}, and {description[-1]} in front of you."

    def detect_objects(self, frame):
        # Run YOLOv8 inference on the frame
        results = self.model(frame)
        
        # Initialize a set to store detected objects in this frame
        current_objects = set()
        current_time = time.time()
        
        # Process the results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence < self.MIN_DETECTION_CONFIDENCE:
                    continue

                # Get the box coordinates
                x1, y1, x2, y2 = box.xyxy[0]  # Get box coordinates in (x1, y1, x2, y2) format
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers

                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                
                # Draw rectangle around detected object
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Calculate distance if height is known
                box_height = y2 - y1
                if class_name in self.object_heights:
                    distance = self.estimate_distance(box_height, self.object_heights[class_name])
                    distance_text = f"{distance:.2f}m"
                else:
                    distance_text = "Unknown"

                # Add label with class name and distance
                label = f"{class_name} ({distance_text})"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Add to current frame's detections
                current_objects.add(class_name)
                
                # Update tracking with timestamp
                self.object_tracking[class_name].append(current_time)
                
                # Remove old timestamps (older than 1 second)
                self.object_tracking[class_name] = [t for t in self.object_tracking[class_name] 
                                                     if current_time - t <= 1.0]
        
        # Modify the speech generation part
        if self.voice_output_enabled and current_time - self.last_speech_time > 5:
            # Count only objects that appear consistently
            reliable_objects = {}
            for obj, timestamps in self.object_tracking.items():
                if len(timestamps) >= self.MIN_DETECTION_COUNT:
                    # Count unique instances within the last second
                    reliable_objects[obj] = min(len(timestamps) // self.MIN_DETECTION_COUNT, 
                                             max(1, len(set([round(t, 1) for t in timestamps]))))
            
            # Clear old tracking data
            self.object_tracking.clear()
            
            # Generate description using reliable_objects instead of recent_detections
            if reliable_objects:
                description = []
                for obj, count in reliable_objects.items():
                    if count > 1:
                        description.append(f"{count} {obj}s")
                    else:
                        description.append(f"one {obj}")
                
                if len(description) == 0:
                    final_description = ""
                elif len(description) == 1:
                    final_description = f"There is {description[0]} in front of you."
                elif len(description) == 2:
                    final_description = f"There are {description[0]} and {description[1]} in front of you."
                else:
                    final_description = f"There are {', '.join(description[:-1])}, and {description[-1]} in front of you."
                
                if final_description:
                    self.engine.say(final_description)
                    self.engine.runAndWait()
                    self.last_speech_time = current_time

        # Return the processed frame and list of detected objects
        return frame, list(current_objects)

    def toggle_voice_output(self):
        # Toggle voice output functionality
        self.voice_output_enabled = not self.voice_output_enabled
        print(f"Voice output {'enabled' if self.voice_output_enabled else 'disabled'}")

    def run(self):
        while True:
            # Read a frame from the video capture
            ret, frame = self.cap.read()
            if not ret:
                break

            # Detect objects
            processed_frame, detected_objects = self.detect_objects(frame)

            # Display the frame
            cv2.imshow('Object Detection', processed_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture and close all windows
        self.cap.release()
        cv2.destroyAllWindows()

