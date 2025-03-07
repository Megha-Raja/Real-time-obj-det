**Project Overview**

The project is an object detection system that leverages machine learning and computer vision techniques to identify and estimate the distance of objects in real-time using a webcam feed. The system utilizes the YOLOv8 model for object detection and text-to-speech functionality to verbally announce detected objects. The primary goal is to create an interactive, user-friendly interface that simplifies object detection and encourages accessibility for non-technical users.
The core methodology involves capturing video frames, processing them through the YOLOv8 model to detect objects, estimating distances based on object heights, and converting the results into speech.

**Key Features**

- Real-time Object Detection: The system can detect objects in real-time using a webcam feed, allowing users to monitor their environment continuously.
- Distance Estimation: Based on the detected object's height and the focal length of the camera, the system estimates and displays the distance between the object and the camera.
- Text-to-Speech Announcements: The system converts detected object names into speech, enabling users with visual impairments or those focusing on other tasks to receive audio notifications.
- Customizable Confidence Thresholds: Users can adjust the confidence thresholds for object detection and distance estimation, tailoring the system to their specific needs.
- Object Tracking: The system maintains a history of detected objects, providing a more comprehensive understanding of the environment.
- User-friendly Interface: The custom Tkinter-based GUI simplifies user interaction, allowing users to start and stop detection with ease.

**Technologies**

- OpenCV: OpenCV is used for capturing and processing video frames, as well as rendering the camera feed in the GUI.
- pyttsx3: This library provides text-to-speech functionality, enabling the system to verbally announce detected objects.
- NumPy: NumPy is used for numerical computations, particularly when calculating distances based on object heights and focal length.
- YOLOv8: YOLOv8 is the machine learning model used for object detection, providing accurate and efficient detection capabilities.
- Tkinter and customtkinter: These libraries are used for building the user interface, allowing users to interact with the system easily.
