import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from object_detection import ObjectDetection
import threading

class ObjectDetectionInterface:
    def __init__(self):
        # Initialize the object detection system
        self.detector = ObjectDetection()
        self.is_detecting = False
        self.camera_active = False
        
        # Create the main window
        self.root = ctk.CTk()
        self.root.title("Object Detection System")
        self.root.geometry("1200x800")
        
        # Create main container
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left frame for camera feed
        self.camera_frame = ctk.CTkFrame(self.main_container)
        self.camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create right frame for controls and results
        self.control_frame = ctk.CTkFrame(self.main_container)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        
        # Add title label
        self.title_label = ctk.CTkLabel(
            self.camera_frame,
            text="Object Detection Camera Feed",
            font=("Helvetica", 16, "bold")
        )
        self.title_label.pack(pady=10)

        # Update camera frame for OpenCV feed
        self.camera_label = ctk.CTkLabel(
            self.camera_frame,
            text=""
        )
        self.camera_label.pack(pady=10)

        # Add control buttons
        self.button_frame = ctk.CTkFrame(self.control_frame)
        self.button_frame.pack(fill=tk.X, padx=5, pady=5)

        self.start_button = ctk.CTkButton(
            self.button_frame,
            text="Start Detection",
            command=self.start_detection
        )
        self.start_button.pack(fill=tk.X, padx=5, pady=5)

        self.stop_button = ctk.CTkButton(
            self.button_frame,
            text="Stop Detection",
            command=self.stop_detection,
            state="disabled"
        )
        self.stop_button.pack(fill=tk.X, padx=5, pady=5)

        self.voice_button = ctk.CTkButton(
            self.button_frame,
            text="Toggle Voice",
            command=self.toggle_voice
        )
        self.voice_button.pack(fill=tk.X, padx=5, pady=5)

        # Add results section
        self.results_label = ctk.CTkLabel(
            self.control_frame,
            text="Detection Results",
            font=("Helvetica", 14, "bold")
        )
        self.results_label.pack(pady=10)

        self.results_text = ctk.CTkTextbox(
            self.control_frame,
            width=300,
            height=400
        )
        self.results_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # Start camera thread
        self.camera_thread = None
        
    def update_camera_feed(self):
        if self.camera_active:
            ret, frame = self.cap.read()
            if ret:
                # Run object detection if active
                if self.is_detecting:
                    frame, detected_objects = self.detector.detect_objects(frame)
                    # Update results text
                    self.update_results(detected_objects)
                
                # Convert frame for display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (800, 600))
                photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                
                self.camera_label.configure(image=photo)
                self.camera_label.image = photo
            
            # Schedule the next update
            self.root.after(10, self.update_camera_feed)

    def update_results(self, detected_objects):
        # Clear previous results
        self.results_text.delete("1.0", "end")
        
        # Update with new detections
        if detected_objects:
            result_text = "Detected Objects:\n"
            # Create a dictionary to count occurrences of each object
            object_counts = {}
            for obj in detected_objects:
                object_counts[obj] = object_counts.get(obj, 0) + 1
            
            # Display counts
            for obj, count in object_counts.items():
                result_text += f"• {obj}: {count}\n"
            self.results_text.insert("1.0", result_text)

    def start_detection(self):
        if not self.camera_active:
            self.camera_active = True
            self.is_detecting = True
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            
            # Start camera feed
            self.update_camera_feed()
            
            self.results_text.insert("1.0", "Detection Started...\n")

    def stop_detection(self):
        self.camera_active = False
        self.is_detecting = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        
        if self.camera_thread:
            self.camera_thread.join()
        
        self.results_text.insert("1.0", "Detection Stopped...\n")

    def toggle_voice(self):
        # Implement voice control functionality
        if hasattr(self.detector, 'toggle_voice_output'):
            self.detector.toggle_voice_output()
            self.results_text.insert("1.0", "Voice Output Toggled...\n")

    def on_closing(self):
        self.camera_active = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

if __name__ == "__main__":
    app = ObjectDetectionInterface()
    app.run()