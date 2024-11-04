import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from mtcnn import MTCNN
import pyttsx3
import numpy as np
from collections import deque
import face_recognition

KNOWN_FACE_WIDTH = 15.0  # Average face width in cm
FOCAL_LENGTH = 600       # Calibrated focal length in pixels (adjust as necessary)
 
# Distance smoothing
distance_history = deque(maxlen=5)  # Store the last 5 distances for smoothing

# Text-to-Speech setup
engine = pyttsx3.init()

def smoothed_distance(new_distance):
    distance_history.append(new_distance)
    return sum(distance_history) / len(distance_history)

def calculate_distance(pixel_width):
    """
    Calculate distance using focal length formula:
    Distance = (Real Width * Focal Length) / Pixel Width
    """
    if pixel_width == 0:
        return float('inf')
    return (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / pixel_width

def speak_text(text):
    """
    Speak the text using TTS.
    """
    engine.say(text)
    engine.runAndWait()

def open_laptop_camera():
    # Load MTCNN detector for face detection
    detector = MTCNN()
    cap = cv2.VideoCapture(0)

    # Face recognition setup
    known_face_encodings = []  # Add known face encodings here
    known_face_names = []      # Corresponding names for the known faces

    # Initialize Tkinter window and label
    global root, display_label  # Declare as global to use in the function
    root = tk.Tk()
    display_label = tk.Label(root)
    display_label.pack()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Detect faces using MTCNN
        results = detector.detect_faces(frame)
        if not results:  # Check if no faces are detected
            continue  # Skip to the next frame if no faces are found

        for result in results:
            x, y, width, height = result['box']

            # Calculate smoothed distance based on face width in pixels
            distance = smoothed_distance(calculate_distance(width))
            distance_text = f"Distance: {distance:.2f} cm"
            
            # Draw bounding box and display distance on the frame
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, distance_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Perform face recognition
            face_encoding = face_recognition.face_encodings(frame, [(y, x + height, y + height, x + width)])  # Corrected parameters
            name = "Unknown"
            if face_encoding:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding[0])
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

            # Draw the name of the person (if known)
            cv2.putText(frame, name, (x, y + height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Speak the name and distance if recognized
            if name != "Unknown":
                speak_text(f"{name} detected at {distance:.2f} centimeters")

        # Convert frame to ImageTk format for Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        imgtk = ImageTk.PhotoImage(image=img)

        # Display the image in Tkinter window
        display_label.imgtk = imgtk
        display_label.config(image=imgtk)

        # Refresh the Tkinter window
        root.update()

    cap.release()
    cv2.destroyAllWindows() 