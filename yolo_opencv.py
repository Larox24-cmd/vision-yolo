import cv2
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import numpy as np
from collections import deque

# Load YOLOv3 model
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# Initialize YOLO
net, classes, output_layers = load_yolo()

# Known parameters for distance calculation (for demo purposes)
KNOWN_DISTANCE = 50.0  # cm
KNOWN_OBJECT_WIDTH = 15.0  # cm (adjust as needed for known object width)

distance_history = deque(maxlen=5)  # Distance smoothing

# Function to calculate focal length (fixed for demonstration)
FOCAL_LENGTH = 600  # Calibrated focal length (adjust as needed)

# Function to calculate distance based on object width and focal length
def calculate_distance(pixel_width):
    if pixel_width == 0:
        return float('inf')
    return (KNOWN_OBJECT_WIDTH * FOCAL_LENGTH) / pixel_width

# Function to smooth distance output
def smoothed_distance(new_distance):
    distance_history.append(new_distance)
    return sum(distance_history) / len(distance_history)

# Process each frame for YOLO detection and display
def process_frame(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Process detection outputs
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Get object coordinates and size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    descriptions = []  # To store object descriptions for each frame
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        distance = calculate_distance(w)
        distance_smooth = smoothed_distance(distance)

        # Append object description
        descriptions.append(f"{label}: {confidence*100:.2f}% - Distance: {distance_smooth:.2f} cm")

        # Draw bounding box and label on frame
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {distance_smooth:.2f}cm", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, descriptions

# Tkinter GUI Setup
root = tk.Tk()
root.title("YOLO Object Detection")
root.geometry("1024x768")

# Text box for displaying object descriptions
description_text = scrolledtext.ScrolledText(root, width=60, height=15, font=("Arial", 12))
description_text.pack(side="bottom", padx=10, pady=10)

# Video capture and display in Tkinter
def update_frame():
    ret, frame = cap.read()
    if ret:
        # Process frame for YOLO object detection
        processed_frame, descriptions = process_frame(frame)

        # Update the description text box
        description_text.delete("1.0", tk.END)
        for desc in descriptions:
            description_text.insert(tk.END, f"{desc}\n")
        description_text.yview(tk.END)  # Scroll to latest text

        # Display frame in Tkinter window
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        display_label.imgtk = imgtk
        display_label.config(image=imgtk)

    # Schedule the next frame update
    display_label.after(10, update_frame)

# Start video capture
cap = cv2.VideoCapture(0)

# Display label for video frames
display_label = tk.Label(root)
display_label.pack()

# Start the update loop
update_frame()

# Run the Tkinter main loop
root.mainloop()

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
