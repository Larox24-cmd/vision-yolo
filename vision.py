import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from mtcnn import MTCNN
from collections import deque


KNOWN_DISTANCE = 50.0  # Distance to object in cm during calibration
KNOWN_FACE_WIDTH = 15.0  # Width of the object or face in cm

# Initialize detector and distance history for smoothing
detector = MTCNN()
distance_history = deque(maxlen=5)


FOCAL_LENGTH_LAPTOP = None
FOCAL_LENGTH_IPHONE = None

# Function to calculate focal length
def calibrate_focal_length(camera_source):
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"Error: Cannot open camera source {camera_source}")
        return None
    
    print("Starting calibration... Place the object at the known distance and press 'q' to capture.")

    focal_length = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Detect face or object for calibration
        results = detector.detect_faces(frame)
        for result in results:
            x, y, width, height = result['box']
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, f"Width in Pixels: {width}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Calculate focal length on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                focal_length = (width * KNOWN_DISTANCE) / KNOWN_FACE_WIDTH
                print(f"Calibrated Focal Length: {focal_length:.2f}")
                break

        cv2.imshow("Calibration Frame", frame)
        if focal_length is not None:
            break

    cap.release()
    cv2.destroyAllWindows()
    return focal_length

# Calibrate both cameras and store their focal lengths
FOCAL_LENGTH_LAPTOP = calibrate_focal_length(0)  # Laptop camera (0)
print(f"Calibrated FOCAL_LENGTH_LAPTOP: {FOCAL_LENGTH_LAPTOP}")

# Replace <IPHONE_IP> with the actual IP for DroidCam
FOCAL_LENGTH_IPHONE = calibrate_focal_length("http://192.168.0.50:4747/video")  # iPhone camera
print(f"Calibrated FOCAL_LENGTH_IPHONE: {FOCAL_LENGTH_IPHONE}")

# Function to calculate distance based on pixel width and focal length
def calculate_distance(pixel_width, focal_length):
    if pixel_width == 0:
        return float('inf')
    return (KNOWN_FACE_WIDTH * focal_length) / pixel_width

# Function to smooth distance output
def smoothed_distance(new_distance):
    distance_history.append(new_distance)
    return sum(distance_history) / len(distance_history)

# Update frame in Tkinter GUI
def update_frame():
    global cap, running, current_focal_length
    if not running:
        return

    ret, frame = cap.read()
    if ret:
        # Resize frame for iPhone to reduce network issues
        frame = cv2.resize(frame, (320, 240) if iphone_mode else (640, 480))

        if iphone_mode and frame.shape[0] > frame.shape[1]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Process every 3rd frame
        if update_frame.counter % 3 == 0:
            results = detector.detect_faces(frame)

            
            if results:
                largest_result = max(results, key=lambda res: res['box'][2] * res['box'][3])
                x, y, width, height = largest_result['box']

                # Calculate distances for the largest object only
                distance_focal = calculate_distance(width, current_focal_length)
                distance_smooth = smoothed_distance(distance_focal)

                # Display distance and label on GUI
                print(f"Distance: {distance_smooth:.2f} cm (using focal length {current_focal_length:.2f})")
                distance_text = f"Object 1 - Distance: {distance_smooth:.2f} cm"
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, distance_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Convert frame to ImageTk format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the display label
        display_label.imgtk = imgtk
        display_label.config(image=imgtk)

    update_frame.counter += 1
    # Slightly increase delay to reduce flickering
    display_label.after(30, update_frame)

# Initialize counter for update_frame
update_frame.counter = 0

# Open camera function
def open_camera(camera_source, focal_length, is_iphone=False):
    global cap, running, current_focal_length, iphone_mode
    if running:
        return

    iphone_mode = is_iphone
    current_focal_length = focal_length
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open camera.")
        return

    running = True
    update_frame()

# Close camera function
def close_camera():
    global cap, running
    running = False
    if cap:
        cap.release()
    cv2.destroyAllWindows()

# End program function to close the application
def end_program():
    close_camera()
    root.destroy()

# Tkinter GUI Setup
root = tk.Tk()
root.title("Camera Selector")
root.attributes("-fullscreen", True)

# Button Frame for camera selection
button_frame = tk.Frame(root)
button_frame.pack(side="top", pady=10)

# Buttons for selecting camera
laptop_button = tk.Button(button_frame, text="Laptop Camera",
                          command=lambda: open_camera(0, FOCAL_LENGTH_LAPTOP, is_iphone=False))
laptop_button.pack(side="left", padx=10)

iphone_button = tk.Button(button_frame, text="iPhone Camera",
                          command=lambda: open_camera("http://192.168.0.50:4747/video", FOCAL_LENGTH_IPHONE, is_iphone=True))
iphone_button.pack(side="left", padx=10)

# End Button to close the program
end_button = tk.Button(button_frame, text="End", command=end_program)
end_button.pack(side="left", padx=10)

# Label for displaying camera feed
display_label = tk.Label(root)
display_label.pack(expand=True)

# Closing event
root.protocol("WM_DELETE_WINDOW", end_program)
running = False
cap = None
iphone_mode = False

root.mainloop()
