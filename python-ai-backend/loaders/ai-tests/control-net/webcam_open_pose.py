import cv2
import os
import time
import threading

# This is a directory where you want to save the snapshots.
SAVE_DIR = "snapshots"

# Create a directory if it doesn't exist.
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Open a connection to the first webcam.
cap = cv2.VideoCapture(0)

# Set the desired resolution (512x512).
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

# Create an index for the saved images.
img_index = 0

# Variable to track if countdown is running.
countdown_running = False

def countdown_timer():
    global countdown_running, img_index

    for i in range(5, 0, -1):
        ret, frame = cap.read()

        if not ret:
            return

        countdown_frame = frame.copy()
        countdown_text = f"Taking picture in {i} seconds..."
        text_position = (100, 100)
        text_color = (255, 0, 0)  # Green color
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 1
        text_thickness = 2
        cv2.putText(countdown_frame, countdown_text, text_position, text_font, text_scale, text_color, text_thickness)
        cv2.imshow("Webcam Stream", countdown_frame)
        cv2.waitKey(1000)  # Wait for 1 second

    ret, frame = cap.read()

    if not ret:
        return

    img_name = f"{SAVE_DIR}/snapshot_{img_index}.png"
    cv2.imwrite(img_name, frame)
    print(f"Saved {img_name}")
    img_index += 1

    countdown_running = False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Webcam Stream", frame)

    key = cv2.waitKey(1)

    if key == 32 and not countdown_running:
        countdown_running = True
        threading.Thread(target=countdown_timer).start()

    elif key in [81, 113]:  # ASCII for 'Q' and 'q'
        break

cap.release()
cv2.destroyAllWindows()
