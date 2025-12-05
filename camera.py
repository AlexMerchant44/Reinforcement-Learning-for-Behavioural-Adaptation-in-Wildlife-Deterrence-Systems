# camera.py
from picamera2 import Picamera2
import cv2

picam2 = None

def init_camera():
    global picam2
    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration())
    picam2.start()

def get_frame():
    """Return a BGR frame (numpy array) from the camera."""
    frame = picam2.capture_array()
    # Picamera2 gives RGB; OpenCV uses BGR
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame_bgr
