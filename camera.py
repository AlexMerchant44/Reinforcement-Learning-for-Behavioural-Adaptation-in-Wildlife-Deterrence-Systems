from picamera2 import Picamera2
import cv2

picam2 = None

def init_camera():
    global picam2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (1080, 1920)},
    )
    picam2.configure(config)
    picam2.start()

init_camera()

def get_frame():
    """Return a BGR frame (numpy array) from the camera."""
    frame = picam2.capture_array()
    # Picamera2 gives RGB; OpenCV uses BGR
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    rotated = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    return rotated