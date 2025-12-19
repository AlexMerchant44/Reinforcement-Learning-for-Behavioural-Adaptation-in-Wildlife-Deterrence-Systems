from picamera2 import Picamera2
import cv2

picam2 = None

def init_camera():
    global picam2
    picam2 = Picamera2()

    # Single main stream, colour, rotated later in get_frame()
    config = picam2.create_preview_configuration(
        main={"size": (1080, 1920), "format": "RGB888"},
        display="main",
    )

    picam2.configure(config)
    picam2.start()

init_camera()

def get_frame():
    """
    Return a rotated BGR frame from the main stream.
    """
    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    rotated_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    return rotated_bgr