from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
import cv2
import os

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

def start_recording(path):
    global picam2
    encoder = H264Encoder(bitrate=10_000_000)  # good quality
    output = FileOutput(path)
    picam2.start_recording(encoder, output)
    print(f"[Camera] Started recording → {path}")


def stop_recording():
    global picam2
    try:
        picam2.stop_recording()
        print("[Camera] Stopped recording")
    except Exception as e:
        print(f"[Camera] stop_recording error: {e}")