from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
import cv2

picam2 = None
current_encoder = None  # keep a reference so it isn't garbage-collected


def init_camera():
    global picam2
    picam2 = Picamera2()

    # Use a video configuration and add a lores stream for detection
    config = picam2.create_video_configuration(
        main={"size": (1080, 1920)},   # used for recording
        lores={"size": (720, 1280)},    # used for detection frames
        display="main",
    )
    picam2.configure(config)
    picam2.start()


# Initialise on import
init_camera()


def get_frame():
    """
    Return a BGR frame (numpy array) from the camera (lores stream).
    """
    # Explicitly use the lores stream so it doesn't clash with the encoder
    frame = picam2.capture_array("lores")
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    rotated = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    return rotated


def start_recording(path):
    """
    Start recording and save to the given path.
    """
    global picam2, current_encoder

    # If for some reason we were already recording, stop first (defensive)
    try:
        picam2.stop_recording()
    except Exception:
        pass

    current_encoder = H264Encoder(bitrate=10_000_000)
    output = FileOutput(path)
    picam2.start_recording(current_encoder, output)
    print(f"[Camera] Started recording → {path}")


def stop_recording():
    """
    Stop recording previously started video.
    """
    global picam2, current_encoder
    try:
        picam2.stop_recording()
        print("[Camera] Stopped recording")
    except Exception as e:
        print(f"[Camera] stop_recording error: {e}")
    finally:
        current_encoder = None
