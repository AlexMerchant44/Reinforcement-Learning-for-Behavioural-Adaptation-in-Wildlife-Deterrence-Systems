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

def start_recording(path):
    """
    Start recording a video to the specified .h264 or .mp4 path.
    """
    if not path.endswith(".h264") and not path.endswith(".mp4"):
        raise ValueError("Video path must end with .h264 or .mp4")

    picam2.start_recording(path)
    print(f"[Camera] Started recording → {path}")


def stop_recording():
    """
    Stop recording the active video.
    """
    try:
        picam2.stop_recording()
        print("[Camera] Stopped recording")
    except Exception as e:
        print(f"[Camera] stop_recording error: {e}")