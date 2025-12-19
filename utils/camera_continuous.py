# camera_test_fullwide.py
import cv2
from picamera2 import Picamera2

def main():
    picam2 = Picamera2()

    # Full sensor resolution for Camera Module 3 Wide
    config = picam2.create_preview_configuration(
        main={"size": (1080, 1920)},
    )
    picam2.configure(config)
    picam2.start()

    print("Camera started at 4608x2592. Press 'q' to quit.")
    cv2.namedWindow("Cam3 Wide – full FOV", cv2.WINDOW_NORMAL)

    while True:
        frame = picam2.capture_array()

        # Convert RGB → BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Rotate manually since libcamera rotation is unavailable
        rotated = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
        # If wrong direction, use ROTATE_90_COUNTERCLOCKWISE instead

        cv2.imshow("Cam3 Wide – full FOV", rotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
