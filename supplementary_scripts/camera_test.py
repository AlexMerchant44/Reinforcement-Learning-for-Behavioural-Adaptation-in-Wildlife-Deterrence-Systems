# testcamera.py
import cv2
from picamera2 import Picamera2
import time

def main():
    picam2 = Picamera2()

    config = picam2.create_preview_configuration(
        main={"size": (4608, 2592)},
        transform=Transform(rotation=90)
    )
    picam2.configure(config)
    picam2.start()

    print("Press 'q' to quit")

    while True:
        frame = picam2.capture_array()

        # Convert RGB→BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Rotate 90° clockwise
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Show in **one** window called "Camera"
        cv2.imshow("Camera", frame)

        # break when 'q' pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
