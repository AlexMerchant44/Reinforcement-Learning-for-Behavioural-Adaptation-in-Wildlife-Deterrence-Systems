# testcamera.py
# Streams the Pi camera feed and rotates 90 degrees clockwise

import cv2
from picamera2 import Picamera2

def main():
    picam2 = Picamera2()

    # Basic preview configuration
    config = picam2.create_preview_configuration()
    picam2.configure(config)
    picam2.start()

    print("Press 'q' to quit")

    while True:
        frame = picam2.capture_array()

        # Picamera2 outputs RGB → convert to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Rotate 90 degrees clockwise
        rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        cv2.imshow("Camera Stream (Rotated 90° CW)", rotated)

        # Quit when 'q' pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
