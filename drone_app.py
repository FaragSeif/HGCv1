from hg_classifier import HGClassifier
from drone_controller import DroneController
from utils import GestureFilter
from djitellopy import Tello
import cv2


def main():

    # hand gesture classifier
    hg_classifier = HGClassifier(
        commands=None,
        model_path="models/keypoint_classifier.tflite",
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    # drone controller
    drone_controller = DroneController(Tello())
    # gesture filter
    gesture_filter = GestureFilter()

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    try:
        # drone takeoff
        drone_controller.tello.takeoff()
        while True:

            # camera capture
            ret, frame = cap.read()
            if not ret:
                continue

            # detect the hand gesture
            command, image, _ = hg_classifier.detect(frame, draw_on_image=True)

            # Show the prediceted landmarks and command to screen
            image = cv2.flip(image, 1)
            cv2.putText(
                image,
                "command: " + command,
                (0, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.imshow("OpenCV Feed", image)

            # filter the getures to avoid false positives
            gesture_filter.add_gesture(command)

            # Send command to drone
            drone_controller.command_control(gesture_filter)

            # break the loop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        # release the camera
        cap.release()
        cv2.destroyAllWindows()

    # drone land and disconnect
    except KeyboardInterrupt:
        drone_controller.tello.land()
        drone_controller.tello.end()


if __name__ == "__main__":
    main()
