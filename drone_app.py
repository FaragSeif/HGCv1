import cv2
from djitellopy import Tello

from utils import GestureFilter
from models.hg_classifier import HGClassifier
from examples.drone_controller import DroneController


def main():

    # hand gesture classifier
    hg_classifier = HGClassifier(
        commands=None, model_path="models/keypoint_classifier.tflite"
    )
    # drone controller
    drone_controller = DroneController(Tello())
    # gesture filter
    gesture_filter = GestureFilter()

    try:
        while True:
            # detect the hand gesture
            command, image = hg_classifier.detect(draw_on_image=True)
            # Show the prediceted landmarks and command to screen
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

        cv2.destroyAllWindows()

    finally:
        # drone land and disconnect
        drone_controller.tello.land()
        drone_controller.tello.end()


if __name__ == "__main__":
    main()
