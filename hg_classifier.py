from kp_classifier import KPClassifier
import mediapipe as mp
import numpy as np
from utils import normalize_landmarks, draw_landmarks
import cv2
import os


class HGClassifier:
    """
    Class to use mediapipe hand gesture classifier model
    to detect hand gesture from a single image
    pass it to tensor flow model and get the labels
    then map the labels to commands and return the command

    parameters:

    commands: list of commands to map the labels to them.
    model_path: path to the keypoint classifier model.
    min_detection_confidence: minimum confidence value for hand detection.
    min_tracking_confidence: minimum confidence value for hand tracking.

    """

    default_map = {
        0: "Forward",
        1: "STOP",
        2: "UP",
        3: "LAND",
        4: "DOWN",
        5: "Back",
        6: "Left",
        7: "Right",
    }

    def __init__(
        self,
        commands: list,
        model_path: str,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        # deifne model path, and mediapipe model parameters
        self._init_model(
            model_path,
            min_detection_confidence,
            min_tracking_confidence,
        )
        # if commands is not given by the user, use the default commands
        if commands is None:
            self.command_map = self.default_map
        else:
            self.command_map = {num: command for num, command in enumerate(commands)}

    def detect(
        self,
        img: np.ndarray,
        draw_on_image: bool,
    ):
        """
        detect hand gesture from a single image
        pass it to tensor flow model and get the labels
        map the labels to command and return the command

        parameters:
            img: image to detect hand gesture from it.
            draw_on_image: if True, draw the landmarks on the image.

        returns:
            command: the predicted command.
            image: the image with the landmarks drawn on it.
            resutls: the detected landmarks resutls.

        """
        # Make mediapipe detections
        img, results = self._mp_detection(img)

        # if hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # get the normalized landmarks
                landmark_list = normalize_landmarks(img, hand_landmarks)

                # call kp_classifier to get the predicted labels
                label = self.kp_classifier.predict(landmark_list)

                # draw the hand landmarks on the image
                if draw_on_image:
                    img = draw_landmarks(img, hand_landmarks)

                # convert command number to command string
                command = self.command_map[label]

            return command, img, results
        return "", img, None

    def _mp_detection(self, image: np.ndarray):
        """
        load mediapipe model, detect hand landmarks and return the image and results
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hand_model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def _init_model(
        self, model_path: str, min_detection_confidence, min_tracking_confidence
    ):
        """
        check the model_path is valid then initialize KPClassifier model
        and mediapipe model
        """
        if model_path is None:
            raise ValueError("model_path is None")
        if not os.path.exists(model_path):
            raise ValueError("model_path is not valid")
        else:
            # initialize the kp_classifier model
            self.kp_classifier = KPClassifier(model_path)

        # initialize mediapipe model
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hand_model = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
