import os

from models.kp_classifier import KPClassifier
from utils import MPDetectionStream
from utils import normalize_landmarks


class HGClassifier:
    """
    Class to use mediapipe hand gesture classifier model
    to detect hand gesture from a single image
    pass it to tensor flow model and get the labels
    then map the labels to commands and return the command

    parameters:

    commands: list of commands to map the labels to them.
    model_path: path to the keypoint classifier model.
    """

    default_map = {
        0: "LAND",
        1: "STOP",
        2: "FORWARD",
        3: "BACK",
        4: "UP",
        5: "LEFT",
        6: "RIGHT",
        7: "TAKEOFF",
    }

    def __init__(
        self,
        commands: list,
        model_path: str,
    ) -> None:

        self.stream = MPDetectionStream(src=0)

        # if commands is not given by the user, use the default commands
        if commands is None:
            self.command_map = self.default_map
        else:
            self.command_map = {num: command for num, command in enumerate(commands)}

        # check the model_path is valid
        if model_path is None:
            raise ValueError("model_path is None")
        if not os.path.exists(model_path):
            raise ValueError("model_path is not valid")
        else:
            # initialize the kp_classifier model
            self.kp_classifier = KPClassifier(model_path)

    def detect(
        self,
        draw_on_image: bool,
    ):
        """
        detect hand landmarks from a single image, normalize it,
        then pass it to tensor flow model and get the labels,
        map the labels to command and return the command

        parameters:
            draw_on_image: if True, draw the landmarks on the image.

        returns:
            command: the predicted command.
            image: the image with the landmarks drawn on it.
        """
        # Make mediapipe detections
        img, landmarks = self.stream.read_frame(draw_landmarks=draw_on_image)
        # get the normalized landmarks
        landmark_list = normalize_landmarks(img, landmarks)

        # if no landmarks detected, return empty string and img
        if landmark_list == []:
            return "", img

        # call kp_classifier to get the predicted labels
        label = self.kp_classifier.predict(landmark_list)

        # convert command number to command string
        command = self.command_map[label]

        return command, img
