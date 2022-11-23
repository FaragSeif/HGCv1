import os
import csv
import time
import copy
import logging
import itertools
from threading import Thread

import cv2
import numpy as np
import mediapipe as mp

from testt import get_custom_style, get_custom_connections_style
from mp_stream import MPDetectionStream

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)


class DataCollector:
    def __init__(
        self,
        num_labels,
        samples_per_label: int,
        label_names,
        dataset_path,
    ):
        self.num_labels = num_labels
        self.samples_per_label = samples_per_label
        self.label_names = label_names
        self.dataset_path = dataset_path
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.model = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        self.mp_stream = MPDetectionStream(src=0)
        thread = Thread(target=self.show_mp_detections, daemon=True)
        thread.start()

    def show_mp_detections(self):
        """Update and show detections."""
        while True:
            self.frame, self.landmarks = self.mp_stream.read_frame(draw_landmarks=True)
            # Draw the hand annotations on the image.
            cv2.imshow("MediaPipe Hands", self.frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    def get_landmarks(self, count: int = float("inf")):
        i = 0
        while i < count:
            if self.landmarks == []:
                time.sleep(1 / 50)
                continue
            landmark_list = self.normalize_landmarks(self.frame, self.landmarks)
            self.landmarks = []
            yield landmark_list
            i += 1

    def collect(self):
        with open(self.dataset_path, "a", newline="") as f:
            for label_id in range(self.num_labels):
                val = input(
                    f'Starting Data Collection for "{self.label_names[label_id].upper()}" gesture, Ready? (y/n): '
                )
                if val.lower() != "y":
                    return

                for normalized_landmarks in self.get_landmarks(self.samples_per_label):
                    self.write_landmarks(f, label_id, normalized_landmarks)

    def write_landmarks(self, file, label_id, landmark_list):
        """Write landmarks to a csv dataset file

        Args:
            file (file): File object returned by open()
            label_id (int): ID of the label to be put at the first column
            landmark_list ([int]): list of normalized landmarks for detection
        """
        if 0 <= label_id <= self.num_labels - 1:
            writer = csv.writer(file)
            writer.writerow([label_id, *landmark_list])

    def normalize_landmarks(self, image, landmarks):
        norm_landmark_list = []
        for landmark in landmarks:
            landmark_list = self.validate_landmarks_bounds(image, landmark)
            temp_landmark_list = copy.deepcopy(landmark_list)

            # Convert to relative coordinates
            base_x, base_y = 0, 0
            for index, landmark_point in enumerate(temp_landmark_list):
                if index == 0:
                    base_x, base_y = landmark_point[0], landmark_point[1]

                temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
                temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

            # Convert to a one-dimensional list (flatten the list)
            flat_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

            # Normalization
            max_value = max(list(map(abs, flat_landmark_list)))

            norm_landmark_list += [n / max_value for n in flat_landmark_list]

        return norm_landmark_list

    def validate_landmarks_bounds(self, image, landmarks):
        landmark_point = []
        image_width, image_height = image.shape[1], image.shape[0]

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])

        return landmark_point
