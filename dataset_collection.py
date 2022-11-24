import os
import csv
import time
import logging
from threading import Thread

import cv2
import mediapipe as mp

from utils import normalize_landmarks
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
            landmark_list = normalize_landmarks(self.frame, self.landmarks)
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
