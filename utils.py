from collections import Counter, deque
import numpy as np
import itertools
import copy


class GestureFilter:
    def __init__(self, filter_len=10):
        self.filter_len = filter_len
        self._buffer = deque(maxlen=filter_len)

    def add_gesture(self, gesture_id):
        self._buffer.append(gesture_id)

    def get_gesture(self):
        counter = Counter(self._buffer).most_common()
        if counter[0][1] >= (self.filter_len - 4):
            self._buffer.clear()
            return counter[0][0]
        else:
            return


def draw_landmarks(image: np.ndarray, hand_landmarks, mp_hands, mp_draw):
    """
    draw the landmarks on the image
    """
    mp_draw.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        get_custom_style(),
        get_custom_connections_style(),
    )
    return image


def normalize_landmarks(image: np.ndarray, results):
    """
    normalize the landmarks to be in the range of 0 to 1
    """
    # get the landmarks
    landmark_list = validate_landmarks_bounds(image, results)

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

    def normalize_(n):
        return n / max_value

    norm_landmark_list = list(map(normalize_, flat_landmark_list))

    # return the normalized landmarks
    return norm_landmark_list


def validate_landmarks_bounds(img: np.ndarray, landmarks):
    """
    extract the hand landmarks from the mediapipe results
    append the landmarks to a list and return it
    """
    landmark_point = []

    image_width, image_height = img.shape[1], img.shape[0]
    # Extracting the key points
    for lm in landmarks.landmark:
        landmark_x = min(int(lm.x * image_width), image_width - 1)
        landmark_y = min(int(lm.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    # return list of hand landmarks
    return landmark_point
