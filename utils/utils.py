import copy
import itertools
from typing import Mapping, Tuple
from collections import Counter, deque

import numpy as np
from mediapipe.python.solutions import drawing_styles as ds


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


def normalize_landmarks(image, landmarks):
    norm_landmark_list = []
    for landmark in landmarks:
        landmark_list = validate_landmarks_bounds(image, landmark)
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


def validate_landmarks_bounds(image, landmarks):
    landmark_point = []
    image_width, image_height = image.shape[1], image.shape[0]

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


DrawingSpec = ds.DrawingSpec
_RADIUS = ds._RADIUS - 1
_RED = ds._RED
_GREEN = ds._GREEN
_BLUE = ds._BLUE
_YELLOW = ds._YELLOW
_GRAY = ds._GRAY
_PURPLE = ds._PURPLE
_PEACH = ds._PEACH
_WHITE = ds._WHITE

# Hands
_THICKNESS_WRIST_MCP = ds._THICKNESS_WRIST_MCP - 1
_THICKNESS_FINGER = ds._THICKNESS_FINGER - 1
_THICKNESS_DOT = ds._THICKNESS_DOT - 1


_HAND_LANDMARK_STYLE = {
    ds._PALM_LANMARKS: DrawingSpec(
        color=_GRAY, thickness=_THICKNESS_DOT, circle_radius=_RADIUS
    ),
    ds._THUMP_LANDMARKS: DrawingSpec(
        color=_PEACH, thickness=_THICKNESS_DOT, circle_radius=_RADIUS
    ),
    ds._INDEX_FINGER_LANDMARKS: DrawingSpec(
        color=_PURPLE, thickness=_THICKNESS_DOT, circle_radius=_RADIUS
    ),
    ds._MIDDLE_FINGER_LANDMARKS: DrawingSpec(
        color=_YELLOW, thickness=_THICKNESS_DOT, circle_radius=_RADIUS
    ),
    ds._RING_FINGER_LANDMARKS: DrawingSpec(
        color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS
    ),
    ds._PINKY_FINGER_LANDMARKS: DrawingSpec(
        color=_BLUE, thickness=_THICKNESS_DOT, circle_radius=_RADIUS
    ),
}

_HAND_CONNECTION_STYLE = {
    ds.hands_connections.HAND_PALM_CONNECTIONS: DrawingSpec(
        color=_WHITE, thickness=_THICKNESS_WRIST_MCP
    ),
    ds.hands_connections.HAND_THUMB_CONNECTIONS: DrawingSpec(
        color=_WHITE, thickness=_THICKNESS_FINGER
    ),
    ds.hands_connections.HAND_INDEX_FINGER_CONNECTIONS: DrawingSpec(
        color=_WHITE, thickness=_THICKNESS_FINGER
    ),
    ds.hands_connections.HAND_MIDDLE_FINGER_CONNECTIONS: DrawingSpec(
        color=_WHITE, thickness=_THICKNESS_FINGER
    ),
    ds.hands_connections.HAND_RING_FINGER_CONNECTIONS: DrawingSpec(
        color=_WHITE, thickness=_THICKNESS_FINGER
    ),
    ds.hands_connections.HAND_PINKY_FINGER_CONNECTIONS: DrawingSpec(
        color=_WHITE, thickness=_THICKNESS_FINGER
    ),
}


def get_custom_style() -> Mapping[int, DrawingSpec]:
    """Returns the default hand landmarks drawing style.
    Returns:
        A mapping from each hand landmark to its default drawing spec.
    """
    hand_landmark_style = {}
    for k, v in _HAND_LANDMARK_STYLE.items():
        for landmark in k:
            hand_landmark_style[landmark] = v
    return hand_landmark_style


def get_custom_connections_style() -> Mapping[Tuple[int, int], DrawingSpec]:
    """Returns the default hand connections drawing style.
    Returns:
        A mapping from each hand connection to its default drawing spec.
    """
    hand_connection_style = {}
    for k, v in _HAND_CONNECTION_STYLE.items():
        for connection in k:
            hand_connection_style[connection] = v
    return hand_connection_style
