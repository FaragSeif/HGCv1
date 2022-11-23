import cv2
import mediapipe as mp


class MPDetectionStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.model = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

    def read_frame(self, draw_landmarks=False):
        frame, landmarks = self._detect_landmarks()
        if draw_landmarks:
            for landmark in landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    landmark,
                    self.mp_hands.HAND_CONNECTIONS,
                    # TODO:
                )
        frame = cv2.flip(frame, 1)
        return frame, landmarks

    def _detect_landmarks(self):
        ret, frame = self.stream.read()
        if not ret:
            return None, []
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.model.process(image)
        landmarks = results.multi_hand_landmarks

        if landmarks is None:
            landmarks = []
        return frame, landmarks
