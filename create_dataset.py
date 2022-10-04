from unittest import result
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_draw_style = mp.solutions.drawing_styles


def mp_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


cap = cv2.VideoCapture(3)
with mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
) as Hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        image, results = mp_detection(frame, Hands)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw_style.get_default_hand_landmarks_style(),
                    mp_draw_style.get_default_hand_connections_style(),
                )

        cv2.imshow("Raw feed", cv2.flip(image, 1))
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# results.right_hand_landmarks
