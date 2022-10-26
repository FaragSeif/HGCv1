import cv2
import time
import numpy as np
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)

volume = cast(interface, POINTER(IAudioEndpointVolume))
range = volume.GetVolumeRange()
mute = range[0]
maxVol = range[1]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
previous_time = 0

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


with mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
) as Hands:
    while cap.isOpened():
        successful, frame = cap.read()
        if not successful:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        image, results = mp_detection(frame, Hands)
        if results.multi_hand_landmarks:
            # print(results.multi_hand_landmarks[0].landmark[0])
            for hand_landmarks in results.multi_hand_landmarks:
                tips = []
                mp_draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw_style.get_default_hand_landmarks_style(),
                    mp_draw_style.get_default_hand_connections_style(),
                )
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, _ = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id == 4:
                        cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                        tips.append([cx, cy])
                    if id == 8:
                        cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                        tips.append([cx, cy])
                cv2.line(
                    image,
                    (tips[0][0], tips[0][1]),
                    (tips[1][0], tips[1][1]),
                    (255, 0, 255),
                    3,
                )
                length = math.hypot(tips[1][0] - tips[0][0], tips[1][1] - tips[0][1])
                vol = np.interp(length, [50, 150], [mute, maxVol])
                barLength = np.interp(length, [50, 150], [400, 150])
                volume.SetMasterVolumeLevel(vol, None)
                cv2.rectangle(image, (50, 150), (85, 400), (139, 139, 0), 3)
                cv2.rectangle(
                    image, (50, int(barLength)), (85, 400), (139, 139, 0), cv2.FILLED
                )

        cv2.imshow("Volume Control", cv2.flip(image, 1))
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
