import cv2
import os
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_draw_style = mp.solutions.drawing_styles

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

data_path = os.path.join("dataset")
actions = np.array(["front", "back", "right", "left", "stop"])
no_sequences = 5
sequence_length = 5


for action in actions:
    try:
        os.makedirs(os.path.join(data_path, action))
    except:
        pass


def mp_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


all_kp_values = []
key_points = []


# Initialize mediapipe
with mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
) as Hands:

    # loop through actions
    for action in actions:
        # loop through sequences
        for sequence in range(no_sequences):
            # lopp through sequence length
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()
                print(frame.shape)

                if not ret:
                    print("Ignoring empty camera frame.")
                    continue

                # Make detections
                image, results = mp_detection(frame, Hands)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks of the hand
                        mp_draw.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_draw_style.get_default_hand_landmarks_style(),
                            mp_draw_style.get_default_hand_connections_style(),
                        )
                        # Extracting the key points
                        for lm in hand_landmarks.landmark:
                            key_point = np.array([lm.x, lm.y, lm.z])
                            key_points.append(key_point)
                    # flatten the output of all the key points
                    all_kp_values = np.array(key_points).flatten()

                # take a break for 3 second it will help to collect the data
                if frame_num == 0:
                    cv2.putText(
                        image,
                        "STARTING COLLECTION",
                        (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        4,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        image,
                        "Collecting frames for {} Video Number {}".format(
                            action, sequence
                        ),
                        (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    # cv2.waitKey(500)
                else:
                    cv2.putText(
                        image,
                        "Collecting frames for {} Video Number {}".format(
                            action, sequence
                        ),
                        (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.waitKey(500)

                # show the image
                cv2.imshow("Raw feed", image)

                # export the key points
                np_path = os.path.join(data_path, action, str(sequence)+str(frame_num))
                np.save(np_path, all_kp_values)

                # break the loop if 'q' is pressed
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

    # release the camera
    cap.release()
    cv2.destroyAllWindows()
