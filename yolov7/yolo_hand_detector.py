import cv2
from yolov7_package import Yolov7Detector
from time import perf_counter

# function to detect hands in images
def yolo_detection(image, model):
    """
    :param image: image to detect hands
    :param model: YOLOv7 model
    :return: image with bounding boxes

    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 640))
    image.flags.writeable = False
    # start = perf_counter()
    classes, boxes, scores = model.detect(image)
    # end = perf_counter()
    # print(f"Time taken: {end - start}")
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = [classes, boxes, scores]
    return results

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load YOLOv7 model
model = Yolov7Detector(traced=False)
while cap.isOpened():

    # Read feed
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    # Make detections
    results = yolo_detection(frame, model)

    # draw bounding boxes and score on the image
    if results[1] is not None:
        for i in range(len(results[1])):
            if results[2][i][0] > 0.5:
                frame = model.draw_on_image(frame, results[1][i], results[2][i], results[0][i])
    
    
    # Display the resulting frame
    cv2.imshow('Yolo Detection', frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()