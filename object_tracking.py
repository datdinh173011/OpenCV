import imutils
import time
import cv2
from keras.models import load_model
import numpy as np

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

tracker = OPENCV_OBJECT_TRACKERS["csrt"]()
initBB = None
model = load_model('model.h5')
vs = cv2.VideoCapture(0)
drawed = []
result = ""
delay_text = 0

while True:
    _, frame = vs.read()
    if frame is None:
        break

    frame = imutils.resize(frame, width=500)

    frame = cv2.flip(frame, 1)
    draw_frame = frame.copy()

    for i in range(1, len(drawed)):
        cv2.line(draw_frame, (drawed[i - 1][0], drawed[i - 1][1]), (drawed[i][0], drawed[i][1]), (0, 255, 0), 3)

    H, W, _ = frame.shape
    if initBB is not None:
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(draw_frame, (x, y), (x + w, y + h),
                          (255, 0, 0), 2)
            cv2.circle(draw_frame, (x + w // 2, y + h // 2), 1, (255, 0, 0), -1)
            drawed.append([x + w // 2, y + h // 2])

    if delay_text > 0:
        cv2.putText(draw_frame, result, (W // 2 - 30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Frame", draw_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("d"):

        delay_text = 20

        xmin, ymin = np.min(drawed, 0)
        xmax, ymax = np.max(drawed, 0)

        table_width = xmax - xmin + 20
        table_height = ymax - ymin + 20

        table = np.zeros((table_height, table_width, 3))
        for i in range(1, len(drawed)):
            cv2.line(table, (drawed[i - 1][0] - xmin + 10, drawed[i - 1][1] - ymin + 10),
                     (drawed[i][0] - xmin + 10, drawed[i][1] - ymin + 10), (255, 255, 255), 3)
        table = cv2.resize(table, (28, 28))
        table = table[:, :, 0] / 255.
        table = np.expand_dims(table, 0)
        table = np.expand_dims(table, -1)
        predictions = np.argmax(model.predict(table)[0])
        if predictions == 0:
            result = "Square"
        elif predictions == 1:
            result = "Circle"
        else:
            result = "Triangle"

        drawed = []

    elif key == ord("s"):
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                               showCrosshair=True)
        tracker.init(frame, initBB)
    elif key == ord("q"):
        break

    delay_text -= 1
vs.release()

cv2.destroyAllWindows()