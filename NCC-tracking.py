import cv2
import numpy as np

# To capture video from webcam.
cap = cv2.VideoCapture(0)

# select region
while True:
    # Read the frame and Convert to grayscale
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    top_left = (200, 300)
    bottom_right = (400, 400)
    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

    # Display
    cv2.imshow('img', img)

    # press the 'a' key
    if cv2.waitKey(33) == ord('a'):
        break

# create a template
template = gray[300:400, 200:400]
w, h = template.shape[::-1]
cv2.imshow('template', template)
cv2.waitKey(0)  # press any keys

# tracking
while True:
    # Read the frame and Convert to grayscale
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corr_map = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr_map)

    # take minimum
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # drawa
    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

    # Display
    cv2.imshow('img', img)

    # press the 'a' key
    if cv2.waitKey(33) == ord('a'):
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()