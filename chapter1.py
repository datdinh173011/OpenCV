import cv2
import numpy as np

print("Package Imported")

img =cv2.imread("Resources/face.jpg")
kernel = np.ones((5,5),np.uint8)

# cv2.imshow("Output", img)
# cv2.waitKey(0)
# cap =cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)
#
# while True:
#     success, img = cap.read()
#     cv2.imshow("Video", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
imgBlur = cv2.GaussianBlur(imgGray,(7,7),0)
imgCanny = cv2.Canny(img,150,200)
imgDialation = cv2.dilate(imgCanny,kernel,iterations=5)
imgEroded = cv2.erode(imgDialation,kernel,iterations=1)

cv2.imshow("Gray Image",imgGray)
cv2.imshow("Blur Image",imgBlur)
cv2.imshow("Canny Image",imgCanny)
cv2.imshow("Dialation Image",imgDialation)
cv2.imshow("Erored Image",imgEroded)
cv2.waitKey(0)