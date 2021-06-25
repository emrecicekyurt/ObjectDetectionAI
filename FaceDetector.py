import cv2
from random import randrange

# Taking the data that is xml file provided by opencv
faceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the image
# img = cv2.imread("ElonMusk.jpg")
img = cv2.imread("MultipleFace.jpg")

# The image must be converted to white and black since the computer detects properly in this manner
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detecting the faces by using detectMultiScale
face_cr = faceData.detectMultiScale(grayImg)
# print(face_cr) <- It shows the coordinates of the face detected
# Create rectangle around of the faces detected. Coordinates come from face_cr obj
for (x, y, w, h) in face_cr:
    cv2.rectangle(img, (x, y), (x+w,y+h), (randrange(256),randrange(256),randrange(256)), 3)

# It shows the output image. By using waitkey(0) we can see the output continuously.
cv2.imshow("Output", img)
cv2.waitKey(0)
print("cv2 is imported")