import cv2


# Taking the data that is xml file provided by opencv
faceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Using webcam
webcam = cv2.VideoCapture(0)

while True:
    success, img = webcam.read()

    # The image must be converted to white and black since the computer detects properly in this manner
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detecting the faces by using detectMultiScale
    face_cr = faceData.detectMultiScale(grayImg)
    # Create rectangle around of the faces detected. Coordinates come from face_cr obj
    for (x, y, w, h) in face_cr:
        cv2.rectangle(img, (x, y), (x+w,y+h), (255,0,255), 3)
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break








