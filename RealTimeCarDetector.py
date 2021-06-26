import cv2

# Load image
img_file = "car.jpg"

# Pre-trained car classification
data_file = "cars.xml"

# Initializing the img
img = cv2.imread(img_file)

# Converting the image to black white version
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Car classification
car_tracker = cv2.CascadeClassifier(data_file)

video = cv2.VideoCapture("images/cars_video.mp4")

while True:
    success, frame = video.read()
    if success:
        # Converting the frame to black and white
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detecting the coordinates of cars
    car_coord = car_tracker.detectMultiScale(gray_frame)
    for (x, y, w, h) in car_coord:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # Displaying the img
    cv2.imshow("Result", frame)
    cv2.waitKey(1)

print("Code is working well")

