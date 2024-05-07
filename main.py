import cv2
import numpy as np

# Load face and eye classifiers
face_classifier = cv2.CascadeClassifier('./inputs/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('./inputs/haarcascade_eye_tree_eyeglasses.xml')

# Load the image
image_path = "./inputs/images/person2eyes.jpg"
image = cv2.imread(image_path)

# Resize the image
if image.shape[0] > 500:
    width = 500
    ratio = float(width) / image.shape[1]
    height = int(image.shape[0] * ratio)
    image = cv2.resize(image, (width, height))
# copy image

faked_image = np.copy(image)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faked_gray_image = cv2.cvtColor(faked_image, cv2.COLOR_BGR2GRAY)

# Apply eye detection ti the fake image to see wrong results
fake_eyes = eye_classifier.detectMultiScale(faked_gray_image)
for (ex, ey, ew, eh) in fake_eyes:
    cv2.rectangle(faked_image, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
# ----------------------------------------------------------------
# Detect faces first
faces = face_classifier.detectMultiScale(gray_image)
for (x, y, w, h) in faces:
    # Detect eyes within the face region
    face_space = gray_image[y:y + h, x:x + w]
    eyes = eye_classifier.detectMultiScale(face_space)

    # Draw a rectangle around the face
    cv2.rectangle(faked_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    print(len(eyes))
    if len(eyes) > 0:
        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (ex, ey, ew, eh) in eyes:
        # Check if eye coordinates are within the face bounding box
        cv2.rectangle(image, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 0, 255), 2)

# Display the result
cv2.imshow("fake_results", faked_image)
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()