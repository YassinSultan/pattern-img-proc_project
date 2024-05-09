import cv2
import numpy as np

# Load face and eye classifiers
face_classifier = cv2.CascadeClassifier('./inputs/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('./inputs/haarcascade_eye_tree_eyeglasses.xml')
nose_classifier = cv2.CascadeClassifier('./inputs/haarcascade_mcs_nose.xml')

# Load the image
image_path = "./inputs/images/person6.jpg"
image = cv2.imread(image_path)

# Resize the image
if image.shape[0] > 500:
    width = 500
    ratio = float(width) / image.shape[1]
    height = int(image.shape[0] * ratio)
    image = cv2.resize(image, (width, height))

# ------------------------Start -> Fake Code ----------------------------------------
faked_image = np.copy(image)
faked_gray_image = cv2.cvtColor(faked_image, cv2.COLOR_BGR2GRAY)

# Apply face detection to the fake image to see wrong results
fake_faces = face_classifier.detectMultiScale(faked_gray_image)
for (fx, fy, fw, fh) in fake_faces:
    cv2.rectangle(faked_image, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)

# Apply eye detection to the fake image to see wrong results
fake_eyes = eye_classifier.detectMultiScale(faked_gray_image)
for (ex, ey, ew, eh) in fake_eyes:
    cv2.rectangle(faked_image, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

# Apply nose detection to the fake image to see wrong results
fake_nose = nose_classifier.detectMultiScale(faked_gray_image)
for (nx, ny, nw, nh) in fake_nose:
    cv2.rectangle(faked_image, (nx, ny), (nx + nw, ny + nh), (255, 255, 255), 2)
# ---------------------------- Start -> Original Code -----------------------------------
# Detect faces first
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(gray_image)
for (fx, fy, fw, fh) in faces:
    # Detect eyes and nose within the face region
    face_space = gray_image[fy:fy + fh, fx:fx + fw]
    eyes = eye_classifier.detectMultiScale(face_space)
    nose = nose_classifier.detectMultiScale(face_space)

    # check if this face have eyes
    if len(eyes) > 0:
        # Draw a rectangle around the face
        cv2.rectangle(image, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)

    # draw the rectangle around the eyes
    for (ex, ey, ew, eh) in eyes:
        # draw the rectangle around the nose
        for (nx, ny, nw, nh) in nose:
            if ny > ey:
                cv2.rectangle(image, (fx + ex, fy + ey), (fx + ex + ew, fy + ey + eh), (0, 0, 255), 2)
                # cv2.rectangle(image, (x + nx, y + ny), (x + nx + nw, y + ny + nh), (255, 0, 255), 2)


# Display the result
cv2.imshow("fake_results", faked_image)
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
