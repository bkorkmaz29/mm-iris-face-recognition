from functions.face.face_extraction import get_face_encodings, get_face_landmarks, get_face_locations
import numpy as np
import PIL.Image
#import imutils
import dlib
import cv2


def rect_to_css(rect):

    return rect.top(), rect.right(), rect.bottom(), rect.left()


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def css_to_rect(css):

    return dlib.rectangle(css[3], css[0], css[1], css[2])


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


# construct the argument parser and parse the arguments
#im = PIL.Image.open("face.jpg")
im = cv2.imread("face.jpg")

cv2.imshow("original", im)
cv2.waitKey(0)
image = np.array(im)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("grayscale", gray)
cv2.waitKey(0)
rect = get_face_locations(gray)
for r in rect:
    (x, y, w, h) = rect_to_bb(r)
    cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("landmarks", gray)
cv2.waitKey(0)

shape = get_face_landmarks(gray)
print(len(shape))
if shape:
    shapenp = shape_to_np(shape[0])

    for (x, y) in shapenp:
        cv2.circle(gray, (x, y), 1, (0, 0, 255), 3)


cv2.imshow("landmarks", gray)
cv2.waitKey(0)
