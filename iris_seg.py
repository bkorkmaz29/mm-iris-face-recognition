from functions.iris.iris_extraction import iris_extract
from functions.iris.iris_line import findline
from functions.iris.iris_normalize import normalize
from functions.iris.iris_segment import segment
import numpy as np
import cv2
im = cv2.imread("data/CASIA1/012/1/012_1_1.bmp")

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_array = np.array(gray)
radial_res = 32
angular_res = 240


def perform_iris_segmentation(eye_image):
    # Preprocess the image (convert to grayscale, enhance contrast, etc.)
    gray_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)

    # Apply any necessary preprocessing steps (e.g., smoothing, histogram equalization)
    blurred_img = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Apply iris segmentation technique (e.g., edge detection, thresholding)
    # Example: Canny edge detection
    threshold1 = np.mean(blurred_img) * 0.2
    threshold2 = np.mean(blurred_img) * 1.2
    edges = cv2.Canny(blurred_img, threshold1, threshold2)

    # Find contours of the iris region
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and select the largest iris contour
    iris_contour = max(contours, key=cv2.contourArea)

    # Draw the iris contour on the original image for visualization
    iris_image = cv2.drawContours(
        eye_image.copy(), [iris_contour], -1, (0, 255, 0), 2)

    return iris_image, iris_contour


# Iris segmentation
i, p, imwn = segment(im_array, 80)
cv2.imshow('Processed image', imwn)
cv2.waitKey(0)
# Drawing circles for iris and pupil
circles = [i, p]
for (x, y, r) in circles:
    cv2.circle(im, (x, y), r, (0, 255, 0), 2)
    cv2.rectangle(im, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
cv2.imshow('Iris and pupil circles', im)
cv2.waitKey(0)

lines = findline(im_array)
for line in lines:
    cos_theta = line[0]
    sin_theta = line[1]
    rho = -line[2]

    x0 = cos_theta * rho
    y0 = sin_theta * rho
    scale = im.shape[0] + im.shape[1]

    x1 = int(x0 + scale * (-sin_theta))
    y1 = int(y0 + scale * cos_theta)
    x2 = int(x0 - scale * (-sin_theta))
    y2 = int(y0 - scale * cos_theta)

    # BGR format (Blue: 0, Green: 0, Red: 255), thickness = 2
    cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)


# Display the image with the lines
cv2.imshow("Lines", im)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Normalize the iris section
iris_x, iris_y, iris_r = i
pupil_x, pupil_y, pupil_r = p
polar_array, noise = normalize(
    imwn, iris_x, iris_y, iris_r, pupil_x, pupil_y, pupil_r, 20, 240)


polar_array = cv2.resize(polar_array, (320, 120))


cv2.imshow('Normalized iris', polar_array)
cv2.waitKey(0)


cv2.destroyAllWindows()
