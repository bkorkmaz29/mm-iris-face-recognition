import numpy as np
from scipy import signal
import cv2


def search_inner_bound(img):
    # Integro-Differential operator coarse (jump-level precision)
    Y = img.shape[0]
    X = img.shape[1]
    sect = X/4 		# Width of the external margin for which search is excluded
    minrad = 10
    maxrad = sect*0.8
    jump = 1 		# Precision of the coarse search, in pixels

    # Hough Space (y,x,r)
    sz = np.array([np.floor((Y-2*sect)/jump),
                   np.floor((X-2*sect)/jump),
                   np.floor((maxrad-minrad)/jump)]).astype(int)

    integrationprecision = 1
    angs = np.arange(0, 2*np.pi, integrationprecision)
    x, y, r = np.meshgrid(np.arange(sz[1]),
                          np.arange(sz[0]),
                          np.arange(sz[2]))
    y = sect + y*jump
    x = sect + x*jump
    r = minrad + r*jump
    hs = countour_circular(img, y, x, r, angs)

    # Hough Space Partial Derivative R
    hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2]-1), 0, 0)]

    # Blur
    sm = 3 		# Size of the blurring mask
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm, sm, sm]), mode="same")
    indmax = np.argmax(hspdrs.ravel())
    y, x, r = np.unravel_index(indmax, hspdrs.shape)

    inner_y = sect + y*jump
    inner_x = sect + x*jump
    inner_r = minrad + (r-1)*jump

    # Integro-Differential operator fine (pixel-level precision)
    integrationprecision = 0.1 		# Resolution of the circular integration
    angs = np.arange(0, 2*np.pi, integrationprecision)
    x, y, r = np.meshgrid(np.arange(jump*2),
                          np.arange(jump*2),
                          np.arange(jump*2))
    y = inner_y - jump + y
    x = inner_x - jump + x
    r = inner_r - jump + r
    hs = countour_circular(img, y, x, r, angs)

    # Hough Space Partial Derivative R

    hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2]-1), 0, 0)]

    # Bluring
    sm = 3 		# Size of the blurring mask
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm, sm, sm]), mode="same")
    indmax = np.argmax(hspdrs.ravel())
    y, x, r = np.unravel_index(indmax, hspdrs.shape)

    inner_y = inner_y - jump + y
    inner_x = inner_x - jump + x
    inner_r = inner_r - jump + r - 1

    return inner_y, inner_x, inner_r


def searchInnerBound2(img):

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # Apply adaptive threshold to enhance the edges
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Apply morphological operations to remove small objects and fill in gaps
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # Perform edge detection
    edges = cv2.Canny(opening, 50, 150)
    # find contours of the iris region
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = max(contours, key=cv2.contourArea)
    # select the largest contour as the iris region

    (ix, iy), rad = cv2.minEnclosingCircle(
        max_contour)
    iris_center_x = int(ix)
    iris_center_y = int(iy)
    normalized_radius = int(rad)
    return iris_center_x, iris_center_y, normalized_radius


def search_outer_bound(img, inner_y, inner_x, inner_r):
    # Maximum displacement
    maxdispl = np.round(inner_r*0.15).astype(int)

    # 0.1 - 0.8
    minrad = np.round(inner_r/0.7).astype(int)
    maxrad = np.round(inner_r/0.2).astype(int)

    # Integration region, avoiding eyelids
    intreg = np.array([[2/6, 4/6], [8/6, 10/6]]) * np.pi

    # Resolution of the circular integration
    integrationprecision = 0.05
    angs = np.concatenate([np.arange(intreg[0, 0], intreg[0, 1], integrationprecision),
                           np.arange(intreg[1, 0], intreg[1, 1], integrationprecision)],
                          axis=0)
    x, y, r = np.meshgrid(np.arange(2*maxdispl),
                          np.arange(2*maxdispl),
                          np.arange(maxrad-minrad))
    y = inner_y - maxdispl + y
    x = inner_x - maxdispl + x
    r = minrad + r
    hs = countour_circular(img, y, x, r, angs)

    # Hough Space Partial Derivative R
    hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2]-1), 0, 0)]

    # Blur
    sm = 7 	# Size of the blurring mask
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm, sm, sm]), mode="same")

    indmax = np.argmax(hspdrs.ravel())
    y, x, r = np.unravel_index(indmax, hspdrs.shape)

    outer_y = inner_y - maxdispl + y + 1
    outer_x = inner_x - maxdispl + x + 1
    outer_r = minrad + r - 1

    return outer_y, outer_x, outer_r


def searchOuterBound2(eye_image, inner_x, inner_y, inner_r):
    scaling_factor = 2.3

    outer_r = inner_r * scaling_factor

    return inner_x, inner_y, outer_r


def countour_circular(imagen, y_0, x_0, r, angs):
    # Get y, x
    y = np.zeros([len(angs), r.shape[0], r.shape[1], r.shape[2]], dtype=int)
    x = np.zeros([len(angs), r.shape[0], r.shape[1], r.shape[2]], dtype=int)
    for i in range(len(angs)):
        ang = angs[i]
        y[i, :, :, :] = np.round(y_0 - np.cos(ang) * r).astype(int)
        x[i, :, :, :] = np.round(x_0 + np.sin(ang) * r).astype(int)

    # Adapt y
    ind = np.where(y < 0)
    y[ind] = 0
    ind = np.where(y >= imagen.shape[0])
    y[ind] = imagen.shape[0] - 1

    # Adapt x
    ind = np.where(x < 0)
    x[ind] = 0
    ind = np.where(x >= imagen.shape[1])
    x[ind] = imagen.shape[1] - 1

    hs = imagen[y, x]
    hs = np.sum(hs, axis=0)
    return hs.astype(float)
