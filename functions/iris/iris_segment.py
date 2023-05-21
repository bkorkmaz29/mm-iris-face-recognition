import numpy as np
from functions.iris.iris_boundary import searchInnerBound, searchOuterBound
from functions.iris.iris_line import findline, linecoords
import multiprocessing as mp
import cv2


def segment(eye_image, eyelash_threshold):
    rowp, colp, rp = searchInnerBound(eye_image)
    row, col, r = searchOuterBound(eye_image, rowp, colp, rp)

    # Package pupil and iris boundaries
    rowp = np.round(rowp).astype(int)
    colp = np.round(colp).astype(int)
    rp = np.round(rp).astype(int)
    row = np.round(row).astype(int)
    col = np.round(col).astype(int)
    r = np.round(r).astype(int)
    pupil_circle = [rowp, colp, rp]
    iris_circle = [row, col, r]

    # Find top and bottom eyelid
    image_shape = eye_image.shape
    irl = np.round(row - r).astype(int)
    iru = np.round(row + r).astype(int)
    icl = np.round(col - r).astype(int)
    icu = np.round(col + r).astype(int)
    if irl < 0:
        irl = 0
    if icl < 0:
        icl = 0
    if iru >= image_shape[0]:
        iru = image_shape[0] - 1
    if icu >= image_shape[1]:
        icu = image_shape[1] - 1
    iris_image = eye_image[irl: iru + 1, icl: icu + 1]

    mask_top = findTopEyelid(image_shape, iris_image, irl, icl, rowp, rp)
    mask_bottom = findBottomEyelid(image_shape, iris_image, rowp, rp, irl, icl)
    # Mask the eye image
    img_noise = eye_image.astype(float)
    img_noise = eye_image + mask_top + mask_bottom
    # Eliminate eyelashes by threshold
    ref = eye_image < eyelash_threshold
    coords = np.where(ref == 1)
    img_noise[coords] = np.nan

    return iris_circle, pupil_circle, img_noise


def findTopEyelid(imsz, iris_image, irl, icl, rowp, rp, ret_top=None):
    top_eyelid = iris_image[0: rowp - irl - rp, :]
    lines = findline(top_eyelid)
    mask = np.zeros(imsz, dtype=float)

    if lines.size > 0:
        xl, yl = linecoords(lines, top_eyelid.shape)
        yl = np.round(yl + irl - 1).astype(int)
        xl = np.round(xl + icl - 1).astype(int)

        yla = np.max(yl)
        y2 = np.arange(yla)

        mask[yl, xl] = np.nan
        grid = np.meshgrid(y2, xl)
        mask[tuple(grid)] = np.nan

    if ret_top is not None:
        ret_top[0] = mask
    return mask


def findBottomEyelid(imsz, imageiris, rowp, rp, irl, icl, ret_bot=None):
    bottomeyelid = imageiris[rowp - irl + rp - 1: imageiris.shape[0], :]
    lines = findline(bottomeyelid)
    mask = np.zeros(imsz, dtype=float)

    if lines.size > 0:
        xl, yl = linecoords(lines, bottomeyelid.shape)
        yl = np.round(yl + rowp + rp - 3).astype(int)
        xl = np.round(xl + icl - 2).astype(int)
        yla = np.min(yl)
        y2 = np.arange(yla-1, imsz[0])

        mask[yl, xl] = np.nan
        grid = np.meshgrid(y2, xl)
        mask[tuple(grid)] = np.nan

    if ret_bot is not None:
        ret_bot[0] = mask
    return mask
