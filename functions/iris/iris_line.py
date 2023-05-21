from functions.iris.utilities import iris_preprocess
import numpy as np
from skimage.transform import radon


def findline(img):
    # Radon transformation

    img_processed = iris_preprocess(img)
    theta = np.arange(180)
    R = radon(img_processed, theta, circle=False)
    sz = R.shape[0] // 2
    xp = np.arange(-sz, sz+1, 1)

    # Find for the strongest edge
    maxv = np.max(R)
    if maxv > 25:
        i = np.where(R.ravel() == maxv)
        i = i[0]
    else:
        return np.array([])

    R_vect = R.ravel()
    ind = np.argsort(-R_vect[i])
    u = i.shape[0]
    k = i[ind[0: u]]
    y, x = np.unravel_index(k, R.shape)
    t = -theta[x] * np.pi / 180
    r = xp[y]

    lines = np.vstack([np.cos(t), np.sin(t), -r]).transpose()
    cx = img.shape[1] / 2 - 1
    cy = img.shape[0] / 2 - 1
    lines[:, 2] = lines[:, 2] - lines[:, 0]*cx - lines[:, 1]*cy
    return lines


def linecoords(lines, imsize):
    xd = np.arange(imsize[1])
    yd = (-lines[0, 2] - lines[0, 0] * xd) / lines[0, 1]

    coords = np.where(yd >= imsize[0])
    coords = coords[0]
    yd[coords] = imsize[0]-1
    coords = np.where(yd < 0)
    coords = coords[0]
    yd[coords] = 0

    x = xd
    y = yd
    return x, y
