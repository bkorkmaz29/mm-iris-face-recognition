from utilities.iris.utilities import iris_preprocess
import cv2
from scipy import signal
import numpy as np
from skimage.transform import radon


def get_circle_coords(c, r, imgsize, nsides=600):
    a = np.linspace(0, 2*np.pi, 2*nsides+1)
    xd = np.round(r * np.cos(a) + c[0])
    yd = np.round(r * np.sin(a) + c[1])

    #  Get rid of values larger than image
    xd2 = xd
    coords = np.where(xd >= imgsize[1])
    xd2[coords[0]] = imgsize[1] - 1
    coords = np.where(xd < 0)
    xd2[coords[0]] = 0

    yd2 = yd
    coords = np.where(yd >= imgsize[0])
    yd2[coords[0]] = imgsize[0] - 1
    coords = np.where(yd < 0)
    yd2[coords[0]] = 0

    x = np.round(xd2).astype(int)
    y = np.round(yd2).astype(int)
    return x, y


def search_inner_bound(img):
    Y = img.shape[0]
    X = img.shape[1]
    sect = X/4
    minrad = 10
    maxrad = sect*0.8
    jump = 1

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
    integrationprecision = 0.1
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


def find_top_eyelid(imsz, iris_image, irl, icl, rowp, rp, ret_top=None):
    top_eyelid = iris_image[0: rowp - irl - rp, :]
    lines = find_line(top_eyelid)
    mask = np.zeros(imsz, dtype=float)

    if lines.size > 0:
        xl, yl = get_line_coords(lines, top_eyelid.shape)
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


def find_bottom_eyelid(imsz, imageiris, rowp, rp, irl, icl, ret_bot=None):
    bottomeyelid = imageiris[rowp - irl + rp - 1: imageiris.shape[0], :]
    lines = find_line(bottomeyelid)
    mask = np.zeros(imsz, dtype=float)

    if lines.size > 0:
        xl, yl = get_line_coords(lines, bottomeyelid.shape)
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


def find_line(img):
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


def get_line_coords(lines, imsize):
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


def gabor_convolve(im, minWaveLength, mult, sigmaOnf):
    rows, ndata = im.shape
    logGabor = np.zeros(ndata)
    filterbank = np.zeros([rows, ndata], dtype=complex)
    # Frequency values 0 - 0.5
    radius = np.arange(ndata/2 + 1) / (ndata/2) / 2
    radius[0] = 1
    # Initialize filter wavelength
    wavelength = minWaveLength
    # Calculate the radial filter component
    fo = 1 / wavelength
    logGabor[0: int(ndata/2) + 1] = np.exp((-(np.log(radius/fo))
                                            ** 2) / (2 * np.log(sigmaOnf)**2))
    logGabor[0] = 0
    for r in range(rows):
        signal = im[r, 0:ndata]
        imagefft = np.fft.fft(signal)
        filterbank[r, :] = np.fft.ifft(imagefft * logGabor)
    return filterbank


def shift_bits(template, noshifts):
    new_template = np.zeros(template.shape)
    width = template.shape[1]
    s = 2 * np.abs(noshifts)
    p = width - s

    # Shift
    if noshifts == 0:
        new_template = template

    elif noshifts < 0:
        x = np.arange(p)
        new_template[:, x] = template[:, s + x]
        x = np.arange(p, width)
        new_template[:, x] = template[:, x - p]

    else:
        x = np.arange(s, width)
        new_template[:, x] = template[:, x - s]
        x = np.arange(s)
        new_template[:, x] = template[:, p + x]

    return new_template
