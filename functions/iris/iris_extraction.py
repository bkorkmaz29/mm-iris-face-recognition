import cv2
from functions.iris.iris_segment import segment
from functions.iris.iris_normalize import normalize
from functions.iris.iris_encode import encode


def iris_extract(im_filename):
    # Normalisation parameters
    radial_res = 20
    angular_res = 240
    eyelashes_thres = 80
    # Feature encoding parameters
    minWaveLength = 18
    mult = 1
    sigmaOnf = 0.5

    # Perform segmentation
    im = cv2.imread(im_filename, 0)
    iris_circle, pupil_circle, imwithnoise = segment(im, eyelashes_thres)

    # Perform normalization
    polar_array, noise_array = normalize(imwithnoise, iris_circle[1], iris_circle[0], iris_circle[2],
                                         pupil_circle[1], pupil_circle[0], pupil_circle[2],
                                         radial_res, angular_res)

    # Perform feature encoding
    template, mask = encode(polar_array, noise_array,
                            minWaveLength, mult, sigmaOnf)

    return template, mask, im_filename
