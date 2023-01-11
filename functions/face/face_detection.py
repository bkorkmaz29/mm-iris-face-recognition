import numpy as np
import dlib
import PIL.Image
from PIL import ImageFile

face_detector = dlib.get_frontal_face_detector()


def _rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _trim_css_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


def load_image_file(file):
    im = PIL.Image.open(file)
    im = im.convert('RGB')
    return np.array(im)


def _raw_face_locations(img, number_of_times_to_upsample=1):
    return face_detector(img, number_of_times_to_upsample)


def face_locations(img, number_of_times_to_upsample=1):
    return [_trim_css_to_bounds(_rect_to_css(face), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample)]
