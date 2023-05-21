import PIL.Image
import numpy as np
from scipy.io import savemat
import os
from functions.face.face_extraction import get_face_encodings


def face_add(dir, idx, data_dir):
    im = PIL.Image.open(dir)
    im2 = im.convert('RGB')
    image = np.array(im2)
    features = get_face_encodings(image)
    all_zeros = np.all(features == 0)
    if all_zeros:
        return 0
    else:
        basename = idx + "/f" + idx
        out_file = data_dir + basename + ".npy"
        np.save(out_file, features)
        #savemat(out_file, mdict={'features': features})
        image_dir = data_dir + idx
        im.save(os.path.join(image_dir, 'face.png'))
        return 1
