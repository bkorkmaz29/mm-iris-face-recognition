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
    basename = idx + "/f" + idx
    out_file = data_dir + basename + ".mat"
    savemat(out_file, mdict={'features': features})
    image_dir = data_dir + idx
    im.save(os.path.join(image_dir, 'face.png'))
