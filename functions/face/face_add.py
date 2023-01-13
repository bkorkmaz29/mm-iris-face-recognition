from functions.face.face_extraction import face_encodings

import PIL.Image
import numpy as np
from scipy.io import savemat

def face_add(dir, idx, data_dir):
    im = PIL.Image.open(dir)
    im = im.convert('RGB')
    image = np.array(im)
    features = face_encodings(image)
    basename = idx + "/f" + idx
    out_file = data_dir + basename + ".mat"
    savemat(out_file, mdict={'features': features})
