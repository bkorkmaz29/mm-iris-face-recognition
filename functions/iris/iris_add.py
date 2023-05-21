from scipy.io import savemat
import PIL.Image
import os
from functions.iris.iris_extraction import iris_extract
import numpy as np


def iris_add(dir, idx, data_dir):
    template, mask, fn = iris_extract(dir)
    basename = idx + "/i" + idx

    out_file = data_dir + basename
    np.savez(out_file, template=template, mask=mask)
    #savemat(out_file, mdict={'template': template, 'mask': mask})
    im = PIL.Image.open(fn)
    image_dir = data_dir + idx
    im.save(os.path.join(image_dir, 'iris.bmp'))
