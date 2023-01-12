from functions.iris.iris_extraction import iris_extract
from functions.iris.iris_matching import calHammingDist

from scipy.io import savemat

DATA_DIR = "data/fusion/"

def iris_add(dir, idx):
    template, mask, _ = iris_extract(dir)
    basename = idx + "/i" + idx
    out_file = DATA_DIR + basename + ".mat"
    savemat(out_file, mdict={'template': template, 'mask': mask})