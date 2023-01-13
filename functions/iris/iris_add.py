from scipy.io import savemat

from functions.iris.iris_extraction import iris_extract


def iris_add(dir, idx, data_dir):
    template, mask, _ = iris_extract(dir)
    basename = idx + "/i" + idx
    out_file = data_dir + basename + ".mat"
    savemat(out_file, mdict={'template': template, 'mask': mask})