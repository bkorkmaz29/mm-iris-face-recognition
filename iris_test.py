from functions.iris.iris_matching import iris_distance
from glob import glob
import scipy.io as sio
import os
import numpy as np
from functions.iris.iris_add import iris_add
from functions.iris.iris_extraction import iris_extract
DATA_DIR = "data/"
TEST_DIR = "data/test_iris2"


def add_all_iris():

    images = sorted(
        glob(os.path.join(DATA_DIR, "CASIA1/*/2/*_2_1.bmp")), key=len)

    for index, image in enumerate(images):
        template, mask, _ = iris_extract(image)
        save_dir = TEST_DIR + "/" + str(index+1)
        if not os.path.exists(save_dir):
            print("makedirs", save_dir)
            os.makedirs(save_dir)
        out_file = save_dir + "/" + str(index+1) + ".mat"
        sio.savemat(out_file, mdict={'template': template, 'mask': mask})


def test_iris_matching():

    files1 = sorted(
        glob(os.path.join(DATA_DIR, "test_iris1/*/*.mat")), key=len)
    files2 = sorted(
        glob(os.path.join(DATA_DIR, "test_iris2/*/*.mat")), key=len)

    preds = []
    true = 0
    false = 0
    for index, file1 in enumerate(files1):
        data_template = sio.loadmat(file1)
        template_search = data_template['template']
        mask_search = data_template['mask']
        dist_list = []

        for file2 in files2:

            data_template = sio.loadmat(file2)
            template = data_template['template']
            mask = data_template['mask']
            dist_list.append(iris_distance(
                template_search, mask_search, template, mask))
        print(np.argmin(dist_list))
        print(index)
        if np.argmin(dist_list) == index:
            true += 1
        else:
            false += 1

    print(true/(true+false) * 100)


test_iris_matching()
