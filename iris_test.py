from functions.iris.iris_matching import iris_distance
from glob import glob
import scipy.io as sio
import os
import numpy as np
from functions.iris.iris_add import iris_add
from functions.iris.iris_extraction import iris_extract
DATA_DIR = "data/"
TEST_DIR = "data/test_iris/iris_2_2"


def add_all_iris():

    images = sorted(
        glob(os.path.join(DATA_DIR, "CASIA1/*/2/*_2_2.bmp")), key=len)

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
        glob(os.path.join(DATA_DIR, "test_iris/iris_1_1/*/*.mat")), key=len)
    files2 = sorted(
        glob(os.path.join(DATA_DIR, "test_iris/iris_2_2/*/*.mat")), key=len)

    true_acc = 0
    rejected = 0
    false_acc = 0
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
        if min(dist_list) > 0.40:
            rejected += 1
            print("iris no:", index + 1, "REJECTED")
        elif np.argmin(dist_list) == index:
            true_acc += 1
            print("iris no:", index + 1, "correctly accepted: ",
                  np.argmin(dist_list) + 1, "CORRECT")
        else:
            false_acc += 1
            print("iris no:", index + 1, "falsely accepted: ",
                  np.argmin(dist_list) + 1, "FALSE")
    print("No of correct acceptance:", true_acc)
    print("No of rejectance:", rejected)
    print("No of false acceptance:", false_acc)
    print("FAR % :", false_acc/(true_acc+rejected) * 100)
    print("FRR % :", rejected/(true_acc+rejected) * 100)


# add_all_iris()
test_iris_matching()
