import PIL.Image
from glob import glob
import os
import numpy as np
import scipy.io as sio
from functions.face.face_matching import face_distance
from functions.face.face_extraction import get_face_encodings
DATA_DIR = "data/"
TEST_DIR = "data/test_face/test_glasses"


def face_add_all():

    images = sorted(
        glob(os.path.join(DATA_DIR, "YALE/*/subject*.glasses")), key=len)

    for index, image in enumerate(images):
        im = PIL.Image.open(image)
        im = im.convert('RGB')
        image_arr = np.array(im)
        features = get_face_encodings(image_arr)
        save_dir = TEST_DIR + "/" + str(index+1)
        if not os.path.exists(save_dir):
            print("makedirs", save_dir)
            os.makedirs(save_dir)

        out_file = save_dir + "/" + str(index+1) + ".mat"
        sio.savemat(out_file, mdict={'features': features})


def test_face(set1, set2):
    true = 0
    false = 0
    files1 = sorted(
        glob(os.path.join(DATA_DIR, f"test_face/test_{set1}/*/*.mat")), key=len)
    files2 = sorted(
        glob(os.path.join(DATA_DIR, f"test_face/test_{set2}/*/*.mat")), key=len)

    for index, file1 in enumerate(files1):
        data_template1 = sio.loadmat(file1)

        features1 = data_template1['features']
        result_list = []
        for file2 in files2:
            data_template2 = sio.loadmat(file2)
            features2 = data_template2['features']
            result_list.append(face_distance(features1, features2))

        dist_list = np.array([result_list[i][0]
                             for i in range(len(result_list))])

        if np.argmin(dist_list) == index:
            true += 1
            print("face no:", index + 1, "prediction:",
                  np.argmin(dist_list) + 1, "CORRECT")
        else:
            false += 1
            print("face no:", index + 1, "prediction:",
                  np.argmin(dist_list) + 1, "FALSE")

    return true/(true+false) * 100


# face_add_all()
test_sets = ["glasses", "sad", "happy", "cl", "rl", "wink"]
test_face("glasses", "happy")
