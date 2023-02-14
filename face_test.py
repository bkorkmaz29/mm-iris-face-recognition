from functions.face.face_extraction import face_encodings
import PIL.Image
from glob import glob
import os
import numpy as np
import scipy.io as sio
from functions.face.face_matching import face_distance
DATA_DIR = "data/"
TEST_DIR = "data/test_face2"


def add_all_face():

    images = sorted(
        glob(os.path.join(DATA_DIR, "YALE/subject*.happy")), key=len)

    for index, image in enumerate(images):
        im = PIL.Image.open(image)
        im = im.convert('RGB')
        image_arr = np.array(im)
        features = face_encodings(image_arr)
        save_dir = TEST_DIR + "/" + str(index+1)
        if not os.path.exists(save_dir):
            print("makedirs", save_dir)
            os.makedirs(save_dir)

        out_file = save_dir + "/" + str(index+1) + ".mat"
        sio.savemat(out_file, mdict={'features': features})


def test_face():
    files1 = sorted(
        glob(os.path.join(DATA_DIR, "test_face1/*/*.mat")), key=len)
    files2 = sorted(
        glob(os.path.join(DATA_DIR, "test_face2/*/*.mat")), key=len)

    face_encoding = sio.loadmat("data/test_face1/3/3.mat")
    features1 = face_encoding['features']

    for file1 in files1:
        data_template1 = sio.loadmat(file1)
        features1 = data_template1['features']
        result_list = []
        for file2 in files2:
            data_template2 = sio.loadmat(file2)
            features2 = data_template2['features']
            result_list.append(face_distance(features1, features2))

        dist_list = np.array([result_list[i][0]
                             for i in range(len(result_list))])

        print(np.argmin(dist_list) + 1)


test_face()
