import scipy.io as sio
from glob import glob
import os
from os import listdir
from itertools import repeat
from multiprocessing import cpu_count, Pool
import numpy as np
import PIL.Image
import numpy as np
import PIL.Image
from functions.face.face_extraction import face_encodings
from functions.iris.iris_extraction import iris_extract
from functions.face.face_matching import matching, face_distance, matchings
from functions.iris.iris_matching import matchingPool

data_dir = "data/fusion/"
files = glob(os.path.join(data_dir, "*"))


def weighted_score(score, min):
    return min / score


def get_weighted_scores(scores):
    weighted_scores = []
    for score in scores:
        weighted_scores.append(weighted_score(score, np.argmin(scores)))
    return weighted_scores


def cal_fusion_scores(face_scores, iris_scores):
    return np.add(get_weighted_scores(face_scores), get_weighted_scores(iris_scores))


def load_image_file(file):
    im = PIL.Image.open(file)
    im = im.convert('RGB')
    image = np.array(im)
    return image


def get_iris_scores(template_extr, mask_extr, data_dir, threshold=0.38):
    args = zip(
        sorted(listdir(data_dir)),
        repeat(template_extr),
        repeat(mask_extr),
        repeat(data_dir),
    )
    with Pool(processes=cpu_count()) as pools:
        result_list = pools.starmap(matchingPool, args)

    filenames = [result_list[i][0] for i in range(len(result_list))]
    hm_dists = np.array([result_list[i][1] for i in range(len(result_list))])

    return hm_dists


def get_face_scores(face, data_dir):

    args = zip(
        sorted(listdir(data_dir)),
        repeat(face),
        repeat(data_dir),
    )
    with Pool(processes=cpu_count()) as pools:
        result_list = pools.starmap(matchings, args)

    dist_list = np.array([result_list[i][0] for i in range(len(result_list))])
    return dist_list


def get_face(dir):
    im = PIL.Image.open(dir)
    im = im.convert('RGB')
    image = np.array(im)
    features = face_encodings(image)
    return features


def get_iris(dir):
    template, mask, _ = iris_extract(dir)
    return template, mask


def get_fusion_scores(face_dir, iris_dir):
    face_encoding = get_face(face_dir)
    template, mask = get_iris(iris_dir)
    face_scores = get_face_scores(face_encoding, data_dir)
    iris_scores = get_iris_scores(template, mask, data_dir)
    fusion_scores = cal_fusion_scores(face_scores, iris_scores)
    index_max = np.argmax(fusion_scores) + 1
    return fusion_scores, index_max
