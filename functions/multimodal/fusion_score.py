import scipy.io as sio
from glob import glob
import os
from itertools import repeat
from multiprocessing import cpu_count, Pool
import numpy as np
import PIL.Image

from functions.face.face_extraction import face_encodings

from functions.iris.iris_extraction import iris_extract
from functions.face.face_matching import matchings, face_distance
from functions.iris.iris_matching import matchingPool, iris_distance


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
        sorted(os.listdir(data_dir)),
        repeat(template_extr),
        repeat(mask_extr),
        repeat(data_dir),
    )
    with Pool(processes=cpu_count()) as pools:
        result_list = pools.starmap(matchingPool, args)

    hm_dists = np.array([result_list[i][1] for i in range(len(result_list))])

    return hm_dists


def get_iris_scores_nm(template_search, mask_search, data_dir):
    
    files = glob(os.path.join(data_dir, "*/i*.mat"))
    dist_list = []
    for file in files:
        data_template = sio.loadmat(file)
        template = data_template['template']
        mask = data_template['mask']
        dist_list.append(iris_distance(
            template_search, mask_search, template, mask))
    return dist_list
    
def get_face_scores_nm(face_encoding, data_dir):
    
    files = glob(os.path.join(data_dir, "*/f*.mat"))
    result_list = []
    for file in files:
        data_template = sio.loadmat(file)
        features = data_template['features']
        result_list.append(face_distance(face_encoding, features))
    dist_list = np.array([result_list[i][0] for i in range(len(result_list))])
    return dist_list


def get_face_scores(face, data_dir):

    args = zip(
        sorted(os.listdir(data_dir)),
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


def get_fusion_scores(face_dir, iris_dir, data_dir):
    face_encoding = get_face(face_dir)
    template, mask = get_iris(iris_dir)
    face_scores = get_face_scores_nm(face_encoding, data_dir)
    iris_scores = get_iris_scores_nm(template, mask, data_dir)
    fusion_scores = cal_fusion_scores(face_scores, iris_scores)
    index_max = np.argmax(fusion_scores) + 1
    print(fusion_scores)
    print(index_max)
    return fusion_scores, index_max

