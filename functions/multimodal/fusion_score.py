import scipy.io as sio
from glob import glob
import os
import numpy as np
import PIL.Image

from functions.face.face_extraction import get_face_encodings
from functions.iris.iris_extraction import iris_extract
from functions.face.face_matching import face_distance
from functions.iris.iris_matching import iris_distance


def get_weighted_scores(scores, weight):
    weighted_scores = []
    for score in scores:
        weighted_scores.append(score * weight)
    return weighted_scores


def get_normalized_score(scores):
    norm_scores = []
    print("initial", scores)
    for score in scores:

        norm_score = (score - np.min(scores)) / \
            (np.max(scores) - np.min(scores))

        norm_scores.append(norm_score)

    print("normalized", norm_scores)
    return norm_scores


def cal_fusion_scores(face_scores, iris_scores):
    norm_face = get_normalized_score(face_scores)
    norm_iris = get_normalized_score(iris_scores)

    return np.add(norm_face, norm_iris)


def get_iris_scores(template_search, mask_search, data_dir):

    files = glob(os.path.join(data_dir, "*/i*.mat"))
    dist_list = []
    for file in files:
        data_template = sio.loadmat(file)
        template = data_template['template']
        mask = data_template['mask']
        dist_list.append(iris_distance(
            template_search, mask_search, template, mask))
    return dist_list


def get_face_scores(face_encoding, data_dir):
    files = glob(os.path.join(data_dir, "*/f*.mat"))
    result_list = []
    for file in files:
        data_template = sio.loadmat(file)
        features = data_template['features']
        result_list.append(face_distance(face_encoding, features))
    dist_list = np.array([result_list[i][0] for i in range(len(result_list))])
    return dist_list


def get_face(dir):
    im = PIL.Image.open(dir)
    im = im.convert('RGB')
    image = np.array(im)
    features = get_face_encodings(image)
    return features


def get_iris(dir):
    template, mask, _ = iris_extract(dir)
    return template, mask


def fusion_matching(face_dir, iris_dir, data_dir):
    face_encoding = get_face(face_dir)
    template, mask = get_iris(iris_dir)
    face_scores = get_face_scores(face_encoding, data_dir)
    iris_list = get_iris_scores(template, mask, data_dir)
    iris_scores = np.array(iris_list)
    fusion_scores = cal_fusion_scores(face_scores, iris_scores)

    print(fusion_scores)
    if (face_scores < 0.5).any() or (iris_scores < 0.3).any() and min(fusion_scores) <= 0.2:
        index = np.argmin(fusion_scores) + 1
    else:
        index = 0
    return fusion_scores, index
