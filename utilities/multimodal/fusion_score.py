from glob import glob
import os
import numpy as np
import PIL.Image

from functions.face.face_extraction import get_face_encodings
from functions.iris.iris_extraction import iris_extract
from functions.face.face_matching import face_distance
from functions.iris.iris_matching import iris_distance
from functions.multimodal.utilities import get_normalized_score, get_normalized_scores
IRIS_TOLARENCE = 0.41
FACE_TOLARENCE = 0.5
FUSION_TOLARENCE = 0.2


def cal_fusion_scores(face_scores, iris_scores):
    norm_face = get_normalized_scores(face_scores)
    norm_iris = get_normalized_scores(iris_scores)

    return np.add(norm_face, norm_iris)


def cal_fusion_score(face_score, iris_score):
    norm_face = get_normalized_score(face_score)
    norm_iris = get_normalized_score(iris_score)

    return norm_face + norm_iris


def get_face_features(dir):
    im = PIL.Image.open(dir)
    im = im.convert('RGB')
    image = np.array(im)
    features = get_face_encodings(image)

    return features


def get_iris_features(dir):
    template, mask, _ = iris_extract(dir)
    return template, mask


def get_iris_scores(template_search, mask_search, data_dir):

    files = glob(os.path.join(data_dir, "*/i*.npz"))
    dist_list = []
    for file in files:
        print(file)
        #data_template = sio.loadmat(file)
        #template = data_template['template']
        #mask = data_template['mask']
        iris_data = np.load(file)
        template = iris_data['template']
        mask = iris_data['mask']
        dist_list.append(iris_distance(
            template_search, mask_search, template, mask))
    return dist_list


def get_face_scores(face_encoding, data_dir):
    files = glob(os.path.join(data_dir, "*/f*.npy"))
    result_list = []
    for file in files:
        features = np.load(file)
        result_list.append(face_distance(face_encoding, features))
    dist_list = np.array([result_list[i][0] for i in range(len(result_list))])
    return dist_list


def fusion_matching(face_dir, iris_dir, data_dir):
    face_encoding = get_face_features(face_dir)
    template, mask = get_iris_features(iris_dir)
    face_scores = get_face_scores(face_encoding, data_dir)
    iris_list = get_iris_scores(template, mask, data_dir)
    iris_scores = np.array(iris_list)
    fusion_scores = cal_fusion_scores(face_scores, iris_scores)
    print(fusion_scores)
    if (face_scores < FACE_TOLARENCE).any() or (iris_scores < IRIS_TOLARENCE).any() and min(fusion_scores) <= FUSION_TOLARENCE:
        index = np.argmin(fusion_scores) + 1
    else:
        index = 0
    return fusion_scores, index


def fusion_score_bayesian(face1, temp1, mask1, face2, temp2, mask2):

    face_score = face_distance(face1, np.array(face2))
    iris_score = iris_distance(temp1, mask1, temp2, mask2)
    s = face_score * iris_score + \
        (1 - face_score) * (1 - iris_score) + (face_score * iris_score)

    return s


def fusion_score_product(face1, temp1, mask1, face2, temp2, mask2):
    face_score = face_distance(face1, np.array(face2))
    iris_score = iris_distance(temp1, mask1, temp2, mask2)

    return np.multiply(face_score, iris_score)


def fusion_score_sum(face1, temp1, mask1, face2, temp2, mask2):
    face_score = face_distance(face1, np.array(face2))
    iris_score = iris_distance(temp1, mask1, temp2, mask2)

    return np.add(face_score, iris_score)


def weighted_sum(face1, temp1, mask1, face2, temp2, mask2, weight):
    face_score = face_distance(face1, np.array(face2))
    iris_score = iris_distance(temp1, mask1, temp2, mask2)

    return weight * face_score + (1 - weight) * iris_score


def weighted_product(face1, temp1, mask1, face2, temp2, mask2, weight):
    face_score = face_distance(face1, np.array(face2))
    iris_score = iris_distance(temp1, mask1, temp2, mask2)

    return (face_score ** weight) * (iris_score ** (1 - weight))
