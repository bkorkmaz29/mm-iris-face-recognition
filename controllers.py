from score_fusion import get_fusion_scores
from functions.face.face_add import face_add
from functions.iris.iris_add import iris_add

import os
from glob import glob

def on_add(face_dir, iris_dir):
    files = glob("data/fusion/*")
    idx = str(len(files) + 1)
    save_dir = "data/fusion/" + idx

    if not os.path.exists(save_dir):
        print("makedirs", save_dir)
        os.makedirs(save_dir)

    face_add(face_dir, idx)
    iris_add(iris_dir, idx)


def on_rec(face_dir, iris_dir):
    return get_fusion_scores(face_dir, iris_dir)

