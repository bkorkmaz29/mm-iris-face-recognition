import os
from glob import glob
from cv2 import imencode
from functions.multimodal.fusion_score import fusion_matching
from functions.face.face_add import face_add
from functions.iris.iris_add import iris_add
from PIL import Image
import numpy as np

DATA_DIR = "data/db/"


def on_add(face_dir, iris_dir):
    files = glob("data/db/*")
    idx = str(len(files) + 1)
    save_dir = DATA_DIR + idx

    if not os.path.exists(save_dir):
        print("makedirs", save_dir)
        os.makedirs(save_dir)

    face_add(face_dir, idx, DATA_DIR)
    iris_add(iris_dir, idx, DATA_DIR)

    return idx


def on_rec(face_dir, iris_dir):
    scores, index_max = fusion_matching(face_dir, iris_dir, DATA_DIR)
    return scores, index_max


def get_image(dir, full):
    im = Image.open(dir)
    if full:
        im.thumbnail((320, 240), Image.Resampling.LANCZOS)
    else:
        im.thumbnail((160, 120), Image.Resampling.LANCZOS)
    im = im.convert('RGB')
    image = np.array(im)
    imgbytes = imencode('.ppm', image)[1].tobytes()

    return imgbytes


def get_db_image(id, full):
    iris_dir = DATA_DIR + id + "/iris.bmp"
    face_dir = DATA_DIR + id + "/face.png"

    iris_image = get_image(iris_dir, full)
    face_image = get_image(face_dir, full)

    return face_image, iris_image


def get_ns():
    return get_image("data/no_result.jpg", 0)
