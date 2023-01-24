import os
from glob import glob
from cv2 import imencode
from functions.multimodal.fusion_score import get_fusion_scores
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
 
 
def on_rec(face_dir, iris_dir):
    scores, index_max = get_fusion_scores(face_dir, iris_dir, DATA_DIR)
    return scores, index_max
 
def get_image(dir):
    im = Image.open(dir)
    im.thumbnail((320,243), Image.Resampling.LANCZOS)
    
    im = im.convert('RGB')
    image = np.array(im)
    imgbytes = imencode('.ppm', image)[1].tobytes()

    return imgbytes