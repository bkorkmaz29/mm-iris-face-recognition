from functions.face.face_extraction import face_encodings

from cv2 import imread
import argparse
import os
from glob import glob
from tqdm import tqdm
from time import time
from scipy.io import savemat
from multiprocessing import cpu_count, Pool
import glob
import scipy.io as sio
temp_dir = "C:/Users/BK/Documents/GitHub/mm-iris-face-recognition/data/tempr"


def enroll_face(img, file):
    features = face_encodings(img)
    basename = os.path.basename(file)
    out_file = os.path.join(temp_dir, "%s.mat" % (basename))
    savemat(out_file, mdict={'features': features})
