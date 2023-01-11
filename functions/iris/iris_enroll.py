from functions.iris.iris_extraction import extractFeature
from functions.iris.iris_matching import calHammingDist
import cv2 
import argparse, os
from glob import glob
from tqdm import tqdm
from time import time
from scipy.io import savemat
from multiprocessing import cpu_count, Pool
import glob
import scipy.io as sio
temp_dir = "C:/Users/BK/Documents/GitHub/mm-iris-face-recognition/data/templates"

def enroll(file):
        template, mask, _ = extractFeature(file, use_multiprocess=False)
        basename = os.path.basename(file)
        out_file = os.path.join(temp_dir, "%s.mat" % (basename))
        savemat(out_file, mdict={'template': template, 'mask': mask})

def pool_extract_feature(args):
    
    template, mask, im_filename = extractFeature(
        im_filename=im_filename,
        eyelashes_thres=eyelashes_thres,
        use_multiprocess=use_multiprocess,
    )
    return template, mask, im_filename

def pool_calHammingDist(args):
    template1, mask1, template2, mask2 = args
    dist = calHammingDist(template1, mask1, template2, mask2)
    return dist
