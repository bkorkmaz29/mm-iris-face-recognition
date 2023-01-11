from functions.iris.iris_extraction import extractFeature
from functions.iris.iris_enroll import enroll, pool_calHammingDist, pool_extract_feature
from functions.face.face_enroll import enroll_face
from functions.face.face_extraction import face_encodings
from functions.face.face_detection import load_image_file
from functions.face.face_matching import matching, face_distance
from cv2 import imread, imshow, waitKey
import cv2
import argparse, os
from glob import glob
from tqdm import tqdm
from time import time
from scipy.io import savemat
from multiprocessing import cpu_count, Pool
import glob
import numpy as np
import random
from itertools import repeat
from functions.iris.iris_matching import matchingPool
from functions.face.face_matching import matching
from os import listdir
import scipy.io as sio
N_IMAGES = 4

if __name__ == '__main__':
    data_dir = "data/CASIA1/"
    temp_dir = "C:/Users/BK/Documents/GitHub/mm-iris-face-recognition/data/templates"
    '''
    start = time()

    # Check the existence of temp_dir
    if not os.path.exists(temp_dir):
        print("makedirs", temp_dir)
        os.makedirs(temp_dir)

    print(os.path.join(data_dir, "*/1/*_1_*.bmp"))
    # Get list of files for enrolling template, just "xxx_1_x.jpg" files are selected
    files = glob.glob(os.path.join(data_dir, "*/1/*.bmp"))
    n_files = len(files)
    print("Number of files for enrolling:", n_files)
    # Parallel pools to enroll templates
    print("Start enrolling...")
    pools = Pool(processes=cpu_count())
    for _ in tqdm(pools.imap_unordered(enroll, files), total=n_files):
        pass

    end = time()
    print('\n>>> Enrollment time: {} [s]\n'.format(end-start))
   
    '''
   
    data_di = "data/YALE"
    temp_di = "C:/Users/BK/Documents/GitHub/mm-iris-face-recognition/data/tempr/"
    
    '''

    # Check the existence of temp_dir
    if not os.path.exists(temp_di):
        print("makedirs", temp_di)
        os.makedirs(temp_di)

    print(os.path.join(data_di, "subject*.normal"))
    # Get list of files for enrolling template, just "xxx_1_x.jpg" files are selected
    files = glob.glob(os.path.join(data_di, "subject*.normal"))
    n_files = len(files)
    print("Number of files for enrolling:", n_files)
    # Parallel pools to enroll templates
    
    print("Start enrolling...")

  
    for file in files:
        arr = load_image_file(file)

        enroll_face(arr, file)
    
    '''
    
    face = load_image_file("C:/Users/BK/Documents/GitHub/mm-iris-face-recognition/data/YALE/subject05.happy")
    face_encoding = face_encodings(face)
    files = glob.glob(os.path.join(temp_di, "subject*.normal.mat"))
    
    
    for file in files:
        basename = os.path.basename(file)
        data_template = sio.loadmat('%s%s'% (temp_di, basename))
        faces = data_template['features']
        res = face_distance(faces, face_encoding)
        print(res)
        
     
    
    
    '''
        pools = Pool(processes=cpu_count())
    for _ in tqdm(pools.imap_unordered(enroll_face, files), total=n_files):
        pass
    
    file = "data/YALE/subject01.normal.jpg"
    arr = load_image_file(file)
    
    encodes = face_encodings(arr)
    print(encodes)
   
    file = "data/CASIA1/009/1/009_1_1.bmp"
    start = time()
    print('>>> Start verifying {}\n'.format(file))
    template, mask, file = extractFeature(file)
    
    def match_iris(template_extr, mask_extr, file, threshold=0.38):
    # Get the number of accounts in the database

        # Use all cores to calculate Hamming distances
        args = zip(
            sorted(listdir(temp_dir)),
            repeat(template_extr),
            repeat(mask_extr),
            repeat(temp_dir),
        )
        with Pool(processes=cpu_count()) as pools:
            result_list = pools.starmap(matchingPool, args)

        filenames = [result_list[i][0] for i in range(len(result_list))]
        hm_dists = np.array([result_list[i][1] for i in range(len(result_list))])

        # Remove NaN elements
        ind_valid = np.where(hm_dists>0)[0]
        hm_dists = hm_dists[ind_valid]
        filenames = [filenames[idx] for idx in ind_valid]

        # Threshold and give the result ID
        ind_thres = np.where(hm_dists<=threshold)[0]

        # Return
        if len(ind_thres)==0:
            return 0
        else:
            hm_dists = hm_dists[ind_thres]
            filenames = [filenames[idx] for idx in ind_thres]
            ind_sort = np.argsort(hm_dists)
            return [filenames[idx] for idx in ind_sort]


        
        


    result = match_iris(template, mask, temp_dir, 0.38)
    print(result)
    '''