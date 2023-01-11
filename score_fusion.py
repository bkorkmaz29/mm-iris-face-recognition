import scipy.io as sio
from glob import glob
import os
from os import listdir
from itertools import repeat
from multiprocessing import cpu_count, Pool
import numpy as np

from functions.face.face_detection import load_image_file
from functions.face.face_extraction import face_encodings
from functions.iris.iris_extraction import iris_extract
from functions.face.face_matching import matching, face_distance, matchings
from functions.iris.iris_matching import matchingPool

data_dir = "data/fusion/"
files = glob(os.path.join(data_dir, "*"))


def weighted_score(score, min):
    return min / score


def weighted_scores(scores):
    wscores= []
    for score in scores:
        wscores.append(weighted_score(score, np.argmin(scores)))
    return wscores
        

def fusion_scores(face_scores, iris_scores):
    scores = weighted_scores(face_scores) + weighted_scores(iris_scores)
    return scores


face = load_image_file("C:/Users/BK/Documents/GitHub/mm-iris-face-recognition/data/YALE/subject02.happy")
eye = "data/CASIA1/002/2/002_2_2.bmp"


if __name__ == '__main__':
    
    template, mask, file = iris_extract(eye)   
    face_encoding = face_encodings(face)
    
    face_files = glob(os.path.join(data_dir, "*/f*.mat"))

    
    '''   
    face_scores = []
    
    for idx, file in enumerate(face_files):
        basename = os.path.basename(file)
        name = str(idx+1) + "/" + basename
        data_template = sio.loadmat('%s%s'% (data_dir, name))
        faces = data_template['features']
        res1 = face_distance(faces, face_encoding)
        face_scores.append(res1[0])
    
    print(face_scores)      
    '''   
    iris_files = glob(os.path.join(data_dir, "*/i*.mat"))
    
    '''
    for idx, file in enumerate(iris_files):
        basename = os.path.basename(file)
        name = str(idx+1) + "/" + basename
        data_template = sio.loadmat('%s%s'% (data_dir, name))
        templates = data_template['template']
        masks = data_template['mask']
    '''    
    
    
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
            print(dist_list)

            return dist_list
    
    face_scores = get_face_scores(face_encoding, data_dir)
    iris_scores = get_iris_scores(template, mask, data_dir)
    
    scores = fusion_scores(face_scores, iris_scores)
    print(scores)
    index_max = np.argmax(scores)   
       
    print(index_max + 1)
    