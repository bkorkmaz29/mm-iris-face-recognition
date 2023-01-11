import numpy as np
import dlib
import PIL.Image
from PIL import ImageFile
import re
from pkg_resources import resource_filename
import scipy.io as sio

def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def matching(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)


def matchings(file_temp_name, face, temp_dir):
    # Load each account
    #idx = int(re.search(r'\d+', file_temp_name).group())
    #temp_name = temp_dir + str(idx) + "/f" + file_temp_name + ".mat"
    
    idx = int(re.search(r'\d+', file_temp_name).group())
    temp_name = temp_dir + str(idx) + "/f" + file_temp_name + ".mat"
    data_template = sio.loadmat(temp_name)
    faces = data_template['features']
    # Calculate the Hamming distance
    res = face_distance(faces, face)
    
    return  res
