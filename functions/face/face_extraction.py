import numpy as np
import dlib

from pkg_resources import resource_filename
from functions.face.face_detection import _raw_face_locations

face_recognition_model = resource_filename(
    __name__, "/models/dlib_face_recognition_resnet_model_v1.dat")
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)
face_predictor_model= resource_filename(
    __name__, "models/shape_predictor_68_face_landmarks.dat")
face_predictor = dlib.shape_predictor(face_predictor_model)


def _css_to_rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def _raw_face_landmarks(face_image, face_locations):
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [_css_to_rect(face_location)
                      for face_location in face_locations]
        
    pose_predictor = face_predictor
    return [pose_predictor(face_image, face_location) for face_location in face_locations]


def face_landmarks(face_image, face_locations=None):
    landmarks = _raw_face_landmarks(face_image, face_locations)
    landmark_tuple = [[(p.x, p.y) for p in landmark.parts()]
                           for landmark in landmarks]
    return [{
        "chin": points[0:17],
        "left_eyebrow": points[17:22],
        "right_eyebrow": points[22:27],
        "nose_bridge": points[27:31],
        "nose_tip": points[31:36],
        "left_eye": points[36:42],
        "right_eye": points[42:48],
        "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
        "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
    } for points in landmark_tuple]


def face_encodings(face_image, db_face_locations=None, num_jitters=1):
    raw_landmarks = _raw_face_landmarks(
        face_image, db_face_locations)
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]
