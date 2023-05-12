import numpy as np
import dlib
from pkg_resources import resource_filename

face_detector = dlib.get_frontal_face_detector()

face_predictor_model = resource_filename(
    __name__, "models/shape_predictor_68_face_landmarks.dat")
face_predictor = dlib.shape_predictor(face_predictor_model)

face_recognition_model = resource_filename(
    __name__, "/models/dlib_face_recognition_resnet_model_v1.dat")
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


def get_face_locations(img):
    return face_detector(img)


def get_face_landmarks(face_image):
    face_locations = get_face_locations(face_image)

    return [face_predictor(face_image, face_location) for face_location in face_locations]


def get_face_encodings(face_image):
    face_landmarks = get_face_landmarks(
        face_image)

    return [np.array(face_encoder.compute_face_descriptor(face_image, landmark_set, 1)) for landmark_set in face_landmarks]
