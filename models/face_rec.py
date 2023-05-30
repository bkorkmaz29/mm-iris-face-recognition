import os
import numpy as np
import dlib as dlib
import PIL.Image
from pkg_resources import resource_filename


class FaceRec:
    # Import face predictor and encoder models
    face_predictor_model = resource_filename(
        __name__, "../utilities/face/trained_models/shape_predictor_68_face_landmarks.dat")
    face_predictor = dlib.shape_predictor(face_predictor_model)
    face_recognition_model = resource_filename(
        __name__, "../utilities/face/trained_models/dlib_face_recognition_resnet_model_v1.dat")
    face_encoder = dlib.face_recognition_model_v1(
        face_recognition_model)
    face_detector = dlib.get_frontal_face_detector()

    @classmethod
    def detect(cls, img):
        return cls.face_detector(img)

    @classmethod
    def get_landmarks(cls, face_image):
        face_locations = cls.detect(face_image)

        return [cls.face_predictor(face_image, face_location) for face_location in face_locations]

    @classmethod
    def get_features(cls, face_image):
        face_landmarks = cls.get_landmarks(
            face_image)

        return [np.array(cls.face_encoder.compute_face_descriptor(face_image, landmark_set, 1)) for landmark_set in face_landmarks]

    @classmethod
    def cal_distance(cls, face_features1, face_features2):
        face_features1 = np.array(face_features1)
        face_features2 = np.array(face_features2)
        return np.linalg.norm(face_features1 - face_features2, axis=1)
