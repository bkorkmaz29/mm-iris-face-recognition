import numpy as np

FACE_TOLERANCE = 0.5


def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return 1

    if len(face_to_compare) == 0:
        return 1

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def matching(known_face_encodings, face_encoding_to_check, FACE_TOLERANCE):
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= FACE_TOLERANCE)
