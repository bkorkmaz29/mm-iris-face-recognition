import os
from glob import glob
import numpy as np
import cv2
from PIL import Image
import shutil
import io
from models.face_rec import FaceRec
from models.iris_rec import IrisRec


class Controller:
    def __init__(self, multimodal):
        self.multimodal = multimodal

    def on_enroll(self, face_dir, iris_dir, name, surname):
        return self.multimodal.enroll_subject(face_dir, iris_dir, name, surname)

    def on_delete(self, idx):
        dir = self.data_dir + idx

        if os.path.exists(dir):
            shutil.rmtree(dir)
            return idx
        else:
            return 0

    def on_rec(self, face_dir, iris_dir, mode, modality):
        self.multimodal.set_mode(mode)
        self.multimodal.set_modality(modality)

        face_img_searched = self.multimodal.get_image(face_dir, 0)
        iris_img_searched = self.multimodal.get_image(iris_dir, 0)

        index_max = self.multimodal.matching(
            face_dir, iris_dir)
        print("index", index_max)
        subject = self.multimodal.get_at_index(index_max)[3:]
        get_face_img_matched, iris_img_matched = self.multimodal.get_db_image(
            subject, 0)
        matched_dir = self.multimodal.get_info(index_max + 1)

        print("dir", matched_dir)
        subject = self.multimodal.format_name(matched_dir)
        return index_max, subject, face_img_searched, iris_img_searched, get_face_img_matched, iris_img_matched

    def on_display(self, subject):
        face_img, iris_img = self.multimodal.get_db_image(subject, 1)
        subjectInfo = self.multimodal.format_name(subject)
        return face_img, iris_img, subjectInfo
