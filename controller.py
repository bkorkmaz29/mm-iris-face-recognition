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
        self.data_dir = multimodal.data_dir

    def on_add(self, face_dir, iris_dir, name, surname):
        files = glob("data/db/*")
        idx = str(len(files) + 1)
        save_dir = self.data_dir + idx + "-" + name + "-" + surname

        if not os.path.exists(save_dir):
            print("makedirs", save_dir)
            os.makedirs(save_dir)
            result = FaceRec.face_add(
                face_dir, idx, save_dir)
        if not result:
            os.rmdir(save_dir)
            return 0
        else:
            IrisRec.iris_add(iris_dir, idx, save_dir)

            return idx

    def on_delete(self, idx):
        dir = self.data_dir + idx

        if os.path.exists(dir):
            shutil.rmtree(dir)
            return idx
        else:
            return 0

    def on_rec(self, face_dir, iris_dir):

        index_max = self.multimodal.fusion_matching(
            face_dir, iris_dir, self.data_dir)
        matched_dir = self.get_info(index_max)
        matched_dir = matched_dir[8:]
        subject = self.format_name(matched_dir)
        return index_max, subject, matched_dir

    def get_image(self, dir, full):
        im = Image.open(dir)

        if full:
            im.thumbnail((320, 240), Image.Resampling.LANCZOS)
        else:
            im.thumbnail((240, 160), Image.Resampling.LANCZOS)

        im_rgb = im.convert('RGB')
        imgbytes = io.BytesIO()
        im_rgb.save(imgbytes, format='PPM')
        imgbytes = imgbytes.getvalue()

        return imgbytes

    def get_db_image(self, subject_dir, full):

        iris_dir = self.data_dir + subject_dir + "/iris.png"
        face_dir = self.data_dir + subject_dir + "/face.png"

        iris_image = self.get_image(iris_dir, full)

        face_image = self.get_image(face_dir, full)

        return face_image, iris_image

    def set_order(self, order):
        if order != self.multimodal.order:
            self.multimodal.order = order

    def set_mode(self, mode):
        if mode != self.multimodal.mode:
            self.multimodal.mode = mode

    def format_name(self, subject):
        parts = subject.split("-")
        return f"#{parts[0]}  {parts[1].title()} {parts[2].title()}"

    def get_info(self, id):
        file = glob(f"data/db/{id}-*")
        return file[0]
