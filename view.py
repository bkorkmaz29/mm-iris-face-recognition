from glob import glob
import os
import numpy as np
from PIL import Image
import io
import shutil
from models.face_rec import FaceRec
from models.iris_rec import IrisRec


class MMR2:
    def __init__(self, data_dir):
        self.FACE_TOLERANCE = 0.41
        self.IRIS_TOLERANCE = 0.41
        self.FUSION_TOLERANCE = 0.444
        self.modality = 1
        self.mode = 2
        self.data_dir = data_dir

    def set_modality(self, modality):
        self.modality = modality

    def set_mode(self, mode):
        self.mode = mode
        if mode == 1:
            # Precision (F0.5)
            self.FACE_TOLERANCE = 0.454
            self.IRIS_TOLERANCE = 0.404
            self.FUSION_TOLERANCE = 0.424
        elif mode == 2:
            # Balanced (F1)
            self.FACE_TOLERANCE = 0.481
            self.IRIS_TOLERANCE = 0.414
            self.FUSION_TOLERANCE = 0.444
        else:
            # Recall (F2)
            self.FACE_TOLERANCE = 0.545
            self.IRIS_TOLERANCE = 0.424
            self.FUSION_TOLERANCE = 0.461

    def get_at_index(self, index):
        files = glob(f"{self.data_dir}/*")
        if len(files) < index:
            return -1
        else:
            return files[index]

    def find_id(self, files):
        id_numbers = []
        for file in files:
            id_numbers.append(int(file[3:][:1]))
        smallest_missing_number = 1
        while smallest_missing_number in id_numbers:
            smallest_missing_number += 1

        return smallest_missing_number

    def format_name(self, subject):
        parts = subject.split("-")
        return f"#{parts[0]} {parts[1].title()} {parts[2].title()}"

    def get_info(self, id):
        file = glob(f"{self.data_dir}/{id}-*")
        if file:
            return file[0]
        else:
            return ""

    def get_image(self, dir, full):
        if dir == "":
            im = Image.new("RGB", (600, 533), "white")
        else:
            im = Image.open(dir)
            im.resize((320, 240))
            original_width, original_height = im.size

            crop_size = min(original_width, original_height)
            # Calculate the left, upper, right, and lower coordinates for the crop
            left = (original_width - crop_size) // 2
            upper = (original_height - crop_size) // 2
            right = left + crop_size
            lower = upper + crop_size

            # Crop the image
            im = im.crop((left, upper, right, lower))

        if full:
            im.thumbnail((240, 240), Image.Resampling.LANCZOS)

        else:
            im.thumbnail((160, 160), Image.Resampling.LANCZOS)

        im_rgb = im.convert('RGB')
        imgbytes = io.BytesIO()
        im_rgb.save(imgbytes, format='PPM')
        imgbytes = imgbytes.getvalue()
        im.close()
        return imgbytes

    def get_db_image(self, subject_dir, full):
        iris_dir = self.data_dir + subject_dir + "/iris.png"
        face_dir = self.data_dir + subject_dir + "/face.png"

        iris_image = self.get_image(iris_dir, full)
        face_image = self.get_image(face_dir, full)

        return face_image, iris_image

    def get_face_features(self, dir):
        im = Image.open(dir)
        im = im.convert('RGB')
        image = np.array(im)
        im.close()

        features = FaceRec.get_features(image)

        return features

    def get_iris_features(self, dir):
        template, mask, _ = IrisRec.get_features(dir)

        return template, mask

    def get_face_scores(self, face_encoding, data_dir):
        files = glob(os.path.join(data_dir, "*/f*.npy"))
        result_list = []
        for file in files:
            features = np.load(file)
            result_list.append(FaceRec.cal_distance(face_encoding, features))
        dist_list = np.array([result_list[i][0]
                             for i in range(len(result_list))])

        return dist_list

    def face_add(self, image_dir, idx, data_dir):
        im = Image.open(image_dir)
        im2 = im.convert('RGB')
        image = np.array(im2)
        features = FaceRec.get_features(image)
        all_zeros = np.all(features == 0)
        '''
        if all_zeros:
            im.close()
            return 0
            
        else:
        '''
        basename = "/f" + idx
        out_file = data_dir + basename + ".npy"
        np.save(out_file, features)
        image_dir = data_dir
        im.save(os.path.join(image_dir, 'face.png'),
                ICC_PROFILE=im.info.get('icc_profile'))
        im.close()
        return 1

    def iris_add(self, dir, idx, data_dir):
        # Extract iris features
        template, mask, fn = IrisRec.get_features(dir)
        basename = "/i" + idx
        out_file = data_dir + basename
        # Save iris features
        np.savez(out_file, template=template, mask=mask)
        # Save iris image
        im = Image.open(fn)
        image_dir = data_dir
        im.save(os.path.join(image_dir, 'iris.png'))
        im.close()

    def enroll_subject(self, face_dir, iris_dir, name, surname):
        files = glob(f"{self.data_dir}/*")
        id = str(self.find_id(files))
        save_dir = self.data_dir + id + "-" + name + "-" + surname
        face_img = self.get_image(face_dir, 1)
        iris_img = self.get_image(iris_dir, 1)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            result = self.face_add(
                face_dir, id, save_dir)
        '''
        if not result:
            os.rmdir(save_dir)
            return 0
        else:
            '''
        self.iris_add(iris_dir, id, save_dir)

        return id, face_img, iris_img

    def matching(self, face_dir, iris_dir):

        if self.modality == 1:
            face_encoding = self.get_face_features(face_dir)
            template, mask = self.get_iris_features(iris_dir)
            face_scores = self.get_face_scores(face_encoding, self.data_dir)
            iris_list = self.get_iris_scores(template, mask, self.data_dir)
            iris_scores = np.array(iris_list)
            if (face_scores < self.FACE_TOLERANCE).any() and (iris_scores < self.IRIS_TOLERANCE).any():
                matched_faces = np.where(face_scores < self.FACE_TOLERANCE)[0]
                print(matched_faces)
                matched_iris = np.where(iris_scores < self.IRIS_TOLERANCE)[0]
                print(matched_iris)
                common_indexes = [
                    element for element in matched_faces if element in matched_iris]
                print(common_indexes)
                if len(common_indexes) != 0:
                    fusion_scores = self.cal_weighted_product(
                        face_scores, iris_scores)
                    if (fusion_scores < self.FUSION_TOLERANCE).any() and np.argmin(fusion_scores) in common_indexes:
                        return np.argmin(fusion_scores)
                else:
                    return -1
            else:
                return -1

        elif self.modality == 2:
            face_encoding = self.get_face_features(face_dir)
            face_scores = self.get_face_scores(face_encoding, self.data_dir)
            if (face_scores < self.FACE_TOLERANCE).any():
                return np.argmin(face_scores)
            else:
                return -1

        elif self.modality == 3:
            template, mask = self.get_iris_features(iris_dir)
            iris_list = self.get_iris_scores(template, mask, self.data_dir)
            iris_scores = np.array(iris_list)
            if (iris_scores < self.IRIS_TOLERANCE).any():
                return np.argmin(iris_scores)
            else:
                return -1
