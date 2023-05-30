import unittest
from unittest.mock import MagicMock
from PIL import Image
from glob import glob
import numpy as np
import io
import dlib
from models.mm import MMR
from models.iris_rec import IrisRec
from models.face_rec import FaceRec
from controller import Controller
from view import MMR2


class FaceRecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load a test image
        test_image_path = "test_data/test_face"
        image = Image.open(test_image_path)
        image = image.convert('RGB')

        cls.test_image = np.array(image)

    def test_detect(self):
        image_array = np.array(self.test_image)
        face_locations = FaceRec.detect(image_array)
        self.assertGreater(len(face_locations), 0)

    def test_get_landmarks(self):
        face_landmarks = FaceRec.get_landmarks(self.test_image)
        self.assertGreater(len(face_landmarks), 0)
        self.assertIsInstance(face_landmarks[0], dlib.full_object_detection)

    def test_get_features(self):
        face_features = FaceRec.get_features(self.test_image)
        self.assertGreater(len(face_features), 0)
        self.assertIsInstance(face_features[0], np.ndarray)

    def test_cal_distance(self):
        fake_features1 = [np.random.rand(128)]
        fake_features2 = [np.random.rand(128)]

        distance = FaceRec.cal_distance(fake_features1, fake_features2)
        self.assertIsInstance(distance.item(), float)

    @classmethod
    def tearDownClass(cls):
        pass


class IrisRecTestCase(unittest.TestCase):

    def setUp(self):
        self.image_dir = "test_data/test_iris"

    def tearDown(self):

        pass

    def test_segment(self):
        eye_image = np.zeros((100, 100))
        eyelash_threshold = 100

        iris_circle, pupil_circle, noisy_image = IrisRec.segment(
            eye_image, eyelash_threshold)

        self.assertIsNotNone(iris_circle)
        self.assertIsNotNone(pupil_circle)
        self.assertIsNotNone(noisy_image)

    def test_normalize(self):

        image = np.zeros((100, 100))
        x_iris, y_iris, r_iris = 50, 50, 30
        x_pupil, y_pupil, r_pupil = 45, 45, 10

        polar_array, polar_noise = IrisRec.normalize(
            image, x_iris, y_iris, r_iris, x_pupil, y_pupil, r_pupil)

        self.assertIsNotNone(polar_array)
        self.assertIsNotNone(polar_noise)

    def test_encode(self):
        polar_array = np.zeros((100, 100))

        noise_array = np.zeros((100, 100), dtype=bool)

        template, mask = IrisRec.encode(polar_array, noise_array)

        self.assertIsNotNone(template)
        self.assertIsNotNone(mask)

    def test_get_features(self):

        image_dir = "test_data/test_iris"

        feature, feature_mask, _ = IrisRec.get_features(image_dir)

        self.assertIsNotNone(feature)
        self.assertIsNotNone(feature_mask)

    def test_cal_distance(self):

        feature1 = np.zeros((100, 200))
        mask1 = np.zeros((100, 200), dtype=bool)
        feature2 = np.zeros((100, 200))
        mask2 = np.zeros((100, 200), dtype=bool)

        distance = IrisRec.cal_distance(feature1, mask1, feature2, mask2)

        self.assertIsNotNone(distance)


class TestController(unittest.TestCase):
    def setUp(self):
        self.multimodal = MagicMock()
        self.controller = Controller(self.multimodal)

    def test_enroll(self):
        face_dir = "test_data/test_face"
        iris_dir = "test_data/test_iris"
        name = "John"
        surname = "Doe"
        expected_result = (1, "face_image_bytes", "iris_image_bytes")

        self.multimodal.enroll_subject.return_value = expected_result
        result = self.controller.enroll(face_dir, iris_dir, name, surname)

        self.multimodal.enroll_subject.assert_called_once_with(
            face_dir, iris_dir, name, surname)
        self.assertEqual(result, expected_result)

    def test_delete(self):
        subject_dir = "1-John-Doe"
        expected_result = (1, "#1 John Doe")

        self.multimodal.delete_subject.return_value = expected_result
        result = self.controller.delete(subject_dir)

        self.multimodal.delete_subject.assert_called_once_with(subject_dir)
        self.assertEqual(result, expected_result)

    def test_recognize_1(self):
        face_dir = "test_data/test_face"
        iris_dir = "test_data/test_iris"
        mode = 2
        modality = 1
        expected_result = (1, "#1 John Doe", "image_bytes",
                           "image_bytes", "image_bytes", "image_bytes")

        self.multimodal.set_mode.return_value = None
        self.multimodal.set_modality.return_value = None
        self.multimodal.get_image.return_value = "image_bytes"
        self.multimodal.matching.return_value = 1
        self.multimodal.get_at_index.return_value = "db/1-John-Doe"
        self.multimodal.get_db_image.return_value = (
            "image_bytes", "image_bytes")
        self.multimodal.format_name.return_value = "#1 John Doe"

        result = self.controller.recognize(face_dir, iris_dir, mode, modality)

        self.multimodal.set_mode.assert_called_once_with(mode)
        self.multimodal.set_modality.assert_called_once_with(modality)
        self.assertEqual(self.multimodal.get_image.call_count, 2)
        self.multimodal.matching.assert_called_once_with(face_dir, iris_dir)
        self.multimodal.get_at_index.assert_called_once_with(1)
        self.assertEqual(self.multimodal.get_db_image.call_count, 1)
        self.multimodal.format_name.assert_called_once_with("1-John-Doe")

        self.assertEqual(result, expected_result)

    def test_recognize_2(self):
        face_dir = "test_data/test_face"
        iris_dir = "test_data/test_iris"
        mode = 2
        modality = 2
        expected_result = (1, "#1 John Doe", "image_bytes",
                           "image_bytes", "image_bytes", "image_bytes")

        self.multimodal.set_mode.return_value = None
        self.multimodal.set_modality.return_value = None
        self.multimodal.get_image.return_value = "image_bytes"
        self.multimodal.matching.return_value = 1
        self.multimodal.get_at_index.return_value = "db/1-John-Doe"
        self.multimodal.get_db_image.return_value = (
            "image_bytes", "image_bytes")
        self.multimodal.format_name.return_value = "#1 John Doe"

        result = self.controller.recognize(face_dir, iris_dir, mode, modality)

        self.multimodal.set_mode.assert_called_once_with(mode)
        self.multimodal.set_modality.assert_called_once_with(modality)
        self.assertEqual(self.multimodal.get_image.call_count, 2)
        self.multimodal.matching.assert_called_once_with(face_dir, iris_dir)
        self.multimodal.get_at_index.assert_called_once_with(1)
        self.assertEqual(self.multimodal.get_db_image.call_count, 1)
        self.multimodal.format_name.assert_called_once_with("1-John-Doe")

        self.assertEqual(result, expected_result)

    def test_recognize_3(self):
        face_dir = "test_data/test_face"
        iris_dir = "test_data/test_iris"
        mode = 2
        modality = 3
        expected_result = (1, "#1 John Doe", "image_bytes",
                           "image_bytes", "image_bytes", "image_bytes")

        self.multimodal.set_mode.return_value = None
        self.multimodal.set_modality.return_value = None
        self.multimodal.get_image.return_value = "image_bytes"
        self.multimodal.matching.return_value = 1
        self.multimodal.get_at_index.return_value = "db/1-John-Doe"
        self.multimodal.get_db_image.return_value = (
            "image_bytes", "image_bytes")
        self.multimodal.format_name.return_value = "#1 John Doe"

        result = self.controller.recognize(face_dir, iris_dir, mode, modality)

        self.multimodal.set_mode.assert_called_once_with(mode)
        self.multimodal.set_modality.assert_called_once_with(modality)
        self.assertEqual(self.multimodal.get_image.call_count, 2)
        self.multimodal.matching.assert_called_once_with(face_dir, iris_dir)
        self.multimodal.get_at_index.assert_called_once_with(1)
        self.assertEqual(self.multimodal.get_db_image.call_count, 1)
        self.multimodal.format_name.assert_called_once_with("1-John-Doe")

        self.assertEqual(result, expected_result)

    def test_display(self):
        subject = "1-John-Doe"
        expected_result = ("face_image_bytes",
                           "iris_image_bytes", "#1 John Doe")

        self.multimodal.get_db_image.return_value = (
            "face_image_bytes", "iris_image_bytes")
        self.multimodal.format_name.return_value = "#1 John Doe"

        result = self.controller.display(subject)

        self.multimodal.get_db_image.assert_called_once_with(subject, 1)
        self.multimodal.format_name.assert_called_once_with(subject)
        self.assertEqual(result, expected_result)


class MMRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mmr = MMR("db/")

    def test_get_image_full(self):
        image_dir = "test_data/test_face"
        result_full_bytes = self.mmr.get_image(image_dir, full=True)
        result_full_image = Image.open(io.BytesIO(result_full_bytes))
        self.assertEqual(result_full_image.size, (240, 240))
        result_full_image.close()

    def test_get_image_not_full(self):
        image_dir = "test_data/test_face"
        result_not_full_bytes = self.mmr.get_image(image_dir, full=False)
        result_not_full_image = Image.open(io.BytesIO(result_not_full_bytes))
        self.assertEqual(result_not_full_image.size, (160, 160))
        result_not_full_image.close()

    def test_get_image(self):
        image_dir = ""
        result_empty_bytes = self.mmr.get_image(image_dir, full=True)
        result_empty_image = Image.open(io.BytesIO(result_empty_bytes))
        im_array = np.array(result_empty_image)
        unique_colors = np.unique(
            im_array.reshape(-1, im_array.shape[2]), axis=0)
        self.assertEqual(len(unique_colors), 1)  # Only one unique color
        self.assertEqual(tuple(unique_colors[0]), (255, 255, 255))
        result_empty_image.close()

    def test_get_db_image(self):
        subject_dir = "1-Cenk-Mert"
        full = True

        expected_face = self.mmr.get_image("db/1-Cenk-Mert/face.png", full)
        expected_iris = self.mmr.get_image("db/1-Cenk-Mert/iris.png", full)
        face_image, iris_image = self.mmr.get_db_image(subject_dir, full)

        self.assertEqual(face_image, expected_face)
        self.assertEqual(iris_image, expected_iris)

    def test_format_name(self):
        subject = "1-John-Doe"
        expected_result = "#1 John Doe"

        result = self.mmr.format_name(subject)
        self.assertEqual(result, expected_result)

    def test_set_modality(self):
        self.mmr.set_modality(1)
        self.assertEqual(self.mmr.modality, 1)

        self.mmr.set_modality(2)
        self.assertEqual(self.mmr.modality, 2)

        self.mmr.set_modality(3)
        self.assertEqual(self.mmr.modality, 3)

    def test_set_mode_1(self):
        self.mmr.set_mode(1)
        self.assertEqual(self.mmr.FACE_TOLERANCE, 0.454)
        self.assertEqual(self.mmr.IRIS_TOLERANCE, 0.404)
        self.assertEqual(self.mmr.FUSION_TOLERANCE, 0.424)

    def test_set_mode_2(self):
        self.mmr.set_mode(2)
        self.assertEqual(self.mmr.FACE_TOLERANCE, 0.481)
        self.assertEqual(self.mmr.IRIS_TOLERANCE, 0.414)
        self.assertEqual(self.mmr.FUSION_TOLERANCE, 0.444)

    def test_set_mode_3(self):
        self.mmr.set_mode(3)
        self.assertEqual(self.mmr.FACE_TOLERANCE, 0.545)
        self.assertEqual(self.mmr.IRIS_TOLERANCE, 0.424)
        self.assertEqual(self.mmr.FUSION_TOLERANCE, 0.461)

    def test_get_at_index(self):
        index = 0
        result = self.mmr.get_at_index(index)
        expected_result = "db\\1-Cenk-Mert"
        self.assertEqual(result, expected_result)

    def test_get_at_index_fail(self):
        index = 100
        result = self.mmr.get_at_index(index)
        self.assertEqual(result, -1)

    def test_find_id(self):
        files = ["db/1-subject-name", "db/2-subject-name", "db/4-subject-name"]
        result = self.mmr.find_id(files)
        expected_result = 3
        self.assertEqual(result, expected_result)

    def test_get_info(self):
        id = 1
        result = self.mmr.get_info(id)
        expected_result = "db\\1-Cenk-Mert"
        self.assertEqual(result, expected_result)

    def test_get_info_fail(self):
        id = 99
        result = self.mmr.get_info(id)
        expected_result = ""
        self.assertEqual(result, expected_result)

    def test_enroll_subject(self):
        face_dir = "test_data\\test_face4"
        iris_dir = "test_data\\test_iris4"
        name = "John"
        surname = "Doe"

        result = self.mmr.enroll_subject(face_dir, iris_dir, name, surname)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], str)
        self.assertIsInstance(result[1], bytes)
        self.assertIsInstance(result[2], bytes)

    def test_delete_subject(self):
        subject_dir = "3-John-Doe"

        result, subject = self.mmr.delete_subject(subject_dir)

        self.assertEqual(result, 1)
        self.assertEqual(subject, "#3 John Doe")

    def test_delete_subject_fail(self):
        subject_dir = "99-Non-Existing"

        result, subject = self.mmr.delete_subject(subject_dir)

        self.assertEqual(result, 0)
        self.assertEqual(subject, "")

    def test_get_face_features(self):
        dir = "db/1-Cenk-Mert/face.png"
        face_features = np.load("db/1-Cenk-Mert/f1.npy")
        result = self.mmr.get_face_features(dir)

        self.assertTrue(np.array_equal(result, face_features))

    def test_get_iris_features(self):
        dir = "db/1-Cenk-Mert/iris.png"
        result_template, result_mask = self.mmr.get_iris_features(dir)
        dir = "db/1-Cenk-Mert/i1.npz"
        iris_data = np.load(dir)
        template = iris_data['template']
        mask = iris_data['mask']
        self.assertTrue(np.array_equal(result_template, template))
        self.assertTrue(np.array_equal(result_mask, mask))

    def test_cal_weighted_product(self):
        face_scores = np.array([0.2, 0.4, 0.6])
        iris_scores = np.array([0.3, 0.5, 0.7])
        expected_product = np.array(
            [0.271829, 0.473589, 0.674243], dtype=np.float32)
        result_product = self.mmr.cal_weighted_product(
            face_scores, iris_scores)
        np.testing.assert_allclose(result_product, expected_product, rtol=1e-5)

    def test_matching_1(self):
        self.mmr.set_modality(1)
        face_dir = "test_data/test_face"
        iris_dir = "test_data/test_iris"

        result = self.mmr.matching(face_dir, iris_dir)

        self.assertEqual(result, 0)

    def test_matching_fail_1(self):
        self.mmr.set_modality(1)
        face_dir = "test_data/test_face2"
        iris_dir = "test_data/test_iris2"

        result = self.mmr.matching(face_dir, iris_dir)

        self.assertIsInstance(result, int)
        self.assertEqual(result, -1)

    def test_matching_2(self):
        self.mmr.set_modality(2)
        face_dir = "test_data/test_face"

        result = self.mmr.matching(face_dir, "")

        self.assertEqual(result, 0)

    def test_matching_fail_2(self):
        self.mmr.set_modality(2)
        face_dir = "test_data/test_face2"

        result = self.mmr.matching(face_dir, "")

        self.assertIsInstance(result, int)
        self.assertEqual(result, -1)

    def test_matching_3(self):
        self.mmr.set_modality(3)
        iris_dir = "test_data/test_iris"

        result = self.mmr.matching("", iris_dir)

        self.assertEqual(result, 0)

    def test_matching_fail_3(self):
        self.mmr.set_modality(3)
        iris_dir = "test_data/test_iris2"

        result = self.mmr.matching("", iris_dir)

        self.assertIsInstance(result, int)
        self.assertEqual(result, -1)

    def test_matching_no_common(self):
        self.mmr.set_modality(1)
        face_dir = "test_data/test_face"
        iris_dir = "test_data/test_iris4"

        result = self.mmr.matching(face_dir, iris_dir)

        self.assertIsInstance(result, int)
        self.assertEqual(result, -1)

    def test_matching_no_face_iris(self):
        self.mmr.set_modality(1)
        face_dir = "test_data/test_face2"
        iris_dir = "test_data/test_iris2"

        result = self.mmr.matching(face_dir, iris_dir)

        self.assertIsInstance(result, int)
        self.assertEqual(result, -1)

    @classmethod
    def tearDownClass(cls):
        pass


class MMR2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mmr = MMR2("db/")

    def test_get_image_full(self):
        image_dir = "test_data/test_face"
        result_full_bytes = self.mmr.get_image(image_dir, full=True)
        result_full_image = Image.open(io.BytesIO(result_full_bytes))
        self.assertEqual(result_full_image.size, (240, 240))
        result_full_image.close()

    def test_get_image_not_full(self):
        image_dir = "test_data/test_face"
        result_not_full_bytes = self.mmr.get_image(image_dir, full=False)
        result_not_full_image = Image.open(io.BytesIO(result_not_full_bytes))
        self.assertEqual(result_not_full_image.size, (160, 160))
        result_not_full_image.close()

    def test_get_image(self):
        image_dir = ""
        result_empty_bytes = self.mmr.get_image(image_dir, full=True)
        result_empty_image = Image.open(io.BytesIO(result_empty_bytes))
        im_array = np.array(result_empty_image)
        unique_colors = np.unique(
            im_array.reshape(-1, im_array.shape[2]), axis=0)
        self.assertEqual(len(unique_colors), 1)  # Only one unique color
        self.assertEqual(tuple(unique_colors[0]), (255, 255, 255))
        result_empty_image.close()

    def test_get_db_image(self):
        subject_dir = "1-Cenk-Mert"
        full = True

        expected_face = self.mmr.get_image("db/1-Cenk-Mert/face.png", full)
        expected_iris = self.mmr.get_image("db/1-Cenk-Mert/iris.png", full)
        face_image, iris_image = self.mmr.get_db_image(subject_dir, full)

        self.assertEqual(face_image, expected_face)
        self.assertEqual(iris_image, expected_iris)

    def test_format_name(self):
        subject = "1-John-Doe"
        expected_result = "#1 John Doe"

        result = self.mmr.format_name(subject)
        self.assertEqual(result, expected_result)

    def test_set_modality(self):
        self.mmr.set_modality(1)
        self.assertEqual(self.mmr.modality, 1)

        self.mmr.set_modality(2)
        self.assertEqual(self.mmr.modality, 2)

        self.mmr.set_modality(3)
        self.assertEqual(self.mmr.modality, 3)

    def test_set_mode_1(self):
        self.mmr.set_mode(1)
        self.assertEqual(self.mmr.FACE_TOLERANCE, 0.454)
        self.assertEqual(self.mmr.IRIS_TOLERANCE, 0.404)
        self.assertEqual(self.mmr.FUSION_TOLERANCE, 0.424)

    def test_set_mode_2(self):
        self.mmr.set_mode(2)
        self.assertEqual(self.mmr.FACE_TOLERANCE, 0.481)
        self.assertEqual(self.mmr.IRIS_TOLERANCE, 0.414)
        self.assertEqual(self.mmr.FUSION_TOLERANCE, 0.444)

    def test_set_mode_3(self):
        self.mmr.set_mode(3)
        self.assertEqual(self.mmr.FACE_TOLERANCE, 0.545)
        self.assertEqual(self.mmr.IRIS_TOLERANCE, 0.424)
        self.assertEqual(self.mmr.FUSION_TOLERANCE, 0.461)

    def test_get_at_index(self):
        index = 0
        result = self.mmr.get_at_index(index)
        expected_result = "db\\1-Cenk-Mert"
        self.assertEqual(result, expected_result)

    def test_get_at_index_fail(self):
        index = 100
        result = self.mmr.get_at_index(index)
        self.assertEqual(result, -1)

    def test_find_id(self):
        files = ["db/1-subject-name", "db/2-subject-name", "db/4-subject-name"]
        result = self.mmr.find_id(files)
        expected_result = 3
        self.assertEqual(result, expected_result)

    def test_get_info(self):
        id = 1
        result = self.mmr.get_info(id)
        expected_result = "db\\1-Cenk-Mert"
        self.assertEqual(result, expected_result)

    def test_get_info_fail(self):
        id = 99
        result = self.mmr.get_info(id)
        expected_result = ""
        self.assertEqual(result, expected_result)

    def test_enroll_subject(self):
        face_dir = "test_data\\test_face4"
        iris_dir = "test_data\\test_iris4"
        name = "John"
        surname = "Doe"

        result = self.mmr.enroll_subject(face_dir, iris_dir, name, surname)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], str)
        self.assertIsInstance(result[1], bytes)
        self.assertIsInstance(result[2], bytes)

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == '__main__':
    unittest.main()
