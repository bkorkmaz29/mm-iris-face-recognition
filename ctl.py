import unittest
from unittest.mock import MagicMock
from controller import Controller


class MockMultimodal:
    def __init__(self):
        pass

    def enroll_subject(self, face_dir, iris_dir, name, surname):
        pass

    def delete_subject(self, subject_dir):
        pass

    def get_image(self, image_dir, option):
        pass

    def set_mode(self, mode):
        pass

    def set_modality(self, modality):
        pass

    def matching(self, face_dir, iris_dir):
        pass

    def get_at_index(self, index):
        pass

    def get_db_image(self, subject_dir, option):
        pass

    def format_name(self, subject_dir):
        pass


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.multimodal = MockMultimodal()
        self.controller = Controller(self.multimodal)

    def test_on_enroll(self):
        self.multimodal.enroll_subject = MagicMock(
            return_value=("enrollment_successful", "face_img", "iris_img"))

        result = self.controller.on_enroll(
            "face_dir", "iris_dir", "name", "surname")

        self.assertEqual(
            result, ("enrollment_successful", "face_img", "iris_img"))
        self.multimodal.enroll_subject.assert_called_with(
            "face_dir", "iris_dir", "name", "surname")

    def test_on_delete(self):
        self.multimodal.delete_subject = MagicMock(return_value=(1, "subject"))

        result = self.controller.on_delete("subject_dir")

        self.assertEqual(result, (1, "subject"))
        self.multimodal.delete_subject.assert_called_with("subject_dir")

    def test_on_rec_modality_1(self):
        self.multimodal.set_mode = MagicMock()
        self.multimodal.set_modality = MagicMock()
        self.multimodal.get_image = MagicMock(
            side_effect=[("face_img_searched", "iris_img_searched"), ("", ""), ("", "")])
        self.multimodal.matching = MagicMock(return_value=3)
        self.multimodal.get_at_index = MagicMock(return_value="subject_dir_3")
        self.multimodal.get_db_image = MagicMock(
            return_value=("get_face_img_matched", "iris_img_matched"))
        self.multimodal.format_name = MagicMock(return_value="Subject 3")

        result = self.controller.on_rec("face_dir", "iris_dir", "mode", 1)

        self.assertEqual(result[0], 3)
        self.assertEqual(result[1], "Subject 3")
        self.assertEqual(result[2], "face_img_searched")
        self.assertEqual(result[3], "iris_img_searched")
        self.assertEqual(result[4], "get_face_img_matched")
        self.assertEqual(result[5], "iris_img_matched")

    def test_on_rec_modality_2(self):
        self.multimodal.set_mode = MagicMock()
        self.multimodal.set_modality = MagicMock()
        self.multimodal.get_image = MagicMock(
            side_effect=[("face_img_searched", ""), ("", ""), ("", "")])
        self.multimodal.matching = MagicMock(return_value=2)
        self.multimodal.get_at_index = MagicMock(return_value="subject_dir_2")
        self.multimodal.get_db_image = MagicMock(
            return_value=("get_face_img_matched", "iris_img_matched"))
        self.multimodal.format_name = MagicMock(return_value="Subject 2")

        result = self.controller.on_rec("face_dir", "iris_dir", "mode", 2)

        self.assertEqual(result[0], 2)
        self.assertEqual(result[1], "Subject 2")
        self.assertEqual(result[2], "face_img_searched")
        self.assertEqual(result[3], "")
        self.assertEqual(result[4], "get_face_img_matched")
        self.assertEqual(result[5], "iris_img_matched")

    def test_on_rec_modality_3(self):
        self.multimodal.set_mode = MagicMock()
        self.multimodal.set_modality = MagicMock()
        self.multimodal.get_image = MagicMock(
            side_effect=[("", ""), ("", "iris_img_searched"), ("", "")])
        self.multimodal.matching = MagicMock(return_value=1)
        self.multimodal.get_at_index = MagicMock(return_value="subject_dir_1")
        self.multimodal.get_db_image = MagicMock(
            return_value=("get_face_img_matched", "iris_img_matched"))
        self.multimodal.format_name = MagicMock(return_value="Subject 1")

        result = self.controller.on_rec("face_dir", "iris_dir", "mode", 3)

        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], "Subject 1")
        self.assertEqual(result[2], "")
        self.assertEqual(result[3], "iris_img_searched")
        self.assertEqual(result[4], "get_face_img_matched")
        self.assertEqual(result[5], "iris_img_matched")

    def test_on_display(self):
        self.multimodal.get_db_image = MagicMock(
            return_value=("face_img", "iris_img"))
        self.multimodal.format_name = MagicMock(return_value="Subject")

        result = self.controller.on_display("subject_dir")

        self.assertEqual(result[0], "face_img")
        self.assertEqual(result[1], "iris_img")
        self.assertEqual(result[2], "Subject")


if __name__ == "__main__":
    unittest.main()
