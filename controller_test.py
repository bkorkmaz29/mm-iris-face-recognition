import unittest
import os
import tempfile


from models.mm import MMR
from controller import Controller


class ControllerTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.multimodal = MMR(self.temp_dir)
        self.controller = Controller(self.multimodal)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_on_enroll(self):
        face_dir = "test_face"
        iris_dir = "test_iris"
        name = "John"
        surname = "Doe"

        id, face_img, iris_img = self.controller.on_enroll(
            face_dir, iris_dir, name, surname)

        # Check if enrollment is successful
        self.assertNotEqual(id, 0)

        # Check if the enrollment directory exists
        enrollment_dir = os.path.join(self.temp_dir, str(
            id) + "-" + name + "-" + surname)
        self.assertTrue(os.path.exists(enrollment_dir))

    def test_on_recognize(self):
        face_dir = "test_face"
        iris_dir = "test_iris"
        modality = 1
        mode = 2

        result = self.controller.on_recognize(
            face_dir, iris_dir, modality, mode)

        # Check if the result is as expected
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)
        self.assertIsNotNone(result[0])
        self.assertIsNotNone(result[1])
        self.assertIsNotNone(result[2])
        self.assertIsNotNone(result[3])
        self.assertIsNotNone(result[4])

    def test_on_display(self):
        subject_dir = "1-John-Doe"
        full = True

        result = self.controller.on_display(subject_dir, full)

        # Check if the result is as expected
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        self.assertIsNotNone(result[0])
        self.assertIsNotNone(result[1])
        self.assertIsNotNone(result[2])

    def test_on_delete(self):
        subject_dir = "1-John-Doe"

        # Create a subject directory
        subject_dir_path = os.path.join(self.temp_dir, subject_dir)
        os.makedirs(subject_dir_path)

        result = self.controller.on_delete(subject_dir)

        # Check if the subject directory is deleted
        self.assertFalse(os.path.exists(subject_dir_path))

        # Check if the result is as expected
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], "")


if __name__ == '__main__':
    unittest.main()
