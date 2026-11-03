import unittest
import numpy as np

from isl_data_collection.storage import validate_label
from isl_data_collection.training import _resample_sequence
from isl_data_collection.config import FEATURE_DIM, SEQUENCE_LENGTH


class CoreWorkflowTests(unittest.TestCase):
    def test_alphabet_label_validation(self):
        self.assertEqual(validate_label("alphabet", "a"), ("alphabet", "A"))
        with self.assertRaises(ValueError):
            validate_label("alphabet", "HELLO")

    def test_number_label_validation(self):
        self.assertEqual(validate_label("number", "two"), ("number", "TWO"))
        with self.assertRaises(ValueError):
            validate_label("number", "HELLO")

    def test_word_label_validation(self):
        self.assertEqual(validate_label("word", "thank you"), ("word", "THANK_YOU"))

    def test_resampling_shape(self):
        data = np.random.default_rng(7).random((17, FEATURE_DIM), dtype=np.float32)
        result = _resample_sequence(data)
        self.assertEqual(result.shape, (SEQUENCE_LENGTH, FEATURE_DIM))
        self.assertEqual(result.dtype, np.float32)

    def test_bad_feature_shape_rejected(self):
        with self.assertRaises(ValueError):
            _resample_sequence(np.zeros((5, 254), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

