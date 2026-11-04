import unittest
import numpy as np

from isl_data_collection.storage import validate_label
from isl_data_collection.training import _resample_sequence
from isl_data_collection.config import FEATURE_DIM, SEQUENCE_LENGTH


class CoreWorkflowTests(unittest.TestCase):
    def test_alphabet_label_validation(self):
        self.assertEqual(validate_label("alphabet", "a"), ("alphabet", "A"))
        self.assertEqual(validate_label("alphabet", "z"), ("alphabet", "Z"))
        with self.assertRaises(ValueError):
            validate_label("alphabet", "HELLO")
        with self.assertRaises(ValueError):
            validate_label("alphabet", "1")

    def test_number_label_validation(self):
        # Verify all ten number classes (0 to 9)
        for i in range(10):
            self.assertEqual(validate_label("number", str(i)), ("number", str(i)))

        # Verify source word normalization to digits
        self.assertEqual(validate_label("number", "two"), ("number", "2"))
        self.assertEqual(validate_label("number", "ZERO"), ("number", "0"))
        self.assertEqual(validate_label("number", "nine"), ("number", "9"))

        # Verify invalid and out-of-range labels are rejected
        with self.assertRaises(ValueError):
            validate_label("number", "10")
        with self.assertRaises(ValueError):
            validate_label("number", "HELLO")
        with self.assertRaises(ValueError):
            validate_label("number", "-1")

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
