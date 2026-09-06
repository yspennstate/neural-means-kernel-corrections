import unittest

import numpy as np

from blocked_prediction_pool import PoolError
from streamed_error import RelativeErrorStream


def measure(prediction, target, scale=1.):
    ids = np.array([10, 20, 30, 40, 50])
    stream = RelativeErrorStream(ids, target.shape[1], case_block=2, pixel_block=3)
    for p in range(0, target.shape[1], 3):
        for c in range(0, len(ids), 2):
            stream.add(c, p, ids[c:c+2], scale * prediction[c:c+2, p:p+3], scale * target[c:c+2, p:p+3])
    return stream.finish()


class StreamedErrorTests(unittest.TestCase):
    def test_complete_irregular_tiles_match_direct_norms_and_units(self):
        target = np.arange(1, 36, dtype=float).reshape(5, 7)
        prediction = target + np.arange(35, dtype=float).reshape(5, 7) / 7
        direct = np.mean(np.linalg.norm(prediction-target, axis=1) / np.linalg.norm(target, axis=1))
        self.assertAlmostEqual(measure(prediction, target)["mean_relative_error"], direct, places=14)
        self.assertAlmostEqual(measure(prediction, target, 100.)["mean_relative_error"], direct, places=14)
        self.assertEqual(measure(target, target)["mean_relative_error"], 0.)

    def test_incomplete_or_duplicate_tiles_cannot_report(self):
        stream = RelativeErrorStream([4, 9], 2, case_block=1, pixel_block=1)
        stream.add(0, 0, [4], [[2.]], [[1.]])
        with self.assertRaisesRegex(PoolError, "incomplete"):
            stream.finish()
        with self.assertRaisesRegex(PoolError, "Duplicate"):
            stream.add(0, 0, [4], [[2.]], [[1.]])

    def test_case_order_and_tile_boundaries_are_checked(self):
        stream = RelativeErrorStream([4, 9], 2, case_block=2, pixel_block=2)
        with self.assertRaisesRegex(PoolError, "identities"):
            stream.add(0, 0, [9, 4], np.ones((2, 2)), np.ones((2, 2)))
        for c, p in ((1, 0), (0, 1), (2, 0), (0, 2), (True, 0), (0, .5)):
            with self.subTest(start=(c, p)), self.assertRaises(PoolError):
                stream.add(c, p, [4, 9], np.ones((2, 2)), np.ones((2, 2)))

    def test_zero_target_and_invalid_values_refused_without_marking_tile(self):
        stream = RelativeErrorStream([1], 1)
        with self.assertRaisesRegex(PoolError, "Nonfinite"):
            stream.add(0, 0, [1], [[1e308]], [[1.]])
        self.assertFalse(stream.covered.any())
        stream.add(0, 0, [1], [[1.]], [[0.]])
        with self.assertRaisesRegex(PoolError, "zero target"):
            stream.finish()

    def test_relative_error_overflow_does_not_become_a_scalar_result(self):
        stream = RelativeErrorStream([1], 1)
        stream.add(0, 0, [1], [[1e153]], [[1e-160]])
        with self.assertRaisesRegex(PoolError, "overflowed"):
            stream.finish()


if __name__ == "__main__":
    unittest.main()
