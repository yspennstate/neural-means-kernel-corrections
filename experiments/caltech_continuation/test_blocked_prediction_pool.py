"""Small file/array boundary checks; no model training or production data."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from blocked_prediction_pool import Member, PoolError, PredictionPool


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pool-test-", dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.arrays = [np.arange(48, dtype=np.float32).reshape(6, 2, 4),
                       np.arange(48, dtype=np.float64).reshape(6, 8) + .125]
        self.members = [self.save(str(i), array) for i, array in enumerate(self.arrays)]

    def save(self, name, array):
        path = self.root / (name + ".npy")
        np.save(path, array)
        return Member(name, path, hashlib.sha256(path.read_bytes()).hexdigest(), array.shape, array.dtype.str)

    def test_blocks_match_independent_full_stack_and_preserve_row_order(self):
        pool = PredictionPool(self.members)
        whole = np.stack([a.reshape(6, 8).astype(np.float32) for a in self.arrays])
        rows = np.array([5, 1, 3])
        reconstructed = np.concatenate([pool.block(rows, 0, 3), pool.block(rows, 3, 8)], axis=2)
        np.testing.assert_array_equal(reconstructed, whole[:, rows])
        self.assertEqual(pool.verify_integrity(), {m.name: m.sha256 for m in self.members})

    def test_budget_is_checked_before_output_allocation(self):
        pool = PredictionPool(self.members, max_block_bytes=1)
        with patch("blocked_prediction_pool.np.empty", side_effect=AssertionError("allocated too soon")):
            with self.assertRaisesRegex(PoolError, "budget"):
                pool.block([0], 0, 1)

    def test_changed_file_and_incorrect_hash_fail(self):
        bad = Member("bad", self.members[0].path, "0" * 64, self.members[0].shape, self.members[0].dtype)
        with self.assertRaisesRegex(PoolError, "pin"):
            PredictionPool([bad])
        pool = PredictionPool(self.members)
        with self.members[0].path.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(PoolError, "changed"):
            pool.block([0], 0, 1)
        with self.assertRaisesRegex(PoolError, "integrity"):
            pool.verify_integrity()

    def test_invalid_rows_pixels_and_block_budget_fail(self):
        pool = PredictionPool(self.members)
        for rows in ([], [True], [0., 1.], [-1], [6], [0, 0], [[1]]):
            with self.subTest(rows=rows), self.assertRaises(PoolError):
                pool.block(rows, 0, 1)
        for start, stop in ((False, 2), (0, 1.5), (-1, 2), (2, 2), (0, 9)):
            with self.subTest(bounds=(start, stop)), self.assertRaises(PoolError):
                pool.block([0], start, stop)
        for budget in (0, -1, True, 1.5):
            with self.subTest(budget=budget), self.assertRaises(PoolError):
                PredictionPool(self.members, max_block_bytes=budget)

    def test_header_mismatch_and_unsafe_layout_are_refused(self):
        m = self.members[0]
        mismatch = Member(m.name, m.path, m.sha256, (6, 8), m.dtype)
        with self.assertRaisesRegex(PoolError, "header"):
            PredictionPool([mismatch])
        for name, array in (("integer", np.zeros((6, 8), dtype=np.int64)),
                            ("fortran", np.asfortranarray(np.zeros((6, 8))))):
            with self.subTest(name=name), self.assertRaises(PoolError):
                PredictionPool([self.save(name, array)])

    def test_nonfinite_selected_values_fail_without_reading_other_pixels(self):
        array = np.ones((6, 8), dtype=np.float32)
        array[4, 7] = np.nan
        pool = PredictionPool([self.save("nan", array)])
        np.testing.assert_array_equal(pool.block([0], 0, 1), [[[1.]]])
        with self.assertRaisesRegex(PoolError, "Nonfinite"):
            pool.block([4], 7, 8)


if __name__ == "__main__":
    unittest.main()
