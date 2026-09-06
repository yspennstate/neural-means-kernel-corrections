import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from affine_pool_evaluation import evaluate_affine_models
from blocked_prediction_pool import Member, PoolError, PredictionPool


class AffinePoolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="affine-test-", dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(491)
        self.P = rng.normal(size=(3, 8, 5)).astype(np.float32)
        self.Y = rng.normal(size=(12, 5)) + 2.
        self.pool = PredictionPool([self.save(f"member{i}", row) for i, row in enumerate(self.P)])
        self.targets = PredictionPool([self.save("targets", self.Y)])
        self.weights = {"first": rng.normal(size=(5, 4)), "second": rng.normal(size=(5, 4))}
        self.options = dict(calibration_rows=[2, 3, 5, 6], evaluation_rows=[7, 1, 4, 0],
                            target_rows=[8, 2, 0, 7], weights=self.weights, case_block=3, pixel_block=2)

    def save(self, name, array):
        path = self.root / (name + ".npy")
        np.save(path, array)
        return Member(name, path, hashlib.sha256(path.read_bytes()).hexdigest(), array.shape, array.dtype.str)

    def test_complete_stream_matches_independent_per_pixel_matrix_products(self):
        result = evaluate_affine_models(self.pool, self.targets, **self.options)
        truth = self.Y[self.options["target_rows"]]
        for name, w in self.weights.items():
            prediction = np.empty_like(truth)
            for pixel in range(self.P.shape[2]):
                X = np.column_stack([self.P[:, self.options["evaluation_rows"], pixel].T, np.ones(len(truth))])
                prediction[:, pixel] = X @ w[pixel]
            expected = np.mean(np.linalg.norm(prediction-truth, axis=1) / np.linalg.norm(truth, axis=1))
            self.assertAlmostEqual(result["results"][name]["mean_relative_error"], expected, places=14)
        self.assertEqual(result["n_evaluation"], 4)
        self.assertEqual(result["results"]["first"]["covered_tiles"], 6)

    def test_overlapping_or_mismatched_splits_fail_before_read(self):
        for changes in (dict(calibration_rows=[0, 2]), dict(target_rows=[8, 2]), dict(evaluation_rows=[7, 7, 4, 0])):
            with self.subTest(changes=changes), patch.object(self.pool, "block", side_effect=AssertionError("read too soon")):
                with self.assertRaises(PoolError):
                    evaluate_affine_models(self.pool, self.targets, **(self.options | changes))

    def test_invalid_or_over_budget_weights_fail_before_read(self):
        for changes in (dict(weights={}), dict(weights={"bad": np.full((5, 4), np.nan)}), dict(max_weight_bytes=1)):
            with self.subTest(changes=changes), patch.object(self.pool, "block", side_effect=AssertionError("read too soon")):
                with self.assertRaises(PoolError):
                    evaluate_affine_models(self.pool, self.targets, **(self.options | changes))

    def test_input_change_during_tiles_cannot_return_a_result(self):
        original = self.targets.block
        first = True
        def changing_block(*args, **kwargs):
            nonlocal first
            values = original(*args, **kwargs)
            if first:
                first = False
                with self.pool.members[0].path.open("ab") as handle:
                    handle.write(b"mutation")
            return values
        with patch.object(self.targets, "block", side_effect=changing_block):
            with self.assertRaisesRegex(PoolError, "changed"):
                evaluate_affine_models(self.pool, self.targets, **self.options)


if __name__ == "__main__":
    unittest.main()
