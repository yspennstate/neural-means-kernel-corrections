"""Small adversarial and independent-arithmetic tests; no training or data loader."""
import os
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
import unittest
import numpy as np
from fold_selection import (convex_weights, fold_candidates, loo_ridge_scores,
                            pixel_prediction, ridge_weights, select_shrinkage)


class FoldSelectionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(318)
        self.Y = rng.normal(size=(12, 3)) + 2
        self.P = np.stack([self.Y + .5 * rng.normal(size=self.Y.shape),
                           .9 * self.Y + .4 * rng.normal(size=self.Y.shape),
                           1.1 * self.Y + .4 * rng.normal(size=self.Y.shape)])

    def test_loo_formula_against_actual_leave_one_out_refits(self):
        candidates = np.array([.003, .3, 3.])
        formula = loo_ridge_scores(self.P, self.Y, candidates, pixel_block=2)
        n = len(self.Y)
        direct = []
        for lam in candidates:
            sum_squared = 0.
            for held in range(n):
                keep = np.delete(np.arange(n), held)
                # Hold the full-smoother penalty n*lambda fixed after deletion.
                weights = ridge_weights(self.P[:, keep], self.Y[keep], lam * n / (n - 1))
                prediction = pixel_prediction(self.P[:, held:held + 1], weights)[0]
                sum_squared += np.sum((prediction - self.Y[held]) ** 2)
            direct.append(sum_squared / n)
        np.testing.assert_allclose(formula, direct, rtol=2e-11, atol=1e-11)

    def test_held_labels_cannot_change_fold_candidates(self):
        held = np.array([0, 3, 7]); train = np.setdiff1d(np.arange(12), held)
        first, fit1 = fold_candidates(self.P[:, train], self.Y[train], self.P[:, held],
                                     [.01, .1, 1.], [0., .5, 1.], pixel_block=2)
        changed = self.Y.copy(); changed[held] = 100 * changed[held] + 500
        second, fit2 = fold_candidates(self.P[:, train], changed[train], self.P[:, held],
                                      [.01, .1, 1.], [0., .5, 1.], pixel_block=2)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(fit1, fit2)
        # Positive control: the inherited full-calibration convex fit depends on these labels.
        before = convex_weights(self.P, self.Y)
        after = convex_weights(self.P, changed)
        self.assertGreater(np.max(np.abs(before - after)), 1e-4)

    def test_pixel_blocking_does_not_change_candidates(self):
        train = np.arange(9); held = np.arange(9, 12)
        p1, r1 = fold_candidates(self.P[:, train], self.Y[train], self.P[:, held], [.01, .1, 1.], [0., .5, 1.], 1)
        p2, r2 = fold_candidates(self.P[:, train], self.Y[train], self.P[:, held], [.01, .1, 1.], [0., .5, 1.], 3)
        np.testing.assert_allclose(p1, p2, rtol=1e-10, atol=1e-10)
        self.assertEqual(r1["ridge_lambda"], r2["ridge_lambda"])

    def test_folds_are_disjoint_and_cover_every_case_once(self):
        selected, record = select_shrinkage(self.P, self.Y, [.01, .1, 1.], [0., .5, 1.], n_splits=5, pixel_block=2)
        self.assertIn(selected, [0., .5, 1.])
        seen = []
        for fold in record["folds"]:
            self.assertFalse(set(fold["training_rows"]) & set(fold["held_rows"]))
            self.assertEqual(set(fold["training_rows"]) | set(fold["held_rows"]), set(range(12)))
            seen.extend(fold["held_rows"])
        self.assertEqual(sorted(seen), list(range(12)))
        weighted = sum(len(f["held_rows"]) * np.array(f["mean_relative_error"]) for f in record["folds"]) / 12
        np.testing.assert_allclose(weighted, record["mean_relative_error"], rtol=1e-14)

    def test_nonfinite_and_zero_norm_targets_refused(self):
        for value in [np.nan, np.inf]:
            P = self.P.copy(); P[0, 0, 0] = value
            with self.assertRaises(ValueError):
                select_shrinkage(P, self.Y, [.1])
        Y = self.Y.copy(); Y[0] = 0
        with self.assertRaises(ValueError):
            select_shrinkage(self.P, Y, [.1])

    def test_invalid_pixel_blocks_refused_at_each_public_entry(self):
        for value in [0, -1, True, np.bool_(True), 1.5, 2., None]:
            calls = [lambda: convex_weights(self.P, self.Y, value),
                     lambda: loo_ridge_scores(self.P, self.Y, [.1], value),
                     lambda: fold_candidates(self.P[:, :9], self.Y[:9], self.P[:, 9:], [.1], [.5], value),
                     lambda: select_shrinkage(self.P, self.Y, [.1], pixel_block=value)]
            for call in calls:
                with self.subTest(value=value, call=call):
                    with self.assertRaises(ValueError):
                        call()
        expected = convex_weights(self.P, self.Y, 2)
        np.testing.assert_array_equal(expected, convex_weights(self.P, self.Y, np.int64(2)))


if __name__ == "__main__":
    unittest.main()
