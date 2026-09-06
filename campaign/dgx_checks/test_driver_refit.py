"""Exercise the actual final convex refit statements without the full data loader."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


def final_refit(P, Y):
    tree = ast.parse(Path(__file__).with_name("ens_rmt_dgx.py").read_text())
    begin = next(i for i, n in enumerate(tree.body)
                 if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and isinstance(n.value.func, ast.Name) and n.value.func.id == "stage"
                 and n.value.args and isinstance(n.value.args[0], ast.Constant)
                 and n.value.args[0].value == "shrink toward global convex") + 1
    end = next(i for i in range(begin, len(tree.body))
               if isinstance(tree.body[i], ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "Wp" for t in tree.body[i].targets))
    env = dict(np=np, P60c=P, Ycal=Y, m_=P.shape[0], n_=P.shape[1], D_=P.shape[2],
               cal=np.arange(len(Y)), nte=np.linalg.norm(Y, axis=1))
    code = ast.fix_missing_locations(ast.Module(body=tree.body[begin:end], type_ignores=[]))
    exec(compile(code, "actual-final-convex-refit", "exec"), env)
    return env["wg"], env["Wg"]


class FinalRefitTests(unittest.TestCase):
    def setUp(self):
        self.P = np.stack([np.ones((4, 2)), np.full((4, 2), 4.)])
        self.Y = np.full((4, 2), 3.)

    def test_known_convex_interpolation_has_zero_residual(self):
        weights, augmented = final_refit(self.P, self.Y)
        np.testing.assert_allclose(weights, [1 / 3, 2 / 3], atol=1e-10, rtol=0)
        np.testing.assert_allclose(augmented, [[1 / 3, 2 / 3, 0.]] * 2, atol=1e-10, rtol=0)
        np.testing.assert_allclose(np.einsum("m,mnd->nd", weights, self.P), self.Y, atol=1e-10)

    def test_failed_final_optimizer_is_not_reported_as_a_fit(self):
        failed = SimpleNamespace(success=False, x=np.array([.5, .5]), message="injected failure")
        with patch("fold_selection.minimize", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                final_refit(self.P, self.Y)


if __name__ == "__main__":
    unittest.main()
