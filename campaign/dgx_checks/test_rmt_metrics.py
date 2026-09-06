"""Execute the actual small-calibration PCR reporting path without its data loader."""
import ast
from pathlib import Path
import unittest
import numpy as np


def pcr_record_path(prediction, truth, inherited=False):
    tree = ast.parse(Path(__file__).with_name("ens_rmt_dgx.py").read_text())
    rel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "rel")
    outer = next(n for n in tree.body if isinstance(n, ast.For)
                 and isinstance(n.target, ast.Name) and n.target.id == "ncal")
    inner = next(n for n in outer.body if isinstance(n, ast.For))
    begin = next(i for i, n in enumerate(inner.body)
                 if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                 and isinstance(n.value.func, ast.Name) and n.value.func.id == "pcr_mp_pred")
    end = next(i for i, n in enumerate(inner.body) if isinstance(n, ast.For))
    # Only the actual PCR reporting statements: exclude other estimator fits.
    statements = [n for n in inner.body[begin:end]
                  if isinstance(n, ast.Assign) and (n is inner.body[begin]
                      or any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                             and t.slice.value == "pcr_mp" for t in n.targets))]
    if inherited:
        statements = ast.parse('out["pcr_mp"], out["_pcr_kept"] = pcr_mp_pred(Pc_, Yc_, P60e)').body
    aggregate = next(n for n in outer.body if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Subscript))
    env = dict(np=np, Yte=truth, nte=np.linalg.norm(truth, axis=1), ev=np.arange(len(truth)),
               Pc_=None, Yc_=None, P60e=None, out={}, acc={}, small={}, ncal=120,
               pcr_mp_pred=lambda *unused: (prediction, 2.))
    code = ast.fix_missing_locations(ast.Module(body=[rel] + statements + [inner.body[end], aggregate], type_ignores=[]))
    exec(compile(code, "actual-PCR-record-path", "exec"), env)
    return env["small"]["ncal120"]


class PcrReportingTests(unittest.TestCase):
    def test_perfect_prediction_is_zero_error_not_mean_stress(self):
        truth = np.full((3, 2), 100.)
        corrected = pcr_record_path(truth.copy(), truth)
        self.assertEqual(corrected["pcr_mp"], 0.)
        self.assertEqual(corrected["_pcr_kept"], 2.)
        self.assertEqual(pcr_record_path(truth.copy(), truth, inherited=True)["pcr_mp"], 10000.)

    def test_relative_error_is_invariant_to_joint_units_change(self):
        truth = np.array([[3., 4.], [0., 10.]])
        prediction = np.array([[6., 8.], [0., 11.]])
        # Independently: mean(5/5, 1/10) * 100 = 55 percent.
        self.assertAlmostEqual(pcr_record_path(prediction, truth)["pcr_mp"], 55.)
        self.assertEqual(pcr_record_path(100 * prediction, 100 * truth)["pcr_mp"], 55.)
        self.assertNotEqual(pcr_record_path(prediction, truth, inherited=True)["pcr_mp"],
                            pcr_record_path(100 * prediction, 100 * truth, inherited=True)["pcr_mp"])


if __name__ == "__main__":
    unittest.main()
