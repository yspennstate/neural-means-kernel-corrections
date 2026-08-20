"""Certify the saved prediction arrays against the members' own recorded metrics.

The val-side reconciliation leaves a residual gap on mlp and mlpMSE but not on mlpR or
krr. The hypothesis is test-time augmentation: gen_preds.py writes reflection-averaged
predictions for every architecture (gen_preds.py:88-98), while the mlp/mlpMSE training
JSONs record plain validation error and report TTA separately as test_tta. mlpR trains
with TTA already, so its 'test' IS the TTA number and it reconciles exactly.

Two checks, both like-for-like:
  A. TEST arrays vs the member's own recorded TTA test error. Same estimator, same
     20000 test points, which never move with the split seed. This is exact if the
     arrays are what they claim to be.
  B. the val-side gap against the member's own recorded TTA gain on test. If the gap is
     TTA it must track this, in sign and magnitude.
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, "/srv/aiwork/nmkc10seed/code")
os.environ.setdefault("NMKC_DATA", "/srv/aiwork/nmkc10seed/data/structmech")
from common import load_arrays, canonical_split, rel_l2

ROOT = os.environ.get("NMKC_ROOT", "/srv/aiwork/nmkc10seed")
loads, stress = load_arrays()

for sd in sorted(glob.glob(os.path.join(ROOT, "seeds", "sm_s*"))):
    seed = os.path.basename(sd).replace("sm_s", "")
    if seed == "99":
        continue
    R = os.path.join(sd, "runs")
    os.environ["NMKC_SPLIT_SEED"] = seed
    tr, va, te = canonical_split(n_val=1000, seed=int(seed))
    Yte = stress[te].reshape(len(te), -1).astype(np.float64)
    Yva = stress[va].reshape(len(va), -1).astype(np.float64)
    for j in sorted(glob.glob(os.path.join(R, "*.json"))):
        n = os.path.basename(j)[:-5]
        m = n.split("_")[0]
        d = json.load(open(j))
        if m == "krr":
            pte = os.path.join(R, n + "_pred_test.npy")
            pva = os.path.join(R, n + "_pred_val.npy")
        else:
            pte = os.path.join(R, n + "_predte.npy")
            pva = os.path.join(R, n + "_predva.npy")
        if not (os.path.exists(pte) and os.path.exists(pva)):
            print("s%-2s %-8s NO ARRAYS" % (seed, m))
            continue
        et = rel_l2(np.load(pte).astype(np.float64), Yte)
        ev = rel_l2(np.load(pva).astype(np.float64), Yva)
        # the recorded number that should match an array holding TTA predictions
        ref_te = d.get("test_tta", d.get("test"))
        tta_gain = d.get("test", 0.0) - d.get("test_tta", d.get("test", 0.0))
        val_gap = d["val"] - ev
        print("s%-2s %-8s arr_test %.6f vs recorded_tta %.6f  dA=%.2e |"
              "  val_gap %+.6f vs test_tta_gain %+.6f  dB=%.2e"
              % (seed, m, et, ref_te, abs(et - ref_te), val_gap, tta_gain,
                 abs(val_gap - tta_gain)))
