"""Reconcile the FNO prediction arrays against the right recorded field.

verify_arrays.py looks for `test_tta`. train_fno.py -- and therefore finalize_fno.py,
which reproduces its tail -- records the reflection-averaged test error under
`test_mirror_tta`. So the general checker silently fell back to the PLAIN test error and
reported a ~7e-4 mismatch on every FNO, which is precisely the TTA gain and not a fault
in the data. This compares against the field that actually holds the same estimator the
array holds.

If these do not agree to storage precision, the five-member stack is void, because a
prediction array written on the wrong validation split is exactly the bug that produced
0.28 errors earlier in this campaign and looked like a training failure.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.environ.get("NMKC_ROOT", "/srv/aiwork/nmkc10seed")
sys.path.insert(0, ROOT + "/code")
os.environ.setdefault("NMKC_DATA", ROOT + "/data/structmech")
from common import load_arrays, canonical_split, rel_l2  # noqa: E402

loads, stress = load_arrays()
worst = 0.0
for sd in sorted(glob.glob(os.path.join(ROOT, "seeds", "sm_s[0-9]"))):
    seed = os.path.basename(sd).replace("sm_s", "")
    R = os.path.join(sd, "runs")
    n = "fno_s%s_w64_m14_L4_mir" % seed
    j, a = os.path.join(R, n + ".json"), os.path.join(R, n + "_predte.npy")
    if not (os.path.exists(j) and os.path.exists(a)):
        print("s%-2s fno: no artifact pair" % seed)
        continue
    d = json.load(open(j))
    os.environ["NMKC_SPLIT_SEED"] = seed
    _, _, te = canonical_split(n_val=1000, seed=int(seed))
    Y = stress[te].reshape(len(te), -1).astype(np.float64)
    e = rel_l2(np.load(a).astype(np.float64), Y)
    ref = d["test_mirror_tta"]
    gap = abs(e - ref)
    worst = max(worst, gap)
    print("s%-2s fno  array %.6f  vs recorded test_mirror_tta %.6f   d=%.2e  %s"
          % (seed, e, ref, gap, "OK" if gap < 1e-5 else "MISMATCH"))
print("\nworst gap over the ten FNO members: %.3e" % worst)
print("(float32 storage precision is ~1e-7; anything at 1e-4 or worse means the array and"
      " the metric describe different objects)")
