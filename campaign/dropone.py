"""Marginal value of each architecture in the sixty-predictor pool.

Reads campaign/collected/dgx/seedarch.json (the second-moment matrices of the sixty
normalized residual vectors on the calibration and evaluation halves of the test block)
and reports, for the whole pool and for the pool with one architecture's ten seeds removed,
the root-mean-square error of the convex mixture chosen in hindsight on the evaluation
half, of the convex mixture fitted on the calibration half and read on the evaluation
half, and of the uniform mixture. Also the same three numbers for each architecture on
its own. Writes campaign/collected/dgx/dropone.json.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "collected", "dgx", "seedarch.json")
OUT = os.path.join(HERE, "collected", "dgx", "dropone.json")


def convex(S):
    m = len(S)
    res = minimize(lambda z: z @ S @ z, np.ones(m) / m, jac=lambda z: 2 * S @ z,
                   bounds=[(0, 1)] * m, constraints={"type": "eq", "fun": lambda z: z.sum() - 1},
                   method="SLSQP", options=dict(maxiter=5000, ftol=1e-16))
    w = np.maximum(res.x, 0)
    return w / w.sum()


def study(S_cal, S_ev, keep):
    keep = np.array(sorted(keep))
    Se, Sc = S_ev[np.ix_(keep, keep)], S_cal[np.ix_(keep, keep)]
    w_or = convex(Se)
    w_c = convex(Sc)
    eq = np.ones(len(keep)) / len(keep)
    return dict(oracle_rms=float(np.sqrt(w_or @ Se @ w_or)),
                cal_to_ev_rms=float(np.sqrt(w_c @ Se @ w_c)),
                equal_rms=float(np.sqrt(eq @ Se @ eq)))


def main():
    d = json.load(open(SRC))
    names, arch = d["names"], d["arch"]
    S_ev, S_cal = np.array(d["S_ev"]), np.array(d["S_cal"])
    idx = {a: [i for i, n in enumerate(names) if n.split("_")[0] == a] for a in arch}
    every = list(range(len(names)))
    out = {"all": study(S_cal, S_ev, every)}
    for a in arch:
        r = study(S_cal, S_ev, [i for i in every if i not in idx[a]])
        for k in list(r):
            r["d_" + k] = r[k] - out["all"][k]
        out["drop_" + a] = r
        out["only_" + a] = study(S_cal, S_ev, idx[a])
    json.dump(out, open(OUT, "w"), indent=1)
    print("all sixty: oracle %.4f  fitted %.4f  equal %.4f" % tuple(100 * out["all"][k] for k in ("oracle_rms", "cal_to_ev_rms", "equal_rms")))
    for a in arch:
        r = out["drop_" + a]
        print("without %-7s oracle %+.4f  fitted %+.4f  equal %+.4f   alone %.4f" % (a, 100 * r["d_oracle_rms"], 100 * r["d_cal_to_ev_rms"], 100 * r["d_equal_rms"], 100 * out["only_" + a]["oracle_rms"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
