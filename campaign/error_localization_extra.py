"""Two numbers the draft asserts and has not measured on the four-member ensemble:
the correlation of ensemble disagreement with the corrected surrogate's absolute error,
and the boundary rms of the ensemble as against the best member.
"""
import glob, json, os, sys
import numpy as np

ROOT = os.environ.get("NMKC_ROOT", "/srv/aiwork/nmkc10seed")
sys.path.insert(0, ROOT + "/code")
os.environ.setdefault("NMKC_DATA", ROOT + "/data/structmech")
from common import load_arrays, canonical_split  # noqa: E402

NAMES = ["mlp", "mlpMSE", "mlpR", "krr"]
loads, stress = load_arrays()

for sd in sorted(glob.glob(os.path.join(ROOT, "seeds", "sm_s*"))):
    seed = os.path.basename(sd).replace("sm_s", "")
    if seed == "99":
        continue
    R = os.path.join(sd, "runs")
    pf = os.path.join(R, "pix4_corr_pred_test.npy")
    if not os.path.exists(pf):
        continue
    os.environ["NMKC_SPLIT_SEED"] = seed
    tr, va, te = canonical_split(n_val=1000, seed=int(seed))
    Yte = stress[te].reshape(len(te), -1).astype(np.float64)
    Yva = stress[va].reshape(len(va), -1).astype(np.float64)

    def arr(m, split):
        if m == "krr":
            g = glob.glob(os.path.join(R, "krr_*_pred_%s.npy"
                                       % ("val" if split == "va" else "test")))
        else:
            g = [p for p in glob.glob(os.path.join(R, "*_pred%s.npy" % split))
                 if os.path.basename(p).split("_")[0] == m]
        return np.load(g[0]).astype(np.float64)

    mem_te = np.stack([arr(m, "te") for m in NAMES])
    pred = np.load(pf).astype(np.float64)
    disagree = mem_te.std(0).mean(1)
    err_abs = np.linalg.norm(pred - Yte, axis=1)
    err_rel = err_abs / np.linalg.norm(Yte, axis=1)
    c_abs = float(np.corrcoef(disagree, err_abs)[0, 1])
    c_rel = float(np.corrcoef(disagree, err_rel)[0, 1])

    mem_va = np.stack([arr(m, "va") for m in NAMES])
    ens_va = mem_va.mean(0)
    e = [float(np.sqrt(np.mean(np.linalg.norm(p - Yva, axis=1) ** 2))) for p in mem_va]
    bi = int(np.argmin(e))
    g = 41
    bidx = [i for i in range(g * g) if i // g in (0, g - 1) or i % g in (0, g - 1)]
    iidx = [i for i in range(g * g) if i not in set(bidx)]
    sp_b = np.sqrt(np.mean((mem_va[bi] - Yva) ** 2, axis=0))
    sp_e = np.sqrt(np.mean((ens_va - Yva) ** 2, axis=0))
    print("s%-2s corr(disagree,abs) %+.3f  corr(disagree,rel) %+.3f | "
          "boundary best %.3f ens %.3f | interior best %.3f ens %.3f"
          % (seed, c_abs, c_rel, sp_b[bidx].mean(), sp_e[bidx].mean(),
             sp_b[iidx].mean(), sp_e[iidx].mean()))
