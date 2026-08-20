"""The uncertainty the seed spread does not contain.

All ten seeds are scored on the same fixed 20000-sample test block, so the seed standard
deviation measures training and selection variability only. The test block is itself a
sample, and its contribution to the uncertainty of the reported mean relative error is a
separate quantity that the seed spread cannot see. This computes it:

  * the marginal standard error of the mean relative error over the 20000 test samples,
    which is what a comparison against a published number carries (we cannot pair against
    a method whose per-sample predictions we do not have);
  * the PAIRED standard error of a within-pipeline difference, where both predictors are
    evaluated on the same samples and the shared difficulty of a sample cancels. This is
    the right error bar on the correction's gain.
"""
import glob, json, os, sys
import numpy as np

ROOT = os.environ.get("NMKC_ROOT", "/srv/aiwork/nmkc10seed")
sys.path.insert(0, ROOT + "/code")
os.environ.setdefault("NMKC_DATA", ROOT + "/data/structmech")
from common import load_arrays, canonical_split  # noqa: E402

loads, stress = load_arrays()
NAMES = ["mlp", "mlpMSE", "mlpR", "krr"]

for sd in sorted(glob.glob(os.path.join(ROOT, "seeds", "sm_s*"))):
    seed = os.path.basename(sd).replace("sm_s", "")
    if seed == "99":
        continue
    R = os.path.join(sd, "runs")
    pf = os.path.join(R, "pix4_corr_pred_test.npy")
    if not os.path.exists(pf):
        continue
    os.environ["NMKC_SPLIT_SEED"] = seed
    _, _, te = canonical_split(n_val=1000, seed=int(seed))
    Y = stress[te].reshape(len(te), -1).astype(np.float64)
    den = np.sqrt((Y ** 2).sum(1))
    corr = np.sqrt(((np.load(pf).astype(np.float64) - Y) ** 2).sum(1)) / den

    # the same seed's best single member, for a paired within-pipeline comparison
    g = [p for p in glob.glob(os.path.join(R, "*_predte.npy"))
         if os.path.basename(p).split("_")[0] == "mlpMSE"][0]
    best = np.sqrt(((np.load(g).astype(np.float64) - Y) ** 2).sum(1)) / den

    n = len(te)
    se_marg = corr.std(ddof=1) / np.sqrt(n)
    d = best - corr
    se_pair = d.std(ddof=1) / np.sqrt(n)
    print("s%-2s mean %.5f | per-sample sd %.5f | marginal SE %.5f (%.4f pts) | "
          "paired gain over best member %.5f +- %.5f (%.1f sigma)"
          % (seed, corr.mean(), corr.std(ddof=1), se_marg, 100 * se_marg,
             d.mean(), se_pair, d.mean() / se_pair))
