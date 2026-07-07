"""Shared data loading, splits, and the benchmark metric."""
import numpy as np, torch, pathlib, json, time

DATA = pathlib.Path(__file__).resolve().parent / "data"
RUNS = pathlib.Path(__file__).resolve().parent / "runs"

def load_arrays():
    loads = np.load(DATA / "loads.npy")     # (N,41) float32
    stress = np.load(DATA / "stress.npy")   # (N,41,41) float32
    return loads, stress

def canonical_split(n_val=1000, seed=0):
    """Train pool = samples 0..19999, test = 20000..39999 (de Hoop convention).
    n_val samples for model selection are taken from the train pool with a fixed
    permutation so every script sees the same split."""
    itr = np.load(DATA / "idx_train.npy")
    ite = np.load(DATA / "idx_test.npy")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(itr))
    val = itr[perm[:n_val]]
    tr = itr[perm[n_val:]]
    return tr, val, ite

def rel_l2(yhat, ytrue):
    """Mean relative L2 error over samples; plain grid norm (the convention of
    de Hoop et al. / user's scripts). Cast to float64 for the metric."""
    yhat = np.asarray(yhat, dtype=np.float64).reshape(len(yhat), -1)
    ytrue = np.asarray(ytrue, dtype=np.float64).reshape(len(ytrue), -1)
    num = np.linalg.norm(yhat - ytrue, axis=1)
    den = np.linalg.norm(ytrue, axis=1)
    return float(np.mean(num / den))

def rel_l2_trapz(yhat, ytrue):
    """Same but with trapezoidal quadrature weights on the 41x41 grid."""
    w1 = np.ones(41); w1[0] = w1[-1] = 0.5
    W = np.sqrt(np.outer(w1, w1)).reshape(1, 41, 41)
    yhat = np.asarray(yhat, dtype=np.float64).reshape(len(yhat), 41, 41) * W
    ytrue = np.asarray(ytrue, dtype=np.float64).reshape(len(ytrue), 41, 41) * W
    num = np.linalg.norm(yhat.reshape(len(yhat), -1) - ytrue.reshape(len(ytrue), -1), axis=1)
    den = np.linalg.norm(ytrue.reshape(len(ytrue), -1), axis=1)
    return float(np.mean(num / den))

def save_run(name, payload):
    RUNS.mkdir(exist_ok=True)
    with open(RUNS / f"{name}.json", "w") as f:
        json.dump(payload, f, indent=1)

class Timer:
    def __init__(self): self.t0 = time.time()
    def __call__(self): return time.time() - self.t0
