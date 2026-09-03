"""Shared loaders and metrics for the paper-2 kernel-learning lanes.

Three problems, one interface. Each loader returns a dict with the standardized inputs
Xtr/Xva/Xte (float64), the targets in the space the kernel regresses (Ztr/Zva/Zte), and a
function phys_err(Z_pred, split) giving the paper's error metric in physical units.

  emit(seed, comp, ntrain=0, rank=64)   EMIT lookup table, one component; fresh split as emit_campaign.py
  oco2(band, seed, ntrain=0)            OCO-2 reduced emulation, protocol of jpl_seeded.py (two metrics)
  structmech(seed, ntrain=0)            structural mechanics, canonical split of the NMKC campaign
Environment: EMIT_DATA, NMKC_JPL_DATA, SM_DATA (default ~/nmkc2/data/structmech).
"""
import os, pathlib
import numpy as np

EMIT_DIR = pathlib.Path(os.environ.get("EMIT_DATA", os.path.expanduser("~/p2/data/emit")))
SM_DIR = pathlib.Path(os.environ.get("SM_DATA", os.path.expanduser("~/nmkc2/data/structmech")))
COMPONENTS = ["Y1", "Y2", "Y3", "Y4"]
TEST_REFL = 0.7


def rel_l2(Yt, Yp):
    return float(np.mean(np.linalg.norm(Yt - Yp, axis=1) / np.linalg.norm(Yt, axis=1)))


class Std:
    def __init__(self, A):
        self.m = A.mean(0); self.s = np.sqrt(A.var(0)); self.s[self.s == 0] = 1.0

    def fwd(self, A):
        return (A - self.m) / self.s

    def inv(self, A):
        return A * self.s + self.m


class PCA:
    def __init__(self, Ystd, rank):
        U, S, Vt = np.linalg.svd(Ystd - Ystd.mean(0), full_matrices=False)
        self.c = Ystd.mean(0); self.Vt = Vt[:rank]

    def fwd(self, Y):
        return (Y - self.c) @ self.Vt.T

    def inv(self, Z):
        return Z @ self.Vt + self.c


def emit_split(seed, ntrain=0):
    n_all = np.load(EMIT_DIR / "X.npy", mmap_mode="r").shape[0]
    rng = np.random.RandomState(seed); perm = rng.permutation(n_all)
    n_te = int(round(0.1 * n_all)); idx_te = perm[:n_te]; tr_full = perm[n_te:]
    vperm = np.random.RandomState(seed + 10000).permutation(len(tr_full))
    n_val = int(round(0.1 * len(tr_full)))
    idx_val, idx_tr = tr_full[vperm[:n_val]], tr_full[vperm[n_val:]]
    if ntrain and ntrain < len(idx_tr):
        idx_tr = idx_tr[:ntrain]
    return idx_tr, idx_val, idx_te


def emit(seed, comp, ntrain=0, rank=64):
    X = np.load(EMIT_DIR / "X.npy").astype(np.float64)
    Y = np.load(EMIT_DIR / (comp + ".npy")).astype(np.float64)
    idx_tr, idx_val, idx_te = emit_split(seed, ntrain)
    xs = Std(X[idx_tr]); Xs = xs.fwd(X)
    ys = Std(Y[idx_tr]); pca = PCA(ys.fwd(Y[idx_tr]), rank)
    Z = {k: pca.fwd(ys.fwd(Y[i])) for k, i in (("tr", idx_tr), ("va", idx_val), ("te", idx_te))}
    Yph = {"tr": Y[idx_tr], "va": Y[idx_val], "te": Y[idx_te]}

    def phys_err(Zp, split):
        return rel_l2(Yph[split], ys.inv(pca.inv(Zp)))
    ymean = Yph["tr"].mean(0)
    return dict(problem="emit", tag=f"emit_{comp}_s{seed}", Xtr=Xs[idx_tr], Xva=Xs[idx_val], Xte=Xs[idx_te],
                Ztr=Z["tr"], Zva=Z["va"], Zte=Z["te"], phys_err=phys_err, to_phys=lambda Zp: ys.inv(pca.inv(Zp)),
                Yobj=Yph["tr"] - ymean, ynorm2=(Yph["tr"] ** 2).sum(1),
                idx_te=idx_te, names=["aod", "elev", "h2o", "azim", "solzen", "viewzen"])


def oco2(band, seed, ntrain=0):
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import jpl_data
    from jpl_data import load_band, reconstruction, radiance_error
    if os.environ.get("NMKC_JPL_DATA"):
        jpl_data.DATA = pathlib.Path(os.environ["NMKC_JPL_DATA"])
    sp = load_band(band, seed=seed)
    Xtr, Ytr, Xva, Yva, Xte, Yte = (sp[k] for k in ("Xtr", "Ytr", "Xval", "Yval", "Xte", "Yte"))
    if ntrain and ntrain < len(Xtr):
        Xtr, Ytr = Xtr[:ntrain], Ytr[:ntrain]
    recon = reconstruction(band)
    xs = Std(Xtr)
    Yph = {"tr": Ytr, "va": Yva, "te": Yte}

    def phys_err(Zp, split):
        return rel_l2(Yph[split], Zp)

    def rad_err(Zp, split):
        return radiance_error(Zp, Yph[split], recon)
    return dict(problem="oco2", tag=f"oco_{band}_s{seed}", Xtr=xs.fwd(Xtr), Xva=xs.fwd(Xva), Xte=xs.fwd(Xte),
                Ztr=Ytr, Zva=Yva, Zte=Yte, phys_err=phys_err, rad_err=rad_err, to_phys=lambda Zp: Zp,
                Yobj=Ytr - Ytr.mean(0), ynorm2=(Ytr ** 2).sum(1),
                names=[f"x{j}" for j in range(Xtr.shape[1])])


def structmech(seed, ntrain=0, n_val=1000):
    loads = np.load(SM_DIR / "loads.npy").astype(np.float64)
    stress = np.load(SM_DIR / "stress.npy").astype(np.float32)
    itr = np.load(SM_DIR / "idx_train.npy"); ite = np.load(SM_DIR / "idx_test.npy")
    perm = np.random.default_rng(seed).permutation(len(itr))
    val, tr = itr[perm[:n_val]], itr[perm[n_val:]]
    if ntrain and ntrain < len(tr):
        tr = tr[:ntrain]
    xs = Std(loads[tr])
    Y = {"tr": stress[tr].reshape(len(tr), -1).astype(np.float64), "va": stress[val].reshape(len(val), -1).astype(np.float64),
         "te": stress[ite].reshape(len(ite), -1).astype(np.float64)}

    def phys_err(Zp, split):
        return rel_l2(Y[split], Zp)
    return dict(problem="structmech", tag=f"sm_s{seed}", Xtr=xs.fwd(loads[tr]), Xva=xs.fwd(loads[val]), Xte=xs.fwd(loads[ite]),
                Ztr=Y["tr"], Zva=Y["va"], Zte=Y["te"], phys_err=phys_err, to_phys=lambda Zp: Zp,
                Yobj=Y["tr"] - Y["tr"].mean(0), ynorm2=(Y["tr"] ** 2).sum(1),
                names=[f"load{j}" for j in range(loads.shape[1])])
