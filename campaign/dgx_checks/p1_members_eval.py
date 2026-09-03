"""Paper 1 additions from the DGX lanes, in paper 1's own conventions (structural mechanics, test block of 20,000,
calibration 1,000 / evaluation 19,000 with the seedarch split seed 20260902, per-sample relative L2 in per cent):

  A. the kernel-flow-regularized MLP members (train_mlp.py --kf 0.3 / 1.0): test error per seed from the run records,
     against the plain MLP member of the same seed (the JSONs carry test and test_tta on the full 20,000);
  B. the learned-kernel KRR members (kf_kernels.py: isotropic validation grid, KF-ARD, EB-ARD, additive) per seed on the
     evaluation half, against paper 1's KRR member and MLP member on the same rows;
  C. stacks on the calibration/evaluation split: the six ten-seed means with the paper's ridge (reproduces 4.556), the
     same with the LOO ridge, and the seven means adding the ten-seed mean of the learned kernel (KF-ARD), both ridges.
Writes ~/nmkc2/results/p1_members_eval.json.
"""
import glob, json, os, time
import numpy as np

ROOT = os.path.expanduser("~/nmkc2"); P2 = os.path.expanduser("~/p2")
t0 = time.time()
stress = np.load(ROOT + "/data/structmech/stress.npy"); ite = np.load(ROOT + "/data/structmech/idx_test.npy")
Yte = stress[ite].reshape(len(ite), -1).astype(np.float64); nte = np.linalg.norm(Yte, axis=1)
rng = np.random.default_rng(20260902); perm = rng.permutation(len(ite)); cal, ev = np.sort(perm[:1000]), np.sort(perm[1000:])
ARCH = ["mlp", "mlpMSE", "mlpR", "fno", "unet", "krr"]


def rel(pred, rows):
    return float((np.linalg.norm(pred - Yte[rows], axis=1) / nte[rows]).mean())


def member_file(d, a):
    if a == "krr":
        return d + "/krr_full_matern52_n19000_pred_test.npy"
    c = [p for p in glob.glob(d + "/*_predte.npy") if os.path.basename(p).split("_")[0] == a and "mpprune" not in p]
    assert len(c) == 1, (a, d, c); return c[0]


out = {"A_kf_members": {}, "B_learned_kernels": {}, "C_stacks": {}}
# ---- A: KF-regularized MLP members from the run records
A = {"mlp": [], "mlpKF03": [], "mlpKF1": []}; A_tta = {"mlp": [], "mlpKF03": [], "mlpKF1": []}
for s in range(10):
    d = f"{ROOT}/seeds/sm_s{s}/runs"
    for tag in ("mlp", "mlpKF03", "mlpKF1"):
        fs = glob.glob(f"{d}/{tag}_s{s}_w1024_d4_n19000_mir.json")
        if fs:
            j = json.load(open(fs[0])); A[tag].append(100 * j["test"]); A_tta[tag].append(100 * j["test_tta"])
for tag in A:
    v, vt = np.array(A[tag]), np.array(A_tta[tag])
    out["A_kf_members"][tag] = dict(n=len(v), test_mean=float(v.mean()), test_sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                                   tta_mean=float(vt.mean()), tta_sd=float(vt.std(ddof=1)) if len(vt) > 1 else 0.0, per_seed=[round(float(x), 4) for x in v])
# paired against the plain member on the same seeds
for tag in ("mlpKF03", "mlpKF1"):
    n = len(A[tag]); d_ = np.array(A[tag]) - np.array(A["mlp"][:n])
    out["A_kf_members"][tag + "_minus_mlp"] = dict(mean=float(d_.mean()), sd=float(d_.std(ddof=1)) if n > 1 else 0.0, n=n, positive=int((d_ > 0).sum()))
print("A:", {k: (round(v["test_mean"], 4), v["n"]) for k, v in out["A_kf_members"].items() if "n" in v and "test_mean" in v}, flush=True)

# ---- B: learned-kernel members on the evaluation half, with paper 1's KRR and MLP members on the same rows
B = {}; KF = {}
for s in range(10):
    z = np.load(f"{P2}/results/preds/sm_s{s}_kfk.npz")
    d = f"{ROOT}/seeds/sm_s{s}/runs"
    row = {k: rel(z[k].astype(np.float64)[ev], ev) for k in z.files}
    row["krr_member"] = rel(np.load(member_file(d, "krr")).astype(np.float64)[ev], ev)
    row["mlp_member"] = rel(np.load(member_file(d, "mlp")).astype(np.float64)[ev], ev)
    B[s] = row; KF[s] = z["kf_ard"].astype(np.float32)
keys = sorted({k for r in B.values() for k in r})
out["B_learned_kernels"] = {k: dict(mean=100 * float(np.mean([B[s][k] for s in B if k in B[s]])), sd=100 * float(np.std([B[s][k] for s in B if k in B[s]], ddof=1)),
                                    n=sum(k in B[s] for s in B)) for k in keys}
print("B:", {k: round(v["mean"], 4) for k, v in out["B_learned_kernels"].items()}, f"[{(time.time()-t0)/60:.1f} min]", flush=True)

# ---- C: stacks (the machinery of ens_rmt_dgx.py, batched matmul, chunked evaluation)
def _aug(Pm):
    m_, n_, D_ = Pm.shape
    return np.ascontiguousarray(np.concatenate([Pm, np.ones((1, n_, D_), Pm.dtype)], 0).astype(np.float64).transpose(2, 1, 0))


def fit_pixel(Pm, Y, lam):
    m_, n_, D_ = Pm.shape; Xt = _aug(Pm)
    G = np.matmul(Xt.transpose(0, 2, 1), Xt) / n_; b = np.matmul(Xt.transpose(0, 2, 1), Y.T[:, :, None])[..., 0] / n_
    lam = np.broadcast_to(np.asarray(lam, np.float64), (D_,))
    return np.linalg.solve(G + lam[:, None, None] * np.eye(m_ + 1)[None], b[..., None])[..., 0]


def apply_pixel(Pm, W, chunk=2000):
    out_ = np.empty((Pm.shape[1], Pm.shape[2]))
    for i in range(0, Pm.shape[1], chunk):
        out_[i:i + chunk] = np.matmul(_aug(Pm[:, i:i + chunk]), W[:, :, None])[..., 0].T
    return out_


def loo_lam(Pm, Y, lams=np.logspace(-6, 4, 41)):
    Xt = _aug(Pm); n_ = Xt.shape[1]; best = (np.inf, None)
    for lam in lams:
        W = fit_pixel(Pm, Y, lam); R = Y - apply_pixel(Pm, W)
        G = np.matmul(Xt.transpose(0, 2, 1), Xt) / n_ + lam * np.eye(Xt.shape[2])[None]
        H = (np.matmul(Xt, np.linalg.inv(G)) * Xt).sum(-1).T / n_
        l = float(((R / (1 - H)) ** 2).sum() / n_)
        if l < best[0]:
            best = (l, lam)
    return best[1]


means = []
for a in ARCH:
    means.append(np.mean([np.load(member_file(f"{ROOT}/seeds/sm_s{s}/runs", a)).astype(np.float32) for s in range(10)], axis=0))
means = np.stack(means)                                   # (6, 20000, 1681)
kf_mean = np.mean([KF[s] for s in range(10)], axis=0)[None]  # (1, 20000, 1681)
seven = np.concatenate([means, kf_mean], 0)
Ycal = Yte[cal]
C = {}
for name, P in (("six_means", means), ("seven_means_plus_kfard", seven)):
    Pc, Pe = P[:, cal], P[:, ev]
    C[name + "_ridge1e-3"] = rel(apply_pixel(Pe, fit_pixel(Pc, Ycal, 1e-3)), ev)
    lam = loo_lam(Pc, Ycal); C[name + "_ridge_loo"] = rel(apply_pixel(Pe, fit_pixel(Pc, Ycal, lam)), ev); C[name + "_loo_lam"] = float(lam)
    C[name + "_equal"] = rel(Pe.mean(0), ev)
    print(name, {k: (round(100 * v, 4) if "lam" not in k else v) for k, v in C.items() if k.startswith(name)}, f"[{(time.time()-t0)/60:.1f} min]", flush=True)
C["kf_ard_mean_alone"] = rel(kf_mean[0][ev], ev); C["krr_mean_alone"] = rel(means[5][ev], ev)
out["C_stacks"] = {k: (100 * v if "lam" not in k else v) for k, v in C.items()}
out["minutes"] = (time.time() - t0) / 60
json.dump(out, open(ROOT + "/results/p1_members_eval.json", "w"), indent=1)
print("wrote results/p1_members_eval.json", flush=True)
