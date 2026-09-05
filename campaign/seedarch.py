"""Seeds against architectures: six architectures x ten seeds on the fixed test block.

Every predictor of the complete-schedule campaign has a test prediction array, and the
test block never moved, so the sixty of them can be compared on identical samples:
residual correlations within an architecture (seed to seed) and between architectures,
seed-ensemble curves, convex optima fitted on 1000 cases of the public test
block and scored on the other 19000, hindsight oracles over all sixty, per-pixel affine
stacks, and the block-correlation prediction of the floor. Writes results/seedarch.json.
Environment: NMKC_ROOT (default ~/nmkc2), NMKC_THREADS.
"""
import glob, itertools, json, os, sys, time
import numpy as np

ROOT = os.environ.get("NMKC_ROOT", os.path.expanduser("~/nmkc2"))
sys.path.insert(0, ROOT + "/code")
os.environ.setdefault("NMKC_DATA", ROOT + "/data/structmech")
os.environ["NMKC_SPLIT_SEED"] = "0"
from common import load_arrays, canonical_split  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

t_start = time.time()
loads, stress = load_arrays()
_, _, te = canonical_split(n_val=1000, seed=0)
Y = stress[te].reshape(len(te), -1).astype(np.float32)
ny = np.linalg.norm(Y, axis=1)
ARCH = ["mlp", "mlpMSE", "mlpR", "fno", "unet", "krr"]
SEEDS = list(range(10))
M, K = len(ARCH), len(SEEDS)
N = M * K


def path(a, s):
    d = "%s/seeds/sm_s%d/runs" % (ROOT, s)
    if a == "krr":
        return "%s/krr_full_matern52_n19000_pred_test.npy" % d
    g = [p for p in glob.glob("%s/*_predte.npy" % d) if os.path.basename(p).split("_")[0] == a]
    assert len(g) == 1, (a, s, g)
    return g[0]


# Fixed fitting/evaluation subsets for retrospective pooling. These cases were
# not used by the original model-fitting stages, but the public test block had
# already been inspected in the broader study.
rng = np.random.default_rng(20260902)
perm = rng.permutation(len(te))
cal, ev = np.sort(perm[:1000]), np.sort(perm[1000:])

R = np.empty((N, len(te), Y.shape[1]), dtype=np.float32)
names = []
for i, (a, s) in enumerate(itertools.product(ARCH, SEEDS)):
    P = np.load(path(a, s)).astype(np.float32)
    R[i] = (P - Y) / ny[:, None]
    names.append("%s_s%d" % (a, s))
    del P
print("loaded %s in %.0fs" % (R.shape, time.time() - t_start), flush=True)
arch_of = np.array([ARCH.index(n.split("_s")[0]) for n in names])


def S_of(idx):
    F = R[:, idx, :].reshape(N, -1)
    S = (F @ F.T) / len(idx)
    mu = F.mean(1, keepdims=True)
    Fc = F - mu
    C = (Fc @ Fc.T)
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    del F, Fc
    return S.astype(np.float64), C.astype(np.float64)


S_cal, C_cal = S_of(cal)
S_ev, C_ev = S_of(ev)
print("second moments done %.0fs" % (time.time() - t_start), flush=True)


def e1(weights, idx):
    """mean relative error of the convex/affine combination with the given weights on idx"""
    w = np.asarray(weights, dtype=np.float32)
    nz = np.nonzero(w)[0]
    acc = np.zeros((len(idx), Y.shape[1]), dtype=np.float32)
    for i in nz:
        acc += w[i] * R[i, idx, :]
    return float(np.linalg.norm(acc, axis=1).mean())


def e1_per_sample(weights, idx):
    w = np.asarray(weights, dtype=np.float32)
    nz = np.nonzero(w)[0]
    acc = np.zeros((len(idx), Y.shape[1]), dtype=np.float32)
    for i in nz:
        acc += w[i] * R[i, idx, :]
    return np.linalg.norm(acc, axis=1)


def e2(weights, S):
    w = np.asarray(weights, dtype=np.float64)
    return float(np.sqrt(w @ S @ w))


def simplex_min(S, active=None):
    idx = np.arange(S.shape[0]) if active is None else np.asarray(active)
    Ss = S[np.ix_(idx, idx)]
    m = len(idx)
    res = minimize(lambda z: z @ Ss @ z, np.ones(m) / m, jac=lambda z: 2 * Ss @ z,
                   bounds=[(0.0, 1.0)] * m, constraints={"type": "eq", "fun": lambda z: z.sum() - 1.0},
                   method="SLSQP", options=dict(maxiter=2000, ftol=1e-15))
    z = np.maximum(res.x, 0.0)
    z /= z.sum()
    assert z @ Ss @ z <= np.min(np.diag(Ss)) + 1e-12
    w = np.zeros(S.shape[0])
    w[idx] = z
    return w


def uniform(active):
    w = np.zeros(N)
    w[np.asarray(active)] = 1.0 / len(active)
    return w


out = dict(names=names, arch=ARCH, seeds=SEEDS, n_cal=len(cal), n_ev=len(ev),
           S_ev=S_ev.tolist(), C_ev=C_ev.tolist(), S_cal=S_cal.tolist())

# single predictors on ev
single_e1 = np.array([e1(np.eye(N)[i], ev) for i in range(N)])
single_e2 = np.sqrt(np.diag(S_ev))
out["single"] = {names[i]: dict(e1=float(single_e1[i]), e2=float(single_e2[i])) for i in range(N)}
out["arch_mean_e1"] = {a: float(single_e1[arch_of == j].mean()) for j, a in enumerate(ARCH)}
print("singles done %.0fs" % (time.time() - t_start), flush=True)

# correlations: within an architecture (seed pairs) and between architectures
within = {}
for j, a in enumerate(ARCH):
    ii = np.nonzero(arch_of == j)[0]
    vals = [C_ev[p, q] for p, q in itertools.combinations(ii, 2)]
    within[a] = dict(mean=float(np.mean(vals)), min=float(np.min(vals)), max=float(np.max(vals)))
between = {}
allb = []
for j1, j2 in itertools.combinations(range(M), 2):
    i1 = np.nonzero(arch_of == j1)[0]
    i2 = np.nonzero(arch_of == j2)[0]
    vals = C_ev[np.ix_(i1, i2)].ravel()
    between["%s-%s" % (ARCH[j1], ARCH[j2])] = dict(mean=float(vals.mean()), min=float(vals.min()), max=float(vals.max()))
    allb.extend(vals.tolist())
out["within"] = within
out["between"] = between
rho_w = float(np.mean([within[a]["mean"] for a in ARCH]))
rho_b = float(np.mean(allb))
out["rho_w_mean"], out["rho_b_mean"] = rho_w, rho_b

# seed-ensemble curves per architecture: equal weights over k seeds, E2 exact over all subsets,
# E1 over all subsets when there are at most 45, else over 40 random subsets
curves = {}
rs = np.random.default_rng(7)
for j, a in enumerate(ARCH):
    ii = np.nonzero(arch_of == j)[0]
    row = {}
    for k in (1, 2, 3, 5, 10):
        subsets = list(itertools.combinations(ii, k))
        e2s = [e2(uniform(sub), S_ev) for sub in subsets]
        if len(subsets) > 45:
            pick = [subsets[t] for t in rs.choice(len(subsets), 40, replace=False)]
        else:
            pick = subsets
        e1s = [e1(uniform(sub), ev) for sub in pick]
        row[str(k)] = dict(e1_mean=float(np.mean(e1s)), e1_sd=float(np.std(e1s, ddof=1)) if len(e1s) > 1 else 0.0,
                           e2_mean=float(np.mean(e2s)), n_subsets=len(subsets), n_e1=len(pick))
    # K -> infinity floor of this architecture from its own block
    ebar2 = float(np.mean(np.diag(S_ev)[ii]))
    off = [S_ev[p, q] for p, q in itertools.combinations(ii, 2)]
    row["floor_e2_inf"] = float(np.sqrt(max(np.mean(off), 0.0)))
    row["ebar2"] = ebar2
    curves[a] = row
    print("curve", a, "done %.0fs" % (time.time() - t_start), flush=True)
out["seed_curves"] = curves

# cross-architecture ensembles at one seed (six members, each seed) and all sixty
per_seed = []
for s in SEEDS:
    ii = [i for i in range(N) if names[i].endswith("_s%d" % s)]
    per_seed.append(dict(seed=s, e1=e1(uniform(ii), ev), e2=e2(uniform(ii), S_ev)))
out["six_arch_one_seed_equal"] = per_seed
w_all = uniform(range(N))
out["sixty_equal"] = dict(e1=e1(w_all, ev), e2=e2(w_all, S_ev), e1_full_test=e1(w_all, np.arange(len(te))))
# five architectures without the UNet, one seed each and all fifty
ii50 = [i for i in range(N) if not names[i].startswith("unet")]
out["fifty_equal_no_unet"] = dict(e1=e1(uniform(ii50), ev), e2=e2(uniform(ii50), S_ev))

# Convex optima: fitted on the 1000-case subset, scored on the 19000-case subset.
w_cal = simplex_min(S_cal)
out["sixty_convex_cal_to_ev"] = dict(e1=e1(w_cal, ev), e2_pred_cal=e2(w_cal, S_cal), e2_ev=e2(w_cal, S_ev),
                                     weights={names[i]: float(w_cal[i]) for i in range(N) if w_cal[i] > 1e-4},
                                     n_nonzero=int((w_cal > 1e-4).sum()))
# Hindsight optimizer on evaluation itself; objective optimality must be checked.
w_or = simplex_min(S_ev)
out["sixty_convex_oracle_ev"] = dict(e1=e1(w_or, ev), e2=e2(w_or, S_ev),
                                     weights={names[i]: float(w_or[i]) for i in range(N) if w_or[i] > 1e-4},
                                     n_nonzero=int((w_or > 1e-4).sum()))
# same-architecture ten-seed convex optima (cal -> ev) and their oracles
same = {}
for j, a in enumerate(ARCH):
    ii = np.nonzero(arch_of == j)[0]
    wc = simplex_min(S_cal, ii)
    wo = simplex_min(S_ev, ii)
    same[a] = dict(cal_to_ev_e1=e1(wc, ev), cal_to_ev_e2=e2(wc, S_ev), oracle_e2=e2(wo, S_ev), oracle_e1=e1(wo, ev))
out["same_arch_convex"] = same
# six-architecture one-seed convex optimum (cal -> ev), seed 0..9 mean
six_conv = []
for s in SEEDS:
    ii = [i for i in range(N) if names[i].endswith("_s%d" % s)]
    wc = simplex_min(S_cal, ii)
    six_conv.append(dict(seed=s, e1=e1(wc, ev), e2=e2(wc, S_ev)))
out["six_arch_one_seed_convex_cal_to_ev"] = six_conv
print("convex optima done %.0fs" % (time.time() - t_start), flush=True)

# block-correlation prediction of the floor: e2^2 [rho_b + (rho_w - rho_b)/M + (1 - rho_w)/(M K)]
# using the mean squared error and the mean within/between second moments read off S_ev
ebar2 = float(np.mean(np.diag(S_ev)))
win_S = np.mean([S_ev[p, q] for j in range(M) for p, q in itertools.combinations(np.nonzero(arch_of == j)[0], 2)])
bet_S = np.mean([S_ev[p, q] for p in range(N) for q in range(N) if arch_of[p] != arch_of[q]])
rw, rb = float(win_S / ebar2), float(bet_S / ebar2)
pred = ebar2 * (rb + (rw - rb) / M + (1 - rw) / (M * K))
out["block_model"] = dict(ebar2=ebar2, rho_w_S=rw, rho_b_S=rb,
                          predicted_e2_sixty_uniform=float(np.sqrt(pred)),
                          realized_e2_sixty_uniform=out["sixty_equal"]["e2"],
                          floor_one_arch_inf_seeds=float(np.sqrt(ebar2 * rw)),
                          floor_inf_arch_inf_seeds=float(np.sqrt(ebar2 * rb)),
                          floor_six_arch_inf_seeds=float(np.sqrt(ebar2 * (rb + (rw - rb) / M))))

# Per-pixel affine stacks fitted on the 1000-case subset, scored on evaluation.
def perpixel_affine(member_rows, ridge=1e-3):
    """member_rows: list of (n, D) prediction arrays in original units (not residuals)."""
    Pc = np.stack([m[cal] for m in member_rows]).astype(np.float64)        # (m, n_cal, D)
    Yc = Y[cal].astype(np.float64)
    m_, n_, D_ = Pc.shape
    X = np.concatenate([Pc, np.ones((1, n_, D_))], 0)
    G = np.einsum("mnd,knd->dmk", X, X) / n_ + ridge * np.eye(m_ + 1)[None]
    b = np.einsum("mnd,nd->dm", X, Yc) / n_
    W = np.linalg.solve(G, b[..., None])[..., 0]                           # (D, m+1)
    acc = np.zeros((len(ev), D_), dtype=np.float64)
    for k in range(m_):
        acc += W[:, k][None, :] * member_rows[k][ev]
    acc += W[:, m_][None, :]
    return float((np.linalg.norm(acc - Y[ev], axis=1) / ny[ev]).mean())


# reconstruct predictions in original units from the residuals when needed
def pred_of(i):
    return R[i] * ny[:, None] + Y


# (i) the six architectures' ten-seed means as members
arch_means = []
for j in range(M):
    ii = np.nonzero(arch_of == j)[0]
    arch_means.append(np.mean([pred_of(i) for i in ii], axis=0))
out["perpixel_six_seed_means"] = perpixel_affine(arch_means)
# (ii) all sixty as members (61 parameters per pixel from 1000 calibration samples)
out["perpixel_sixty"] = perpixel_affine([pred_of(i) for i in range(N)])
# (iii) six members at seed 0 (the deployed configuration's shape, fitted on the calibration half)
out["perpixel_six_seed0"] = perpixel_affine([pred_of(i) for i in range(N) if names[i].endswith("_s0")])
print("per-pixel done %.0fs" % (time.time() - t_start), flush=True)

# hard tail: the hardest one percent of evaluation samples for the best single architecture's seed mean
best_arch = min(ARCH, key=lambda a: out["arch_mean_e1"][a])
jb = ARCH.index(best_arch)
ii_b = np.nonzero(arch_of == jb)[0]
ps_best_single = e1_per_sample(np.eye(N)[ii_b[0]], ev)
ps_best_seed10 = e1_per_sample(uniform(ii_b), ev)
ps_sixty = e1_per_sample(w_all, ev)
order = np.argsort(-ps_best_single)
top = order[: max(1, len(ev) // 100)]
out["hard_tail_1pct"] = dict(best_arch=best_arch, best_single=float(ps_best_single[top].mean()),
                             best_arch_ten_seeds=float(ps_best_seed10[top].mean()), sixty_equal=float(ps_sixty[top].mean()),
                             median_best_single=float(np.median(ps_best_single)))
out["minutes"] = (time.time() - t_start) / 60
os.makedirs(ROOT + "/results", exist_ok=True)
with open(ROOT + "/results/seedarch.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print("wrote results/seedarch.json in %.1f min" % out["minutes"], flush=True)
print("sixty equal e1 %.4f e2 %.4f | convex cal->ev e1 %.4f | oracle e2 %.4f | rho_w %.3f rho_b %.3f | block pred e2 %.4f"
      % (out["sixty_equal"]["e1"], out["sixty_equal"]["e2"], out["sixty_convex_cal_to_ev"]["e1"], out["sixty_convex_oracle_ev"]["e2"],
         rho_w, rho_b, out["block_model"]["predicted_e2_sixty_uniform"]), flush=True)
