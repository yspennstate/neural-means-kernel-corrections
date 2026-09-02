"""Numerical checks of the results added in the second revision: the block floor
(Theorem blockfloor), the accuracy-for-correlation exchange rate (Proposition exchange),
the sharpened weight bound with its equality case (Lemma kappa), the sharpness of the
certified bound and the minimax identity (Theorems or and minimax), and the signed-stack
corollary. Synthetic objects with exactly known structure, plus the released sixty-predictor
second-moment matrix where one exists. Prints PASS/FAIL per check and exits nonzero on any FAIL.
"""
import itertools, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1)
fails = 0


def report(name, ok, detail=""):
    global fails
    fails += 0 if ok else 1
    print("  %-58s %s %s" % (name, "PASS" if ok else "FAIL", detail))


def block_bound(S, arch_of, M, K):
    """Theorem blockfloor evaluated with the pool's minima, and the uniform-weight identity."""
    N = len(arch_of)
    e2 = min(S[i, i] for i in range(N))
    win = [S[p, q] for p in range(N) for q in range(p + 1, N) if arch_of[p] == arch_of[q]]
    bet = [S[p, q] for p in range(N) for q in range(N) if arch_of[p] != arch_of[q]]
    rw, rb = min(win) / e2, min(bet) / e2
    return e2 * (rb + (rw - rb) / M + (1 - rw) / (M * K))


print("Theorem blockfloor")
# exact block-equicorrelated S: the display is an identity for every convex w
M, K = 4, 5
N = M * K
arch_of = np.repeat(np.arange(M), K)
e2, rw, rb = 0.0025, 0.97, 0.90
B = (arch_of[:, None] == arch_of[None, :]).astype(float)
S = e2 * ((1 - rw) * np.eye(N) + (rw - rb) * B + rb * np.ones((N, N)))
worst = 0.0
for _ in range(2000):
    w = rng.dirichlet(np.ones(N) * rng.uniform(0.2, 3.0))
    W = np.array([w[arch_of == a].sum() for a in range(M)])
    rhs = e2 * (rb + (rw - rb) * (W ** 2).sum() + (1 - rw) * (w ** 2).sum())
    worst = max(worst, abs(w @ S @ w - rhs))
report("exact block model: quadratic form equals the display", worst < 1e-15, "max |diff| %.1e" % worst)
lo = e2 * (rb + (rw - rb) / M + (1 - rw) / (M * K))
u = np.ones(N) / N
report("uniform weights attain the floor in the exact model", abs(u @ S @ u - lo) < 1e-15)
# perturbed S: entries at or above the block minima; every convex w must sit above the bound
S2 = S.copy()
S2 += e2 * np.abs(rng.normal(0, 0.01, (N, N)))
S2 = (S2 + S2.T) / 2
np.fill_diagonal(S2, np.diag(S) + e2 * np.abs(rng.normal(0, 0.02, N)))
bound = block_bound(S2, arch_of, M, K)
mn = min(w @ S2 @ w for w in [rng.dirichlet(np.ones(N) * rng.uniform(0.2, 3.0)) for _ in range(20000)])
report("perturbed pool: random convex mixtures never pass the bound", mn >= bound - 1e-15, "min %.6f bound %.6f" % (mn, bound))
# the released sixty-predictor matrix, if present
sa = os.path.join(HERE, "collected", "dgx", "seedarch.json")
if os.path.exists(sa):
    d = json.load(open(sa, encoding="utf-8"))
    Sev = np.array(d["S_ev"]); A = d["arch"]; ao = np.array([A.index(n.split("_s")[0]) for n in d["names"]])
    bnd = block_bound(Sev, ao, len(A), len(d["seeds"]))
    mn = min(w @ Sev @ w for w in [rng.dirichlet(np.ones(60) * rng.uniform(0.2, 3.0)) for _ in range(20000)])
    orc = d["sixty_convex_oracle_ev"]["e2"] ** 2
    report("sixty predictors: bound below every sampled convex mixture", mn >= bnd - 1e-15, "sqrt: min %.4f bound %.4f" % (100 * mn ** 0.5, 100 * bnd ** 0.5))
    report("sixty predictors: bound below the hindsight optimum", orc >= bnd - 1e-12, "oracle %.4f" % (100 * orc ** 0.5))
    ebar2 = np.mean(np.diag(Sev))
    win = np.mean([Sev[p, q] for p in range(60) for q in range(p + 1, 60) if ao[p] == ao[q]])
    bet = np.mean([Sev[p, q] for p in range(60) for q in range(60) if ao[p] != ao[q]])
    ident = ebar2 * (bet / ebar2 + (win / ebar2 - bet / ebar2) / 6 + (1 - win / ebar2) / 60)
    uu = np.ones(60) / 60
    report("sixty predictors: mean-entry form equals the uniform mixture", abs(ident - uu @ Sev @ uu) < 1e-12)

print("Proposition exchange")
def V(e1, e2, rho):
    return e1 * e1 * e2 * e2 * (1 - rho * rho) / (e1 * e1 + e2 * e2 - 2 * rho * e1 * e2)
worst_t = worst_r = worst_ind = 0.0
positive = True
for _ in range(2000):
    t = rng.uniform(0.5, 2.0)
    # interior points kept a little away from the admission boundary rho = min(t, 1/t)
    rho = rng.uniform(-0.9, min(t, 1 / t) - 0.02)
    e1 = 1.0
    h = 1e-6
    dVt = (V(e1, (t + h) * e1, rho) - V(e1, (t - h) * e1, rho)) / (2 * h)
    dVr = (V(e1, t * e1, rho + h) - V(e1, t * e1, rho - h)) / (2 * h)
    D = 1 + t * t - 2 * rho * t
    ft = 2 * t * (1 - rho * rho) * (1 - rho * t) / D ** 2
    fr = 2 * t * t * (t - rho) * (1 - rho * t) / D ** 2
    worst_t = max(worst_t, abs(dVt - ft) / max(abs(ft), 1e-9))
    worst_r = max(worst_r, abs(dVr - fr) / max(abs(fr), 1e-9))
    positive = positive and ft > 0 and fr > 0
    # indifference direction: along (dt, drho) with drho = -(1-rho^2) dt/(t(t-rho)) the first-order change
    # vanishes, so the remainder against the first-order terms must fall linearly with the step
    ratios = []
    for dt in (1e-4, 1e-5, 1e-6):
        dr = -(1 - rho * rho) * dt / (t * (t - rho))
        dV = V(e1, (t + dt) * e1, rho + dr) - V(e1, t * e1, rho)
        ratios.append(abs(dV) / (abs(ft * dt) + abs(fr * dr)))
    worst_ind = max(worst_ind, ratios[2] * 1e6 / (ratios[0] * 1e4 + 1e-30) if ratios[0] > 1e-12 else 0.0)
report("partial derivatives match finite differences", worst_t < 1e-5 and worst_r < 1e-5, "rel err %.1e %.1e" % (worst_t, worst_r))
report("both derivatives positive on the interior region", positive)
report("the stated direction is indifferent to first order (remainder falls with the step)", worst_ind < 2.0, "worst normalized remainder ratio %.2f (1 = exactly second order)" % worst_ind)

print("Lemma kappa (sharpened)")
def matern52(D2, s):
    r = np.sqrt(np.maximum(D2, 0)) / s
    a = np.sqrt(5) * r
    return (1 + a + (5 / 3) * r * r) * np.exp(-a)
worst_ratio = 0.0
for _ in range(300):
    n = rng.integers(2, 60)
    dim = rng.integers(1, 6)
    X = rng.normal(size=(n, dim)) * rng.uniform(0.2, 3.0)
    s = rng.uniform(0.2, 3.0)
    lam = 10 ** rng.uniform(-8, 0)
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    Kmat = matern52(D2, s)
    for _ in range(5):
        u = rng.normal(size=dim) * rng.uniform(0.2, 3.0)
        ku = matern52(((X - u) ** 2).sum(-1), s)
        c = np.linalg.solve(Kmat + n * lam * np.eye(n), ku)
        worst_ratio = max(worst_ratio, np.abs(c).sum() / (0.5 * np.sqrt(1.0 / lam)))
report("random Matern designs: ||c||_1 <= (1/2) sqrt(k(u,u)/lambda)", worst_ratio <= 1 + 1e-12, "largest ratio %.4f" % worst_ratio)
# the equality case: cos(x-y) kernel, two points at 1 - cos theta = 2 lambda, u = (theta + pi)/2
eq_worst = 0.0
for lam in (0.01, 0.1, 0.3):
    theta = np.arccos(1 - 2 * lam)
    x = np.array([0.0, theta]); n = 2
    Kc = np.cos(x[:, None] - x[None, :]); u = (theta + np.pi) / 2
    ku = np.cos(u - x)
    c = np.linalg.solve(Kc + n * lam * np.eye(n), ku)
    eq_worst = max(eq_worst, abs(np.abs(c).sum() - 0.5 / np.sqrt(lam)))
report("cosine kernel example attains the bound", eq_worst < 1e-12, "max |gap| %.1e" % eq_worst)

print("Theorems or and minimax")
worst_id = worst_ord = worst_sharp = 0.0
for _ in range(200):
    # small well-spread designs keep K comfortably positive definite, which the identity assumes
    n = rng.integers(3, 12); dim = rng.integers(2, 4)
    X = rng.normal(size=(n, dim)) * 2.0; s = rng.uniform(0.5, 1.5); lam = 10 ** rng.uniform(-4, -1)
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    Kmat = matern52(D2, s)
    mu, Vec = np.linalg.eigh(Kmat)
    if mu.min() < 1e-4:
        continue
    t = n * lam; A_ = Kmat + t * np.eye(n)
    u = rng.normal(size=dim) * 2.0; ku = matern52(((X - u) ** 2).sum(-1), s); kuu = 1.0
    beta = Vec.T @ ku
    P0 = kuu - float((beta ** 2 / mu).sum())
    w = np.linalg.solve(A_, ku)
    Pl = kuu - ku @ w
    Pt = Pl - t * (w @ w)
    ident = float((beta ** 2 * t * t / (mu * (mu + t) ** 2)).sum())
    # the two sides are differences of O(1) quantities; compare on the scale of P_lambda^2 itself
    worst_id = max(worst_id, abs((Pt - P0) - ident) / max(Pl, 1e-14))
    worst_ord = max(worst_ord, max(P0 - Pt, Pt - Pl) / max(Pl, 1e-14))
    # sharpness: the residual g_u = k(u,.) - sum_i w_i k(u_i,.) attains |r(u) - rhat(u)| = ||g_u||^2 = Pt exactly
    g_at_X = ku - Kmat @ w                       # g_u evaluated at the design
    rhat = ku @ np.linalg.solve(A_, g_at_X)      # ridge correction built from g_u's own training values
    g_at_u = kuu - ku @ w
    err = g_at_u - rhat                           # r(u) - rhat(u) for r = g_u
    norm2 = kuu - 2 * w @ ku + w @ Kmat @ w       # ||g_u||^2_H
    worst_sharp = max(worst_sharp, abs(err - norm2) / max(abs(norm2), 1e-14))
report("identity Ptilde^2 - P0^2 = (n lambda)^2 k' A^-2 K^-1 k", worst_id < 1e-6, "rel err %.1e" % worst_id)
report("ordering P0 <= Ptilde <= P", worst_ord < 1e-9)
report("sharpness: r = g_u attains the certified bound with equality", worst_sharp < 1e-8, "rel err %.1e" % worst_sharp)

print("Corollary signed")
worst = 0.0
for _ in range(200):
    m = rng.integers(2, 8)
    G = rng.normal(size=(m, m)); Sm = G @ G.T + 0.1 * np.eye(m)
    one = np.ones(m); w = np.linalg.solve(Sm, one); w /= one @ w
    val = w @ Sm @ w
    mn = min(v @ Sm @ v for v in [rng.dirichlet(np.ones(m)) for _ in range(3000)])
    worst = max(worst, val - 1 / (one @ np.linalg.solve(Sm, one)))
    if mn < val - 1e-12:
        worst = 1.0
report("signed optimum = 1/(1'S^-1 1) and never above the convex minimum", worst < 1e-12)

print("%d check(s) failed" % fails)
sys.exit(1 if fails else 0)
