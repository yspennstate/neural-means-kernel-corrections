"""Numeric check of Proposition prop:affine (per-pixel affine stacking).

Synthetic model with known population: members = truth + correlated noise,
per-coordinate ridge fit on a validation sample, excess risk measured against
the population ridge/ball optimum estimated on a large holdout. Verifies, over
many trials, that the measured average excess never exceeds the stated bound
(the bound holds with probability 1-delta and is loose by design, so a single
violation at these settings means the statement is misassembled, not unlucky).
Also verifies the cor:percoordw identity numerically.

    python campaign/check_affine_bound.py
"""
import numpy as np

rng = np.random.default_rng(0)
q, M, m, N = 20, 6, 1000, 50000
delta, lam = 0.05, 1e-3
D = np.sqrt(M + 1)

viol = 0
worst = 0.0
for trial in range(100):
    tr = np.random.default_rng(1000 + trial)
    # population: truth in [-1,1], members = truth + biased correlated noise
    G = np.tanh(tr.normal(size=(N, q)))
    common = tr.normal(size=(N, q))
    F = np.stack([0.8 * G + 0.3 * common + 0.2 * tr.normal(size=(N, q)) + 0.05 * k
                  for k in range(M)])                       # (M,N,q)
    b = max(np.abs(G).max(), np.abs(F).max())
    X = np.concatenate([F, np.full((1, N, q), b)], 0)       # intercept scale b

    val = tr.choice(N, m, replace=False)
    excesses = []
    Wmax = 0.0
    for j in range(q):
        Xj = X[:, :, j].T                                   # (N, M+1)
        # ridge fit on validation
        A = Xj[val].T @ Xj[val] / m + lam * np.eye(M + 1)
        w_hat = np.linalg.solve(A, Xj[val].T @ G[val, j] / m)
        # population optimum (unregularized least squares on all N)
        Ap = Xj.T @ Xj / N + 1e-12 * np.eye(M + 1)
        w_star = np.linalg.solve(Ap, Xj.T @ G[:, j] / N)
        W_j = max(np.linalg.norm(w_hat), np.linalg.norm(w_star))
        Wmax = max(Wmax, W_j)
        R = lambda w: float(((Xj @ w - G[:, j]) ** 2).mean())
        excesses.append(R(w_hat) - R(w_star))
    excess_avg = float(np.mean(excesses))
    W = Wmax
    bound = (8 * W * b * b * D * (1 + W * D)
             + 2 * b * b * (1 + W * D) ** 2 * np.sqrt(0.5 * np.log(2 * q / delta))) \
        / np.sqrt(m) + lam * W * W
    worst = max(worst, excess_avg / bound)
    if excess_avg > bound:
        viol += 1
print(f"prop:affine check: trials=100 violations={viol} worst excess/bound={worst:.4f}")
assert viol == 0, "bound violated -- statement misassembled"

# cor:percoordw identity: weighted metric decomposes coordinate-wise
a = 1.0 / (np.linalg.norm(G, axis=1) ** 2 + 1e-30)
w_diag = rng.uniform(0.5, 2.0, size=q)
fv = F[0]
lhs = float((a * ((w_diag * (fv - G)) ** 2).sum(1)).mean())
rhs = float(sum(w_diag[j] ** 2 * (a * (fv[:, j] - G[:, j]) ** 2).mean() for j in range(q)))
print(f"cor:percoordw identity: lhs={lhs:.10f} rhs={rhs:.10f} diff={abs(lhs-rhs):.2e}")
assert abs(lhs - rhs) < 1e-9 * max(1.0, abs(lhs))
# Jensen direction: mean relative error <= sqrt(mean squared relative error)
rho = np.linalg.norm(fv - G, axis=1) / np.linalg.norm(G, axis=1)
print(f"jensen: E|rho|={rho.mean():.6f} sqrt(E rho^2)={np.sqrt((rho**2).mean()):.6f}")
assert rho.mean() <= np.sqrt((rho ** 2).mean()) + 1e-12
print("ALL CHECKS PASS")
