"""End-to-end cost of the structural-mechanics kernel stages at the campaign's shapes, measured after review.

Reports, on the campaign host: the stored coefficient count of the residual correction (n x q) and of the
kernel member, the peak resident memory of the block-filled 19000 x 19000 Gram build plus in-place Cholesky
(the released memory-safe path) and of the naive full-matrix build, the factorization and triangular-solve
times with q right-hand sides, and the batched query latency of the kernel stage (kernel row evaluation plus
the coefficient product) for batches of 1, 100 and 1000 queries. Shapes are the campaign's; values are
synthetic, so timings measure the arithmetic, not the data.

    python cost_check.py [--n 19000] [--d 41] [--q 1681] [--threads 8]
"""
import argparse, json, os, resource, sys, time
# the BLAS thread cap must be in the environment before numpy is imported (a later setdefault is ignored by
# an already-initialized OpenBLAS); the first release set it after the import, which is why the timings are re-measured
_t = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--threads" and i + 1 < len(sys.argv)), "8")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = _t
import numpy as np
from scipy.linalg import cho_factor, cho_solve

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=19000)
p.add_argument("--d", type=int, default=41)
p.add_argument("--q", type=int, default=1681)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--out", default="results/cost_check.json")
args = p.parse_args()
os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0


def sqdist(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def m52(D2, s):
    r2 = D2 / (s * s); r = np.sqrt(r2); a = np.sqrt(5.0) * r
    return (1.0 + a + (5.0 / 3.0) * r2) * np.exp(-a)


rng = np.random.default_rng(0)
n, d, q = args.n, args.d, args.q
X = rng.standard_normal((n, d)); R = rng.standard_normal((n, q)) * 1e-3
s = float(np.sqrt(np.median(sqdist(X[:2000], X[:2000])[np.triu_indices(2000, 1)])))
out = dict(n=n, d=d, q=q, threads=args.threads, coefficients_correction=n * q, coefficients_kernel_member=n * q,
           coefficient_bytes_float64=2 * n * q * 8, baseline_rss_gb=rss_gb())

# block-filled Gram (the released path), in-place factorization, solve with q right-hand sides
t0 = time.time()
K = np.empty((n, n))
for k in range(0, n, 1000):
    K[k:k + 1000] = m52(sqdist(X[k:k + 1000], X), 2.0 * s)
K.flat[::n + 1] += 1e-3 * n
out["gram_block_build_s"] = time.time() - t0; t1 = time.time()
c = cho_factor(K, lower=True, check_finite=False, overwrite_a=True)
out["cholesky_s"] = time.time() - t1; t2 = time.time()
alpha = cho_solve(c, R, check_finite=False)
out["solve_q_rhs_s"] = time.time() - t2
out["peak_rss_gb_block_path"] = rss_gb()
del K, c

# query latency of the kernel stage: kernel row evaluation against the design plus the coefficient product
for b in (1, 100, 1000):
    U = rng.standard_normal((b, d)); reps = 20 if b < 1000 else 5; t3 = time.time()
    for _ in range(reps):
        pred = m52(sqdist(U, X), 2.0 * s) @ alpha
    dt = (time.time() - t3) / reps
    out[f"query_batch{b}_s"] = dt; out[f"query_batch{b}_ms_per_query"] = 1000 * dt / b
del alpha

# the naive full-matrix build for comparison (several n x n temporaries)
t4 = time.time()
K2 = m52(sqdist(X, X), 2.0 * s); K2.flat[::n + 1] += 1e-3 * n
out["gram_naive_build_s"] = time.time() - t4
out["peak_rss_gb_after_naive_build"] = rss_gb()
del K2
json.dump(out, open(args.out, "w"), indent=1)
for k, v in out.items():
    print(f"{k:34s} {v}")
