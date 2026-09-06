"""Top-k eigenvalues of the FULL 19000-row Matern-5/2 Gram matrix by ARPACK
Lanczos (scipy.sparse.linalg.eigsh on the dense float64 matrix), an
implementation-independent route for the leading modes next to the LAPACK
evd of full_design_spectrum.py. Reads the standardized design saved by that
script (design_full.npz), rebuilds the Gram matrix in row blocks, and writes
spectrum/full_design_19000/topk_lanczos.json with the top eigenvalues, the
trace fractions, and interval bounds on d_eff(lambda) that use only the top-k
eigenvalues and the exact trace (a check on the Cholesky-trace figure).

    python topk_lanczos_cpu.py --k 64 --threads 4
"""
import argparse, hashlib, json, os, time
from datetime import datetime, timezone
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(name, "4")
import numpy as np
import scipy
from scipy.sparse.linalg import eigsh, LinearOperator
from threadpoolctl import threadpool_limits

OUT = Path("C:/Users/owner/GOAL20H_20260906/nmkc_gpu_experiments/spectrum/full_design_19000")
REC6000 = Path("C:/Users/owner/GOAL20H_20260906/nmkc_gpu_experiments/records/campaign/collected/spectrum_20260906")
NUGGET = 1e-5


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def m52_from_r2(r2):
    radial = np.sqrt(5 * r2)
    return (1 + radial + (5 / 3) * r2) * np.exp(-radial)


def gram_blocks(z, ls, block=2000):
    n = len(z)
    v = np.sum(z * z, axis=1)
    K = np.empty((n, n), dtype=np.float64)
    for s in range(0, n, block):
        e = min(s + block, n)
        d2 = np.maximum(v[s:e, None] + v[None, :] - 2 * z[s:e] @ z.T, 0)
        K[s:e] = m52_from_r2(d2 / ls ** 2)
    return K


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--tol", type=float, default=1e-10)
    a = p.parse_args()
    t0 = time.time()
    rec = dict(kind="topk_lanczos_cpu", started_at=datetime.now(timezone.utc).isoformat(),
               source_sha256=sha(Path(__file__)), design_full_sha256=sha(OUT / "design_full.npz"),
               numpy=np.__version__, scipy=scipy.__version__, k=a.k, tol=a.tol, nugget=NUGGET)
    d = np.load(OUT / "design_full.npz", allow_pickle=False)
    z = d["standardized_loads"].astype(np.float64); ls = float(d["length_scale"])
    n = len(z)
    rec.update(n_rows=n, length_scale=ls)
    with threadpool_limits(a.threads):
        K = gram_blocks(z, ls)
        rec["gram_seconds"] = time.time() - t0
        tr = float(np.trace(K))
        rec["gram_trace"] = tr
        matvecs = [0]

        def mv(x):
            matvecs[0] += 1
            return K @ x

        op = LinearOperator((n, n), matvec=mv, dtype=np.float64)
        rng = np.random.default_rng(0)
        t1 = time.time()
        ev = eigsh(op, k=a.k, which="LA", tol=a.tol, v0=rng.standard_normal(n), maxiter=20000, return_eigenvectors=False)
        ev = np.sort(ev)[::-1]
        rec["lanczos_seconds"] = time.time() - t1
        rec["matvecs"] = matvecs[0]
        # residual check on the top eight with explicit eigenvectors (a second eigsh call, small k)
        t2 = time.time()
        w, V = eigsh(op, k=8, which="LA", tol=a.tol, v0=rng.standard_normal(n), maxiter=20000)
        order = np.argsort(w)[::-1]; w = w[order]; V = V[:, order]
        resid = [float(np.linalg.norm(K @ V[:, i] - w[i] * V[:, i]) / w[i]) for i in range(8)]
        rec["top8_residual_check"] = dict(eigenvalues=w.tolist(), relative_residuals=resid, seconds=time.time() - t2,
                                          max_abs_diff_vs_first_call=float(np.max(np.abs(w - ev[:8]))))
        del K
    cum = np.cumsum(ev)
    top8 = float(cum[7] / tr)
    # interval on d_eff from the top-k and the trace: the remainder eigenvalues sum to tr - cum[k-1],
    # each lies in [0, ev[k-1]]; x/(x + n lam) is concave increasing, so the remainder's contribution
    # is at most (tr - cum[k-1])/(ev[k-1] + n lam) x ev[k-1]/ev[k-1] ... bounded by putting the mass
    # at the largest allowed eigenvalue (fewest, largest modes: lower bound of contribution is at that
    # extreme? no: for concave f with f(0)=0, spreading mass over MORE modes gives MORE f; the
    # maximum is at the smallest possible eigenvalues (limit: mass/(n lam)), the minimum at the
    # largest allowed ones (mass/ev[k-1] modes of size ev[k-1]).
    mass = tr - cum[a.k - 1]
    lam = NUGGET
    top_part = float(np.sum(ev / (ev + n * lam)))
    rem_min = float(mass / ev[a.k - 1] * (ev[a.k - 1] / (ev[a.k - 1] + n * lam)))
    rem_max = float(min(mass / (n * lam), n - a.k))
    rec.update(top_eigenvalues=ev.tolist(), top8_eigenvalues_over_n=(ev[:8] / n).tolist(),
               top8_trace_fraction=top8, top16_trace_fraction=float(cum[15] / tr),
               topk_trace_fraction=float(cum[a.k - 1] / tr),
               deff_lower_bound_from_topk=top_part + rem_min, deff_upper_bound_from_topk=top_part + rem_max,
               deff_topk_part=top_part, remainder_mass=float(mass))
    rec6 = json.loads((REC6000 / "result.json").read_text(encoding="utf-8"))
    ev6 = np.sort(np.load(REC6000 / "spectrum.npz")["eigenvalues"])[::-1]
    rec["subsample_6000"] = dict(top8_eigenvalues_over_n=(ev6[:8] / 6000).tolist(),
                                 top8_trace_fraction=float(ev6[:8].sum() / ev6.sum()),
                                 effective_dimension=rec6["spectral_effective_dimension"],
                                 ratio_top8_over_n_full_over_sub=(ev[:8] / n / (ev6[:8] / 6000)).tolist())
    rec["elapsed_seconds"] = time.time() - t0
    rec["finished_at"] = datetime.now(timezone.utc).isoformat()
    (OUT / "topk_lanczos.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("gram_seconds", "lanczos_seconds", "matvecs", "top8_trace_fraction",
                                            "deff_lower_bound_from_topk", "deff_upper_bound_from_topk")}, indent=1))
    print("top8/n", rec["top8_eigenvalues_over_n"])
    print("top8 resid", rec["top8_residual_check"]["relative_residuals"])
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
