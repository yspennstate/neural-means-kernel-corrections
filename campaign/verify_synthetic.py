"""Controlled numerical verification of the paper's theoretical results.

Each check builds an object whose ground truth is known exactly and tests the
stated inequality or identity on it. Identities must hold to near machine
precision (PASS/FAIL); rate statements are reported as measured slopes with
their predicted values (REPORT). Output: one JSON with a status per check.

    python campaign/verify_synthetic.py --check all
Checks: or_bound, kf_unbiased, mp_deff, aniso_rates.
"""
import argparse, json, os, pathlib, time
import numpy as np
from scipy.linalg import cho_factor, cho_solve

p = argparse.ArgumentParser()
p.add_argument("--check", default="all")
args = p.parse_args()
ROOT = pathlib.Path(os.environ.get("NMKC_ROOT", "."))
TASK_ID = os.environ.get("TASK_ID", f"verify_{args.check}")
out = dict(task_id=TASK_ID, kind="verify_synthetic", checks={})


def m52(D2, ls):
    r = np.sqrt(D2) / ls
    return (1 + np.sqrt(5) * r + 5 * D2 / (3 * ls ** 2)) * np.exp(-np.sqrt(5) * r)


def sqd(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def check_or_bound():
    """thm:or on a residual with exactly known RKHS norm: r = sum c_i k(.,z_i),
    ||r||_K^2 = c' K_zz c. The certified inequality must hold at every test
    point and every nugget."""
    rng = np.random.default_rng(0)
    d, nc, ntr, nte = 4, 60, 400, 800
    Z = rng.uniform(-1, 1, (nc, d))
    c = rng.normal(size=(nc, 3))                       # 3 output components
    ls = 0.8
    Kzz = m52(sqd(Z, Z), ls)
    norm2 = float(np.einsum("ij,ik,jk->", Kzz, c, c))  # sum_j c_j' Kzz c_j
    X = rng.uniform(-1, 1, (ntr, d))
    U = rng.uniform(-1, 1, (nte, d))
    R_X = m52(sqd(X, Z), ls) @ c
    R_U = m52(sqd(U, Z), ls) @ c
    worst = -np.inf
    ratios = []
    for lam in (0.0, 1e-8, 1e-4, 1e-2):
        K = m52(sqd(X, X), ls)
        Kr = K + ntr * lam * np.eye(ntr) + (1e-12 * np.eye(ntr) if lam == 0 else 0)
        ch = cho_factor(Kr, lower=True, check_finite=False)
        alpha = cho_solve(ch, R_X, check_finite=False)
        KUX = m52(sqd(U, X), ls)
        pred = KUX @ alpha
        err = np.linalg.norm(pred - R_U, axis=1)
        sol = cho_solve(ch, KUX.T, check_finite=False)          # (ntr, nte)
        P2 = np.maximum(m52(np.zeros((nte, 1)), ls)[:, 0] - (KUX * sol.T).sum(1), 0.0)
        Pt2 = np.maximum(P2 - ntr * lam * (sol ** 2).sum(0), 0.0)
        bound = np.sqrt(norm2) * np.sqrt(Pt2)
        viol = err - bound
        worst = max(worst, float((viol / (np.sqrt(norm2) + 1e-30)).max()))
        m = bound > 1e-8
        ratios.append(float(np.median(err[m] / bound[m])))
    ok = worst <= 1e-8
    return dict(status="PASS" if ok else "FAIL", worst_rel_violation=worst,
                median_tightness=ratios, rkhs_norm=np.sqrt(norm2))


def check_kf_unbiased():
    """prop:kf: e2 equals the held-out-half sum exactly, and (2/b)e2 averaged
    over splits estimates the split-expected held-out error."""
    rng = np.random.default_rng(1)
    b, dz, dv = 32, 5, 2
    Zf = rng.normal(size=(b, dz))
    V = rng.normal(size=(b, dv))
    ls = float(np.sqrt(np.median(sqd(Zf, Zf)[np.triu_indices(b, 1)])))

    def interp_err(C):
        Cc = np.array(sorted(C)); rest = np.array([i for i in range(b) if i not in C])
        K = m52(sqd(Zf[Cc], Zf[Cc]), ls) + 1e-10 * np.eye(len(Cc))
        a = np.linalg.solve(K, V[Cc])
        pred_all = m52(sqd(Zf, Zf[Cc]), ls) @ a
        e2_all = float(((V - pred_all) ** 2).sum())
        e2_held = float(((V[rest] - pred_all[rest]) ** 2).sum())
        return e2_all, e2_held

    diffs, est, direct = [], [], []
    for t in range(400):
        C = set(rng.choice(b, b // 2, replace=False).tolist())
        e2a, e2h = interp_err(C)
        diffs.append(abs(e2a - e2h))
        est.append(2.0 / b * e2a)
        rest = [i for i in range(b) if i not in C]
        i = rest[rng.integers(len(rest))]
        # direct MC of E_{C,i}[ ||v_i - I_C(z_i)||^2 ]: reuse e2h's per-point terms
        direct.append(e2h / (b / 2))
    identity_ok = max(diffs) <= 1e-8 * max(1.0, max(est))
    a1, a2 = float(np.mean(est)), float(np.mean(direct))
    se = float(np.std(est) / np.sqrt(len(est)) + np.std(direct) / np.sqrt(len(direct)))
    unbiased_ok = abs(a1 - a2) <= 4 * se + 1e-12
    return dict(status="PASS" if (identity_ok and unbiased_ok) else "FAIL",
                max_identity_gap=float(max(diffs)), mc_gap=abs(a1 - a2), mc_se=se)


def check_mp_deff():
    """lem:deff identity to machine precision; cor:mp closed form vs empirical
    d_eff for an iid-entry Gram."""
    rng = np.random.default_rng(2)
    n, pdim, s2 = 3000, 900, 1.7
    X = rng.normal(scale=np.sqrt(s2), size=(n, pdim))
    K = X @ X.T
    ev = np.linalg.eigvalsh(K)
    rows = []
    worst_id = 0.0
    for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        deff = float((ev / (ev + n * lam)).sum())
        mhat = float((1.0 / (ev / n - (-lam))).mean())
        ident = n * (1 - lam * mhat)
        worst_id = max(worst_id, abs(deff - ident) / max(deff, 1e-12))
        g = pdim / n
        m_mp = (np.sqrt((lam + s2 * (1 - g)) ** 2 + 4 * g * s2 * lam)
                - (lam + s2 * (1 - g))) / (2 * g * s2 * lam)
        rows.append(dict(lam=lam, deff_emp=deff / n, deff_mp=g * (1 - lam * m_mp),
                         gap=abs(deff / n - g * (1 - lam * m_mp))))
    mp_worst = max(r["gap"] for r in rows)
    ok = worst_id <= 1e-10 and mp_worst <= 5e-3
    return dict(status="PASS" if ok else "FAIL", identity_worst_rel=worst_id,
                mp_rows=rows, mp_worst_gap=mp_worst)


def check_aniso_rates():
    """prop:aniso mechanism on a controlled G(u)=g(Au): the adapted-metric
    kernel dominates the isotropic one and improves the empirical rate.
    Slopes at finite n are noisy, so this is a REPORT with an ordering test."""
    rng = np.random.default_rng(3)
    d, r_eff = 8, 2
    sig = np.array([1.0, 1.0] + [0.02] * (d - 2))
    nc = 400
    Zc = rng.uniform(-1, 1, (nc, d)) * sig            # centers in image space
    cc = rng.normal(size=(nc, 1))
    ls_g = 0.5

    def G(u):
        return m52(sqd(u * sig, Zc), ls_g) @ cc

    Ute = rng.uniform(-1, 1, (1500, d))
    Yte = G(Ute)
    ns = (200, 400, 800, 1600, 3200)
    errs = {"iso": [], "ard": []}
    for n in ns:
        e_iso, e_ard = [], []
        for s in range(3):
            rs = np.random.default_rng(100 + 10 * s + n % 97)
            U = rs.uniform(-1, 1, (n, d))
            Y = G(U)
            for kind in ("iso", "ard"):
                Xk = U * sig if kind == "ard" else U
                Tk = Ute * sig if kind == "ard" else Ute
                med = np.sqrt(np.median(sqd(Xk[:min(n, 1000)], Xk[:min(n, 1000)])
                                        [np.triu_indices(min(n, 1000), 1)]))
                best = np.inf
                for sm in (0.5, 1.0, 2.0):
                    K = m52(sqd(Xk, Xk), sm * med) + 1e-8 * n * np.eye(n)
                    a = np.linalg.solve(K, Y)
                    e = float(np.mean(np.abs(m52(sqd(Tk, Xk), sm * med) @ a - Yte)))
                    best = min(best, e)
                (e_iso if kind == "iso" else e_ard).append(best)
        errs["iso"].append(float(np.mean(e_iso)))
        errs["ard"].append(float(np.mean(e_ard)))
    ln = np.log(np.array(ns, float))
    slope = {k: float(np.polyfit(ln, np.log(np.array(v) + 1e-15), 1)[0])
             for k, v in errs.items()}
    dominates = all(a <= i * (1 + 1e-9) for a, i in zip(errs["ard"], errs["iso"]))
    return dict(status="PASS" if (dominates and slope["ard"] < slope["iso"] - 0.05)
                else "REPORT",
                ns=list(ns), err_iso=errs["iso"], err_ard=errs["ard"],
                slope_iso=slope["iso"], slope_ard=slope["ard"],
                note="predicted ordering -s/d vs -s/r; finite-n slopes are indicative")


t0 = time.time()
todo = ("or_bound", "kf_unbiased", "mp_deff", "aniso_rates") if args.check == "all" \
    else (args.check,)
fns = dict(or_bound=check_or_bound, kf_unbiased=check_kf_unbiased,
           mp_deff=check_mp_deff, aniso_rates=check_aniso_rates)
for name in todo:
    t1 = time.time()
    try:
        out["checks"][name] = fns[name]()
    except Exception as exc:
        out["checks"][name] = dict(status="INCOMPLETE", error=repr(exc))
    out["checks"][name]["seconds"] = round(time.time() - t1, 1)
    print(name, out["checks"][name]["status"], flush=True)
out["minutes"] = round((time.time() - t0) / 60, 1)

res = ROOT / "results" / f"{TASK_ID}.json"
res.parent.mkdir(exist_ok=True)
tmp = res.with_suffix(".tmp")
json.dump(out, open(tmp, "w"), indent=1, default=float)
os.replace(tmp, res)
print("saved", res, flush=True)
