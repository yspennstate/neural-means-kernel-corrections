"""Post-hoc analyses per seed directory: the application-facing results.

Everything here is computed from artifacts the campaign already saved
(per-sample errors, member predictions); nothing retrains. For an OCO seed
directory it produces tail quantiles and exceedance curves against the
kernel-flow emulator, a certified surrogate-fallback rule (split conformal on
a calibration subset of the test block, with the flag's precision against the
worst-decile errors), the prospective second-moment scoreboard over member
pairs, and errors stratified by the named physical state coordinates. For a
structural-mechanics seed directory it produces the tails and the scoreboard.

    python campaign/analyze_extras.py --seed-dir <root>/seeds/oco_o2_s3
Writes <seed-dir>/extras.json.
"""
import argparse, json, os, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

p = argparse.ArgumentParser()
p.add_argument("--seed-dir", required=True)
args = p.parse_args()
SD = pathlib.Path(args.seed_dir)
name = SD.name
out = dict(seed_dir=name)

QS = (50, 90, 95, 99, 100)


def quantiles(e):
    q = np.percentile(e, QS)
    return {f"p{a}": float(v) for a, v in zip(QS, q)}


def exceedance(e, grid):
    return {f"{t:g}": float((e <= t).mean()) for t in grid}


def secmom_scoreboard(residuals, errs):
    """Prospective test of the two-member rule: mixing the weaker member in
    helps iff the residual correlation is below the error ratio. Predictions
    from validation, outcomes from test."""
    names = list(residuals["val"].keys())
    rows, correct = [], 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if errs["val"][a] > errs["val"][b]:
                a, b = b, a
            rv_a, rv_b = residuals["val"][a], residuals["val"][b]
            rho_v = float((rv_a * rv_b).sum() / (np.linalg.norm(rv_a) * np.linalg.norm(rv_b) + 1e-30))
            predict_helps = rho_v < errs["val"][a] / max(errs["val"][b], 1e-30)
            # outcome on test: does the optimal convex 2-mix beat the better member
            rt_a, rt_b = residuals["test"][a], residuals["test"][b]
            e1s = float((rt_a ** 2).mean()); e2s = float((rt_b ** 2).mean())
            c = float((rt_a * rt_b).mean())
            den = e1s + e2s - 2 * c
            w = (e2s - c) / den if den > 1e-30 else 1.0
            w = min(1.0, max(0.0, w))
            mix = float(((w * rt_a + (1 - w) * rt_b) ** 2).mean())
            helps = mix < e1s * (1 - 1e-6)
            rows.append(dict(pair=[a, b], rho_val=round(rho_v, 4),
                             ratio=round(errs["val"][a] / max(errs["val"][b], 1e-30), 4),
                             predict=bool(predict_helps), outcome=bool(helps)))
            correct += int(predict_helps == helps)
    return dict(pairs=rows, n=len(rows), correct=correct)


if name.startswith("oco_"):
    band, seed = name.split("_")[1], int(name.split("_s")[1])
    import jpl_data
    if os.environ.get("NMKC_JPL_DATA"):
        jpl_data.DATA = pathlib.Path(os.environ["NMKC_JPL_DATA"])
    from jpl_data import load_band, reconstruction, to_radiance
    import h5py

    per = np.load(SD / "per_sample_errors.npz")
    mp = np.load(SD / "member_preds.npz")
    Yte = mp["Yte"].astype(np.float64); Yval = mp["Yval"].astype(np.float64)
    recon = reconstruction(band)

    # 1. tails and exceedance, both metrics, all models
    models = sorted(set(k[:-4] for k in per.files if k.endswith("_red")))
    tails = {}
    grid_red = np.geomspace(0.005, 1.0, 25)
    grid_rad = np.geomspace(1e-5, 3e-3, 25)
    for m in models:
        tails[m] = dict(reduced=quantiles(per[m + "_red"]),
                        radiance=quantiles(per[m + "_rad"]),
                        exceed_red=exceedance(per[m + "_red"], grid_red),
                        exceed_rad=exceedance(per[m + "_rad"], grid_rad))
    out["tails"] = tails

    # 2. certified fallback: conformal on radiance error of the combination,
    # scaled by member disagreement in radiance space
    members = [k[3:] for k in mp.files
               if k.startswith("te_") and k[3:] not in ("kernel_flow", "combined", "combined_w")]
    # disagreement scale uses only the deployed ensemble members; the raw-input
    # kernel rows are diagnostics and would inflate the bands
    deployed = [m for m in members if m not in ("kernel_raw", "kernel_ard")]
    R_mem = np.stack([to_radiance(mp["te_" + m].astype(np.float64), recon) for m in deployed])
    disagree = R_mem.std(0).mean(1)
    den = np.linalg.norm(to_radiance(Yte, recon), axis=1)
    disagree = disagree / den * len(R_mem[0][0]) ** 0  # scale-free enough; relative units
    err = per["combined_rad"]
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(len(err))
    cal, ev = perm[:500], perm[500:]
    conf = {}
    for alpha in (0.1, 0.05):
        import math
        k = math.ceil((1 - alpha) * (len(cal) + 1))
        q = np.sort(err[cal] / (disagree[cal] + 1e-30))[min(k, len(cal)) - 1]
        band_w = q * (disagree[ev] + 1e-30)
        cover = float((err[ev] <= band_w).mean())
        # flag rule: cutoff fixed on CALIBRATION band widths (causal), applied
        # to evaluation; compared against the realized worst decile
        thresh = np.percentile(q * (disagree[cal] + 1e-30), 90)
        flag = band_w >= thresh
        worst = err[ev] >= np.percentile(err[ev], 90)
        prec = float((flag & worst).sum() / max(flag.sum(), 1))
        rec = float((flag & worst).sum() / max(worst.sum(), 1))
        conf[f"a{alpha:g}"] = dict(coverage=cover, target=1 - alpha,
                                   flag_precision=prec, flag_recall=rec)
    out["conformal_fallback"] = conf

    # 3. second-moment scoreboard on this band-seed
    resid = dict(val={}, test={})
    errs = dict(val={}, test={})
    nv = np.linalg.norm(Yval, axis=1, keepdims=True)
    nt = np.linalg.norm(Yte, axis=1, keepdims=True)
    for m in members:
        rv = (mp["val_" + m].astype(np.float64) - Yval) / nv
        rt = (mp["te_" + m].astype(np.float64) - Yte) / nt
        resid["val"][m] = rv.ravel(); resid["test"][m] = rt.ravel()
        errs["val"][m] = float(np.sqrt((rv ** 2).sum(1)).mean())
        errs["test"][m] = float(np.sqrt((rt ** 2).sum(1)).mean())
    out["secmom"] = secmom_scoreboard(resid, errs)

    # floor bracket over the deployed members; the realized point is the
    # uniform convex mix, the feasible point the pinch bound evaluates
    S_ = np.empty((len(deployed), len(deployed)))
    for i2, a2 in enumerate(deployed):
        for j2, b3 in enumerate(deployed):
            S_[i2, j2] = float((resid["test"][a2] * resid["test"][b3]).mean())
    eb2 = float(np.diag(S_).min())
    offd_ = S_[~np.eye(len(deployed), dtype=bool)] / eb2
    lo_, hi_ = float(offd_.min()), float(offd_.max())
    dh_ = float(np.diag(S_).max() / eb2)
    u_ = np.full(len(deployed), 1.0 / len(deployed))
    out["floor_bracket"] = dict(
        members=deployed, bracket_units="ebar2",
        bracket=[round(lo_, 5), round(hi_ + (dh_ - hi_) / len(deployed), 5)],
        realized_uniform_mix=round(float(u_ @ S_ @ u_ / eb2), 5))

    # 4. stratified by named state coordinates
    sp = load_band(band, seed=seed)
    Xte = sp["Xte"]
    names_all, xind = None, None
    with h5py.File(jpl_data.DATA / "dimred_data_4_mono.jld", "r") as h:
        if "red_names" in h:
            try:
                names_all = [x.decode() if isinstance(x, bytes) else str(x)
                             for x in np.asarray(h["red_names"]).ravel()]
            except Exception:
                names_all = None
        key = f"{band}_xind"
        if key in h:
            xind = np.asarray(h[key]).ravel().astype(int) - 1   # julia is 1-based
    coord_names = ([names_all[i] for i in xind] if (names_all is not None and xind is not None
                   and max(xind) < len(names_all)) else
                   [f"x{j}" for j in range(Xte.shape[1])])
    e = per["combined_red"]
    strat = []
    for j in range(Xte.shape[1]):
        qs = np.quantile(Xte[:, j], [0.2, 0.4, 0.6, 0.8])
        binned = np.digitize(Xte[:, j], qs)
        means = [float(e[binned == b].mean()) for b in range(5)]
        strat.append(dict(coord=coord_names[j], quintile_err=[round(x, 5) for x in means],
                          spread=round(max(means) - min(means), 5)))
    strat.sort(key=lambda r: -r["spread"])
    out["stratified_top"] = strat[:6]

    # prop:selectcoord evaluated on this band-seed: the deployed per-coordinate
    # selection (M=6, q=40, m_v=2000) against the test oracle, in both the
    # unweighted and the exact squared-relative-error metrics, next to the
    # bound with the measured range proxy L. Isolated: this block has not yet
    # run against a finished band-seed, and a failure here must not cost the
    # established sections above.
    try:
      import math as _math
      Pv_o = np.stack([mp["val_" + m].astype(np.float64) for m in deployed])
      Pt_o = np.stack([mp["te_" + m].astype(np.float64) for m in deployed])
      M_o, m_v, q_o = len(deployed), Yval.shape[0], Yval.shape[1]
      delta_o = 0.05
      a_val = 1.0 / (np.linalg.norm(Yval, axis=1) ** 2 + 1e-30)
      a_te = 1.0 / (np.linalg.norm(Yte, axis=1) ** 2 + 1e-30)
      sc = {}
      for tag, av, at in (("unweighted", np.ones(m_v), np.ones(len(Yte))),
                          ("weighted", a_val, a_te)):
          Rv = np.einsum("n,mnj->mj", av, (Pv_o - Yval) ** 2) / m_v
          Rt = np.einsum("n,mnj->mj", at, (Pt_o - Yte) ** 2) / len(Yte)
          win = Rv.argmin(0)
          realized = float(np.mean(Rt[win, np.arange(q_o)] - Rt.min(0)))
          L_meas = float(max((av[None, :, None] * (Pv_o - Yval) ** 2).max(),
                             (at[None, :, None] * (Pt_o - Yte) ** 2).max()))
          sc[tag] = dict(
              realized_excess=realized, L_meas=L_meas,
              bound=2 * L_meas * _math.sqrt(_math.log(2 * M_o * q_o / delta_o)
                                            / (2 * m_v)),
              oracle_risk=float(Rt.min(0).mean()),
              deployed_risk=float(Rt[win, np.arange(q_o)].mean()))
      out["bounds"] = dict(M=M_o, q=q_o, m_v=m_v, delta=delta_o,
                           members=deployed, selectcoord=sc)
    except Exception as exc:                                   # noqa: BLE001
      out["bounds"] = dict(error=f"{type(exc).__name__}: {exc}")
      print(f"BOUNDS BLOCK FAILED on {name}: {exc}", flush=True)

elif name.startswith("sm_"):
    per = np.load(SD / "per_sample_errors.npz")
    out["tails"] = dict(stack=quantiles(per["stack"]), corr=quantiles(per["corr"]),
                        exceed_corr=exceedance(per["corr"], np.geomspace(0.01, 0.5, 25)))

    def floor_bracket(residuals_test, stack_weights):
        """cor:floor / cor:floorpinch on the measured S-matrix, in ebar^2
        units (scale-invariant, so the flattened-residual normalization drops
        out). The realized point is the fitted convex stack's w'Sw."""
        names_ = list(residuals_test.keys())
        M_ = len(names_)
        S = np.empty((M_, M_))
        for i, a in enumerate(names_):
            for j2, b2 in enumerate(names_):
                S[i, j2] = float((residuals_test[a] * residuals_test[b2]).mean())
        ebar2 = float(np.diag(S).min())
        offd = S[~np.eye(M_, dtype=bool)] / ebar2
        s_off_lo, s_off_hi = float(offd.min()), float(offd.max())
        s_diag_hi = float(np.diag(S).max() / ebar2)
        lower = s_off_lo
        upper = s_off_lo + (s_off_hi - s_off_lo) + (s_diag_hi - s_off_hi) / M_
        w = np.array([stack_weights.get(n_, 0.0) for n_ in names_])
        realized = float(w @ S @ w / ebar2) if w.sum() > 0 else None
        return dict(members=names_, bracket_units="ebar2",
                    bracket=[round(lower, 5), round(upper, 5)],
                    realized_convex_stack=None if realized is None else round(realized, 5),
                    inside=None if realized is None else
                    bool(lower - 1e-9 <= realized <= upper + 1e-9))
    runs = SD / "runs"
    hj = json.load(open(runs / "hpix.json"))
    members = hj["members"]
    import common
    os.environ.setdefault("NMKC_SPLIT_SEED", name.split("_s")[1])
    from common import load_arrays, canonical_split
    loads, stress = load_arrays()
    tr, va, te = canonical_split(n_val=1000, seed=int(name.split("_s")[1]))
    Yva = stress[va].reshape(len(va), -1).astype(np.float64)
    Yte_s = stress[te].reshape(len(te), -1).astype(np.float64)
    nv = np.linalg.norm(Yva, axis=1, keepdims=True)
    nt = np.linalg.norm(Yte_s, axis=1, keepdims=True)
    resid = dict(val={}, test={})
    errs = dict(val={}, test={})
    for m in members:
        fva = runs / (f"{m}_predva.npy" if m != "krr" else "krr_full_matern52_n19000_pred_val.npy")
        fte = runs / (f"{m}_predte.npy" if m != "krr" else "krr_full_matern52_n19000_pred_test.npy")
        rv = (np.load(fva).astype(np.float64) - Yva) / nv
        rt = (np.load(fte).astype(np.float32).astype(np.float64) - Yte_s) / nt
        resid["val"][m] = rv.ravel(); resid["test"][m] = rt.ravel()
        errs["val"][m] = float(np.sqrt((rv ** 2).sum(1)).mean())
        errs["test"][m] = float(np.sqrt((rt ** 2).sum(1)).mean())
    out["secmom"] = secmom_scoreboard(resid, errs)
    hstk = json.load(open(runs / "hstk.json"))
    out["floor_bracket"] = floor_bracket(resid["test"],
                                         hstk["report"].get("weights", {}))
    # certificate constant W: refit the per-pixel ridge on the validation
    # members (deterministic given saved arrays) and record the weight norms
    Pv = np.stack([(np.load(runs / (f"{m}_predva.npy" if m != "krr" else
                    "krr_full_matern52_n19000_pred_val.npy"))).astype(np.float64)
                   for m in members])
    Xa = np.concatenate([Pv, np.ones((1,) + Pv.shape[1:])], 0)
    Gm = np.einsum("mnd,knd->dmk", Xa, Xa) / Pv.shape[1]
    bv = np.einsum("mnd,nd->dm", Xa, Yva) / Pv.shape[1]
    Gm += 1e-3 * np.eye(len(members) + 1)[None]
    Wd = np.linalg.solve(Gm, bv[..., None])[..., 0]
    wn = np.linalg.norm(Wd, axis=1)
    out["certificate_constants"] = dict(
        W_max=float(wn.max()), W_median=float(np.median(wn)),
        b=float(max(np.abs(Yva).max(), max(float(np.abs(p).max()) for p in Pv))),
        note="kappa (correction weight l1 norm) computed in the collection wave")

    # finite-sample bounds next to realized excesses (verification appendix).
    # The oracle of prop:affine is estimated by an unregularized per-coordinate
    # refit on the test arrays (M+1 parameters on 19000 samples; its optimism
    # is O((M+1)/n) and it can only make the reported ratios conservative).
    # Member test predictions are streamed in row chunks; loading all six at
    # once is the 19 GB mistake the stack stages already fixed.
    import math as _math
    hstk_w = hstk["report"].get("weights", {})
    w_glob = np.array([hstk_w.get(m, 0.0) for m in members])
    have_glob = float(w_glob.sum()) > 0
    files_te = [runs / (f"{m}_predte.npy" if m != "krr" else
                        "krr_full_matern52_n19000_pred_test.npy") for m in members]
    mm = [np.load(f, mmap_mode="r") for f in files_te]
    S_dep = np.load(runs / "hpix_stack_te.npy", mmap_mode="r")
    C_dep = np.load(runs / "hpix_corr_pred_test.npy", mmap_mode="r")
    n_te, Dq = Yte_s.shape
    M_ = len(members)
    Gm_te = np.zeros((Dq, M_ + 1, M_ + 1))
    bv_te = np.zeros((Dq, M_ + 1))
    y2_c = np.zeros(Dq); r2_stack_c = np.zeros(Dq); r2_corr_c = np.zeros(Dq)
    e_glob = np.empty(n_te) if have_glob else None
    b_meas = float(np.abs(Yva).max())
    for k0 in range(0, n_te, 1000):
        sl = slice(k0, min(k0 + 1000, n_te))
        P = np.stack([np.asarray(m_[sl], dtype=np.float64) for m_ in mm])
        b_meas = max(b_meas, float(np.abs(P).max()))
        Pa = np.concatenate([P, np.ones((1,) + P.shape[1:])], 0)
        Gm_te += np.einsum("mnd,knd->dmk", Pa, Pa)
        Y = Yte_s[sl]
        b_meas = max(b_meas, float(np.abs(Y).max()))
        bv_te += np.einsum("mnd,nd->dm", Pa, Y)
        y2_c += (Y ** 2).sum(0)
        r2_stack_c += ((np.asarray(S_dep[sl], np.float64) - Y) ** 2).sum(0)
        r2_corr_c += ((np.asarray(C_dep[sl], np.float64) - Y) ** 2).sum(0)
        if have_glob:
            Rg = np.einsum("m,mnd->nd", w_glob, P) - Y
            e_glob[sl] = np.linalg.norm(Rg, axis=1) / np.linalg.norm(Y, axis=1)
    Gm_te /= n_te; bv_te /= n_te; y2_c /= n_te
    Gm_te += 1e-9 * np.eye(M_ + 1)[None]
    W_star = np.linalg.solve(Gm_te, bv_te[..., None])[..., 0]
    R_oracle = float(np.mean(y2_c - np.einsum("dm,dm->d", W_star, bv_te)))
    R_stack = float(np.mean(r2_stack_c / n_te))
    R_corr = float(np.mean(r2_corr_c / n_te))

    delta_ = 0.05
    W_b, D_b, m_b, q_b, lam_b = float(wn.max()), _math.sqrt(M_ + 1), len(Yva), Dq, 1e-3
    aff_bound = ((8 * W_b * b_meas ** 2 * D_b * (1 + W_b * D_b)
                  + 2 * b_meas ** 2 * (1 + W_b * D_b) ** 2
                  * _math.sqrt(0.5 * _math.log(4 * q_b / delta_)))
                 / _math.sqrt(m_b)) + lam_b * W_b ** 2
    # lem:select on the honesty comparison (K=2 on half the validation split)
    sel = dict(K=2, m_half=len(Yva) // 2,
               holdout_pix=hj.get("holdout_pix"), holdout_glob=hj.get("holdout_glob"),
               used_perpixel=bool(hj.get("used", True)))
    e_pix_mean = float(per["stack"].mean())
    if have_glob:
        e_glob_mean = float(e_glob.mean())
        chosen = e_pix_mean if sel["used_perpixel"] else e_glob_mean
        B_hat = float(max(per["stack"].max(), e_glob.max()))
        sel.update(test_perpixel=e_pix_mean, test_global=e_glob_mean,
                   realized_regret=chosen - min(e_pix_mean, e_glob_mean),
                   B_meas=B_hat,
                   bound=B_hat * _math.sqrt(2 * _math.log(2 * 2 / delta_)
                                            / sel["m_half"]))
    out["bounds"] = dict(
        delta=delta_, b_meas=b_meas, W=W_b, D=D_b, m=m_b, q=q_b,
        n_grid=9,
        select=sel,
        affine=dict(bound=aff_bound, ridge_term=lam_b * W_b ** 2,
                    R_deployed_stack=R_stack, R_test_refit_oracle=R_oracle,
                    realized_excess=R_stack - R_oracle),
        certificate=dict(R_deployed_corrected=R_corr,
                         realized_excess_vs_oracle=R_corr - R_oracle,
                         note="bound assembled at collection once kappa joins"))
else:
    raise SystemExit(f"unrecognized seed dir kind: {name}")

tmp = SD / "extras.json.tmp"
json.dump(out, open(tmp, "w"), indent=1)
os.replace(tmp, SD / "extras.json")

# a copy lands in <root>/results/ under a unique name so the collector's
# existing results pull picks extras up without a second transport path
out["kind"] = "extras"
root_res = SD.parent.parent / "results"
if root_res.is_dir():
    tmp2 = root_res / f"extras_{name}.json.tmp"
    json.dump(out, open(tmp2, "w"), indent=1)
    os.replace(tmp2, root_res / f"extras_{name}.json")
sb = out.get("secmom")
print(f"{name}: extras written" +
      (f"; secmom rule correct {sb['correct']}/{sb['n']}" if sb else ""), flush=True)
