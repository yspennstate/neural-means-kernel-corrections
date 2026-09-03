"""Pull campaign results from the three boxes and aggregate seed statistics.

Run on the Windows workstation (uses ssh/scp with the configured hosts).
Copies <root>/results/*.json (and small per-sample archives) into
campaign/collected/<host>/ then writes campaign/summary.json and prints a
LaTeX macro block for the paper.

    python campaign/collect.py            # pull + aggregate
    python campaign/collect.py --no-pull  # aggregate what is already local
"""
import argparse, json, math, pathlib, statistics, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
COLL = HERE / "collected"
# The host list is machine-local and not part of the repository:
# campaign/boxes.local.json holds rows of [name, scp_argv, host], e.g.
#   [["box1", ["scp", "-o", "ConnectTimeout=20"], "my-box-host"]]
_BOXES_FILE = HERE / "boxes.local.json"
BOXES = ([tuple(r) for r in json.load(open(_BOXES_FILE))]
         if _BOXES_FILE.exists() else [])
ROOT = "/srv/aiwork/nmkc10seed"

ap = argparse.ArgumentParser()
ap.add_argument("--no-pull", action="store_true")
args = ap.parse_args()

if not args.no_pull:
    if not BOXES:
        sys.exit("no boxes configured: create campaign/boxes.local.json or run --no-pull")
    for name, scp, host in BOXES:
        dst = COLL / name
        dst.mkdir(parents=True, exist_ok=True)
        cmd = scp + [f"{host}:{ROOT}/results/*.json", str(dst) + "/"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        n = len(list(dst.glob("*.json")))
        print(f"{name}: pulled, {n} result files present"
              + ("" if r.returncode == 0 else f" (scp rc={r.returncode}: {r.stderr.strip()[:200]})"))

rows = []
for f in sorted(COLL.glob("*/*.json")):
    try:
        d = json.load(open(f))
        d["_host"] = f.parent.name
        rows.append(d)
    except ValueError:
        print("unparseable:", f)


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    n = len(xs)
    mu = statistics.fmean(xs)
    sd = statistics.stdev(xs) if n > 1 else 0.0
    # two-sided 95% t interval for the mean
    T = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    half = T.get(n - 1, 1.96) * sd / math.sqrt(n) if n > 1 else float("nan")
    return dict(n=n, mean=mu, sd=sd, ci95=[mu - half, mu + half],
                min=min(xs), max=max(xs))


summary = {}

# structural mechanics seeds
sm_all = [r for r in rows if r.get("kind") == "structmech_seed" and r.get("seed", 99) < 90]
# Two structural-mechanics campaigns share the seed numbers: the box lanes and the
# complete-schedule six-member campaign (collected/dgx). Keep one row per seed, the
# dgx one when both exist, so the summary describes that campaign at ten seeds.
_by_seed = {}
for r in sorted(sm_all, key=lambda r: r.get("_host") != "dgx"):
    _by_seed.setdefault(r["seed"], r)
if len(_by_seed) < len(sm_all):
    print(f"NOTE: structmech: {len(sm_all) - len(_by_seed)} box-lane rows superseded by the "
          f"complete-schedule campaign at the same seeds")
sm_all = list(_by_seed.values())
sm = [r for r in sm_all if "stack_test" in r]
for r in sm_all:
    if r not in sm:
        print(f"WARNING: skipping incomplete structmech result {r.get('task_id')}")
if sm:
    s = dict(seeds=sorted(r["seed"] for r in sm))
    s["stack"] = stats([r["stack_test"] for r in sm])
    s["corr"] = stats([r["corr_test"] for r in sm])
    s["delta"] = stats([r["delta_corr_minus_stack"] for r in sm])
    s["delta_boot_negative"] = sum(1 for r in sm if r["boot_ci_delta"][1] < 0)
    s["hstk_final"] = stats([r["hstk"]["report"]["final_test"] for r in sm
                             if "report" in r.get("hstk", {})])
    for m in ("mlp", "mlpMSE", "mlpR", "fno", "unet"):
        s[f"member_{m}"] = stats([r["members"][m] for r in sm])
    s["krr"] = stats([r.get("krr") for r in sm])
    cov = [r["uq"].get("a0.1", {}).get("scaled", {}).get("coverage") for r in sm]
    s["uq_cover90_scaled"] = stats(cov)
    summary["structmech"] = s

# OCO-2 seeds. The a2_* queue lane partially re-ran seeds the 08-09 lane had
# already produced; keep one row per (band, seed), preferring the complete
# original lane, so the summary is ten seeds and not a mixture of the two.
for band in ("o2", "wco2", "sco2"):
    br_all = [r for r in rows if r.get("kind") == "oco_seed" and r.get("band") == band
              and r.get("seed", 99) < 90]
    by_seed = {}
    for r in sorted(br_all, key=lambda r: r.get("task_id", "").startswith("a2_")):
        by_seed.setdefault(r["seed"], r)
    br = list(by_seed.values())
    dropped = len(br_all) - len(br)
    if dropped:
        print(f"NOTE: oco_{band}: {dropped} duplicate a2_* lane rows excluded from stats")
    if not br:
        continue
    s = dict(seeds=sorted(r["seed"] for r in br))
    models = sorted({k for r in br for k in r["results"]})
    for mname in models:
        for metric in ("reduced", "radiance"):
            vals = [r["results"].get(mname, {}).get(metric) for r in br]
            s[f"{mname}_{metric}"] = stats(vals)
    summary[f"oco_{band}"] = s

# ClimSim points
cs = [r for r in rows if r.get("kind") == "climsim_point"]
if cs:
    pts = {}
    for r in cs:
        key = f"n{r['n']:07d}"
        pts.setdefault(key, {"n": r["n"], "mean_r2": [], "kernel_r2": [],
                             "mean_rel": []})
        pts[key]["mean_r2"].append(r.get("mean_r2"))
        pts[key]["kernel_r2"].append(r.get("kernel_r2"))
        pts[key]["mean_rel"].append(r.get("mean_rel"))
    summary["climsim"] = {k: dict(n=v["n"], mean_r2=stats(v["mean_r2"]),
                                  kernel_r2=stats(v["kernel_r2"]),
                                  mean_rel=stats(v["mean_rel"]))
                          for k, v in sorted(pts.items())}

# exact law of realized conformal coverage (rem:covlaw): conditional coverage
# is Beta(k, n_cal+1-k), the observed coverage on n_eval disjoint samples is
# Beta-Binomial; every seed is placed against the exact central band.
def _betabinom(N, a, b):
    logB = lambda x, y: math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)
    den = logB(a, b)
    cdf, c = [], 0.0
    for k in range(N + 1):
        c += math.exp(math.lgamma(N + 1) - math.lgamma(k + 1)
                      - math.lgamma(N - k + 1) + logB(k + a, N - k + b) - den)
        cdf.append(min(c, 1.0))
    q = lambda t: next(i for i, cv in enumerate(cdf) if cv >= t)
    return cdf, q(0.025), q(0.975)   # central 95 percent band of the observed evaluation fraction (the paper's band)


if sm:
    cl = {}
    for akey in ("a0.1", "a0.05"):
        seeds_e = [(r["seed"], r["uq"]) for r in sm
                   if isinstance(r.get("uq"), dict) and akey in r["uq"]]
        if not seeds_e:
            continue
        n_cal = seeds_e[0][1]["n_cal"]; n_eval = seeds_e[0][1]["n_eval"]
        target = seeds_e[0][1][akey]["target"]
        alpha = 1 - target
        k = math.ceil(target * (n_cal + 1))
        cdf, qlo, qhi = _betabinom(n_eval, k, n_cal + 1 - k)
        entry = dict(alpha=alpha, k=k, n_cal=n_cal, n_eval=n_eval,
                     beta_mean=k / (n_cal + 1),
                     band95=[qlo / n_eval, qhi / n_eval], seeds=[])
        for s, uq in seeds_e:
            if uq["n_cal"] != n_cal or uq["n_eval"] != n_eval:
                entry["seeds"].append(dict(seed=s, note="size mismatch, skipped"))
                continue
            obs = uq[akey]["scaled"]["coverage"]
            c_obs = round(obs * n_eval)
            entry["seeds"].append(dict(seed=s, coverage=obs,
                                       pit=round(cdf[c_obs], 4),
                                       inside=bool(qlo <= c_obs <= qhi)))
        ok = [e for e in entry["seeds"] if "inside" in e]
        entry["n_inside"] = sum(e["inside"] for e in ok)
        entry["n"] = len(ok)
        cl[akey] = entry
    if cl:
        summary["coverage_law"] = cl

# bound sharpness: extras rows carry the measured constants and realized
# excesses; kappa joins here and closes the certificate chain (|Gamma| is the
# 9-point correction grid plus gamma=0).
kap = {r["seed"]: r.get("kappa") for r in rows
       if r.get("kind") == "kappa" and r.get("seed", 99) < 90}
ex = [r for r in rows if r.get("kind") == "extras"]
sm_ex = [r for r in ex if r.get("seed_dir", "").startswith("sm_s")
         and int(r["seed_dir"].split("_s")[1]) < 90]
if sm_ex:
    per_seed, cols = [], {}
    for r in sm_ex:
        bo = r.get("bounds") or {}
        if "error" in bo or "affine" not in bo:
            continue
        s = int(r["seed_dir"].split("_s")[1])
        row = dict(seed=s,
                   select_bound=bo["select"].get("bound"),
                   select_realized=bo["select"].get("realized_regret"),
                   affine_bound=bo["affine"]["bound"],
                   affine_realized=bo["affine"]["realized_excess"],
                   cert_realized=bo["certificate"]["realized_excess_vs_oracle"],
                   kappa=kap.get(s))
        if row["kappa"] is not None:
            W, D, b2 = bo["W"], bo["D"], bo["b_meas"] ** 2
            g = bo["n_grid"] + 1
            row["cert_bound"] = (2 * (1 + row["kappa"]) ** 2 * b2 * (1 + W * D)
                                 * (4 * W * D + (1 + W * D)
                                    * math.sqrt(0.5 * math.log(4 * bo["q"] * g
                                                               / bo["delta"])))
                                 / math.sqrt(bo["m"])) + bo["affine"]["bound"]
        per_seed.append(row)
    for key in ("select_bound", "select_realized", "affine_bound",
                "affine_realized", "cert_bound", "cert_realized", "kappa"):
        cols[key] = stats([r.get(key) for r in per_seed])
    summary["bounds_sm"] = dict(per_seed=per_seed, **cols)

for band in ("o2", "wco2", "sco2"):
    bx = [r for r in ex if r.get("seed_dir", "").startswith(f"oco_{band}_s")
          and int(r["seed_dir"].rsplit("_s", 1)[1]) < 90]
    rowsb = [r["bounds"]["selectcoord"] for r in bx
             if isinstance(r.get("bounds"), dict) and "selectcoord" in r["bounds"]]
    if rowsb:
        summary[f"bounds_oco_{band}"] = {
            f"{tag}_{f}": stats([rb[tag][f] for rb in rowsb])
            for tag in ("unweighted", "weighted")
            for f in ("bound", "realized_excess")}

# raw-kernel scaling sweep: shared exponent + constant gap is the prop:aniso
# prediction at near-full approximate rank
sc_rows = [r for r in rows if r.get("kind") == "oco_scaling" and r.get("seed", 99) < 90
           and not r.get("task_id", "").startswith("aa_")]
if sc_rows:
    scs = {}
    for band in ("o2", "wco2", "sco2"):
        br = [r for r in sc_rows if r.get("band") == band]
        if not br:
            continue
        sizes = br[0]["sizes"]
        curves = {f"n{n:05d}": dict(
            err_iso=stats([r["rows"][i]["err_iso"] for r in br]),
            err_ard=stats([r["rows"][i]["err_ard"] for r in br]),
            gap=stats([r["rows"][i]["gap"] for r in br]))
            for i, n in enumerate(sizes)}
        scs[band] = dict(
            d=br[0]["d"], n_seeds=len(br), sizes=sizes,
            slope_iso=stats([r["slope_iso"] for r in br]),
            slope_ard=stats([r["slope_ard"] for r in br]),
            slope_ratio=stats([r["slope_ratio"] for r in br]),
            gap_min=stats([r["gap_min"] for r in br]),
            gap_max=stats([r["gap_max"] for r in br]),
            participation_ratio=stats([r["participation_ratio"] for r in br]),
            curves=curves)
    if scs:
        summary["oco_scaling"] = scs

out = HERE / "summary.json"
json.dump(summary, open(out, "w"), indent=1)
print("wrote", out)

# quick console view
def fmt(s, scale=100.0, digits=2):
    if not s:
        return "--"
    return (f"{scale*s['mean']:.{digits}f} +/- {scale*s['sd']:.{digits}f} "
            f"(n={s['n']}, CI {scale*s['ci95'][0]:.{digits}f}..{scale*s['ci95'][1]:.{digits}f})")


if "structmech" in summary:
    s = summary["structmech"]
    print("\nstructmech: stack", fmt(s["stack"]), " corr", fmt(s["corr"]))
    print("  delta(corr-stack)", fmt(s["delta"], digits=3),
          f" per-seed boot CIs excluding 0: {s['delta_boot_negative']}/{s['delta']['n'] if s['delta'] else 0}")
for band in ("o2", "wco2", "sco2"):
    k = f"oco_{band}"
    if k in summary:
        s = summary[k]
        for mname in ("kernel_raw", "kernel_ard", "kernel_flow", "mean_flat",
                      "mean_radx", "dkr_flat", "combined", "combined_w"):
            key = f"{mname}_reduced"
            if key in s and s[key]:
                print(f"{band} {mname:12s} reduced {fmt(s[key])}")
if "climsim" in summary:
    for k, v in summary["climsim"].items():
        print(f"climsim {v['n']:>8d}: mean_r2 {fmt(v['mean_r2'], scale=1, digits=4)}"
              f"  kernel_r2 {fmt(v['kernel_r2'], scale=1, digits=4)}")
if "coverage_law" in summary:
    for akey, e in summary["coverage_law"].items():
        print(f"coverage law {akey}: band95 [{e['band95'][0]:.4f}, {e['band95'][1]:.4f}]"
              f"  inside {e['n_inside']}/{e['n']}")
if "bounds_sm" in summary:
    b = summary["bounds_sm"]
    for nm in ("select", "affine", "cert"):
        bd, rl = b.get(f"{nm}_bound"), b.get(f"{nm}_realized")
        if bd and rl and rl["mean"]:
            print(f"bounds {nm}: bound {bd['mean']:.4g}  realized {rl['mean']:.4g}"
                  f"  ratio {bd['mean']/abs(rl['mean']):.3g}")
if "oco_scaling" in summary:
    for band, v in summary["oco_scaling"].items():
        r = v["slope_ratio"]
        print(f"scaling {band}: slope_iso {fmt(v['slope_iso'], scale=1, digits=3)}"
              f"  slope_ard {fmt(v['slope_ard'], scale=1, digits=3)}"
              f"  ratio {fmt(r, scale=1, digits=3) if r else 'n/a'}"
              f"  gap {fmt(v['gap_min'], scale=1, digits=2)}..{fmt(v['gap_max'], scale=1, digits=2)}")
