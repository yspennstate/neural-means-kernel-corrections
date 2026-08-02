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
BOXES = [
    ("box1", ["scp", "-o", "ConnectTimeout=20"], "research-box"),
    ("box2", ["scp", "-o", "ConnectTimeout=20"], "research-box-02"),
    ("box3", ["scp", "-o", "ConnectTimeout=20"], "research-box-03"),
]
ROOT = "/srv/aiwork/nmkc10seed"

ap = argparse.ArgumentParser()
ap.add_argument("--no-pull", action="store_true")
args = ap.parse_args()

if not args.no_pull:
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
sm = [r for r in rows if r.get("kind") == "structmech_seed" and r.get("seed", 99) < 90]
if sm:
    s = dict(seeds=sorted(r["seed"] for r in sm))
    s["stack"] = stats([r["stack_test"] for r in sm])
    s["corr"] = stats([r["corr_test"] for r in sm])
    s["delta"] = stats([r["delta_corr_minus_stack"] for r in sm])
    s["delta_boot_negative"] = sum(1 for r in sm if r["boot_ci_delta"][1] < 0)
    s["hstk_final"] = stats([r["hstk"]["report"]["final_test"] for r in sm
                             if "hstk" in r])
    for m in ("mlp", "mlpMSE", "mlpR", "fno", "unet"):
        s[f"member_{m}"] = stats([r["members"][m] for r in sm])
    s["krr"] = stats([r.get("krr") for r in sm])
    cov = [r["uq"].get("a0.1", {}).get("scaled", {}).get("coverage") for r in sm]
    s["uq_cover90_scaled"] = stats(cov)
    summary["structmech"] = s

# OCO-2 seeds
for band in ("o2", "wco2", "sco2"):
    br = [r for r in rows if r.get("kind") == "oco_seed" and r.get("band") == band
          and r.get("seed", 99) < 90]
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
