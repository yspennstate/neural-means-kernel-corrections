"""Collect the scaled-experiment records into tables and figures in the paper's
figure style. Reads runs/structmech/*.json (train_mlp_scale.py), runs/jpl/*.json
(jpl_scale.py) and runs/lk/*.json (lk_scale.py); writes analysis/out/scale_tables.json,
scale_tables.md and the figures fig_scale_structmech.{pdf,png},
fig_scale_oco2.{pdf,png}, fig_scale_kernelflows.{pdf,png}.

Conventions (the paper's): errors in percent, mean over seeds with the seed
standard deviation (ddof=1); paired differences against the paper configuration
(w1024 d4 for structural mechanics, w384 d4 for OCO-2) are formed within seed
and reported with their own seed sd; a difference is called resolved when
|mean| > 2 sd/sqrt(n).

    python collect_scale.py
"""
import glob, json, math, os, re
from pathlib import Path
import numpy as np

W = Path("C:/Users/owner/GOAL20H_20260906/nmkc_gpu_experiments")
OUT = W / "analysis" / "out"; OUT.mkdir(parents=True, exist_ok=True)
PAPER_SM = dict(val=0.050077538210512584, test=0.04967147490627154, test_tta=0.04862144406870548, best_ep=79)      # seed 0, metric loss (paper record)
PAPER_SM_MSE = dict(val=0.04757395748275777, test=0.04727689949994896, test_tta=0.047052615512338615, best_ep=119)  # seed 0, MSE (paper record)


def ms(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else float("nan"), n=int(len(v)),
                se=float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan"))


def load(pattern):
    recs = []
    for f in sorted(glob.glob(str(pattern))):
        try:
            recs.append(json.loads(Path(f).read_text(encoding="utf-8")))
        except Exception:
            pass
    return recs


# ---------------------------------------------------------------- structural mechanics
sm = [r for r in load(W / "runs" / "structmech" / "*.json") if r.get("kind") == "mlp_scale"]
by_cfg = {}
for r in sm:
    a = r["args"]
    key = (a["loss"], a["width"], a["depth"], a["amp"])
    by_cfg.setdefault(key, {})[a["seed"]] = r
tables = dict(structmech={}, oco2={}, kernelflows={}, meta=dict(n_structmech_records=len(sm)))
md = ["# Scaled experiments, collected " + __import__("time").strftime("%Y-%m-%d %H:%M"), ""]
md += ["## Structural mechanics: residual MLP, widths and depths (test error, percent, mean +- seed sd [n])", "",
       "| loss | width | depth | amp | params | seeds | val | test | test+TTA | best epoch | min/run | GPU util % | VRAM peak MB |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for key in sorted(by_cfg, key=lambda k: (k[0], k[3], k[2], k[1])):
    loss, w, d, amp = key
    rs = by_cfg[key]
    seeds = sorted(rs)
    row = dict(loss=loss, width=w, depth=d, amp=amp, params=rs[seeds[0]]["params"], seeds=seeds,
               val=ms([100 * rs[s]["val"] for s in seeds]), test=ms([100 * rs[s]["test"] for s in seeds]),
               test_tta=ms([100 * rs[s]["test_tta"] for s in seeds]), best_ep=ms([rs[s]["best_ep"] for s in seeds]),
               minutes=ms([rs[s]["minutes_wall"] for s in seeds]),
               gpu_util=ms([rs[s]["gpu_util_mean"] for s in seeds if rs[s].get("gpu_util_mean") is not None]),
               vram_reserved_peak_mb=max(rs[s]["vram_reserved_peak_mb"] for s in seeds),
               per_seed={s: dict(val=100 * rs[s]["val"], test=100 * rs[s]["test"], test_tta=100 * rs[s]["test_tta"],
                                 best_ep=rs[s]["best_ep"]) for s in seeds})
    # paired difference against the paper configuration within seed (same loss, same amp none)
    base = by_cfg.get((loss, 1024, 4, "none"), {})
    common = [s for s in seeds if s in base]
    if common and key != (loss, 1024, 4, "none"):
        for m in ("test", "test_tta", "val"):
            dif = [100 * (rs[s][m] - base[s][m]) for s in common]
            row[f"paired_{m}_minus_w1024d4"] = ms(dif)
            row[f"paired_{m}_minus_w1024d4"]["resolved"] = bool(len(dif) > 1 and abs(np.mean(dif)) > 2 * np.std(dif, ddof=1) / math.sqrt(len(dif)))
    tables["structmech"][f"{loss}_w{w}_d{d}_{amp}"] = row
    f = lambda k: f"{row[k]['mean']:.3f} +- {row[k]['sd']:.3f}" if row[k]["n"] > 1 else f"{row[k]['mean']:.3f}"
    md.append(f"| {loss} | {w} | {d} | {amp} | {row['params']/1e6:.1f}M | {len(seeds)} | {f('val')} | {f('test')} | {f('test_tta')} | "
              f"{row['best_ep']['mean']:.0f} | {row['minutes']['mean']:.1f} | {row['gpu_util']['mean']:.0f} | {row['vram_reserved_peak_mb']} |")
md += ["", f"Paper's seed-0 records for reference: metric loss val {100*PAPER_SM['val']:.3f} test {100*PAPER_SM['test']:.3f} "
       f"test+TTA {100*PAPER_SM['test_tta']:.3f} (best epoch {PAPER_SM['best_ep']}); MSE val {100*PAPER_SM_MSE['val']:.3f} "
       f"test {100*PAPER_SM_MSE['test']:.3f} test+TTA {100*PAPER_SM_MSE['test_tta']:.3f}.", ""]
md += ["### Paired differences against w1024 d4 (same loss, within seed; percent points, mean +- sd; resolved = |mean| > 2 se)", "",
       "| loss | width | depth | amp | n | test+TTA diff | test diff | resolved |", "|---|---|---|---|---|---|---|---|"]
for k, row in tables["structmech"].items():
    if "paired_test_tta_minus_w1024d4" in row:
        pt, p = row["paired_test_tta_minus_w1024d4"], row["paired_test_minus_w1024d4"]
        md.append(f"| {row['loss']} | {row['width']} | {row['depth']} | {row['amp']} | {pt['n']} | {pt['mean']:+.3f} +- {pt['sd']:.3f} | {p['mean']:+.3f} +- {p['sd']:.3f} | {'yes' if pt['resolved'] else 'no'} |")

# ---------------------------------------------------------------- OCO-2
jp = [r for r in load(W / "runs" / "jpl" / "*.json") if r.get("kind") == "jpl_scale"]
by_j = {}
for r in jp:
    a = r["args"]
    by_j.setdefault((a["band"], a["width"], a["depth"], a["amp"], a["head"]), {})[a["seed"]] = r
md += ["", "## OCO-2 emulation: residual MLP means at widths and depths (percent; reduced / radiance; mean +- seed sd)", "",
       "| band | width | depth | head | seeds | flat reduced | flat radiance | weighted reduced | weighted radiance | combined reduced | combined radiance | kernel-flow reduced / radiance | min/run |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for key in sorted(by_j):
    band, w, d, amp, head = key
    rs = by_j[key]; seeds = sorted(rs)
    def g(model, metric):
        return ms([rs[s]["results"][model][metric] for s in seeds if model in rs[s]["results"]])
    row = dict(band=band, width=w, depth=d, amp=amp, head=head, seeds=seeds, params=rs[seeds[0]]["params"],
               mean_flat=dict(reduced=g("mean_flat", "reduced"), radiance=g("mean_flat", "radiance")),
               mean_weighted=dict(reduced=g("mean_weighted", "reduced"), radiance=g("mean_weighted", "radiance")),
               combined=dict(reduced=g("combined", "reduced"), radiance=g("combined", "radiance")),
               kernel_flow=dict(reduced=g("kernel_flow", "reduced"), radiance=g("kernel_flow", "radiance")),
               minutes=ms([rs[s]["minutes_wall"] for s in seeds]))
    if head != "none":
        row["dkr_flat"] = dict(reduced=g("dkr_flat", "reduced"), radiance=g("dkr_flat", "radiance"))
        row["dkr_weighted"] = dict(reduced=g("dkr_weighted", "reduced"), radiance=g("dkr_weighted", "radiance"))
    base = by_j.get((band, 384, 4, "none", head), {})
    common = [s for s in seeds if s in base]
    if common and (w, d) != (384, 4):
        row["paired_flat_reduced_minus_w384d4"] = ms([rs[s]["results"]["mean_flat"]["reduced"] - base[s]["results"]["mean_flat"]["reduced"] for s in common])
        row["paired_weighted_radiance_minus_w384d4"] = ms([rs[s]["results"]["mean_weighted"]["radiance"] - base[s]["results"]["mean_weighted"]["radiance"] for s in common])
    tables["oco2"][f"{band}_w{w}_d{d}_{amp}_{head}"] = row
    f = lambda m: f"{m['mean']:.3f} +- {m['sd']:.3f}" if m["n"] > 1 else f"{m['mean']:.3f}"
    fr = lambda m: f"{m['mean']:.4f} +- {m['sd']:.4f}" if m["n"] > 1 else f"{m['mean']:.4f}"
    md.append(f"| {band} | {w} | {d} | {head} | {len(seeds)} | {f(row['mean_flat']['reduced'])} | {fr(row['mean_flat']['radiance'])} | "
              f"{f(row['mean_weighted']['reduced'])} | {fr(row['mean_weighted']['radiance'])} | {f(row['combined']['reduced'])} | "
              f"{fr(row['combined']['radiance'])} | {row['kernel_flow']['reduced']['mean']:.3f} / {row['kernel_flow']['radiance']['mean']:.4f} | {row['minutes']['mean']:.1f} |")

# ---------------------------------------------------------------- kernel flows
lk = [r for r in load(W / "runs" / "lk" / "*.json") if r.get("kind") == "lk_scale"]
by_l = {}
for r in lk:
    a = r["args"]
    by_l.setdefault((a["space"], a["ntr"], a["steps"], a["bs"], a["rank"]), {})[a["seed"]] = r
md += ["", "## Kernel flows on OCO-2 O2: learned metrics at larger sizes (test error percent, exact ridge; mean +- seed sd)", "",
       "| space | rows | steps | batch | rank | seeds | iso | ard | lowrank | ard - iso | lowrank - iso | min/run |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for key in sorted(by_l):
    space, ntr, steps, bs, rank = key
    rs = by_l[key]; seeds = sorted(rs)
    row = dict(space=space, ntr=ntr, steps=steps, bs=bs, rank=rank, seeds=seeds, minutes=ms([rs[s]["minutes"] for s in seeds]))
    for kind in ("iso", "ard", "lowrank"):
        row[kind] = ms([rs[s]["results"][kind]["test_pct"] for s in seeds if kind in rs[s]["results"]])
    for kind in ("ard", "lowrank"):
        row[f"{kind}_minus_iso"] = ms([rs[s]["results"][kind]["test_pct"] - rs[s]["results"]["iso"]["test_pct"] for s in seeds if kind in rs[s]["results"] and "iso" in rs[s]["results"]])
    tables["kernelflows"][f"{space}_n{ntr}_st{steps}_bs{bs}_r{rank}"] = row
    f = lambda m: (f"{m['mean']:.3f} +- {m['sd']:.3f}" if m["n"] > 1 else f"{m['mean']:.3f}") if m["n"] else "-"
    md.append(f"| {space} | {ntr} | {steps} | {bs} | {rank} | {len(seeds)} | {f(row['iso'])} | {f(row['ard'])} | {f(row['lowrank'])} | "
              f"{f(row['ard_minus_iso'])} | {f(row['lowrank_minus_iso'])} | {row['minutes']['mean']:.1f} |")

(OUT / "scale_tables.json").write_text(json.dumps(tables, indent=1, default=float) + "\n", encoding="utf-8")
(OUT / "scale_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))

# ---------------------------------------------------------------- figures, paper style
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})
C = {"metric": "#3b5bdb", "mse": "#e8590c", "flat": "#3b5bdb", "weighted": "#e8590c", "iso": "#495057", "ard": "#2b8a3e", "lowrank": "#862e9c"}


def series(loss, axis, amp="none"):
    """points (x, mean, sd, n) along widths (depth 4) or depths (width 1024)."""
    pts = []
    for k, row in tables["structmech"].items():
        if row["loss"] != loss or row["amp"] != amp:
            continue
        if axis == "width" and row["depth"] == 4:
            pts.append((row["width"], row["test_tta"]["mean"], row["test_tta"]["sd"], row["test_tta"]["n"]))
        if axis == "depth" and row["width"] == 1024:
            pts.append((row["depth"], row["test_tta"]["mean"], row["test_tta"]["sd"], row["test_tta"]["n"]))
    return sorted(pts)


if tables["structmech"]:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for ax, axis, xl in ((axes[0], "width", "width (depth 4)"), (axes[1], "depth", "depth (width 1024)")):
        for loss, lab in (("metric", "metric loss, 400 epochs"), ("mse", "normalized MSE, 120 epochs")):
            pts = series(loss, axis)
            if not pts:
                continue
            x = np.array([p[0] for p in pts]); m = np.array([p[1] for p in pts]); sd = np.array([np.nan_to_num(p[2]) for p in pts])
            ax.errorbar(x, m, yerr=sd, color=C[loss], marker="o", ms=4, lw=1.2, capsize=2, label=lab)
            for xi, mi, ni in zip(x, m, [p[3] for p in pts]):
                ax.annotate(f"n={ni}", (xi, mi), textcoords="offset points", xytext=(4, 4), fontsize=6, color=C[loss])
        if axis == "width":
            ax.set_xscale("log", base=2); ax.set_xticks([1024, 2048, 4096]); ax.set_xticklabels(["1024", "2048", "4096"])
        else:
            ax.set_xticks([4, 5, 6])
        ax.set(xlabel=xl, ylabel="test error + TTA (%)", title=f"Structural mechanics: test error against {axis}")
        ax.axhline(100 * PAPER_SM["test_tta"], color=C["metric"], ls=":", lw=0.8)
        ax.axhline(100 * PAPER_SM_MSE["test_tta"], color=C["mse"], ls=":", lw=0.8)
        ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_scale_structmech.{ext}", metadata={"CreationDate": None, "ModDate": None} if ext == "pdf" else None)
    plt.close(fig)

if tables["oco2"]:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for ax, metric, model, yl in ((axes[0], "reduced", "mean_flat", "reduced error, flat MLP (%)"), (axes[1], "radiance", "mean_weighted", "radiance error, weighted MLP (%)")):
        for axis, mk, lab in (("width", "o", "widths (depth 4)"), ("depth", "s", "depths (width 384)")):
            pts = []
            for k, row in tables["oco2"].items():
                if row["band"] != "o2" or row["head"] != "none" or row["amp"] != "none":
                    continue
                if axis == "width" and row["depth"] == 4:
                    pts.append((row["width"], row[model][metric]["mean"], row[model][metric]["sd"]))
                if axis == "depth" and row["width"] == 384:
                    pts.append((row["depth"] * 96, row[model][metric]["mean"], row[model][metric]["sd"]))
            pts.sort()
            if pts:
                x = np.array([p[0] for p in pts]); m = np.array([p[1] for p in pts]); sd = np.array([np.nan_to_num(p[2]) for p in pts])
                ax.errorbar(x, m, yerr=sd, marker=mk, ms=4, lw=1.2, capsize=2, label=lab, color=C["flat"] if axis == "width" else C["weighted"])
        ax.set(xlabel="width (depths plotted at 96 x depth)", ylabel=yl, title=f"OCO-2 O2 band: {metric} metric")
        ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_scale_oco2.{ext}", metadata={"CreationDate": None, "ModDate": None} if ext == "pdf" else None)
    plt.close(fig)

if tables["kernelflows"]:
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 3.0))
    keys = sorted(tables["kernelflows"])
    xs = np.arange(len(keys))
    for i, kind in enumerate(("iso", "ard", "lowrank")):
        m = np.array([tables["kernelflows"][k][kind]["mean"] if tables["kernelflows"][k][kind]["n"] else np.nan for k in keys])
        sd = np.array([np.nan_to_num(tables["kernelflows"][k][kind]["sd"]) if tables["kernelflows"][k][kind]["n"] else 0 for k in keys])
        ax.errorbar(xs + 0.25 * (i - 1), m, yerr=sd, fmt="o", ms=4, capsize=2, color=C[kind], label=kind)
    ax.set_xticks(xs); ax.set_xticklabels(keys, rotation=60, ha="right", fontsize=6)
    ax.set(ylabel="test error, exact ridge (%)", title="Kernel flows on OCO-2 O2: learned metric against the isotropic kernel")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_scale_kernelflows.{ext}", metadata={"CreationDate": None, "ModDate": None} if ext == "pdf" else None)
    plt.close(fig)
print("written", OUT)
