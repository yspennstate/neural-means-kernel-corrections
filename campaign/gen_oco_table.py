"""Aggregate the a2 OCO seed campaign and emit the tab:oco2 rows.

Reads oco_{band}_s{0..9}.json (the per-band-seed result files collected from
the campaign) and prints (i) the O2 table body as LaTeX, (ii) the per-band
summary numbers quoted in the text, and (iii) a flat JSON that
audit_reported_macros.py can check the manuscript against.

Usage: python campaign/gen_oco_table.py <results_dir> [out_json]
"""
import json
import os
import statistics as st
import sys

BANDS = ["o2", "wco2", "sco2"]
SEEDS = range(10)
# (result key, table label, frozen control?)
ROWS = [
    ("kernel_raw",  r"Mat\'ern kernel, raw input, one length scale", False),
    ("kernel_ard",  r"Mat\'ern kernel, raw input, sensitivity-scaled (ARD)", False),
    ("kernel_flow", r"kernel-flow emulator \citep{susiluoto2025radiative}", True),
    ("mean_flat",   r"residual MLP, flat metric", False),
    ("mean_wnum",   r"residual MLP, weighted-coefficient loss", False),
    ("mean_radx",   r"residual MLP, exact radiance loss", False),
    ("dkr_flat",    r"Mat\'ern head on the flat network's features", False),
    ("dkr_wnum",    r"Mat\'ern head on the weighted network's features", False),
    ("dkr_radx",    r"Mat\'ern head on the radiance network's features", False),
    ("combined",    r"per-coordinate combination", False),
]


def load(results_dir):
    data = {}
    for band in BANDS:
        per_seed = {}
        for s in SEEDS:
            path = os.path.join(results_dir, f"oco_{band}_s{s}.json")
            with open(path, encoding="utf-8") as f:
                per_seed[s] = json.load(f)
        data[band] = per_seed
    return data


def stats(per_seed, key, metric):
    vals = [d["results"][key][metric] for d in per_seed.values()]
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0), vals


def fmt_red(mean, sd, frozen):
    if frozen:
        return f"{mean:.2f}\\%"
    return f"${mean:.2f}\\pm{sd:.2f}$\\%"


def fmt_rad(mean, sd, frozen):
    prec = 4 if mean < 0.1 else 3
    if frozen:
        return f"{mean:.{prec}f}\\%"
    return f"${mean:.{prec}f}\\pm{sd:.{prec}f}$\\%"


def main():
    results_dir = sys.argv[1]
    out_json = sys.argv[2] if len(sys.argv) > 2 else None
    data = load(results_dir)
    report = {}

    print("% ---- tab:oco2 body (O2 band, ten seeds, matched 250-epoch budget) ----")
    for key, label, frozen in ROWS:
        rm, rs, _ = stats(data["o2"], key, "reduced")
        am, asd, avals = stats(data["o2"], key, "radiance")
        if key == "kernel_ard":
            # validation ties on the reduced metric between two hyperparameter
            # configurations whose radiance differs 2.2x; a mean would average
            # the mixture, so the table prints the split in the caption instead
            rad_cell = "---"
        else:
            rad_cell = fmt_rad(am, asd, frozen)
        bold = key == "combined"
        red_cell = fmt_red(rm, rs, frozen)
        if bold:
            red_cell = f"\\textbf{{{red_cell}}}"
            rad_cell = f"\\textbf{{{rad_cell}}}"
        print(f"{label} & {red_cell} & {rad_cell} \\\\")
        report[f"o2.{key}.reduced"] = [round(rm, 3), round(rs, 3)]
        report[f"o2.{key}.radiance"] = [round(am, 4), round(asd, 4)]

    print()
    print("% ---- ARD selection split (o2) ----")
    ard = [(d["hyper"]["kernel_ard"]["scale"], d["results"]["kernel_ard"]["radiance"])
           for d in data["o2"].values()]
    for scale in sorted({a[0] for a in ard}):
        vals = [r for s_, r in ard if s_ == scale]
        print(f"%   scale={scale}: {len(vals)} seeds, radiance mean {st.mean(vals):.4f}")
        report[f"o2.kernel_ard.radiance.scale{scale:g}"] = [round(st.mean(vals), 4), len(vals)]

    print()
    print("% ---- per-band: combined vs emulator, radx vs wnum ----")
    for band in BANDS:
        per_seed = data[band]
        kf_red = per_seed[0]["results"]["kernel_flow"]["reduced"]
        kf_rad = per_seed[0]["results"]["kernel_flow"]["radiance"]
        cm_red, cs_red, _ = stats(per_seed, "combined", "reduced")
        cm_rad, cs_rad, _ = stats(per_seed, "combined", "radiance")
        wins = sum(1 for d in per_seed.values()
                   if d["results"]["combined"]["reduced"] < kf_red
                   and d["results"]["combined"]["radiance"] < kf_rad)
        rx = sum(1 for d in per_seed.values()
                 if d["results"]["mean_radx"]["radiance"] < d["results"]["mean_wnum"]["radiance"])
        dx = sum(1 for d in per_seed.values()
                 if d["results"]["dkr_radx"]["radiance"] < d["results"]["dkr_wnum"]["radiance"])
        print(f"% {band}: combined {cm_red:.2f}+-{cs_red:.2f} / {cm_rad:.4f}+-{cs_rad:.4f} "
              f"vs emulator {kf_red:.2f} / {kf_rad:.4f}; beats both at {wins}/10; "
              f"mean_radx<mean_wnum radiance {rx}/10, dkr {dx}/10")
        report[f"{band}.combined.reduced"] = [round(cm_red, 3), round(cs_red, 3)]
        report[f"{band}.combined.radiance"] = [round(cm_rad, 4), round(cs_rad, 4)]
        report[f"{band}.kernel_flow.reduced"] = [round(kf_red, 3), 0.0]
        report[f"{band}.kernel_flow.radiance"] = [round(kf_rad, 4), 0.0]
        report[f"{band}.combined_beats_emulator_both"] = wins
        report[f"{band}.radx_beats_wnum.mean"] = rx
        report[f"{band}.radx_beats_wnum.dkr"] = dx
        for key in ("mean_flat", "dkr_flat", "mean_wnum", "dkr_wnum", "mean_radx", "dkr_radx"):
            m, sd_, _ = stats(per_seed, key, "reduced")
            report[f"{band}.{key}.reduced"] = [round(m, 3), round(sd_, 3)]
            m, sd_, _ = stats(per_seed, key, "radiance")
            report[f"{band}.{key}.radiance"] = [round(m, 4), round(sd_, 4)]

    # 750-epoch diagnostic, if present
    p900 = os.path.join(results_dir, "oco_o2_s900.json")
    if os.path.exists(p900):
        with open(p900, encoding="utf-8") as f:
            d900 = json.load(f)
        print()
        print(f"% ---- 750-epoch diagnostic (o2, one seed) ----")
        for key in ("mean_flat", "dkr_flat", "mean_wnum", "dkr_wnum", "mean_radx",
                    "dkr_radx", "combined"):
            r = d900["results"][key]
            print(f"%   {key}: {r['reduced']:.3f} / {r['radiance']:.4f}")
            report[f"o2_ep750.{key}.reduced"] = round(r["reduced"], 3)
            report[f"o2_ep750.{key}.radiance"] = round(r["radiance"], 4)

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, sort_keys=True)
        print(f"% wrote {out_json}")


if __name__ == "__main__":
    main()
