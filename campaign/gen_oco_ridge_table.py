"""The frozen-feature readout control for the OCO-2 kernel heads (tab:oco2ridge).

Reads the rerun of the seeded OCO-2 campaign that carries the ridge rows
(campaign/jpl_seeded.py with ridge_head; oco_{band}_s{0..9}.json under one
results directory) and prints, per band, the network, the ridge readout refit
on its frozen last-layer features, and the exact Matern head on the same
features, for the flat and the weighted training losses; the per-seed win
counts of the head over the readout; and, as a check that the rerun is the
campaign, the rows the published table also carries, next to the published
values from a second results directory when one is given.

Usage: python campaign/gen_oco_ridge_table.py <rerun_dir> [<campaign_collected_glob>] [out_json]
"""
import glob
import json
import os
import statistics as st
import sys

BANDS = ["o2", "wco2", "sco2"]
BAND_LABEL = {"o2": "O2", "wco2": "WCO2", "sco2": "SCO2"}
SEEDS = range(10)
ROWS = [("mean_flat", "network, flat loss"),
        ("ridge_flat", r"\quad ridge readout on its frozen features"),
        ("dkr_flat", r"\quad Mat\'ern head on its frozen features"),
        ("mean_wnum", "network, weighted-coefficient loss"),
        ("ridge_wnum", r"\quad ridge readout on its frozen features"),
        ("dkr_wnum", r"\quad Mat\'ern head on its frozen features")]


def load(results_dir):
    data = {}
    for band in BANDS:
        data[band] = {}
        for s in SEEDS:
            p = os.path.join(results_dir, f"oco_{band}_s{s}.json")
            if os.path.exists(p):
                data[band][s] = json.load(open(p, encoding="utf-8"))
    return data


def stats(per_seed, key, metric):
    v = [d["results"][key][metric] for d in per_seed.values() if key in d["results"]]
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v)) if v else (float("nan"), float("nan"), 0)


def cell(m, s, metric):
    prec = 2 if metric == "reduced" else (4 if m < 0.1 else 3)
    return f"${m:.{prec}f}\\pm{s:.{prec}f}$\\%"


def main():
    rerun = load(sys.argv[1])
    pub_glob = next((a for a in sys.argv[2:] if "*" in a), None)
    out_json = next((a for a in sys.argv[2:] if a.endswith(".json") and "*" not in a), None)
    report = {}
    print("% ---- tab:oco2ridge: frozen-feature readout control, ten seeds per band (complete tabular) ----")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("model & reduced & radiance \\\\")
    print("\\midrule")
    for band in BANDS:
        ps = rerun[band]
        n = len(ps)
        print(f"\\multicolumn{{3}}{{l}}{{\\emph{{{BAND_LABEL[band]} band}} ({n} seeds)}} \\\\")
        for key, label in ROWS:
            rm, rs, _ = stats(ps, key, "reduced")
            am, asd, _ = stats(ps, key, "radiance")
            print(f"{label} & {cell(rm, rs, 'reduced')} & {cell(am, asd, 'radiance')} \\\\")
            report[f"{band}.{key}.reduced"] = [round(rm, 3), round(rs, 3)]
            report[f"{band}.{key}.radiance"] = [round(am, 4), round(asd, 4)]
        # paired counts on the reduced metric
        for mode in ("flat", "wnum"):
            head_lt_ridge = sum(d["results"][f"dkr_{mode}"]["reduced"] < d["results"][f"ridge_{mode}"]["reduced"] for d in ps.values())
            ridge_lt_net = sum(d["results"][f"ridge_{mode}"]["reduced"] < d["results"][f"mean_{mode}"]["reduced"] for d in ps.values())
            gap = [d["results"][f"ridge_{mode}"]["reduced"] - d["results"][f"dkr_{mode}"]["reduced"] for d in ps.values()]
            print(f"% {band} {mode}: head beats ridge at {head_lt_ridge}/{n} seeds; ridge beats network at {ridge_lt_net}/{n}; "
                  f"ridge minus head {st.mean(gap):.3f} +- {st.stdev(gap) if n > 1 else 0:.3f} points")
            report[f"{band}.{mode}.head_beats_ridge"] = [head_lt_ridge, n]
            report[f"{band}.{mode}.ridge_beats_network"] = [ridge_lt_net, n]
            report[f"{band}.{mode}.ridge_minus_head"] = [round(st.mean(gap), 4), round(st.stdev(gap) if n > 1 else 0, 4)]
        cpr = stats(ps, "combined_plus_ridge", "reduced"); cmb = stats(ps, "combined", "reduced")
        print(f"% {band}: combined {cmb[0]:.3f} vs combined_plus_ridge {cpr[0]:.3f} (reduced)")
        report[f"{band}.combined_plus_ridge.reduced"] = [round(cpr[0], 3), round(cpr[1], 3)]
    print("\\bottomrule")
    print("\\end{tabular}")
    if pub_glob:
        print("% ---- rerun against the published campaign rows (mean over seeds) ----")
        pub = {}
        for f in glob.glob(pub_glob):
            J = json.load(open(f, encoding="utf-8")); pub.setdefault(J["band"], {})[int(J["seed"])] = J
        for band in BANDS:
            for key in ("mean_flat", "dkr_flat", "mean_wnum", "dkr_wnum", "combined"):
                a = stats(rerun[band], key, "reduced"); b = stats(pub.get(band, {}), key, "reduced")
                print(f"% {band} {key}: rerun {a[0]:.3f}+-{a[1]:.3f} (n={a[2]})  published {b[0]:.3f}+-{b[1]:.3f} (n={b[2]})")
                report[f"{band}.{key}.rerun_vs_published"] = [round(a[0], 3), round(b[0], 3)]
    if out_json:
        json.dump(report, open(out_json, "w", encoding="utf-8"), indent=1)
        print(f"% wrote {out_json}")


if __name__ == "__main__":
    main()
