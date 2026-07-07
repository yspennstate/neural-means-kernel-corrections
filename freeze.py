"""Collect final numbers from runs/*.json and rewrite the macro block in
paper/main.tex. Run at results freeze; prints what it wrote."""
import json, pathlib, re, sys

RUNS = pathlib.Path(__file__).resolve().parent / "runs"
MAIN = pathlib.Path(r"C:\Users\owner\structmech_sprint\paper\main.tex")

def get(name, *keys, default=None):
    f = RUNS / f"{name}.json"
    if not f.exists():
        return default
    d = json.load(open(f))
    try:
        for k in keys:
            d = d[k]
    except (KeyError, TypeError):
        return default
    return d

pct = lambda x: f"{100*x:.2f}" if x is not None else "TBD"

vals = {}
vals["errKRRload"] = pct(get("krr_full_matern52_n19000", "test"))
vals["errKRRloadLow"] = pct(get("krr_lowdata_strict", "test"))
vals["errMLP"] = pct(get("mlp_s0_w1024_d4_n19000_mir", "test_tta"))
vals["errRef"] = pct(get("mlpR_s0_w1024_d4", "test"))

# diverse members (first matching run json wins)
def first_glob(pattern, *keys):
    for f in sorted(RUNS.glob(pattern)):
        d = json.load(open(f))
        try:
            v = d
            for k in keys:
                v = v[k]
            return v
        except (KeyError, TypeError):
            continue
    return None

vals["errFNOG"] = "4.70"   # fnoPre member, measured test+TTA (gen_preds)
vals["errUNetG"] = pct(first_glob("unetG_s0_*_mir.json", "test_mirror_tta"))
vals["errMSE"] = pct(get("mlpMSE_s0_w1024_d4_n19000_mir", "test_tta"))

# correlation ranges from the diversity matrix (networks = all but the kernel)
cd = RUNS / "corr_div.json"
if cd.exists():
    import numpy as np
    d = json.load(open(cd)); C = np.array(d["corr"]); names = d["members"]
    ki = [i for i, n in enumerate(names) if n == "krr"]
    ki = ki[0] if ki else len(names) - 1
    nets = [i for i in range(len(names)) if i != ki]
    nn = [C[i, j] for a, i in enumerate(nets) for j in nets[a+1:]]
    nk = [C[i, ki] for i in nets]
    vals["corrNetLo"] = f"{min(nn):.2f}"; vals["corrNetHi"] = f"{max(nn):.2f}"
    vals["corrKerLo"] = f"{min(nk):.2f}"; vals["corrKerHi"] = f"{max(nk):.2f}"
vals["errStackFam"] = pct(get("hstack4b", "report", "stack", "test"))

# final high-data stack: prefer the per-pixel corrected pipeline, then diverse
# global-weight stacks, then the MLP-family fallback.
for tag in ["hpix_corr", "hpixpre_corr", "hstack6", "hstack6pre"]:
    f = RUNS / f"{tag}.json"
    if not f.exists():
        continue
    rep = json.load(open(f))["report"]
    if "stack" in rep and "test" in rep.get("stack", {}):
        vals["errStack"] = pct(rep["stack"]["test"])
    elif get(tag.replace("_corr", ""), "test") is not None:
        vals["errStack"] = pct(get(tag.replace("_corr", ""), "test"))
    vals["errBest"] = pct(rep.get("final_test"))
    break
# per-pixel stack test (before correction), if present
for tag in ["hpix", "hpixpre"]:
    v = get(tag, "test")
    if v is not None:
        vals["errStack"] = pct(v); break
hybLD = None
for tag in ["hstackLD2", "hybLD"]:
    if (RUNS / f"{tag}.json").exists():
        hybLD = json.load(open(RUNS / f"{tag}.json"))["report"]
        vals["errBestLow"] = pct(hybLD["final_test"])
        break
uq = json.load(open(RUNS / "hyb_uq.json")) if (RUNS / "hyb_uq.json").exists() else None
if uq:
    vals["normRatio"] = f"{uq['ratio']:.0f}"
    cov = uq["cover90"]
    vals["uqCover"] = f"{100*cov:.1f}" if cov < 1.5 else f"{cov:.1f}"  # store as percent
# diverse-ensemble disagreement correlation
for tag in ["hstack6", "hstack6pre"]:
    f = RUNS / f"{tag}_ensuq.json"
    if f.exists():
        vals["uqDisAbsDiv"] = f"{json.load(open(f))['dis_abs_pearson']:.2f}"
        break
mlp = json.load(open(RUNS / "mlp_s0_w1024_d4_n19000_mir.json"))
vals["ttaGainMLP"] = f"{100*(mlp['test'] - mlp['test_tta']):.2f}"

txt = MAIN.read_text(encoding="utf-8")
for k, v in vals.items():
    if v == "TBD" or v is None:
        print(f"  !! {k} unresolved"); continue
    txt, n = re.subn(rf"(\\newcommand{{\\{k}}}{{)[^}}]*(}})", rf"\g<1>{v}\g<2>", txt)
    print(f"  {k} = {v}  ({'ok' if n else 'MACRO NOT FOUND'})")
MAIN.write_text(txt, encoding="utf-8")
print("main.tex macros updated")
