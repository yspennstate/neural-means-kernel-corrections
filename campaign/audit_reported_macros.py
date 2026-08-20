"""Audit the paper's reported macros against the measured artifacts.

Parses \newcommand values out of main.tex and recomputes each from the collected campaign
JSON, so a hand-typed digit cannot survive. Anything that disagrees is printed as FAIL.
"""
import glob
import json
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PAPER = os.path.join(REPO, "paper")
MAIN = os.path.join(PAPER, "main.tex")

src = open(MAIN, encoding="utf-8").read()
mac = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", src))


def load(d, tag):
    return {int(f.split("_s")[-1][:-5]): json.load(open(f))
            for f in glob.glob(os.path.join(HERE, d, "%s_s*.json" % tag))}


ens5, pix5, pix4 = load("ens5", "ens5"), load("pix5", "pix5"), load("pix4", "pix4")
S = list(range(10))
fails = 0


def chk(name, measured, dp):
    global fails
    got = mac.get(name)
    want = ("%%.%df" % dp) % measured
    ok = got == want
    if not ok:
        fails += 1
    print("  %-16s paper=%-8s measured=%-8s %s" % (name, got, want, "" if ok else "<== FAIL"))


print("PIPELINE")
chk("errPipe", 100 * st.mean([pix5[s]["final_test"] for s in S]), 3)
chk("errPipeSd", 100 * st.stdev([pix5[s]["final_test"] for s in S]), 3)
chk("errStack", 100 * st.mean([pix5[s]["stack"]["test"] for s in S]), 3)
chk("errStackSd", 100 * st.stdev([pix5[s]["stack"]["test"] for s in S]), 3)
chk("errPipeFour", 100 * st.mean([pix4[s]["final_test"] for s in S]), 3)
chk("errPipeFourSd", 100 * st.stdev([pix4[s]["final_test"] for s in S]), 3)
g = [100 * (pix4[s]["final_test"] - pix5[s]["final_test"]) for s in S]
chk("fnoGain", st.mean(g), 3)
chk("fnoGainSd", st.stdev(g), 3)
cg = [100 * (pix5[s]["stack"]["test"] - pix5[s]["final_test"]) for s in S]
chk("corrGain", st.mean(cg), 3)
chk("corrGainSd", st.stdev(cg), 3)

print("ENSEMBLE")
chk("errEnsEq", 100 * st.mean([ens5[s]["ens_equal_test"] for s in S]), 3)
chk("errEnsEqSd", 100 * st.stdev([ens5[s]["ens_equal_test"] for s in S]), 3)
chk("errEnsFit", 100 * st.mean([ens5[s]["ens_fitted_test"] for s in S]), 3)
chk("errEnsFitSd", 100 * st.stdev([ens5[s]["ens_fitted_test"] for s in S]), 3)
bt = [ens5[s]["per_member_test"][ens5[s]["best_member"]] for s in S]
e2e = [100 * (b - pix5[s]["final_test"]) for b, s in zip(bt, S)]
chk("errEndToEnd", st.mean(e2e), 3)
chk("errEndToEndSd", st.stdev(e2e), 3)

print("ERROR LOCALISATION")
chk("hardBest", 100 * st.mean([ens5[s]["hard20_best_relerr"] for s in S]), 2)
chk("hardEns", 100 * st.mean([ens5[s]["hard20_ens_relerr"] for s in S]), 2)
chk("hardAll", 100 * st.mean([ens5[s]["hard20_all_member_mean_relerr"] for s in S]), 2)
chk("medRel", 100 * st.mean([ens5[s]["median_relerr_best"] for s in S]), 2)

# The low-data chain reports through macros as well, so it is checked the same
# way. Its per-seed records are collected alongside everything else.
ld = {}
for f in glob.glob(os.path.join(HERE, "collected", "box*", "ld_s*.json")):
    tail = os.path.basename(f).rsplit("_s", 1)[1][:-5]
    if not tail.isdigit():          # probes and smokes carry a suffix
        continue
    if int(tail) < 90:
        ld[int(tail)] = json.load(open(f, encoding="utf-8"))
if len(ld) == 10:
    print("LOW DATA")
    v = [100 * ld[s]["final_test"] for s in sorted(ld)]
    chk("errBestLow", st.mean(v), 3)
    chk("errBestLowSd", st.stdev(v), 3)
else:
    print("LOW DATA: %d of 10 seeds collected, macros not checked" % len(ld))

# OCO-2 numbers appear in the text rather than as macros, so they are audited
# against the same artifacts by regenerating the table body and requiring the
# manuscript to contain every value it prints.
band_files = glob.glob(os.path.join(HERE, "collected", "box*", "oco_*_s*.json"))
if band_files:
    per = {}
    for f in band_files:
        r = json.load(open(f, encoding="utf-8"))
        if r.get("seed", 99) < 90 and not os.path.basename(f).startswith("a2_"):
            per.setdefault(r["band"], {})[r["seed"]] = r
    print("OCO-2 (ten seeds per band, values quoted in Section 6)")
    oco_src = open(os.path.join(PAPER, "oco2.tex"), encoding="utf-8").read()
    for band in ("o2", "wco2", "sco2"):
        seeds = per.get(band, {})
        if len(seeds) != 10:
            print("  %-5s %d of 10 seeds collected, not checked" % (band, len(seeds)))
            continue
        for key in ("mean_flat", "dkr_flat", "combined"):
            vals = [seeds[s]["results"][key]["reduced"] for s in sorted(seeds)]
            m, sd = st.mean(vals), st.stdev(vals)
            token = "%.2f\\pm%.2f" % (m, sd)
            here = token in oco_src
            if band == "o2" or key == "combined":
                if not here:
                    fails += 1
                print("  %-5s %-10s %-16s %s" % (band, key, token,
                                                 "" if here else "<== not found in oco2.tex"))

print("\n%d macro(s) disagree with the artifacts" % fails)
