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
MAIN = os.path.join(PAPER, "macros.tex")

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

# The complete-schedule six-member campaign (collected/dgx): pipeline records, the
# per-seed member run records, the second-moment record and the kappa records.
DGX = os.path.join(HERE, "collected", "dgx")
six = {int(os.path.basename(f).split("_s")[-1][:-5]): json.load(open(f, encoding="utf-8"))
       for f in glob.glob(os.path.join(DGX, "sm_seed_s*.json"))}
if len(six) == 10:
    print("SIX-MEMBER CAMPAIGN")
    chk("errSix", 100 * st.mean([six[s]["corr_test"] for s in S]), 3)
    chk("errSixSd", 100 * st.stdev([six[s]["corr_test"] for s in S]), 3)
    chk("errSixStack", 100 * st.mean([six[s]["stack_test"] for s in S]), 3)
    chk("errSixStackSd", 100 * st.stdev([six[s]["stack_test"] for s in S]), 3)
    cg6 = [100 * (six[s]["stack_test"] - six[s]["corr_test"]) for s in S]
    chk("corrGainSix", st.mean(cg6), 3)
    chk("corrGainSixSd", st.stdev(cg6), 3)
    chk("errSixGlob", 100 * st.mean([six[s]["hstk"]["stack"]["test"] for s in S]), 3)
    chk("errSixGlobSd", 100 * st.stdev([six[s]["hstk"]["stack"]["test"] for s in S]), 3)
    chk("errSixGlobCorr", 100 * st.mean([six[s]["hstk"]["plus_corr"]["test"] for s in S]), 3)
    chk("errSixGlobCorrSd", 100 * st.stdev([six[s]["hstk"]["plus_corr"]["test"] for s in S]), 3)
    g6 = [100 * (six[s]["corr_test"] - pix5[s]["final_test"]) for s in S]
    chk("sixGain", -st.mean(g6), 3)
    chk("sixGainSd", st.stdev(g6), 3)

    def run(s, pat):
        f = glob.glob(os.path.join(DGX, "runs", "sm_s%d" % s, pat))
        return json.load(open(f[0], encoding="utf-8")) if f else None

    def tta(rec):
        return rec.get("test_tta", rec.get("test_mirror_tta", rec["test"]))
    fno = [run(s, "fno_s%d_w64_m14_L4_mir.json" % s) for s in S]
    unet = [run(s, "unet_s%d_w48_mir.json" % s) for s in S]
    if all(fno) and all(unet):
        chk("errFNOc", 100 * st.mean(tta(r) for r in fno), 3)
        chk("errFNOcSd", 100 * st.stdev(tta(r) for r in fno), 3)
        chk("errUNet", 100 * st.mean(tta(r) for r in unet), 3)
        chk("errUNetSd", 100 * st.stdev(tta(r) for r in unet), 3)
        fg = [100 * (ens5[s]["per_member_test"]["fno"] - tta(fno[s])) for s in S]
        chk("fnoCompleteGain", st.mean(fg), 3)
        chk("fnoCompleteGainSd", st.stdev(fg), 3)
        members = {s: dict(mlp=tta(run(s, "mlp_s%d_*_mir.json" % s)), mlpMSE=tta(run(s, "mlpMSE_s%d_*_mir.json" % s)),
                           mlpR=tta(run(s, "mlpR_s%d_*.json" % s)), fno=tta(fno[s]), unet=tta(unet[s]),
                           krr=run(s, "krr_full_matern52_n19000.json")["test"]) for s in S}
        e2e6 = [100 * (min(members[s].values()) - six[s]["corr_test"]) for s in S]
        chk("errEndToEndSix", st.mean(e2e6), 3)
        chk("errEndToEndSixSd", st.stdev(e2e6), 3)
    # the same five members without the UNet (hpix5 / hstk5): where the half-split rule
    # declined the per-pixel weights, the deployed pipeline is the corrected global stack
    five = {}
    for s in S:
        h = run(s, "hpix5_corr.json"); g = run(s, "hstk5.json")
        if h is not None:
            five[s] = h["report"]["final_test"]
        elif g is not None:
            five[s] = g["report"]["final_test"]
    if len(five) == 10:
        ug = [100 * (five[s] - six[s]["corr_test"]) for s in S]
        chk("unetGain", st.mean(ug), 3)
        chk("unetGainSd", st.stdev(ug), 3)
        chk("errFiveC", 100 * st.mean(five[s] for s in S), 3)
        chk("errFiveCSd", 100 * st.stdev(five[s] for s in S), 3)
        wash = [100 * (five[s] - pix5[s]["final_test"]) for s in S]
        chk("fnoScheduleWash", st.mean(wash), 3)
        chk("fnoScheduleWashSd", st.stdev(wash), 3)
        sm5c = os.path.join(HERE, "collected", "secmom5c_seeded.json")
        if os.path.exists(sm5c):
            rec5 = json.load(open(sm5c, encoding="utf-8"))
            chk("predRMSFiveC", 100 * st.mean(r["pred_rms"] for r in rec5), 3)
    else:
        print("  unetGain: %d of 10 seeds present, not checked" % len(five))
    cov90 = [100 * six[s]["uq"]["a0.1"]["scaled"]["coverage"] for s in S]
    cov95 = [100 * six[s]["uq"]["a0.05"]["scaled"]["coverage"] for s in S]
    chk("uqCoverSix", st.mean(cov90), 1)
    chk("uqCoverSixSd", st.stdev(cov90), 1)
    chk("uqCoverSixHi", st.mean(cov95), 1)
    chk("uqCoverSixHiSd", st.stdev(cov95), 1)
    sm6 = os.path.join(HERE, "collected", "secmom6_seeded.json")
    if os.path.exists(sm6):
        rec = json.load(open(sm6, encoding="utf-8"))
        chk("predRMSSix", 100 * st.mean(r["pred_rms"] for r in rec), 3)
        chk("predRMSSixSd", 100 * st.stdev(r["pred_rms"] for r in rec), 3)
        chk("dispFacSix", st.mean(r["disp_factor"] for r in rec), 3)
        chk("dispFacSixSd", st.stdev(r["disp_factor"] for r in rec), 3)
        chk("corrBarSix", st.mean(r["rho_mean"] for r in rec), 3)
        chk("corrBarSixSd", st.stdev(r["rho_mean"] for r in rec), 3)
        chk("ensFloorSix", 100 * st.mean(r["floor"] for r in rec), 3)
        chk("ensFloorSixSd", 100 * st.stdev(r["floor"] for r in rec), 3)
        chk("errSixEq", 100 * st.mean(r["equal_weight"]["test"] for r in rec), 3)
        chk("errSixEqSd", 100 * st.stdev(r["equal_weight"]["test"] for r in rec), 3)
        chk("corrNetLoSix", min(r["rho_min"] for r in rec), 2)
        chk("corrNetHiSix", max(r["rho_max"] for r in rec), 2)
    kap = {}
    for f in glob.glob(os.path.join(HERE, "collected", "*", "a5_kappa_s*.json")):
        r = json.load(open(f, encoding="utf-8"))
        if r.get("seed", 99) < 90:
            kap.setdefault(r["seed"], r["kappa"])
    if len(kap) == 10:
        kv = [kap[s] for s in S]
        chk("kappaTen", st.mean(kv), 1)
        chk("kappaTenSd", st.stdev(kv), 1)
        chk("kappaTenLo", min(kv), 1)
        chk("kappaTenHi", max(kv), 1)
else:
    print("SIX-MEMBER CAMPAIGN: %d of 10 seeds collected, macros not checked" % len(six))

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
    oco_src = (open(os.path.join(PAPER, "oco2.tex"), encoding="utf-8").read()
               + open(os.path.join(PAPER, "supp_oco2.tex"), encoding="utf-8").read())
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
