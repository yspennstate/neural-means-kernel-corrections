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
        smA = os.path.join(HERE, "collected", "secmom_seeded5.json")
        if os.path.exists(sm5c) and os.path.exists(smA):
            rec5 = json.load(open(sm5c, encoding="utf-8"))
            recA = json.load(open(smA, encoding="utf-8"))
            chk("predRMSFiveC", 100 * st.mean(r["pred_rms"] for r in rec5), 3)
            # Proposition exchange at the measured values: V = e1^2 e2^2 (1-rho^2)/(e1^2+e2^2-2 rho e1 e2)
            def V(e1, e2, rho):
                return e1 * e1 * e2 * e2 * (1 - rho * rho) / (e1 * e1 + e2 * e2 - 2 * rho * e1 * e2)
            eA1 = st.mean(r["member_err"]["mlpMSE"] for r in recA); eA2 = st.mean(r["member_err"]["fno"] for r in recA)
            chk("fnoRmsA", 100 * eA2, 3)
            chk("tRatioA", eA2 / eA1, 3)
            chk("vTwoA", 100 * V(eA1, eA2, 0.96) ** 0.5, 2)
            chk("exchRateA", (1 - 0.96 ** 2) / (eA2 / eA1 - 0.96), 1)
            nm = rec5[0]["members"]; i1, i2 = nm.index("mlpMSE"), nm.index("fno")
            eB2 = st.mean(r["member_err"]["fno"] for r in rec5)
            chk("fnoRmsB", 100 * eB2, 3)
            chk("fnoImprove", 100 * (eA2 - eB2) / eA2, 1)
            chk("tRatioB", st.mean(r["member_err"]["fno"] / r["member_err"]["mlpMSE"] for r in rec5), 3)
            chk("rhoFnoB", st.mean(r["C"][i1][i2] for r in rec5), 3)
            chk("vTwoB", 100 * st.mean(V(r["member_err"]["mlpMSE"], r["member_err"]["fno"], r["C"][i1][i2]) ** 0.5 for r in rec5), 2)
            chk("exchRateB", st.mean((1 - r["C"][i1][i2] ** 2) / (r["member_err"]["fno"] / r["member_err"]["mlpMSE"] - r["C"][i1][i2]) for r in rec5), 1)
    else:
        print("  unetGain: %d of 10 seeds present, not checked" % len(five))
    sa_path = os.path.join(DGX, "seedarch.json")
    if os.path.exists(sa_path):
        import itertools
        sa = json.load(open(sa_path, encoding="utf-8"))
        A_ = sa["arch"]; nm_ = sa["names"]
        chk("saWithinLo", min(v["min"] for v in sa["within"].values()), 2)
        chk("saWithinHi", max(v["max"] for v in sa["within"].values()), 2)
        chk("saWithinMean", st.mean(v["mean"] for v in sa["within"].values()), 2)
        chk("saBetweenLo", min(v["min"] for v in sa["between"].values()), 2)
        chk("saBetweenHi", max(v["max"] for v in sa["between"].values()), 2)
        chk("saBetweenMean", sa["rho_b_mean"], 2)
        sc = sa["seed_curves"]
        chk("saFnoOne", 100 * sc["fno"]["1"]["e1_mean"], 3)
        chk("saFnoOneRms", 100 * sc["fno"]["1"]["e2_mean"], 3)
        chk("saFnoTen", 100 * sc["fno"]["10"]["e1_mean"], 3)
        chk("saFnoTenRms", 100 * sc["fno"]["10"]["e2_mean"], 3)
        chk("saMseOne", 100 * sc["mlpMSE"]["1"]["e1_mean"], 3)
        chk("saMseTen", 100 * sc["mlpMSE"]["10"]["e1_mean"], 3)
        chk("saMlpGain", 100 * (sc["mlp"]["1"]["e1_mean"] - sc["mlp"]["10"]["e1_mean"]), 2)
        e6 = [r["e1"] for r in sa["six_arch_one_seed_equal"]]; r6 = [r["e2"] for r in sa["six_arch_one_seed_equal"]]
        chk("saSixEq", 100 * st.mean(e6), 3); chk("saSixEqSd", 100 * st.stdev(e6), 3)
        chk("saSixEqRms", 100 * st.mean(r6), 3); chk("saSixEqRmsSd", 100 * st.stdev(r6), 3)
        c6 = [r["e1"] for r in sa["six_arch_one_seed_convex_cal_to_ev"]]
        chk("saSixConvex", 100 * st.mean(c6), 3); chk("saSixConvexSd", 100 * st.stdev(c6), 3)
        chk("saSixtyEq", 100 * sa["sixty_equal"]["e1"], 3); chk("saSixtyEqRms", 100 * sa["sixty_equal"]["e2"], 3)
        chk("saSixtyConvex", 100 * sa["sixty_convex_cal_to_ev"]["e1"], 3); chk("saSixtyConvexRms", 100 * sa["sixty_convex_cal_to_ev"]["e2_ev"], 3)
        chk("saSixtyOracle", 100 * sa["sixty_convex_oracle_ev"]["e1"], 3); chk("saSixtyOracleRms", 100 * sa["sixty_convex_oracle_ev"]["e2"], 3)
        chk("saFiftyEq", 100 * sa["fifty_equal_no_unet"]["e1"], 3)
        chk("saPerpixMeans", 100 * sa["perpixel_six_seed_means"], 3)
        chk("saPerpixSeedZero", 100 * sa["perpixel_six_seed0"], 3)
        chk("saPerpixSixty", 100 * sa["perpixel_sixty"], 3)
        h = sa["hard_tail_1pct"]
        chk("saHardSingle", 100 * h["best_single"], 2); chk("saHardTen", 100 * h["best_arch_ten_seeds"], 2)
        chk("saHardSixty", 100 * h["sixty_equal"], 2); chk("saHardMedian", 100 * h["median_best_single"], 2)
        # Theorem blockfloor with the pool's minima
        Smat = sa["S_ev"]; arch_of = [A_.index(n.split("_s")[0]) for n in nm_]; N_ = len(nm_)
        e2min = min(Smat[i][i] for i in range(N_))
        win = [Smat[p][q] for p in range(N_) for q in range(p + 1, N_) if arch_of[p] == arch_of[q]]
        bet = [Smat[p][q] for p in range(N_) for q in range(N_) if arch_of[p] != arch_of[q]]
        rw, rb = min(win) / e2min, min(bet) / e2min
        chk("saBound", 100 * (e2min * (rb + (rw - rb) / len(A_) + (1 - rw) / N_)) ** 0.5, 3)
        dep = os.path.join(DGX, "deployed_on_ev.json")
        if os.path.exists(dep):
            dv = json.load(open(dep, encoding="utf-8"))["hpix_corr_pred_test.npy"]
            chk("saDeployedEv", 100 * st.mean(dv), 3)
            chk("saDeployedEvSd", 100 * st.stdev(dv), 3)
        pp_path = os.path.join(DGX, "pool_pipeline.json")
        if os.path.exists(pp_path):
            pp = json.load(open(pp_path, encoding="utf-8"))
            chk("saPoolPipeline", 100 * pp["final_ev"], 3)
            chk("saPerpixMeans", 100 * pp["stack_ev"], 3)
        do_path = os.path.join(DGX, "dropone.json")
        if os.path.exists(do_path):
            do = json.load(open(do_path, encoding="utf-8"))
            for a, key in (("mlpR", "saDropMlpR"), ("unet", "saDropUnet"), ("fno", "saDropFno"), ("krr", "saDropKrr")):
                chk(key, 100 * do["drop_" + a]["d_oracle_rms"], 3)
            chk("saOnlyFnoRms", 100 * do["only_fno"]["oracle_rms"], 3)
            chk("saOnlyMlpRRms", 100 * do["only_mlpR"]["oracle_rms"], 3)
            chk("saCorrMlpRUnet", sa["between"]["mlpR-unet"]["mean"], 2)
            chk("saCorrMlpRFno", sa["between"]["mlpR-fno"]["mean"], 2)
            chk("saCorrMseFno", sa["between"]["mlpMSE-fno"]["mean"], 2)
    else:
        print("  seedarch.json absent, sixty-predictor macros not checked")
    lc_path = os.path.join(DGX, "learning_curve.json")
    if os.path.exists(lc_path):
        lc = json.load(open(lc_path, encoding="utf-8"))
        rows = {r["N"]: r for r in lc["rows"]}
        for N, key in ((1000, "lcMlpOneK"), (2000, "lcMlpTwoK"), (4000, "lcMlpFourK"), (8000, "lcMlpEightK"), (12000, "lcMlpTwelveK"), (16000, "lcMlpSixteenK"), (19000, "lcMlpNineteenK")):
            if N in rows and key in mac:
                chk(key, rows[N]["mlp_mean"], 3)
                chk(key + "Sd", rows[N]["mlp_sd"], 3)
        for N, key in ((1000, "lcKrrOneK"), (4000, "lcKrrFourK"), (8000, "lcKrrEightK"), (16000, "lcKrrSixteenK"), (19000, "lcKrrNineteenK")):
            if N in rows and key in mac:
                chk(key, rows[N]["krr"], 3)
        if "slope_mlp_hi" in lc and "lcSlopeMlp" in mac:
            chk("lcSlopeMlp", -lc["slope_mlp_hi"], 3)
            chk("lcSlopeMlpSd", lc["slope_mlp_hi_per_seed_sd"], 3)
        if "slope_krr_hi" in lc and "lcSlopeKrr" in mac:
            chk("lcSlopeKrr", -lc["slope_krr_hi"], 3)
        if "last_step" in lc and "lcLastGain" in mac:
            chk("lcLastGain", lc["last_step"]["gain_mean"], 3)
            chk("lcLastGainSd", lc["last_step"]["gain_sd"], 3)
        if "last_two_steps" in lc and "lcLastTwoGain" in mac:
            chk("lcLastTwoGain", lc["last_two_steps"]["gain_mean"], 3)
            chk("lcLastTwoGainSd", lc["last_two_steps"]["gain_sd"], 3)
    # conformal coverage, three scores on one calibration split: the disagreement-scaled and raw
    # bands are in each seed's pipeline record (conformal_seeded.py); the P_lambda-scaled band is
    # in collected/dgx/uq_plam_seeded.json (uq_conformal_plam.py), keyed <tag>_s<seed>
    for a, L in (("a0.1", ""), ("a0.05", "Hi")):
        dis = [100 * six[s]["uq"][a]["scaled"]["coverage"] for s in S]
        chk("uqDisSix" + L, st.mean(dis), 1)
        chk("uqDisSix" + L + "Sd", st.stdev(dis), 1)
    # the ClimSim kernel budget ladder (added after review): one record per budget, seed 0, n = 1,000,000
    for cap, key, seed in ((6000, "csCapSix", 0), (12000, "csCapTwelve", 0), (24000, "csCapTwentyFour", 0),
                           (24000, "csCapTwentyFourB", 1), (48000, "csCapFortyEight", 0)):
        f = os.path.join(DGX, "climsim_cap", f"climsim_train_n1000000_s{seed}_cap{cap}.json")
        if os.path.exists(f) and key in mac:
            r = json.load(open(f, encoding="utf-8"))
            assert r["kernel_hyper"]["cap"] == cap, (f, r["kernel_hyper"]["cap"])
            chk(key, r["kernel_r2"], 3)
    # after-review diagnostics: the fitted-norm ratios (seed 0, deployed correction kernel), the OCO-2 alignment
    # check at the paper's setting (n = 4000, nugget 1e-8, median length scale), and the kernel-stage cost record
    f = os.path.join(DGX, "sm_norm_check_hpix_s0.json")
    if os.path.exists(f):
        r = json.load(open(f, encoding="utf-8"))
        chk("normRatio", round(r["ratio_aKa"], -2), 0)
        chk("normRatioNorm", r["ratio_aKa_normalized"], 1)
        chk("normEnergyRatio", round(r["energy_ratio"], -1), 0)
    f = os.path.join(DGX, "jpl_alignment_o2_s0.json")
    if os.path.exists(f):
        r = json.load(open(f, encoding="utf-8"))
        by = {(s["rows"], s["nugget"], s["scale"]): s for s in r["settings"]}
        s0 = by[("n4000", 1e-08, 1.0)]
        chk("alignDeffRaw", s0["raw"]["deff"], 0)
        chk("alignDeffFeat", s0["feature"]["deff"], 0)
        chk("alignRatioNorm", s0["ratio_aKa"], 0)
        chk("alignRatioP", s0["ratio_P_median"], 1)
        chk("alignRatioLo", by[("n4000", 1e-08, 0.5)]["ratio_aKa"], 0)
        chk("alignRatioHi", by[("n4000", 1e-08, 2.0)]["ratio_aKa"], 0)
        if ("full", 1e-08, 1.0) in by:
            chk("alignRatioFull", by[("full", 1e-08, 1.0)]["ratio_aKa"], 0)
    f = os.path.join(DGX, "sm_norm_check_hpix_s1.json")
    if os.path.exists(f):
        r = json.load(open(f, encoding="utf-8"))
        chk("normRatioLo", round(r["ratio_aKa"], -2), 0)
        chk("normRatioNormLo", r["ratio_aKa_normalized"], 1)
    f = os.path.join(DGX, "cost_check.json")
    if os.path.exists(f):
        r = json.load(open(f, encoding="utf-8"))
        chk("costCoefM", r["coefficients_correction"] / 1e6, 1)
        chk("costGramS", r["gram_block_build_s"], 0)
        chk("costCholS", r["cholesky_s"], 0)
        chk("costSolveS", r["solve_q_rhs_s"], 0)
        chk("costPeakBlockGb", r["peak_rss_gb_block_path"], 1)
        chk("costPeakNaiveGb", r["peak_rss_gb_after_naive_build"], 1)
        chk("costQueryOneMs", r["query_batch1_ms_per_query"], 0)
        chk("costQueryBatchMs", r["query_batch1000_ms_per_query"], 1)
    plam_f = os.path.join(DGX, "uq_plam_seeded.json")
    if os.path.exists(plam_f):
        plam = json.load(open(plam_f, encoding="utf-8"))
        for tag, T in (("hpix", "Six"), ("hpix5", "Five")):
            for a, L in (("a0.1", ""), ("a0.05", "Hi")):
                for band, B in (("plam", "Plam"), ("raw", "Raw")):
                    v = [100 * plam[f"{tag}_s{s}"][a][band]["coverage"] for s in S]
                    chk(f"uq{B}{T}{L}", st.mean(v), 1)
                    chk(f"uq{B}{T}{L}Sd", st.stdev(v), 1)
        # the raw band is computed by both scripts on the same split: they must agree exactly
        for s in S:
            assert abs(plam[f"hpix_s{s}"]["a0.1"]["raw"]["coverage"] - six[s]["uq"]["a0.1"]["raw"]["coverage"]) < 1e-12, s
        chk("uqWidthRatio", st.mean(plam[f"hpix_s{s}"]["a0.1"]["plam"]["mean_width"]
                                    / plam[f"hpix_s{s}"]["a0.1"]["raw"]["mean_width"] for s in S), 2)
        chk("uqSpearman", st.mean(plam[f"hpix_s{s}"]["spearman_P_err"] for s in S), 2)
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
