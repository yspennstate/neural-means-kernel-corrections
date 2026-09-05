"""Verify a complete evidence package and summarize paired sensitivity runs.

All values are fractions. Seed summaries describe variation across the specified
training procedures on shared benchmark cases; they are not population intervals.
Every reported mean is recomputed from per-case arrays and checked against the
producer's separately stored summary. An incomplete package cannot produce tables.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_mean(values, expected, label):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid per-case values: " + label)
    direct = math.fsum(map(float, values)) / len(values)
    if not math.isclose(direct, float(expected), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"Summary does not reproduce: {label}: {direct} != {expected}")
    return direct


def stats(values):
    values = list(map(float, values))
    if len(values) < 2 or not all(map(math.isfinite, values)):
        raise ValueError("At least two finite seed values required")
    return dict(n=len(values), mean=statistics.fmean(values),
                sd=statistics.stdev(values), minimum=min(values), maximum=max(values),
                per_seed=values)


def validate_indices(values, expected_size, label):
    if (values.ndim != 1 or len(values) != expected_size
            or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0) or len(np.unique(values)) != expected_size):
        raise ValueError("Invalid case indices: " + label)


GRIDS = {
    "recorded_grid": dict(scales="0.5,1,2,4", nuggets="1e-8,1e-6,1e-4"),
    "expanded_grid": dict(scales="0.25,0.5,1,2,4,8,16", nuggets="1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3"),
}
GRID_MODELS = {"kernel_flow", "kernel_raw", "kernel_ard", "combined", "combined_plus_ridge"} | {
    prefix + "_" + mode for prefix in ("mean", "ridge", "dkr") for mode in ("flat", "wnum", "radx")}


def validate_independent_checks(root):
    """Require a complete raw-prediction check before accepting producer scores."""
    check = read(root / "grid_prediction_check.json")
    if (check["kind"] != "completed_grid_prediction_recomputation"
            or check["seeds"] != [0, 1, 2] or check["bands"] != ["o2", "wco2", "sco2"]
            or check["driver_sha256"] != sha(root / "check_completed_grids.py")):
        raise ValueError("Independent grid check has the wrong design or source")
    expected = {(b, s, g) for b in check["bands"] for s in range(3) for g in GRIDS}
    seen = set()
    for row in check["rows"]:
        key = row["band"], row["seed"], row["scenario"]
        if key not in expected or key in seen:
            raise ValueError("Independent check contains a duplicate or unexpected grid")
        seen.add(key)
        if (row["models"] != 14 or row["cases"] != 2000 or set(row["metrics"]) != GRID_MODELS
                or not 0 <= row["max_per_case_difference"] <= 2e-9):
            raise ValueError("Independent prediction coverage or agreement failed")
        folder = root / "oco_grid/seeds" / f"oco_{row['band']}_s{row['seed']}"
        report = read(folder / (row["scenario"] + ".json"))
        for model, values in row["metrics"].items():
            if set(values) != {"reduced", "radiance"}:
                raise ValueError("Independent check omits a metric")
            for metric, value in values.items():
                if not math.isclose(value, report["metrics"][model][metric], rel_tol=2e-9, abs_tol=2e-12):
                    raise ValueError("Independent score differs from collected result")
        if row["selectors"] != report["winners"]:
            raise ValueError("Independent coordinate selector differs")
    if seen != expected:
        raise ValueError("Independent check omits one or more completed grids")
    # The remote check identifies prediction files too; retain those hashes while
    # cross-checking every identified small result file included in this package.
    for remote, identity in check["input_sha256"].items():
        marker = "/nmkc_paper1_20260905/"
        if marker in remote:
            local = root / remote.split(marker, 1)[1]
            if local.is_file() and sha(local) != identity:
                raise ValueError("Raw check and collected inputs have different identities")
    metric = read(root / "benchmark_metric_check.json")
    if (metric["driver_sha256"] != sha(root / "benchmark_metric_check.py")
            or metric["seeds"] != list(range(10)) or len(metric["rows"]) != 10
            or len(metric["member_rows"]) != 60 or metric["pool"] is None):
        raise ValueError("Historical metric/pool check is incomplete or stale")
    if [r["seed"] for r in metric["rows"]] != list(range(10)):
        raise ValueError("Historical metric seeds are reordered or duplicated")
    for row in metric["rows"]:
        if (not all(math.isfinite(row[k]) and row[k] >= 0 for k in ("plain", "trapezoidal"))
                or not 0 <= row["independent_quadrature_max_error"] <= 1e-12
                or not math.isclose(row["trapezoidal"] - row["plain"], row["trapezoidal_minus_plain"], abs_tol=1e-14)):
            raise ValueError("Historical metric independent quadrature failed")
    pool = metric["pool"]
    names = [a + f"_s{s}" for a in ("mlp", "mlpMSE", "mlpR", "fno", "unet", "krr") for s in range(10)]
    matrix = np.asarray(pool["S_ev"])
    if (pool["names"] != names or (pool["n_fit"], pool["n_eval"]) != (1000, 19000)
            or matrix.shape != (60, 60) or not np.isfinite(matrix).all()
            or not np.allclose(matrix, matrix.T, rtol=0, atol=1e-14)
            or np.linalg.eigvalsh(matrix)[0] < -1e-12):
        raise ValueError("Independent pool matrix identity or positivity failed")
    centering = read(root / "centering_prediction_check.json")
    if (centering["kind"] != "completed_centering_prediction_recomputation"
            or centering["seeds"] != list(range(10)) or not centering["complete_seed_set"]
            or centering["driver_sha256"] != sha(root / "check_completed_centering.py")):
        raise ValueError("Centering raw-field check is incomplete or stale")
    expected_centering = {(seed, arm) for seed in range(10) for arm in ("pooled", "local")}
    checked_centering = set()
    for row in centering["rows"]:
        key = row["seed"], row["arm"]
        if key not in expected_centering or key in checked_centering or row["cases"] != 20000 or row["members"] != 6:
            raise ValueError("Centering raw-field coverage failed")
        checked_centering.add(key)
        summary = read(root / "seeds" / f"sm_s{row['seed']}" / "summary.json")
        if not math.isclose(row["mean_relative_l2"], summary["metrics"][row["arm"]]["mean_relative_l2"], rel_tol=2e-11, abs_tol=2e-12):
            raise ValueError("Raw-field and collected centering means disagree")
        limits = dict(relative=2e-12, absolute=1e-8, disagreement=1e-9, scalar_control=2e-12)
        if set(row["maximum_absolute_gaps"]) != set(limits) or any(
                not 0 <= row["maximum_absolute_gaps"][name] <= limit for name, limit in limits.items()):
            raise ValueError("Centering raw-field numerical control failed")
    if checked_centering != expected_centering:
        raise ValueError("Centering raw-field check omitted a seed/arm")
    for remote, identity in centering["input_sha256"].items():
        marker = "/nmkc_paper1_20260905/"
        if marker in remote:
            local = root / remote.split(marker, 1)[1]
            if local.is_file() and sha(local) != identity:
                raise ValueError("Raw centering check and collected inputs have different identities")
    return dict(grid_check_sha256=sha(root / "grid_prediction_check.json"),
                metric_check_sha256=sha(root / "benchmark_metric_check.json"),
                centering_check_sha256=sha(root / "centering_prediction_check.json"),
                centering_arms=len(checked_centering),
                scenarios=len(seen), models_per_scenario=14, cases_per_scenario=2000)


def validate_grid_report(report, errors, scenario):
    if report["scenario"] != scenario or set(report["metrics"]) != GRID_MODELS:
        raise ValueError("Unexpected grid scenario or predictor set")
    expected_keys = {model + "_" + metric for model in GRID_MODELS for metric in ("reduced", "radiance")}
    if set(errors) != expected_keys:
        raise ValueError("Grid error arrays omit or add a predictor/metric")
    for name, values in errors.items():
        if values.shape != (2000,) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Grid errors must cover exactly 2000 finite nonnegative cases: " + name)
    scales = list(map(float, GRIDS[scenario]["scales"].split(",")))
    nuggets = list(map(float, GRIDS[scenario]["nuggets"].split(",")))
    expected_cells = [(s, n) for s in scales for n in nuggets]
    for name in ("kernel_raw", "kernel_ard", "dkr_flat", "dkr_wnum", "dkr_radx"):
        h = report["hyper"][name]
        cells = h["cells"]
        if [(c["scale"], c["nugget"]) for c in cells] != expected_cells:
            raise ValueError("Kernel grid contains missing, reordered, or duplicate cells")
        if any(c["status"] not in ("OK", "CHOLESKY_FAILED", "NONFINITE") for c in cells):
            raise ValueError("Unknown kernel cell outcome")
        successful = [c for c in cells if c["status"] == "OK"]
        if not successful or any(not math.isfinite(c["validation"]) for c in successful):
            raise ValueError("Kernel selection has no finite successful candidates")
        winner = min(successful, key=lambda c: c["validation"])
        if (h["scale"], h["nugget"], h["validation"]) != (winner["scale"], winner["nugget"], winner["validation"]):
            raise ValueError("Kernel winner is not the recorded validation minimum")
        if (h["scale_boundary"] != (h["scale"] in (min(scales), max(scales)))
                or h["nugget_boundary"] != (h["nugget"] in (min(nuggets), max(nuggets)))):
            raise ValueError("Kernel boundary flag is incorrect")


def aggregate(root):
    root = root.resolve()
    manifest = read(root / "evidence_manifest.json")
    if not manifest["complete"] or manifest["centering_seeds"] != list(range(10)) or manifest["mismatch_seeds"] != list(range(10)) or manifest["grid_seeds"] != [0, 1, 2] or manifest["bands"] != ["o2", "wco2", "sco2"]:
        raise ValueError("Unexpected or incomplete campaign design")
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if root.resolve() not in path.parents or not path.is_file() or sha(path) != expected:
            raise ValueError("Evidence identity failed: " + name)
    # A valid hash for an unrelated file cannot stand in for a consumed record.
    # Reject unhashed evidence as well as changed bytes.
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name != "evidence_manifest.json"}
    if actual != set(manifest["files"]):
        raise ValueError("Evidence file set differs from its manifest")
    for name in ("status.json", "followup_status.json"):
        if read(root / name)["status"] != "COMPLETE":
            raise ValueError("Unfinished controller: " + name)
    independent = validate_independent_checks(root)
    centering = []; mismatch = []; folds = []
    for seed in range(10):
        sr = root / "seeds" / f"sm_s{seed}"
        rec = read(sr / "summary.json")
        if rec["seed"] != seed or sha(sr / "paired_errors.npz") != rec["error_archive_sha256"]:
            raise ValueError("Centering seed identity failed")
        with np.load(sr / "paired_errors.npz", allow_pickle=False) as z:
            validate_indices(z["test_indices"], 20000, f"centering {seed}")
            if not np.array_equal(z["test_indices"], np.arange(20000, 40000)):
                raise ValueError("Unexpected structural test block")
            test_reference = z["test_indices"].copy()
            row = dict(seed=seed)
            for arm in ("historical", "pooled", "local"):
                row[arm] = checked_mean(z[arm], rec["metrics"][arm]["mean_relative_l2"], arm)
                rms = math.sqrt(math.fsum(float(v) ** 2 for v in z[arm]) / len(z[arm]))
                if not math.isclose(rms, rec["metrics"][arm]["rms_relative_l2"], rel_tol=1e-10):
                    raise ValueError("RMS does not reproduce")
            delta = z["local"] - z["pooled"]
            if not np.array_equal(delta, z["local_minus_pooled"]):
                raise ValueError("Centering pairing/sign mismatch")
            row["local_minus_pooled"] = checked_mean(delta, rec["local_minus_pooled_mean"], "centering contrast")
            row["local_better_fraction"] = float(np.mean(delta < 0))
            if row["local_better_fraction"] != rec["local_better_fraction"]:
                raise ValueError("Centering win fraction mismatch")
        centering.append(row)
        fold = read(sr / "runs/fold_centering.json")
        if fold["seed"] != seed or len(fold["folds"]) != 4 or fold["ntrain"] != 19000:
            raise ValueError("Fold configuration mismatch")
        if fold["driver_sha256"] != sha(root / "code/campaign/fold_centering_sensitivity.py") or fold["lam"] != 1e-6:
            raise ValueError("Fold source or nugget mismatch")
        for f in fold["folds"]:
            if (f["nfit"], f["nhold"]) != (14250, 4750):
                raise ValueError("Unexpected fold sizes")
            if not 0 <= f["historical_relative_frobenius"] <= 1e-5 or not 0 <= f["historical_max_sample_relative_error"] <= 1e-4:
                raise ValueError("Historical centering reproduction failed")
        folds.extend(fold["folds"])
        row["uq"] = {}
        cal_reference = None
        for arm, directory in (("pooled", "pooled"), ("local", "runs")):
            uq = read(sr / directory / "hpix_uq.json")
            if uq["seed"] != seed or (uq["n_cal"], uq["n_eval"]) != (1000, 19000):
                raise ValueError("Unexpected conformal split")
            with np.load(sr / directory / "hpix_uq.npz", allow_pickle=False) as z:
                cal, ev = z["cal"], z["ev"]
                validate_indices(cal, 1000, "calibration")
                validate_indices(ev, 19000, "evaluation")
                if not np.array_equal(np.sort(np.concatenate([cal, ev])), np.arange(20000)):
                    raise ValueError("Conformal subsets overlap or omit cases")
                if cal_reference is not None and not np.array_equal(cal, cal_reference):
                    raise ValueError("Paired arms used different calibration cases")
                cal_reference = cal.copy()
                err, scale = z["err"], z["disagree"]
                if (err.shape != (20000,) or scale.shape != err.shape
                        or not np.isfinite(err).all() or np.any(err < 0)
                        or not np.isfinite(scale).all() or np.any(scale <= 0)):
                    raise ValueError("Invalid conformal errors or scales")
                arm_uq = {}
                for alpha in (.1, .05):
                    key = f"a{alpha:g}"
                    rank = math.ceil((1-alpha) * (len(cal)+1))
                    for mode, scale_used in (("raw", np.ones_like(scale)), ("scaled", scale)):
                        quantile = float(np.partition(err[cal] / scale_used[cal], rank-1)[rank-1])
                        coverage = int(np.count_nonzero(err[ev] <= quantile * scale_used[ev])) / len(ev)
                        radius = math.fsum(float(quantile*v) for v in scale_used[ev]) / len(ev)
                        for field, value in (("q", quantile), ("coverage", coverage), ("mean_width", radius)):
                            if not math.isclose(value, uq[key][mode][field], rel_tol=1e-10, abs_tol=1e-12):
                                raise ValueError("Conformal measurement does not reproduce")
                        arm_uq[key + "_" + mode] = dict(coverage=coverage, mean_radius=radius)
                row["uq"][arm] = arm_uq
        mr = root / "mismatch" / f"s{seed}"
        rec = read(mr / "summary.json")
        if rec["seed"] != seed or sha(mr / "paired_errors.npz") != rec["errors_sha256"]:
            raise ValueError("Mismatch seed identity failed")
        if (rec["driver_sha256"] != sha(root / "diagnostics/correction_mismatch.py")
                or not rec["kernel_frozen"] or not rec["weights_frozen"] or rec["neural_retraining"]):
            raise ValueError("Mismatch source or frozen-estimator protocol changed")
        row = dict(seed=seed, controls=rec["controls"])
        with np.load(mr / "paired_errors.npz", allow_pickle=False) as z:
            for split, size in (("validation", 1000), ("test", 20000)):
                validate_indices(z[split + "_indices"], size, f"mismatch {split}")
                if split == "test" and not np.array_equal(z["test_indices"], test_reference):
                    raise ValueError("Mismatch and centering test cases differ")
                metric = rec["metrics"][split]
                for field, producer in (("base", "base_mean"), ("historical", "historical_correction_mean"), ("consistent", "consistent_correction_mean")):
                    row[split + "_" + field] = checked_mean(z[split + "_" + field], metric[producer], split + field)
                term = z[split + "_propagated_mismatch"]
                gap = np.abs(z[split + "_consistent"] - z[split + "_historical"])
                if np.any(term < 0) or np.any(gap > term + 2e-12):
                    raise ValueError("Propagated term violates the reverse triangle inequality")
                row[split + "_mismatch_mean"] = checked_mean(term, metric["propagated_mismatch"]["mean"], "propagated term")
                for label, value in (("maximum", np.max(term)), ("p95", np.quantile(term, .95))):
                    if not math.isclose(float(value), metric["propagated_mismatch"][label], rel_tol=1e-10, abs_tol=1e-12):
                        raise ValueError("Mismatch distribution summary failed")
                    row[split + "_mismatch_" + label] = float(value)
                row[split + "_consistent_minus_historical"] = checked_mean(
                    z[split + "_consistent"] - z[split + "_historical"],
                    row[split + "_consistent"] - row[split + "_historical"], "mismatch contrast")
            stage = "consistent" if row["validation_consistent"] < row["validation_base"] else "base"
            row["selected_test"] = row["test_" + stage]
            if not math.isclose(row["selected_test"], rec["metrics"]["inference_consistent_selected_test"], abs_tol=1e-12):
                raise ValueError("Mismatch validation selection failed")
        mismatch.append(row)
    grids = {}
    for band in ("o2", "wco2", "sco2"):
        rows = []
        fixed_test_hashes = None
        for seed in range(3):
            br = root / "oco_grid" / "seeds" / f"oco_{band}_s{seed}"
            rec = read(br / "grid_sensitivity.json")
            if rec["identity"]["seed"] != seed or rec["identity"]["band"] != band:
                raise ValueError("Grid identity mismatch")
            identity = rec["identity"]
            if (identity != read(br / "experiment_identity.json") or identity["grids"] != GRIDS
                    or (identity["epochs"], identity["width"], identity["threads"]) != (250, 384, 8)
                    or identity["driver_sha256"] != sha(root / "diagnostics/jpl_grid_sensitivity.py")):
                raise ValueError("Grid source, cache identity, or fixed design changed")
            test_hashes = {k: identity["data_sha256"][k] for k in ("Xte", "Yte")}
            if fixed_test_hashes is not None and test_hashes != fixed_test_hashes:
                raise ValueError("Grid seeds do not share the same test cases")
            fixed_test_hashes = test_hashes
            row = dict(seed=seed, scenarios={})
            errors = {}
            for scenario in ("recorded_grid", "expanded_grid"):
                report = read(br / f"{scenario}.json")
                if report != rec["results"][scenario] or sha(br / f"{scenario}_errors.npz") != report["errors_sha256"]:
                    raise ValueError("Grid report/array identity failed")
                with np.load(br / f"{scenario}_errors.npz", allow_pickle=False) as z:
                    errors[scenario] = {name: z[name] for name in z.files}
                validate_grid_report(report, errors[scenario], scenario)
                metrics = {}
                for model, scores in report["metrics"].items():
                    metrics[model] = {metric: checked_mean(errors[scenario][model + "_" + metric], expected, model + metric) for metric, expected in scores.items()}
                heads = [h for h in report["hyper"].values() if "cells" in h]
                row["scenarios"][scenario] = dict(metrics=metrics,
                    boundary_heads=sum(h["scale_boundary"] or h["nugget_boundary"] for h in heads),
                    failed_cells=sum(c["status"] != "OK" for h in heads for c in h["cells"]))
            row["expanded_minus_recorded"] = {}
            for name, old in errors["recorded_grid"].items():
                delta = errors["expanded_grid"][name] - old
                row["expanded_minus_recorded"][name] = checked_mean(delta, rec["expanded_minus_recorded"][name]["mean"], name + " grid contrast")
                for key, value in (("p05", np.quantile(delta, .05)), ("p95", np.quantile(delta, .95)),
                                   ("improved_fraction", np.count_nonzero(delta < 0) / len(delta))):
                    if not math.isclose(float(value), rec["expanded_minus_recorded"][name][key], rel_tol=1e-10, abs_tol=1e-12):
                        raise ValueError("Grid paired distribution summary failed")
                if name.startswith(("mean_", "ridge_", "kernel_flow_")) and np.any(delta != 0):
                    raise ValueError("A frozen control changed between grids: " + name)
            rows.append(row)
        models = rows[0]["scenarios"]["recorded_grid"]["metrics"]
        grids[band] = dict(rows=rows, aggregate={
            scenario: {model: {metric: stats(r["scenarios"][scenario]["metrics"][model][metric] for r in rows)
                               for metric in ("reduced", "radiance")} for model in models}
            for scenario in ("recorded_grid", "expanded_grid")},
            differences={name: stats(r["expanded_minus_recorded"][name] for r in rows)
                         for name in rows[0]["expanded_minus_recorded"]})
    return dict(schema=1, units="fractions; multiply by 100 for percent or percentage-point differences",
                seed_uncertainty="sample standard deviation on shared benchmark cases, not a population confidence interval",
                evidence_manifest_sha256=sha(root / "evidence_manifest.json"),
                independent_prediction_checks=independent,
                centering=dict(rows=centering, aggregate={key: stats(r[key] for r in centering)
                    for key in ("historical", "pooled", "local", "local_minus_pooled")},
                    uq={arm: {key: {metric: stats(r["uq"][arm][key][metric] for r in centering)
                                   for metric in ("coverage", "mean_radius")}
                              for key in centering[0]["uq"][arm]}
                        for arm in ("pooled", "local")},
                    field_change_range=[min(f["centering_relative_frobenius"] for f in folds), max(f["centering_relative_frobenius"] for f in folds)],
                    independent_solve_max=max(f["independent_solve_max_absolute_difference"] for f in folds)),
                mismatch=dict(rows=mismatch, aggregate={key: stats(r[key] for r in mismatch)
                    for key in ("test_base", "test_historical", "test_consistent", "selected_test",
                                "test_mismatch_mean", "test_mismatch_p95", "test_mismatch_maximum", "test_consistent_minus_historical")}),
                oco_grids=grids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = aggregate(args.root)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Verified complete evidence and wrote", args.out)


if __name__ == "__main__":
    main()
