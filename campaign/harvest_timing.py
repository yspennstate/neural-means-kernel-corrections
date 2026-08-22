"""Harvest the compute record of the structural-mechanics campaign.

Nothing is measured here. The script reads what the campaign itself wrote:
the per-run records (minutes, best_ep, epochs, threads, params), the pipeline
logs (per-step wall clock, the device line each torch trainer prints, the
cumulative timer the trainers print every ten epochs, the kernel grid's
per-cell factor-and-solve seconds, the out-of-fold fold times), the task
records (thread count, time limit, interpreter) and the failure markers.
It writes one JSON with the raw values per seed and a summary with the seed
count behind each figure, and prints the summary. Memory was never recorded
by any stage; the script reports that absence rather than estimating it.

Run on the campaign tree with NMKC_ROOT pointing at it; NMKC_TIMING_OUT names
the output file.
"""
import glob
import json
import os
import re
import statistics
import sys

ROOT = os.environ.get("NMKC_ROOT", "/srv/aiwork/nmkc10seed/nmkc10seed")
OUT = os.environ.get("NMKC_TIMING_OUT", os.path.join(ROOT, "release_residuals", "timing_harvest.json"))
SEEDS = list(range(10))
MEMBERS = ["mlp", "mlpMSE", "mlpR", "fno", "krr"]

RE_STEP_START = re.compile(r"^\[(?P<task>\S+)\] step (?P<step>\w+): (?P<cmd>.*)$")
RE_STEP_SKIP = re.compile(r"^\[(?P<task>\S+)\] step (?P<step>\w+): outputs present, skip")
RE_STEP_DONE = re.compile(r"^\[(?P<task>\S+)\] step (?P<step>\w+) done in (?P<min>[\d.]+) min")
RE_EPOCH = re.compile(r"^ep\s+(?P<ep>\d+)\s+train\s+(?P<tr>[\d.]+)\s+val\s+(?P<va>[\d.]+)\s+\[(?P<s>\d+)s\](?P<star>\s*\*)?")
RE_CELL = re.compile(r"^\s+smult=\s*(?P<smult>[\d.]+)\s+lam=(?P<lam>\S+)\s+val\s+(?P<va>[\d.]+)\s+\[(?P<s>\d+)s\]")
RE_FOLD = re.compile(r"^fold (?P<f>\d+): (?P<s>\d+)s")
RE_DIST = re.compile(r"^distance matrices: (?P<s>[\d.]+)s")
RE_DEVICE = re.compile(r"^device: (?P<dev>\S+)")
RE_PARAMS = re.compile(r"^params: (?P<p>[\d.]+)M")
RE_TUNE = re.compile(r"corr tune \(sub (?P<n>\d+)\): .*\[(?P<s>\d+)s\]")


def run_records(seed):
    """The trainer-written record of each member at one seed."""
    out = {}
    rundir = os.path.join(ROOT, "seeds", "sm_s%d" % seed, "runs")
    for p in sorted(glob.glob(os.path.join(rundir, "*.json"))):
        base = os.path.basename(p)
        member = "krr" if base.startswith("krr_") else base.split("_")[0]
        if member not in MEMBERS:
            continue
        with open(p) as fh:
            rec = json.load(fh)
        args = rec.get("args", {})
        out[member] = dict(
            file=base, kind=rec.get("kind"), minutes=rec.get("minutes"),
            best_ep=rec.get("best_ep"), epochs=args.get("epochs"),
            threads=args.get("threads"), batch=args.get("batch"),
            params=rec.get("params"), note=rec.get("note"),
            val=rec.get("val"), test=rec.get("test"),
            grid_cells=(len(rec["grid"]) if "grid" in rec else None),
        )
    return out


def parse_log(path):
    """One pipeline log: steps with their wall clock, device, epoch timers."""
    steps = []          # list of dicts in log order; an attempt per 'step X:' line
    cur = None
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = RE_STEP_SKIP.match(line)
            if m:
                steps.append(dict(step=m.group("step"), skipped=True))
                cur = None
                continue
            m = RE_STEP_START.match(line)
            if m:
                cur = dict(step=m.group("step"), cmd=m.group("cmd"), skipped=False,
                           done_min=None, device=None, params_M=None,
                           epochs=[], cells=[], folds=[], dist_s=None, tune=[])
                steps.append(cur)
                continue
            m = RE_STEP_DONE.match(line)
            if m:
                # attach to the most recent open attempt of that step
                for s in reversed(steps):
                    if s.get("step") == m.group("step") and not s.get("skipped"):
                        s["done_min"] = float(m.group("min"))
                        break
                cur = None
                continue
            if cur is None:
                continue
            m = RE_DEVICE.match(line)
            if m:
                cur["device"] = m.group("dev")
                continue
            m = RE_PARAMS.match(line)
            if m:
                cur["params_M"] = float(m.group("p"))
                continue
            m = RE_EPOCH.match(line)
            if m:
                cur["epochs"].append(dict(ep=int(m.group("ep")), train=float(m.group("tr")),
                                          val=float(m.group("va")), s=int(m.group("s")),
                                          best=bool(m.group("star"))))
                continue
            m = RE_CELL.match(line)
            if m:
                cur["cells"].append(dict(smult=float(m.group("smult")), lam=m.group("lam"),
                                         val=float(m.group("va")), s=int(m.group("s"))))
                continue
            m = RE_FOLD.match(line)
            if m:
                cur["folds"].append(int(m.group("s")))
                continue
            m = RE_DIST.match(line)
            if m:
                cur["dist_s"] = float(m.group("s"))
                continue
            m = RE_TUNE.search(line)
            if m:
                cur["tune"].append(dict(sub=int(m.group("n")), s=int(m.group("s"))))
    return steps


def epoch_rate(attempt):
    """Seconds per epoch from consecutive ten-epoch timer prints of one attempt."""
    eps = attempt["epochs"]
    rates = []
    for a, b in zip(eps, eps[1:]):
        if b["ep"] > a["ep"]:
            rates.append((b["s"] - a["s"]) / (b["ep"] - a["ep"]))
    if eps and eps[0]["ep"] > 0 and eps[0]["s"] < 10 * 3600 * 24:
        rates.insert(0, eps[0]["s"] / eps[0]["ep"])     # first print from the timer origin
    return rates


def small_text(path, limit=600):
    try:
        with open(path, errors="replace") as fh:
            return fh.read(limit)
    except OSError as e:
        return "ERR %s" % e


def stats(xs):
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return dict(n=0)
    d = dict(n=len(xs), mean=statistics.fmean(xs), min=min(xs), max=max(xs),
             median=statistics.median(xs))
    d["sd"] = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return d


harvest = dict(source_tree=ROOT, seeds={}, task_records={}, failure_markers={},
               other_logs={}, heartbeat=None)

for seed in SEEDS:
    entry = dict(records=run_records(seed), log=None)
    lp = os.path.join(ROOT, "logs", "a1_sm_s%d.log" % seed)
    if os.path.exists(lp):
        steps = parse_log(lp)
        entry["log"] = steps
        # per-attempt derived quantities
        for s in steps:
            if s.get("skipped"):
                continue
            if s["epochs"]:
                s["epoch_rate_s"] = epoch_rate(s)
                starred = [e for e in s["epochs"] if e["best"]]
                s["last_starred_ep"] = starred[-1]["ep"] if starred else None
                s["last_ep_seen"] = s["epochs"][-1]["ep"]
                s["last_timer_s"] = s["epochs"][-1]["s"]
    harvest["seeds"][str(seed)] = entry

for d in ("done", "tasks", "active", "failed"):
    for p in sorted(glob.glob(os.path.join(ROOT, d, "a1_sm_s*.json"))):
        with open(p) as fh:
            rec = json.load(fh)
        harvest["task_records"][d + "/" + os.path.basename(p)] = dict(
            interpreter=(rec.get("argv") or [None])[0], threads=rec.get("threads"),
            timeout_hours=rec.get("timeout_hours"), env=rec.get("env"),
            keys=sorted(rec.keys()))
for p in sorted(glob.glob(os.path.join(ROOT, "failed", "a1_sm_s*.attempts*"))):
    harvest["failure_markers"][os.path.basename(p)] = small_text(p)
hb = os.path.join(ROOT, "heartbeat.json")
if os.path.exists(hb):
    with open(hb) as fh:
        harvest["heartbeat"] = json.load(fh)
for name in ("finalize_fno.log", "genpreds_box04.log", "pix4.log",
             "genpredfast_s2_fno.log", "genpred_sm_s0_mlp_s0_w1024_d4_n19000_mir.log"):
    p = os.path.join(ROOT, "logs", name)
    if os.path.exists(p):
        harvest["other_logs"][name] = small_text(p, 1500)

# ---- summary with the seed count behind each number -------------------------
summary = dict(members={}, kernel={}, device={}, notes=[])
for member in ["mlp", "mlpMSE", "mlpR", "fno"]:
    mins, best, eps, thr, params, devs, pipe_min = [], [], [], [], [], [], []
    rates, to_best_s, last_seen = [], [], []
    for seed in SEEDS:
        e = harvest["seeds"][str(seed)]
        r = e["records"].get(member)
        if r:
            mins.append(r["minutes"]); best.append(r["best_ep"]); eps.append(r["epochs"])
            thr.append(r["threads"]); params.append(r["params"])
        for s in e["log"] or []:
            if s.get("step") == member and not s.get("skipped"):
                if s["device"]:
                    devs.append(s["device"])
                if s["done_min"] is not None:
                    pipe_min.append(s["done_min"])
                if s["epochs"]:
                    rates.extend(s["epoch_rate_s"])
                    last_seen.append(s["last_ep_seen"])
    summary["members"][member] = dict(
        trainer_minutes=stats(mins), pipeline_step_minutes=stats(pipe_min),
        best_epoch=stats(best), scheduled_epochs=sorted(set(x for x in eps if x is not None)),
        threads=sorted(set(x for x in thr if x is not None)),
        params=sorted(set(x for x in params if x is not None)),
        device_lines=dict((d, devs.count(d)) for d in set(devs)),
        seconds_per_epoch=stats(rates), last_epoch_seen=stats(last_seen),
        seeds_with_record=len(mins))

# kernel: per-cell factor-and-solve seconds, the whole stage, the OOF folds
cells, krr_min, krr_pipe, folds, dist, oof_pipe, tune = [], [], [], [], [], [], []
for seed in SEEDS:
    e = harvest["seeds"][str(seed)]
    r = e["records"].get("krr")
    if r:
        krr_min.append(r["minutes"])
    for s in e["log"] or []:
        if s.get("skipped"):
            continue
        if s["step"] == "krr":
            cells.extend(c["s"] for c in s["cells"])
            if s["done_min"] is not None:
                krr_pipe.append(s["done_min"])
            if s["dist_s"] is not None:
                dist.append(s["dist_s"])
        if s["step"] == "krr_oof":
            folds.extend(s["folds"])
            if s["done_min"] is not None:
                oof_pipe.append(s["done_min"])
        tune.extend(t["s"] for t in s["tune"])
summary["kernel"] = dict(
    cell_seconds=stats(cells), stage_minutes_record=stats(krr_min),
    stage_minutes_pipeline=stats(krr_pipe), distance_seconds=stats(dist),
    oof_fold_seconds=stats(folds), oof_stage_minutes=stats(oof_pipe),
    correction_tune_seconds=stats(tune), seeds_with_record=len(krr_min))

# per-seed fingerprint so host heterogeneity is visible rather than averaged away
per_seed = {}
for seed in SEEDS:
    e = harvest["seeds"][str(seed)]
    row = {}
    for member in MEMBERS:
        r = e["records"].get(member)
        row[member + "_min"] = None if not r else r["minutes"]
    for s in e["log"] or []:
        if not s.get("skipped") and s["step"] == "krr" and s["cells"]:
            row["krr_cell_median_s"] = statistics.median(c["s"] for c in s["cells"])
        if not s.get("skipped") and s["step"] == "fno" and s["epochs"]:
            row.setdefault("fno_attempts", []).append(dict(
                rate_s=statistics.median(s["epoch_rate_s"]) if s["epoch_rate_s"] else None,
                last_ep=s["last_ep_seen"], last_starred=s["last_starred_ep"]))
    per_seed[str(seed)] = row
summary["per_seed"] = per_seed
summary["notes"].append("no stage records peak memory; only the OOM marker files in failed/ speak to it")
summary["notes"].append("no stage records the wall clock of test-block inference for the five members; "
                        "the genpred logs carry the score only")
harvest["summary"] = summary

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(harvest, fh, indent=1)
print(json.dumps(summary, indent=1))
print("wrote", OUT)
