"""Sequential GPU queue for the F5 experiments. Reads runs/control/queue.txt,
one job per line: `<script> <args...>` relative to runs/. Runs jobs one at a
time (or --parallel k at a time) while runs/control/GPU_GO exists and no
runs/control/PAUSE flag is set; otherwise waits. A job whose result file
already exists is skipped by the job itself. Every state change is logged to
runs/control/runner.log with a clock time. Stops on runs/control/STOP.
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

W = Path(os.environ.get("NMKC_W", "C:/Users/owner/GOAL20H_20260906/nmkc_gpu_experiments"))
CTRL = W / "runs" / "control"; CTRL.mkdir(parents=True, exist_ok=True)
LOG = CTRL / "runner.log"
PY = os.environ.get("NMKC_PY", sys.executable)


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def go():
    return (CTRL / "GPU_GO").exists() and not (CTRL / "PAUSE").exists() and not (CTRL / "STOP").exists()


PEAK_MB = {"4096": 2700, "2048": 1800, "1536": 1200, "1024": 1500, "768": 1000, "384": 900}


def expected_peak_mb(line):
    for tok in ("--width",):
        if tok in line:
            w = line.split(tok)[1].split()[0]
            return PEAK_MB.get(w, 2700)
    if "lk_scale" in line:
        return 3000 if "--ntr 12000" in line else 1200
    return 2700


def vram_used_mb():
    if os.environ.get("NMKC_NO_NVSMI"):      # the Caltech DGX: nvidia-smi hangs in D state there, never call it
        return 0
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return 99999


VRAM_CAP_MB = 6500   # never let the card pass this: a near-full WDDM card resets the driver for everyone

MEM_GATE_GB = float(os.environ.get("NMKC_MEM_GATE_GB", "50"))   # owner law 06 Sep 2026: launch only while MemAvailable
                                                                  # stays above this PLUS the job's written peak; never
                                                                  # fill a shared box (the DGX runner's own gate is 50 GB)


def mem_available_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1048576.0
    except Exception:
        return None
    return None


def peak_gb_from_line(line):
    """The peak written on the queue line as a trailing @peak_gb=<x> token (the law: size the peak from
    the inputs and write it on the line); a line without one does not launch on a box with /proc/meminfo."""
    m = re.search(r"@peak_gb=([0-9.]+)", line)
    return float(m.group(1)) if m else None


def read_queue():
    q = CTRL / "queue.txt"
    if not q.exists():
        return []
    lines = [l.strip() for l in q.read_text(encoding="utf-8").splitlines()]
    return [l for l in lines if l and not l.startswith("#")]


MAX_ATTEMPTS = 8          # a line that fails this often is skipped (logged) instead of looping
KILL_RC = 4294967295      # Stop-Process / TerminateProcess(-1): an external kill (CrashGuard), not our bug
KILL_BACKOFF_S = 60       # let the guard de-escalate before the next launch touches the card


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--parallel", type=int, default=1)
    a = ap.parse_args()
    done = set()
    donef = CTRL / "done.txt"
    if donef.exists():
        done = set(l.strip() for l in donef.read_text(encoding="utf-8").splitlines() if l.strip())
    running = {}
    attempts = {}
    skipped = set()
    hold_until = 0.0
    log(f"runner start parallel={a.parallel}")
    while not (CTRL / "STOP").exists():
        # reap
        for line, pr in list(running.items()):
            rc = pr.poll()
            if rc is not None:
                log(f"finished rc={rc}: {line}")
                del running[line]
                if rc == 0:
                    done.add(line)
                    with open(donef, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                else:
                    with open(CTRL / "failed.txt", "a", encoding="utf-8") as f:
                        f.write(f"{time.strftime('%H:%M:%S')} rc={rc} attempt={attempts.get(line, 0)} {line}\n")
                    if rc == KILL_RC:
                        hold_until = time.time() + KILL_BACKOFF_S
                        log(f"external kill (rc {rc}); holding launches {KILL_BACKOFF_S} s")
                    if attempts.get(line, 0) >= MAX_ATTEMPTS:
                        skipped.add(line)
                        log(f"SKIPPED after {attempts[line]} attempts: {line}")
        par = a.parallel
        try:                                   # runs/control/parallel.txt overrides --parallel each cycle (no restart)
            par = max(1, min(2, int((CTRL / "parallel.txt").read_text().strip())))
        except Exception:
            pass
        if go() and len(running) < par and time.time() >= hold_until:
            pending = [l for l in read_queue() if l not in done and l not in running and l not in skipped]
            if pending:
                line = pending[0]
                attempts[line] = attempts.get(line, 0) + 1
                used = vram_used_mb(); need = expected_peak_mb(line)
                if used + need > VRAM_CAP_MB:
                    log(f"VRAM gate: used {used} MB + expected {need} MB > {VRAM_CAP_MB}; waiting")
                    time.sleep(30)
                    continue
                avail = mem_available_gb()
                if avail is not None:                       # a Linux box: the memory gate is the governor
                    peak = peak_gb_from_line(line)
                    if peak is None:
                        skipped.add(line); log(f"SKIPPED (no @peak_gb on the line; the law wants the peak written): {line}")
                        continue
                    if avail < MEM_GATE_GB + peak or peak > avail / 2:
                        log(f"memory gate: MemAvailable {avail:.1f} GB, need gate {MEM_GATE_GB} + peak {peak} GB; waiting")
                        time.sleep(60)
                        continue
                parts = [t for t in line.split() if not t.startswith("@peak_gb=")]
                script = W / "runs" / parts[0]
                if "--tag" in parts:
                    stem = "job_" + parts[parts.index("--tag") + 1]
                else:
                    stem = "job_" + "_".join(parts[1:])
                stem = re.sub(r"[^A-Za-z0-9_.]+", "_", stem.replace("-", ""))[:120]   # a path in --outdir is not a file name
                logf = W / "logs" / (stem + ".log")
                env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
                with open(logf, "a", encoding="utf-8") as lf:
                    pr = subprocess.Popen([PY, "-u", str(script)] + parts[1:], stdout=lf, stderr=subprocess.STDOUT,
                                          cwd=str(W / "runs"), env=env, creationflags=(0x08000000 if os.name == "nt" else 0))  # CREATE_NO_WINDOW
                running[line] = pr
                log(f"started pid={pr.pid}: {line} -> {logf.name}")
            elif not running:
                time.sleep(20)
        time.sleep(5)
    for line, pr in running.items():
        log(f"STOP: leaving running job to finish: {line}")
    log("runner exit")


if __name__ == "__main__":
    main()
