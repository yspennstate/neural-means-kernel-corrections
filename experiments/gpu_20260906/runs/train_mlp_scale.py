"""Structural-mechanics residual MLP at larger widths and depths, on the GPU,
with the paper's protocol held fixed (train_mlp.py of the NMKC repository):

  split      canonical_split(n_val=1000): validation = 1000 rows of the 20000-row
             training block drawn by permutation seed s (NMKC_SPLIT_SEED = s),
             the remaining 19000 train; the 20000-row test block never moves
  network    41 -> w, SiLU, (d-1) residual layers h = h + SiLU(W h), w -> 1681
  training   AdamW lr 1e-3, weight decay 1e-5, cosine to 1e-6, batch 256,
             reflection augmentation with p = 0.5, 400 epochs (metric loss) or
             120 epochs (MSE on standardized targets); torch.manual_seed(s)
  selection  validation error every 10 epochs, best-validation checkpoint
  metrics    common.rel_l2 (float64, unweighted grid norm) on val, test, test+TTA

Additions: --width, --depth, --seed, --amp bf16 (autocast on the forward pass,
labelled), resumable checkpoints every 10 epochs, GPU-utilization sampling
through NVML in-process, a pause on compute distress, saved reflection-
averaged predictions on val and test for the ensemble analysis.

--graph 1 (default): the full-batch training step (forward, loss, backward,
AdamW update) is captured once in a CUDA graph and replayed; the batch is
gathered and mirrored outside the graph exactly as the eager loop does. The
kernels are the same as in eager mode; the warm-up steps needed for capture
are undone in place (parameters restored, optimizer moments and step counters
zeroed) so the trajectory is the eager one. The 56-row last batch of each
epoch runs eagerly. The cosine schedule is driven by the library scheduler on
a shadow optimizer so the learning-rate sequence is bit-identical.
"""
import argparse, hashlib, json, os, sys, threading, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

W = Path(os.environ.get("NMKC_W", "C:/Users/owner/GOAL20H_20260906/nmkc_gpu_experiments"))
os.environ.setdefault("NMKC_DATA", str(W / "data" / "structmech"))
sys.path.insert(0, str(W / "code"))
from common import load_arrays, canonical_split, rel_l2  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
p.add_argument("--width", type=int, default=1024)
p.add_argument("--depth", type=int, default=4)
p.add_argument("--loss", choices=("metric", "mse"), default="metric")
p.add_argument("--epochs", type=int, default=0, help="0 = paper default (400 metric, 120 mse)")
p.add_argument("--batch", type=int, default=256)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=1e-5)
p.add_argument("--amp", choices=("none", "bf16"), default="none")
p.add_argument("--graph", type=int, default=1)
p.add_argument("--duty", type=float, default=0.80, help="GPU duty cycle: after every --duty-window steps the loop syncs and idles (1-duty)/duty of the busy time; the owner's CrashGuard kills python GPU clients at a utilization sample >= 95")
p.add_argument("--duty-window", type=int, default=8)
p.add_argument("--stop-after", type=int, default=0, help="debug: exit after this epoch with the checkpoint left in place (resume control)")
p.add_argument("--tag", default="")
p.add_argument("--outdir", default=str(W / "runs" / "structmech"))
args = p.parse_args()
EPOCHS = args.epochs or (400 if args.loss == "metric" else 120)
name = args.tag or f"mlp{'MSE' if args.loss == 'mse' else ''}_s{args.seed}_w{args.width}_d{args.depth}{'_bf16' if args.amp == 'bf16' else ''}"
OUT = Path(args.outdir); OUT.mkdir(parents=True, exist_ok=True)
CK = OUT / "ckpt"; CK.mkdir(exist_ok=True)
STATE = W / "runs" / "control"
if (OUT / f"{name}.json").exists():
    print("done already:", name); sys.exit(0)

os.environ["NMKC_SPLIT_SEED"] = str(args.seed)
torch.manual_seed(args.seed); np.random.seed(args.seed)
# CPU mode (added 13:0x 6 Sep, owner order: DNN training on the Caltech DGX, whose GPUs are wedged): device is
# cuda when available unless NMKC_DEVICE=cpu; on cpu the CUDA graph, NVML sampler, thermal governor and duty
# cycle are off, threads come from NMKC_THREADS (default 8), and the numerics are the eager path.
ON_CPU = os.environ.get("NMKC_DEVICE", "").lower() == "cpu" or not torch.cuda.is_available()
dev = torch.device("cpu" if ON_CPU else "cuda")
if ON_CPU:
    torch.set_num_threads(int(os.environ.get("NMKC_THREADS", "8")))
else:
    torch.backends.cuda.matmul.allow_tf32 = False
DEVICE_NAME = "cpu:" + str(torch.get_num_threads()) + "thr" if ON_CPU else torch.cuda.get_device_name(0)
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


loads, stress = load_arrays()
tr, va, te = canonical_split(n_val=1000, seed=0)
data_sha = {n: sha(Path(os.environ["NMKC_DATA"]) / f"{n}.npy") for n in ("loads", "stress", "idx_train", "idx_test")}
Xtr = torch.from_numpy(loads[tr]).to(dev); Ytr = torch.from_numpy(stress[tr]).reshape(len(tr), -1).to(dev)
Xva = torch.from_numpy(loads[va]).to(dev); Yva = torch.from_numpy(stress[va]).reshape(len(va), -1).to(dev)
Xte = torch.from_numpy(loads[te]).to(dev); Yte = torch.from_numpy(stress[te]).reshape(len(te), -1).to(dev)
mu_x, sd_x = Xtr.mean(), Xtr.std()
mu_y = Ytr.mean(0, keepdim=True); sd_y = Ytr.std()
idx2d = torch.arange(1681, device=dev).reshape(41, 41); MIR = idx2d.flip(0).reshape(-1)


class MLP(nn.Module):
    def __init__(self, w, d):
        super().__init__()
        self.inp = nn.Linear(41, w)
        self.hid = nn.ModuleList([nn.Linear(w, w) for _ in range(d - 1)])
        self.out = nn.Linear(w, 1681)

    def forward(self, x):
        h = F.silu(self.inp(x))
        for l in self.hid:
            h = h + F.silu(l(h))
        return self.out(h)


model = MLP(args.width, args.depth).to(dev)
n_params = sum(q.numel() for q in model.parameters())
use_graph = bool(args.graph) and not ON_CPU
if use_graph:
    lr_t = torch.tensor(args.lr, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_t, weight_decay=args.wd, capturable=True)
    shadow = torch.optim.SGD([torch.zeros(1)], lr=args.lr)          # carries the library schedule
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(shadow, T_max=EPOCHS, eta_min=1e-6)
else:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)


def sched_step():
    sched.step()
    if use_graph:
        lr_t.fill_(shadow.param_groups[0]["lr"])


def loss_fn(pred_n, ytrue):
    if args.loss == "mse":
        return F.mse_loss(pred_n.float(), (ytrue - mu_y) / sd_y)
    pred = pred_n.float() * sd_y + mu_y
    return (torch.linalg.vector_norm(pred - ytrue, dim=1) / torch.linalg.vector_norm(ytrue, dim=1)).mean()


amp_ctx = (lambda: torch.autocast(dev.type, dtype=torch.bfloat16)) if args.amp == "bf16" else (lambda: torch.autocast(dev.type, enabled=False))


def step_eager(xb, yb):
    with amp_ctx():
        pred_n = model((xb - mu_x) / sd_x)
    loss = loss_fn(pred_n, yb)
    opt.zero_grad(set_to_none=not use_graph)
    loss.backward()
    opt.step()
    return loss


@torch.no_grad()
def predict(X, tta=False, bs=4096):
    model.eval()
    outs = []
    for k in range(0, len(X), bs):
        xb = X[k:k + bs]
        pr = model((xb - mu_x) / sd_x) * sd_y + mu_y
        if tta:
            pr2 = model((torch.flip(xb, dims=[1]) - mu_x) / sd_x) * sd_y + mu_y
            pr = 0.5 * (pr + pr2[:, MIR])
        outs.append(pr.float())
    model.train()
    return torch.cat(outs)


def evaluate(X, Y, tta=False):
    return rel_l2(predict(X, tta).cpu().numpy(), Y.cpu().numpy())


# ---- GPU utilization sampler (NVML in-process; nvidia-smi subprocesses can deadlock from a CUDA process)
util = []
mem_peak = [0]
temp_now = [0]       # last NVML GPU temperature (C); the thermal governor below reads it
temp_max = [0]
def sampler(stop):
    try:
        if ON_CPU:
            return
        import pynvml
        pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
        k = 0
        while not stop.is_set():
            try:
                t = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                temp_now[0] = t; temp_max[0] = max(temp_max[0], t)
                if k % 2 == 0:   # utilization and memory every 4 s, temperature every 2 s
                    util.append(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
                    mem_peak[0] = max(mem_peak[0], pynvml.nvmlDeviceGetMemoryInfo(h).used)
            except Exception:
                pass
            k += 1
            stop.wait(2)
    except Exception as ex:  # noqa: BLE001
        util.append(-1); print("nvml unavailable:", ex, flush=True)
stop = threading.Event(); threading.Thread(target=sampler, args=(stop,), daemon=True).start()


def distress():
    """True while the machine reports distress/throttle or a PAUSE flag exists."""
    if (STATE / "PAUSE_ALL").exists():
        return True
    try:
        d = json.loads(Path(os.environ.get("NMKC_COMPUTE_STATE", "C:/Users/owner/.claude/compute/compute_state.json")).read_text())
        return d.get("status") == "distress" or d.get("mode") == "throttle"
    except Exception:
        return False


# ---- resume
ck = CK / f"{name}.pt"
s = None
start_ep, best_val, best_state, best_ep, paused_s, t_train = 0, 1e9, None, -1, 0.0, 0.0
if ck.exists():
    try:
        s = torch.load(ck, map_location=dev, weights_only=False)
    except Exception as ex:  # a kill during torch.save leaves a truncated zip (w4096 at 10:50:48): start fresh
        bad = ck.with_suffix(".corrupt")
        ck.replace(bad); print(f"checkpoint unreadable ({ex}); moved to {bad.name}; starting fresh", flush=True)
        s = None
if s is not None:
    model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
    if use_graph:
        # capturable AdamW: lr must stay the tensor the graph will read; state tensors on the device
        for g in opt.param_groups:
            g["lr"] = lr_t
        if s.get("shadow_state") is not None:
            shadow.load_state_dict(s["shadow_state"])
        cur = float(sched.get_last_lr()[0])
        shadow.param_groups[0]["lr"] = cur
        lr_t.fill_(cur)
    start_ep, best_val, best_state, best_ep = s["epoch"], s["best_val"], s["best_state"], s["best_ep"]
    t_train = s.get("t_train", 0.0)
    # map_location=dev moved the saved ByteTensors to the card; both setters want CPU ByteTensors
    torch.set_rng_state(s["rng_cpu"].cpu())
    if not ON_CPU and s.get("rng_cuda") is not None:
        torch.cuda.set_rng_state(s["rng_cuda"].cpu())
    resumed = True
    print(f"resumed {name} at epoch {start_ep}", flush=True)
else:
    resumed = False
print(f"{name}: params {n_params/1e6:.2f}M  epochs {EPOCHS}  amp {args.amp}  graph {use_graph}  device {DEVICE_NAME}", flush=True)

N = Xtr.shape[0]
B = args.batch
n_full = N // B                       # full batches per epoch (74); the remainder (56 rows) runs eagerly
graph = None
if use_graph:
    xs = torch.zeros((B, 41), device=dev); ys = torch.zeros((B, 1681), device=dev)
    xs.copy_(Xtr[:B]); ys.copy_(Ytr[:B])
    saved = [q.detach().clone() for q in model.parameters()]
    # on a resume the optimizer already carries moments and step counts: keep them across the
    # warm-up (a fresh run has an empty state, which the zeroing below reproduces)
    saved_opt = {id(p): {k2: v.detach().clone() for k2, v in opt.state[p].items() if torch.is_tensor(v)}
                 for p in model.parameters() if p in opt.state} if resumed else {}
    side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step_eager(xs, ys)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph):
        with amp_ctx():
            pred_s = model((xs - mu_x) / sd_x)
        loss_s = loss_fn(pred_s, ys)
        loss_s.backward()
        opt.step()
    # undo the warm-up in place: parameters back to the initialization, moments and steps to zero
    with torch.no_grad():
        for q, q0 in zip(model.parameters(), saved):
            q.copy_(q0)
        for p in model.parameters():
            st = opt.state[p]
            keep = saved_opt.get(id(p))
            for k2 in ("exp_avg", "exp_avg_sq", "step"):
                if k2 in st:
                    if keep is not None and k2 in keep:
                        st[k2].copy_(keep[k2])      # resumed: moments and step counts restored in place
                    else:
                        st[k2].zero_()
    del saved, saved_opt
    torch.cuda.synchronize()
    print("CUDA graph captured; warm-up undone in place", flush=True)

t0 = time.time()
loss = None
# GPU duty cycle: every DW steps the loop waits for the card and idles (1-DUTY)/DUTY of the busy
# time, so a utilization sample reads about 100*DUTY. The owner's CrashGuard (C:/Users/owner/CrashGuard)
# kills python GPU clients when a sample reaches its red tier (95); his order is 80 percent.
DUTY = min(max(args.duty, 0.05), 1.0); DW = max(1, args.duty_window)
# Thermal governor (added 11:2x 6 Sep): CrashGuard's red tier is ALSO gpu_temp_c >= 87 sustained 20 s
# (configs/thresholds.json) and a trend-predictor forecast of 88 C - it killed a python client at 11:06:23
# on util 88 / 86 C. The laptop's card sits at 80-86 C under the other agents' CPU load, so a fixed
# 80 percent duty would cross the line. The busy fraction is scaled down linearly from T_SOFT to T_HARD,
# and at T_HARD the loop holds (2 s naps) until the card cools below T_HARD - 1. Recorded in the result.
T_SOFT = float(os.environ.get("NMKC_T_SOFT", "80")); T_HARD = float(os.environ.get("NMKC_T_HARD", "84"))
# The predictor (crashguard-gpu-trend-predictor.ps1) regresses temperature over a 900 s window and goes
# red when the fitted slope reaches 88 C within 180 s at >= 75 C: a fast ramp is red even at 85 C. So
# the busy fraction also ramps from DUTY_MIN to DUTY over RAMP_S seconds of wall time (about 0.5 C/min).
RAMP_S = float(os.environ.get("NMKC_RAMP_S", "480"))
DUTY_MIN = 0.20
hold_s = 0.0
t_launch = time.time()
def duty_now():
    t = temp_now[0]
    cap = DUTY_MIN + (DUTY - DUTY_MIN) * min(1.0, (time.time() - t_launch) / RAMP_S)
    if t <= T_SOFT:
        return cap
    if t >= T_HARD:
        return DUTY_MIN
    return min(cap, DUTY - (DUTY - DUTY_MIN) * (t - T_SOFT) / (T_HARD - T_SOFT))
n_steps, idle_s, t_w = 0, 0.0, time.perf_counter()
if args.stop_after:
    assert args.stop_after % 10 == 0, "--stop-after must be a checkpoint epoch (multiple of 10)"
for ep in range(start_ep, EPOCHS):
    while distress():
        time.sleep(30); paused_s += 30
    te0 = time.time(); t_w = time.perf_counter()
    perm = torch.randperm(N, device=dev)
    for k in range(0, N, B):
        idx = perm[k:k + B]
        xb, yb = Xtr[idx], Ytr[idx]
        if torch.rand(()) < 0.5:
            xb = torch.flip(xb, dims=[1]); yb = yb[:, MIR]
        if graph is not None and len(idx) == B:
            xs.copy_(xb); ys.copy_(yb)
            graph.replay()
            loss = loss_s
        else:
            loss = step_eager(xb, yb)
        n_steps += 1
        if DUTY < 1.0 and n_steps % DW == 0 and not ON_CPU:
            torch.cuda.synchronize()
            d = duty_now()
            nap = (time.perf_counter() - t_w) * (1.0 - d) / d
            time.sleep(nap); idle_s += nap
            while temp_now[0] >= T_HARD:          # thermal hold: the card is at CrashGuard's kill line
                time.sleep(2.0); hold_s += 2.0; idle_s += 2.0
                if temp_now[0] < T_HARD - 1:
                    break
            t_w = time.perf_counter()
    sched_step()
    t_train += time.time() - te0
    if (ep + 1) % 10 == 0:
        e_va = evaluate(Xva, Yva)
        note = ""
        if e_va < best_val:
            best_val, best_ep = e_va, ep
            best_state = {k2: v.detach().clone() for k2, v in model.state_dict().items()}
            note = " *"
        print(f"ep {ep+1:4d}  train {loss.item():.4f}  val {e_va:.4f}  [{time.time()-t0:.0f}s]  "
              f"gpu {temp_now[0]}C duty {duty_now():.2f} hold {hold_s:.0f}s{note}", flush=True)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep + 1,
                        best_val=best_val, best_state=best_state, best_ep=best_ep, t_train=t_train,
                        shadow_lr=(shadow.param_groups[0]["lr"] if use_graph else None),
                        shadow_state=(shadow.state_dict() if use_graph else None),
                        rng_cpu=torch.get_rng_state(), rng_cuda=(None if ON_CPU else torch.cuda.get_rng_state())), ck.with_suffix(".tmp"))
        os.replace(ck.with_suffix(".tmp"), ck)        # atomic: a kill mid-write never leaves a half checkpoint
        if args.stop_after and ep + 1 >= args.stop_after:
            print(f"stop-after {ep+1}: checkpoint left in place", flush=True)
            stop.set(); sys.exit(3)

if best_state is not None:
    model.load_state_dict(best_state)
e_va = evaluate(Xva, Yva); e_te = evaluate(Xte, Yte); e_tta = evaluate(Xte, Yte, tta=True)
e_va_tta = evaluate(Xva, Yva, tta=True)
print(f"FINAL  val {e_va:.4f}  test {e_te:.4f}  test+TTA {e_tta:.4f}", flush=True)
np.save(OUT / f"{name}_predva.npy", predict(Xva, tta=True).cpu().numpy().astype(np.float32))
np.save(OUT / f"{name}_predte.npy", predict(Xte, tta=True).cpu().numpy().astype(np.float32))
torch.save(model.state_dict(), OUT / f"{name}.pt")
stop.set()
u = [x for x in util if x >= 0]
rec = dict(kind="mlp_scale", name=name, args=vars(args), epochs=EPOCHS, params=n_params,
           val=e_va, val_tta=e_va_tta, test=e_te, test_tta=e_tta, best_ep=best_ep,
           minutes_wall=(time.time() - t0) / 60, minutes_train=t_train / 60, paused_seconds=paused_s,
           duty=DUTY, duty_window=DW, idle_seconds=idle_s, resumed_from_epoch=(start_ep if resumed else None),
           thermal_soft_c=T_SOFT, thermal_hard_c=T_HARD, thermal_hold_seconds=hold_s, gpu_temp_max_c=temp_max[0],
           gpu_util_mean=float(np.mean(u)) if u else None, gpu_util_median=float(np.median(u)) if u else None,
           gpu_util_samples=len(u), gpu_mem_peak_card_mb=mem_peak[0] // 2 ** 20,
           vram_reserved_peak_mb=(0 if ON_CPU else torch.cuda.max_memory_reserved() // 2 ** 20),
           data_sha256=data_sha, split_seed=args.seed, torch=torch.__version__, cuda_graph=use_graph,
           device=DEVICE_NAME, threads=torch.get_num_threads(), tf32=False, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
           script_sha256=sha(Path(__file__)))
(OUT / f"{name}.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
if ck.exists():
    ck.unlink()
print("saved", name, flush=True)
