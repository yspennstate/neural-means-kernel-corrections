# Running the NMKC training queue on the Caltech DGX (CPU only)

Owner order, 6 September 2026 ~12:0x his clock: DNN training for the kernel work runs on the Caltech box, not on
the laptop. The DGX's four GPUs are wedged (never call nvidia-smi there), so everything below is CPU.

## Layout on the box

    /raid/yitz/nmkc_gpu_experiments/        <- untar nmkc_bundle.tar here (it contains the nmkc_gpu_experiments/ tree)
        code/           the paper's code (common.py, train_mlp.py, jpl_data.py, ...)
        data/           structmech (264 MB) and jpl_oco2 (173 MB); sha256 of every file in PROVENANCE.json
        runs/           train_mlp_scale.py, jpl_scale.py, lk_scale.py, queue_runner.py
        runs/control/   queue_dgx_cpu.txt (147 lines, cheapest first); the runner writes runner.log, done.txt, failed.txt
        records/        the paper's own run records, for the comparisons
        analysis/       collect_scale.py (tables and figures from runs/*/*.json)

## Environment (every process)

    export NMKC_W=/raid/yitz/nmkc_gpu_experiments      # workspace root (the scripts default to a Windows path)
    export NMKC_DEVICE=cpu                             # no CUDA graph, no NVML, no duty cycle, CPU RNG only
    export NMKC_THREADS=8                              # torch threads per job (the box rule: <= 10 cores of ours)
    export NMKC_NO_NVSMI=1                             # the runner's VRAM gate never calls nvidia-smi
    export NMKC_COMPUTE_STATE=/nonexistent             # no laptop compute-state file; distress() then returns False
    export NMKC_DATA=$NMKC_W/data/structmech
    export NMKC_PY=$NMKC_W/../venv/bin/python           # whichever python has torch (2.13.0+cpu is fine) and numpy/scipy
    export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

## One job by hand (the first line of the queue: the paper's own MSE script on CPU, the provenance control)

    cd $NMKC_W/code && NMKC_RUNS=$NMKC_W/runs/controls/paper_script_cpu nice -n 15 $NMKC_PY train_mlp.py --mse 1 --epochs 120 --seed 0 --threads 6 --tag mlpMSE
    # expected: about 1 s/epoch on 6-8 threads (the record took 1.9 min on the paper's machine); compare the json
    # in runs/controls/paper_script_cpu with records/runs/mlpMSE_s0_w1024_d4_n19000_mir.json (test 0.047277, TTA 0.047053)

## The queue (one job at a time, nice 15)

    cd $NMKC_W/runs && touch control/GPU_GO && echo 1 > control/parallel.txt
    cp control/queue_dgx_cpu.txt control/queue.txt
    nohup nice -n 15 $NMKC_PY queue_runner.py --parallel 1 > ../logs/queue_runner_dgx.log 2>&1 &
    # pause: touch control/PAUSE ; stop after the running job: touch control/STOP
    # each finished job writes runs/structmech/<name>.json (or runs/jpl, runs/lk) with data sha256, script sha256,
    # device "cpu:8thr", threads, wall minutes; checkpoints every 10 epochs (atomic), resume is bit-exact.

Memory: a w1024 d4 MLP job needs about 1.5 GB RSS; w4096 d6 about 6 GB; lk_scale --ntr 12000 holds one
12000 x 12000 float64 matrix (1.2 GB) plus its factor, about 4 GB. All under the box's 10 GB rule.

Expected time per job on 8 threads (from the paper's CPU records at ~1 s/epoch for w1024): w1024 metric 400 epochs
7-10 min, w1024 MSE 120 epochs 2-3 min, w2048 about 4x, w4096 about 16x (hours). The queue is ordered so the seed
blocks at w1024 and the 6000-row kernel-flows land first.

## Reporting back

Copy `runs/structmech/*.json`, `runs/jpl/*.json`, `runs/lk/*.json` (small) back to the laptop workspace
`C:\Users\owner\GOAL20H_20260906\nmkc_gpu_experiments\runs\` under the same names; `analysis/collect_scale.py`
rebuilds the tables from them. Weights (.pt) and predictions (.npy) can stay on /raid.
