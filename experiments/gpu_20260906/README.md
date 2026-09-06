# GPU and CPU experiments of 6 September 2026

Scripts, result records and figures from the re-runs that extend the paper's numerics: the spectrum of the
Matern-5/2 smoother on the full 19,000-row structural-mechanics design (spectrum/), and the residual MLP
re-runs at larger widths and depths with the paper's protocol held fixed (runs/). REPORT.md summarises the
results; README_DGX.md describes how the remaining queue runs on the Caltech DGX's CPUs.

Contents

    runs/train_mlp_scale.py     residual MLP trainer (CUDA-graph step on a GPU, eager on CPU), resumable, records provenance
    runs/jpl_scale.py           OCO-2 residual MLP at larger sizes
    runs/lk_scale.py            kernel-flows learned metrics on OCO-2 O2 at larger sizes (float64)
    runs/queue_runner.py        sequential job queue
    runs/structmech/*.json      finished runs (seed 0: w1024 and w2048, metric and MSE losses)
    runs/controls/              the graph-versus-eager pairs and the paper's own script re-run on CUDA
    runs/resumetest/            the stop-at-10-and-resume control against the uninterrupted run
    spectrum/full_design_spectrum.py, topk_lanczos_cpu.py, gpu_blocked_routes.py
    spectrum/full_design_19000/ result.json, spectrum_full.npz (all 19,000 eigenvalues), topk_lanczos.json,
                                power_law_fits.json, the two figures
    analysis/                   collect_scale.py (tables), ensemble_floor.py; analysis/out holds the current tables
    queue/                      the laptop queue and the CPU-ordered DGX queue, with the runner's logs
    logs/                       job logs
    PROVENANCE.json             hashes of the inputs pinned at the start

The data files (loads, targets, the OCO-2 arrays) are not in the repository; their sha256 values are in
PROVENANCE.json and in every result record. Weights and prediction arrays stay with the workspace.
