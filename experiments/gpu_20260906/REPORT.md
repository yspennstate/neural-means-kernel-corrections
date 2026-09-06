# NMKC numerics extended: the full 19,000-row spectrum and the MLP re-runs

F5 (claude-f5-gputrain-dd39), 6 September 2026, written 13:3x. Workspace
`C:\Users\owner\GOAL20H_20260906\nmkc_gpu_experiments\` (every number below has a JSON record there with the
input hashes, the script hash and the machine state).

## 1. The full-design spectrum (done, handed to the paper worker, accepted)

The paper's spectral diagnostic of the Matern-5/2 smoother was computed on a 6,000-row subsample of the 19,000
training rows. The full design was computed here in float64 on the CPU, three ways:

| route | effective dimension at nugget 1e-5 (shift n x nugget) |
|---|---|
| LAPACK dsyevd on the 19,000 x 19,000 Gram (all eigenvalues) | 103.43044450254591 |
| Cholesky trace of the inverse (no eigenvalues) | 103.43044450254092 |
| ARPACK Lanczos, top 64 eigenvalues, independent code | top eigenvalues agree with LAPACK to about 1e-12 relative |

The two full routes agree to 5e-12. The paper's 6,000-row value, 103.11885064181605, was reproduced from the
packet's own design file to 7.7e-12, and the 6,000-row eigenvalues match the packet's spectrum file to 3.6e-12.

So the full design has effective dimension 103.43 against the subsample's 103.12, a ratio of 1.0030. The top
mode carries 0.9427 of the trace; eight modes give 99 percent, forty-five give 99.9 percent; the smallest
eigenvalue is 5.2e-7 and none is negative; the trace is 19,000 to machine precision. Nothing in the paper's
conclusions moves; the full-design figure can be quoted beside the subsample one. One convention trap: at the
6,000-row shift (0.06 instead of 0.19) the full design would read 178.96; the two must not be mixed.

Figure: `spectrum/full_design_19000/spectra_full_vs_subsample_loglog.pdf` (normalized spectra of both designs,
and the effective dimension as a function of the nugget with the 1e-5 point marked; a linear-axis version sits
beside it).

Seen on logarithmic axes the two spectra are one curve: normalized by n, the subsample's eigenvalues lie on the
full design's over three decades (ratio 0.9967 in the median over indices 10 to 1000) and fall as index^-2.1
(full -2.14, subsample -2.15), the subsample departing only in its last thousand modes where it runs out of rows.
The effective dimension is a power law in the nugget over five decades, nugget^-0.46, which is what an index^-2.14
spectrum predicts (-1/2.14 = -0.47): 299 at 1e-6, 103.4 at 1e-5, 38.4 at 1e-4. The subsample diagnostic in the
paper is therefore the full design's spectrum, not an approximation of it; the 0.3 percent at 1e-5 is the tail.
Fits: `power_law_fits.json`.

## 2. The MLP re-runs on the laptop GPU (seed 0, before the 12:0x order)

Protocol held fixed from the paper: canonical split (1,000 validation rows by permutation seed s, 19,000 train,
the 20,000-row test block untouched), AdamW 1e-3, weight decay 1e-5, cosine to 1e-6, batch 256, reflection
augmentation with p = 0.5, 400 epochs for the metric loss and 120 for the MSE loss, best-validation checkpoint,
relative L2 error in float64. Training step captured in a CUDA graph; the eager twin agrees to 3e-6 on predictions.

| configuration | params | val | test | test + TTA | best epoch |
|---|---|---|---|---|---|
| w1024 d4, metric loss (the paper's) | 4.9M | 0.050078 | 0.049671 | 0.048621 | 79 |
| paper's record | | 0.050078 | 0.049671 | 0.048621 | 79 |
| w2048 d4, metric loss | 16.1M | 0.05215 | 0.05185 | 0.04951 | 59 |
| w1024 d4, MSE loss | 4.9M | 0.04769 | 0.04735 | 0.04712 | 119 |
| paper's MSE record | | 0.04757 | 0.04728 | 0.04705 | 119 |
| w2048 d4, MSE loss | 16.1M | 0.04782 | 0.04747 | 0.04720 | 119 |

The paper's metric-loss run is reproduced to 7.6e-7 relative. Doubling the width does not help at seed 0: the
metric-loss w2048 run is worse than w1024 and the MSE w2048 run is no better. The seed-resolved table (seeds 1 to 4,
widths to 4096, depths 5 and 6) is queued and will replace this one.

The MSE record is a provenance question, not a correction: the packet's own `train_mlp.py --mse 1`, run unchanged
on the GPU, gives test 0.047355 and TTA 0.047121, the same as the re-implementation to 4e-7, and 2.5e-3 relative
away from the record's 0.047277 and 0.047053. The record's own metadata says a six-thread CPU run; a CPU run of
the shipped script is the first line of the queue below. The gap is 0.008 percentage points on a 4.7 percent error.

Checkpointing and resumption were verified exact: a run stopped at epoch 10 and resumed gives predictions
identical to the bit to the uninterrupted run.

## 3. Where the queue runs now

At about 12:0x his clock the owner ordered that DNN training for the kernel work happens on the Caltech box, with
this laptop's GPU reserved for the films at 70 percent. The remaining 147 jobs (structural-mechanics MLPs at widths
1024 to 4096 and depths 4 to 6 over five seeds, the OCO-2 residual MLPs, and the kernel-flows learned-metric
comparisons on 6,000 and 12,000 rows in float64) were made CPU-capable and ordered by cost for the DGX's CPUs (its
four GPUs have been wedged since 28 August). They run one at a time under the DGX claim holder; a w1024 lane is
expected to take minutes on eight threads, a w4096 lane hours. Results are appended to
`analysis/out/scale_tables.md` as they land and handed to the paper worker in the same form as section 1.

## 4. What went wrong on the laptop, for the record

The three training deaths at 10:46 to 10:50 were the owner's CrashGuard killing python GPU clients at its red tier
(a 57M-parameter job held the card at 100 percent), and a 12:28 change of two feeders to Normal priority on the
display GPU coincided with the freeze that prompted the order in section 3. Both are written to the brain.
