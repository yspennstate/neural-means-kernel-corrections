# Refiner reflection averaging: implementation and guarantee

The recorded kernel-conditioned refiner evaluates a fitted network `F` twice:

```
b(u) = [F(u, h(u)) + T F(Su, T h(u))] / 2.
```

Here `h(u)` is the supplied kernel prediction, `S` reverses the load samples,
and `T` reverses the first coordinate of the output grid. The second branch
permutes the supplied field. It does not recompute `h(Su)`. Both
`train_mlp_refine.py:evaluate` and the `mlpR` branch of `gen_preds.py:predict`
use this convention, including normalization and output denormalization.

The risk theorem concerns averaging a complete predictor. For
`f(u) = F(u, h(u))`, that average is

```
f_S(u) = [F(u, h(u)) + T F(Su, h(Su))] / 2.
```

The condition `h(Su) = T h(u)` is sufficient for equality of these two
operations. We have not established it for the fitted, centered,
finite-sample kernel predictor. Symmetry of the target and input law does
not establish symmetry of the fitted kernel channel. Randomly reflected
training triples also do not force exact equivariance of the trained
network. Thus the risk theorem alone does not guarantee non-increase for
the recorded refiner operation.

The supplement now gives a relative-risk bound with the explicit branch
defect. Its two-point counterexample has raw and full-map risk zero but
supplied-field risk one. The example attains equality in the defect bound
with channel Lipschitz constant one. It disproves an unconditional
guarantee; it is not a benchmark simulation or evidence that the recorded
models worsened. No branch-defect or Lipschitz-bound measurement was
performed on those models.

## Reproduce the correction's checks

From the repository root, with Python 3 and its standard library:

```
python campaign/refiner_symmetry_diagnostic.py --out refiner_symmetry_check.json
```

The script checks the finite counterexample with rational arithmetic, a
separate scalar derivation, and an equivariant-channel positive control.
It then reads the ten retained `mlpR` logs under
`campaign/collected/dgx/runs/sm_s<seed>/`, hashes each input, and computes
`100 * (test_notta - test)` in percentage points using both binary
arithmetic and 50-digit decimal arithmetic. It validates the two paper
macros used in the supplement.

All ten recorded differences are positive. Their mean is
0.042187818239107014 percentage points and their sample standard deviation
is 0.001892694803349329 percentage points; the manuscript prints
0.0422 +/- 0.0019. These are retained-log measurements. The script does not
train models, regenerate predictions, or replay a large benchmark archive.
The prior statement that the effect was absent by construction has been
removed from the manuscript.

## Version and review scope

This correction changes the mathematical scope and explanatory text, and
adds reproducibility checks. It does not change historical model weights,
prediction arrays, or reported ensemble errors. Historical review votes
remain attached to their original source and PDF packets. A changed
manuscript requires review of its own exact bytes.


## Presentation edits following the independent V1 review

The independent F2 report recommends publication of the immutable V1 packet
(source commit 47c29c8a32ade3b1131d4c77999973ad171e666a), with minor edits.
This revision identifies the Marchenko--Pastur simulation figure as one
realization, names the separate supplement.pdf, foregrounds the sufficient
conditioning-field equivariance condition, and writes all four evaluations
in the two-point counterexample. Numerical records and theorem statements
are unchanged. The paired build and author visual verification have completed;
a changed-packet reassessment remains required.

The review's requested "if and only if" wording was not adopted. Equivariance
of h is sufficient, but need not be necessary when F ignores or discards part
of that channel. Pointwise equality of the two averages is equivalent to
F(Su, T h(u)) = F(Su, h(Su)); it does not imply equality of the two inputs
without injectivity of F(Su, .). The reviewer has been sent the constant-F
countercase. The previous V1 report and packet remain preserved separately.
