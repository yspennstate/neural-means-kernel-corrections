# Why the deep kernel head wins: it is alignment, not conditioning

The same exact Matern kernel scores 40% on the raw OCO-2 O2 input and 3.8% on
the trained network's features. Two candidate explanations, tested directly.

## Not the spectrum

Effective dimension of the Matern Gram (n = 4000, matched median length scale):

| nugget | raw input | features |
|-------:|----------:|---------:|
| 1e-8 | 3995.5 | 3993.9 |
| 1e-6 | 3614.1 | 3542.1 |
| 1e-4 | 692.8 | 795.5 |

Top-10 eigenvalue share: raw 0.669, features 0.692. The two Gram matrices have
almost the same spectrum; the feature kernel is if anything slightly less
concentrated. The effective-dimension lemma does not explain the gap, so the
win is not a conditioning or capacity effect.

## It is alignment

What changes is how well the target lives in the kernel's space. The RKHS
interpolation norm of the target (n = 4000, matched median length scale):

| kernel | RKHS norm of the target |
|--------|------------------------:|
| raw input | 1.64e6 |
| learned features | 3.87e4 |

a factor of 42.5. This is the mechanism. The optimal-recovery bound already in
the paper is error ≤ ‖G − m‖ · P_λ, a product of the target's RKHS norm and a
design factor that depends only on the Gram spectrum. The spectrum barely
moves between the two kernels (the effective dimensions above are nearly
equal), so the design factor is essentially fixed; the entire representational
gap is the first factor, which the features cut by more than an order of
magnitude. The network reshapes the input geometry so the map becomes a
low-norm element of the kernel's native space, and the exact solve recovers
it. The observed error ratio (about tenfold) is smaller than the norm ratio,
as an upper bound should be.
