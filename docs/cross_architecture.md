# The ensembling floor is a property of the architecture class

The WCO2 reduced-radiance error sits at 16% because independently seeded
copies of the residual MLP correlate at 0.97, and the second-moment floor
caps any ensemble of them near the single-member error. That correlation is
what makes the floor; it is not a property of the problem, but of training the
same architecture twice.

Four genuinely different architectures on WCO2, and their pairwise residual
correlations:

| architecture | test error |
|--------------|-----------:|
| residual SiLU MLP | 16.4% |
| ReLU MLP | 19.6% |
| wide shallow MLP | 41.7% |
| random Fourier features | (bandwidth-dependent) |

| pair | correlation |
|------|------------:|
| ReLU x SiLU | 0.840 |
| ReLU x wide | 0.341 |
| SiLU x wide | 0.402 |
| ReLU x Fourier | 0.079 |
| SiLU x Fourier | 0.114 |

Against 0.97 for seeds, cross-architecture correlations run from 0.84 (two
plain MLPs, still similar) down to 0.08 (the Fourier-feature network against
either MLP). The floor of the corollary is therefore not fundamental: a
different inductive bias breaks the correlation that sets it.

The catch is the other factor in the floor. A decorrelated member lowers the
ensemble only if it is also accurate: the two-member condition is
`corr < e_reference / e_member`, so a member three times worse must correlate
below one third to help. The Fourier network at the first bandwidth tried sat
exactly on that boundary. The bandwidth sweep in `box_fourier_sweep.py` looks
for the setting where the Fourier member is both accurate and decorrelated,
which is what a genuinely lower floor requires.
