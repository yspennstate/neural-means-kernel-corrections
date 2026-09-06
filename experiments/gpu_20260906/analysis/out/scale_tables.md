# Scaled experiments, collected 2026-09-06 12:58

## Structural mechanics: residual MLP, widths and depths (test error, percent, mean +- seed sd [n])

| loss | width | depth | amp | params | seeds | val | test | test+TTA | best epoch | min/run | GPU util % | VRAM peak MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metric | 1024 | 4 | none | 4.9M | 1 | 5.008 | 4.967 | 4.862 | 79 | 2.4 | 42 | 958 |
| metric | 2048 | 4 | none | 16.1M | 1 | 5.215 | 5.185 | 4.951 | 59 | 19.5 | 63 | 1470 |
| mse | 1024 | 4 | none | 4.9M | 1 | 4.769 | 4.735 | 4.712 | 119 | 1.0 | 41 | 994 |
| mse | 2048 | 4 | none | 16.1M | 1 | 4.782 | 4.747 | 4.720 | 119 | 1.9 | 72 | 1390 |

Paper's seed-0 records for reference: metric loss val 5.008 test 4.967 test+TTA 4.862 (best epoch 79); MSE val 4.757 test 4.728 test+TTA 4.705.

### Paired differences against w1024 d4 (same loss, within seed; percent points, mean +- sd; resolved = |mean| > 2 se)

| loss | width | depth | amp | n | test+TTA diff | test diff | resolved |
|---|---|---|---|---|---|---|---|
| metric | 2048 | 4 | none | 1 | +0.089 +- nan | +0.217 +- nan | no |
| mse | 2048 | 4 | none | 1 | +0.008 +- nan | +0.012 +- nan | no |

## OCO-2 emulation: residual MLP means at widths and depths (percent; reduced / radiance; mean +- seed sd)

| band | width | depth | head | seeds | flat reduced | flat radiance | weighted reduced | weighted radiance | combined reduced | combined radiance | kernel-flow reduced / radiance | min/run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Kernel flows on OCO-2 O2: learned metrics at larger sizes (test error percent, exact ridge; mean +- seed sd)

| space | rows | steps | batch | rank | seeds | iso | ard | lowrank | ard - iso | lowrank - iso | min/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
