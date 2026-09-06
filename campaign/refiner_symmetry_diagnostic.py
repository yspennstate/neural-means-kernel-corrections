"""Check supplied-field reflection separately from full-map symmetrization.

No training or benchmark predictions are generated. The two-point construction
is a mathematical counterexample. The empirical lane reads ten retained logs.
"""
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def witness():
    # Explicit vector operations, with the output reflection a coordinate swap.
    target = (1, 1)
    h = {1: (1, 1), -1: (3, 3)}
    def reflect(z):
        return z[::-1]
    def refiner(u, z):
        return z if u == 1 else tuple(4 - v for v in z)
    def average(a, b):
        return tuple(Fraction(x + y, 2) for x, y in zip(a, b))
    rows = []
    for u in (1, -1):
        plain = refiner(u, h[u])
        full = average(plain, reflect(refiner(-u, h[-u])))
        supplied = average(plain, reflect(refiner(-u, reflect(h[u]))))
        assert reflect(target) == target and h[-u] != reflect(h[u])
        assert plain == full == target and supplied == (2, 2)
        numerator = sum((v - g) ** 2 for v, g in zip(supplied, target))
        denominator = sum(g * g for g in target)
        assert numerator / denominator == 1
        left = refiner(-u, reflect(h[u]))
        right = refiner(-u, h[-u])
        defect_squared = sum((a-b)**2 for a, b in zip(left, right))
        channel_squared = sum((a-b)**2 for a, b in zip(reflect(h[u]), h[-u]))
        # Both maps F(u,.) have operator norm one; the relative defect is 2.
        assert Fraction(defect_squared, denominator) == 4
        assert defect_squared == channel_squared
        rows.append(dict(input=u, plain=list(plain),
                         full_map=list(map(float, full)),
                         supplied_field=list(map(float, supplied))))
    # A separate scalar derivation: both coordinates equal. The plain branches
    # are 1 and 4-3=1; the two implemented averages are (1+3)/2=2.
    scalar_plain = (1, 4-3)
    scalar_full = (sum(scalar_plain)/2,) * 2
    scalar_supplied = ((1+(4-1))/2, ((4-3)+3)/2)
    assert scalar_plain == scalar_full == (1, 1)
    assert scalar_supplied == (2, 2)
    # Positive control: a truly equivariant supplied field restores equality
    # of the two averaging rules, for arbitrary refiner outputs here.
    equivariant_h = {1: (1, 2), -1: (2, 1)}
    for u in (1, -1):
        assert equivariant_h[-u] == reflect(equivariant_h[u])
        direct = refiner(u, equivariant_h[u])
        full = average(direct, reflect(refiner(-u, equivariant_h[-u])))
        supplied = average(direct, reflect(refiner(-u, reflect(equivariant_h[u]))))
        assert full == supplied
    return dict(rows=rows, plain_relative_risk=0, full_map_relative_risk=0,
                supplied_field_relative_risk=1,
                refiner_channel_lipschitz_constant=1,
                mean_relative_branch_defect=2,
                risk_bound=1,
                defect_bound_equality='PASS',
                independent_scalar_derivation='PASS',
                equivariant_channel_control='PASS',
                scope='Exact finite counterexample, not a benchmark simulation')


def retained_records():
    rows = []
    with localcontext() as context:
        context.prec = 50
        decimal_deltas = []
        for seed in range(10):
            path = ROOT / f'campaign/collected/dgx/runs/sm_s{seed}/mlpR_s{seed}_w1024_d4.json'
            raw = path.read_text(encoding='utf-8')
            binary = json.loads(raw)
            decimal = json.loads(raw, parse_float=Decimal)
            assert binary['kind'] == 'mlpR' and binary['args']['seed'] == seed
            delta = 100 * (binary['test_notta'] - binary['test'])
            exact = Decimal(100) * (decimal['test_notta'] - decimal['test'])
            assert abs(delta - float(exact)) < 1e-14 and delta > 0
            decimal_deltas.append(exact)
            rows.append(dict(seed=seed, source=path.relative_to(ROOT).as_posix(),
                sha256=sha(path), test_tta=binary['test'],
                test_plain=binary['test_notta'], improvement_pp=delta,
                decimal_improvement_pp=str(exact)))
        mean = math.fsum(row['improvement_pp'] for row in rows) / 10
        sd = statistics.stdev(row['improvement_pp'] for row in rows)
        dm = sum(decimal_deltas) / Decimal(10)
        ds = (sum((value-dm)**2 for value in decimal_deltas)/Decimal(9)).sqrt()
        assert abs(mean-float(dm)) < 1e-14 and abs(sd-float(ds)) < 1e-14
    macros = dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}',
                            (ROOT/'paper/macros.tex').read_text(encoding='utf-8')))
    assert macros['ttaGainRefiner'] == f'{dm:.4f}'
    assert macros['ttaGainRefinerSd'] == f'{ds:.4f}'
    text = (ROOT/'paper/supp_experiments.tex').read_text(encoding='utf-8')
    assert r'\ttaGainRefiner\pm\ttaGainRefinerSd' in text
    return dict(rows=rows, n=10, mean_improvement_pp=mean,
                sample_sd_improvement_pp=sd, decimal_mean_pp=str(dm),
                decimal_sample_sd_pp=str(ds), improved_seeds=10,
                paper_macros='PASS',
                scope='Retained per-seed log metrics; no fresh training or prediction reconstruction')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = dict(producer_sha256=sha(Path(__file__)),
                  input_code_sha256={name: sha(ROOT/name) for name in
                                     ('train_mlp_refine.py', 'gen_preds.py')},
                  paper_sha256={name: sha(ROOT/'paper'/name) for name in
                                ('macros.tex', 'supp_experiments.tex', 'supp_theory_stages.tex')},
                  witness=witness(), retained_records=retained_records())
    args.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='PASS', output_sha256=sha(args.out),
                         counterexample_risks=[0, 0, 1],
                         mean_improvement_pp=result['retained_records']['mean_improvement_pp'],
                         sample_sd_pp=result['retained_records']['sample_sd_improvement_pp'])))


if __name__ == '__main__':
    main()
