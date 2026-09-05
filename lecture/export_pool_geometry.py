"""Bind the sixty-predictor lecture figures to the released second moments."""
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'campaign'))
from check_published_matrices import solve_simplex


def main():
    source = ROOT/'campaign/collected/dgx/seedarch.json'
    raw = source.read_bytes()
    data = json.loads(raw.decode('utf-8'))
    matrix = np.asarray(data['S_ev'], dtype=np.float64)
    if matrix.shape != (60,60) or (data['n_cal'],data['n_ev']) != (1000,19000):
        raise ValueError('Unexpected pool identity')
    configurations = list(dict.fromkeys(name.rsplit('_s',1)[0] for name in data['names']))
    groups = [[i for i,name in enumerate(data['names']) if name.rsplit('_s',1)[0] == a]
              for a in configurations]
    assert [len(g) for g in groups] == [10]*6
    diagonal = min(matrix.diagonal())
    within = min(matrix[i,j] for group in groups for i,j in combinations(group,2))
    between = min(matrix[i,j] for a,b in combinations(groups,2) for i in a for j in b)
    assert 0 <= between <= within <= diagonal
    floor2 = between+(within-between)/6+(diagonal-within)/60
    # Second expression uses the theorem's normalized coefficients.
    normalized = diagonal*(between/diagonal+(within/diagonal-between/diagonal)/6+
                           (1-within/diagonal)/60)
    np.testing.assert_allclose(floor2, normalized, atol=1e-16)
    optimum = solve_simplex(matrix)
    curves = {}
    for a,group in zip(configurations,groups):
        curve = []
        for k in (1,2,3,5,10):
            scores = [math.sqrt(float(matrix[np.ix_(ix,ix)].sum())/k**2)
                      for ix in combinations(group,k)]
            mean = math.fsum(scores)/len(scores)
            published = data['seed_curves'][a][str(k)]['e2_mean']
            np.testing.assert_allclose(mean,published,atol=5e-8,rtol=0)
            curve.append(dict(k=k,mean_rms=mean,n_subsets=len(scores)))
        curves[a] = curve
    assert source.read_bytes() == raw
    out = ROOT/'lecture/assets/pool_geometry.json'
    out.write_text(json.dumps(dict(source=source.relative_to(ROOT).as_posix(),
        source_sha256=hashlib.sha256(raw).hexdigest(), names=data['names'], configurations=configurations,
        n_cal=data['n_cal'],n_ev=data['n_ev'],second_moments=matrix.tolist(),curves=curves,
        block=dict(diagonal_min=float(diagonal),within_min=float(within),between_min=float(between),
                   rms_lower_bound=math.sqrt(floor2)),optimum=optimum,
        equal_rms=math.sqrt(float(matrix.sum())/3600),
        interpretation='Historical evaluation second moments; hindsight optimization on these same cases. Not population confidence bounds.',
        units='relative-error fractions; multiply RMS by100 for percentages'),indent=2),encoding='utf-8')
    print(json.dumps(dict(floor_rms=math.sqrt(floor2),optimum_rms=optimum['rms_upper'],
                          gap_percentage_points=100*(optimum['rms_upper']-math.sqrt(floor2)))))


if __name__=='__main__':
    main()
