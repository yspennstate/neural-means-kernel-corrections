"""Compare a field-level float64 reconstruction with the retained pool analysis."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
from check_published_matrices import solve_simplex


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def floor(matrix,labels):
    d=float(np.min(np.diag(matrix)))
    w=min(float(matrix[i,j]) for i in range(60) for j in range(i) if labels[i]==labels[j])
    b=min(float(matrix[i,j]) for i in range(60) for j in range(i) if labels[i]!=labels[j])
    if not 0<=b<=w<=d:raise ValueError('Block-floor ordering fails')
    return math.sqrt(b+(w-b)/6+(d-w)/60)


def compare(check_path,published_path):
    hashes={str(p):sha(p) for p in (check_path,published_path)}
    check=json.loads(check_path.read_text(encoding='utf-8'))
    old=json.loads(published_path.read_text(encoding='utf-8'))
    pool=check['pool']
    if pool['names']!=old['names'] or (pool['n_fit'],pool['n_eval'])!=(old['n_cal'],old['n_ev']):
        raise ValueError('Predictor order or split sizes differ')
    labels=[name.rsplit('_s',1)[0] for name in pool['names']]
    new_matrix=np.asarray(pool['S_ev']);old_matrix=np.asarray(old['S_ev'])
    if new_matrix.shape!=(60,60) or old_matrix.shape!=(60,60):raise ValueError('Wrong matrix shape')
    new_optimum=solve_simplex(new_matrix);old_optimum=solve_simplex(old_matrix)
    new_floor=floor(new_matrix,labels);old_floor=floor(old_matrix,labels)
    # Check the precision actually displayed in the pool figure, while retaining
    # the full discrepancy. The original arrays and matrices are never replaced.
    for a,b in ((new_optimum['rms_upper'],old_optimum['rms_upper']),(new_floor,old_floor)):
        if f'{100*a:.4f}'!=f'{100*b:.4f}':raise ValueError('Displayed pool result requires revision')
    mean_gap=max(abs(pool['member_evaluation_means'][name]-old['single'][name]['e1']) for name in pool['names'])
    if mean_gap>1e-7:raise ValueError('Reconstructed individual means differ materially')
    if [r['seed'] for r in check['rows']]!=list(range(10)) or len(check['member_rows'])!=60:
        raise ValueError('Incomplete metric check')
    metrics={}
    for key in ('plain','trapezoidal','trapezoidal_minus_plain'):
        values=[r[key] for r in check['rows']]
        metrics[key]=dict(mean=math.fsum(values)/10,sd=statistics.stdev(values),per_seed=values)
    result=dict(input_sha256=hashes,driver_sha256=sha(Path(__file__)),
        solver_source_sha256=sha(Path(__file__).with_name('check_published_matrices.py')),
        matrix_max_absolute_difference=float(np.max(np.abs(new_matrix-old_matrix))),
        matrix_relative_frobenius_difference=float(np.linalg.norm(new_matrix-old_matrix)/np.linalg.norm(old_matrix)),
        individual_mean_max_difference=mean_gap,
        reconstructed_block_floor=new_floor,retained_block_floor=old_floor,
        reconstructed_optimum=new_optimum,retained_optimum=old_optimum,
        optimum_difference_percentage_points=100*(new_optimum['rms_upper']-old_optimum['rms_upper']),
        historical_corrected_pipeline_metrics=metrics,
        scope='Same historical field predictions and fixed pool split; float64 streamed reconstruction. Trapezoidal norms are a separate diagnostic; quoted comparators were not rescored.',
        displayed_pool_results_unchanged=True,units='fractions except explicitly named percentage-point difference')
    if hashes!={str(p):sha(p) for p in (check_path,published_path)}:raise ValueError('Inputs changed')
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--check',type=Path,required=True)
    ap.add_argument('--published',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise ValueError('Refuse to overwrite a comparison')
    result=compare(args.check,args.published)
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('reconstructed_optimum','retained_optimum')},indent=2))


if __name__=='__main__':main()
