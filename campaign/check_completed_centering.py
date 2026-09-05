"""Recompute completed centering scores and disagreement from saved fields.

No fitting or model selection occurs. Predictions are streamed in 200-case
blocks. Ensemble variance uses pairwise differences, independently of the
producer's subtraction of the ensemble mean. A named seed subset is useful
for early checks but is never labelled a completed ten-seed campaign.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--historical-root', required=True, type=Path)
    parser.add_argument('--seeds', default=','.join(map(str, range(10))))
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    seeds = list(map(int, args.seeds.split(',')))
    if seeds != sorted(set(seeds)) or not seeds or not set(seeds) <= set(range(10)):
        raise ValueError('Expected unique ordered seeds in 0..9')
    if args.out.exists():
        raise ValueError('Refusing to replace an existing check')
    root = args.root.resolve()
    data = args.historical_root / 'data/structmech'
    identities = {}

    def pin(path):
        key = str(path)
        identity = sha(path)
        if key in identities and identities[key] != identity:
            raise ValueError('Input changed while checking: ' + key)
        identities[key] = identity
        return path

    target = np.load(pin(data / 'stress.npy'), mmap_mode='r')
    indices = np.load(pin(data / 'idx_test.npy'))
    if not np.array_equal(indices, np.arange(20000, 40000)):
        raise ValueError('Unexpected mechanics test ordering')
    rows = []
    for seed in seeds:
        folder = root / 'seeds' / f'sm_s{seed}'
        summary = read(pin(folder / 'summary.json'))
        error_path = pin(folder / 'paired_errors.npz')
        if summary['seed'] != seed or summary['error_archive_sha256'] != sha(error_path):
            raise ValueError('Completed centering receipt does not match')
        with np.load(error_path, allow_pickle=False) as z:
            if not np.array_equal(z['test_indices'], indices):
                raise ValueError('Paired errors use a different test order')
            errors = {arm: z[arm].copy() for arm in ('pooled', 'local')}
        for arm, subdir in (('pooled', 'pooled'), ('local', 'runs')):
            runs = folder / subdir
            prediction = np.load(pin(runs / 'hpix_corr_pred_test.npy'), mmap_mode='r')
            uq_path = pin(runs / 'hpix_uq.npz')
            uq_report = read(pin(runs / 'hpix_uq.json'))
            names = uq_report['members']
            if len(names) != 6 or len(set(names)) != 6 or 'krr' not in names:
                raise ValueError('Unexpected six-member disagreement pool')
            members = [np.load(pin(runs / ('krr_full_matern52_n19000_pred_test.npy'
                       if name == 'krr' else name + '_predte.npy')), mmap_mode='r')
                       for name in names]
            if any(array.shape != (20000, 1681) for array in [prediction] + members):
                raise ValueError('Unexpected prediction shape')
            with np.load(uq_path, allow_pickle=False) as z:
                absolute = z['err'].copy()
                disagreement = z['disagree'].copy()
            if errors[arm].shape != (20000,) or absolute.shape != (20000,) or disagreement.shape != (20000,):
                raise ValueError('Incomplete per-case score arrays')
            values = []
            gaps = dict(relative=0., absolute=0., disagreement=0., scalar_control=0.)
            for start in range(0, len(indices), 200):
                stop = min(start + 200, len(indices))
                y = np.asarray(target[indices[start:stop]], dtype=np.float64).reshape(-1, 1681)
                residual = np.asarray(prediction[start:stop], dtype=np.float64) - y
                norm = np.sqrt(np.einsum('ij,ij->i', residual, residual))
                relative = norm / np.sqrt(np.einsum('ij,ij->i', y, y))
                block = [np.asarray(member[start:stop], dtype=np.float64) for member in members]
                pair_sum = np.zeros_like(y)
                for a, b in itertools.combinations(block, 2):
                    pair_sum += (a - b) ** 2
                scale = np.sqrt(pair_sum / len(block) ** 2).mean(axis=1)
                for key, calculated, reported in (
                    ('relative', relative, errors[arm][start:stop]),
                    ('absolute', norm, absolute[start:stop]),
                    ('disagreement', scale, disagreement[start:stop])):
                    if not np.isfinite(calculated).all() or not np.allclose(calculated, reported, rtol=2e-11, atol=2e-12):
                        raise ValueError(f'Saved-field disagreement: seed={seed}, arm={arm}, metric={key}')
                    gaps[key] = max(gaps[key], float(np.max(np.abs(calculated - reported))))
                for case in (0, 9999, 19999):
                    if start <= case < stop:
                        j = case - start
                        direct = math.sqrt(math.fsum(float(v) ** 2 for v in residual[j]) /
                                           math.fsum(float(v) ** 2 for v in y[j]))
                        gaps['scalar_control'] = max(gaps['scalar_control'], abs(direct-relative[j]))
                values.extend(relative)
            mean = math.fsum(map(float, values)) / len(values)
            if not math.isclose(mean, summary['metrics'][arm]['mean_relative_l2'], rel_tol=2e-11, abs_tol=2e-12):
                raise ValueError('Reconstructed mean does not reproduce the completed summary')
            if gaps['scalar_control'] > 2e-12:
                raise ValueError('Direct scalar control failed')
            rows.append(dict(seed=seed, arm=arm, cases=len(values), members=len(members),
                             mean_relative_l2=mean, maximum_absolute_gaps=gaps))
            print('CHECKED', seed, arm, json.dumps(gaps), flush=True)
    for name, identity in identities.items():
        if sha(Path(name)) != identity:
            raise ValueError('Input changed during verification: ' + name)
    result = dict(kind='completed_centering_prediction_recomputation', seeds=seeds, rows=rows,
                  complete_seed_set=seeds == list(range(10)),
                  campaign_completion_claim=False,
                  computation='Float64 field errors; pairwise-difference population variance; scalar fsum controls',
                  input_sha256=identities, driver_sha256=sha(Path(__file__)))
    args.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
