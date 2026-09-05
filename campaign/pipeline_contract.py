"""Bind a resumable campaign to explicit settings, code and input bytes."""
import hashlib
import json
from pathlib import Path


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def make_contract(settings, files):
    """Named inputs are stable across moves; contents determine identity."""
    return dict(schema='nmkc-pipeline-contract-v1', settings=settings,
                files={name: file_sha(path) for name, path in sorted(files.items())})


def require_contract(runs, expected):
    """Call while holding the seed lock, before any stage can skip outputs."""
    runs = Path(runs)
    receipt = runs / 'pipeline_contract.json'
    if receipt.exists():
        observed = json.loads(receipt.read_text(encoding='utf-8'))
        if observed != expected:
            raise RuntimeError('Resume refused: settings, source or data differ from '
                               'pipeline_contract.json. Use a separate campaign root.')
        return
    if any(runs.iterdir()):
        raise RuntimeError('Resume refused: existing outputs have no pipeline contract. '
                           'Preserve them and use an empty campaign root; do not infer '
                           'their centering convention from filenames.')
    with receipt.open('x', encoding='utf-8') as handle:
        json.dump(expected, handle, indent=2, sort_keys=True)
        handle.write('\n')


def verify_oof(runs, centering, split_seed, producer_sha256):
    """The actual OOF field must match its mode, split and producer receipt."""
    runs = Path(runs)
    metadata = json.loads((runs / 'krr_oof_train.json').read_text(encoding='utf-8'))
    if (metadata['target_centering'] != centering
            or metadata['split_seed'] != split_seed
            or metadata['driver_sha256'] != producer_sha256
            or metadata['field_sha256'] != file_sha(runs / 'krr_oof_train.npy')):
        raise RuntimeError('OOF field provenance differs from the requested run')
