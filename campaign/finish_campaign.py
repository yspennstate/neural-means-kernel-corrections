"""One bounded continuation of the already-fixed Caltech campaign.

Wait for completion, check raw predictions on one CPU, then collect evidence.
No training, retry queue, GPU work or modification of executed model code.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/yitz/nmkc_paper1_20260905')
PYTHON = '/home/yitz/nmkc_venv/bin/python'
START = time.time()
DEADLINE = START + 5400
DRIVER = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
STATE = ROOT / 'finalization_status.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def state(phase, **detail):
    payload = dict(phase=phase, started_at=START, updated_at=time.time(),
                   deadline=DEADLINE, pid=os.getpid(), cpu=39,
                   driver_sha256=DRIVER, **detail)
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(STATE)
    print(json.dumps(payload), flush=True)


def headroom():
    def cpu():
        with open('/proc/stat') as f:
            return list(map(int, f.readline().split()[1:9]))
    a = cpu(); time.sleep(2); b = cpu()
    delta = [y-x for x,y in zip(a,b)]
    idle = os.cpu_count()*(delta[3]+delta[4])/sum(delta)
    with open('/proc/meminfo') as f:
        memory = {line.split(':')[0]:int(line.split()[1]) for line in f}
    return dict(idle_cpu_equivalents=idle, available_gib=memory['MemAvailable']/2**20)


def run(command, log_name, maximum_seconds):
    remaining = DEADLINE-time.time()
    if remaining <= 0:
        raise TimeoutError('Finalization deadline reached')
    with (ROOT/'logs'/log_name).open('x', encoding='utf-8') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                       timeout=min(maximum_seconds, remaining), cwd=ROOT)


def validate_grid_check():
    report = read(ROOT/'grid_prediction_check.json')
    expected = {(b,s,g) for b in ('o2','wco2','sco2') for s in range(3)
                for g in ('recorded_grid','expanded_grid')}
    actual = {(r['band'],r['seed'],r['scenario']) for r in report['rows']}
    if (report['kind']!='completed_grid_prediction_recomputation'
            or report['seeds']!=[0,1,2] or report['bands']!=['o2','wco2','sco2']
            or report['driver_sha256']!=sha(ROOT/'check_completed_grids.py')
            or actual!=expected or len(report['rows'])!=18):
        raise ValueError('Raw grid check has incomplete or stale identity')
    for r in report['rows']:
        if r['models']!=14 or r['cases']!=2000 or not 0<=r['max_per_case_difference']<=2e-9:
            raise ValueError('Raw grid reconstruction failed')


def main():
    # Atomic launch ownership. An old lock requires inspection, never deletion
    # or an automatic replacement worker.
    (ROOT/'finalization.lock').mkdir()
    os.sched_setaffinity(0, {39})
    os.nice(19)
    for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    try:
        state('WAITING_FOR_FIXED_CAMPAIGN')
        while True:
            if time.time()>=DEADLINE:
                raise TimeoutError('Campaign did not finish within the bounded continuation')
            statuses = [read(ROOT/name).get('status') for name in ('status.json','followup_status.json')]
            if all(s=='COMPLETE' for s in statuses):
                break
            if any(s in ('FAILED','ERROR','STOPPED') for s in statuses):
                raise RuntimeError('An upstream campaign stopped: '+repr(statuses))
            time.sleep(30)
        for s in range(3):
            for b in ('o2','wco2','sco2'):
                if not (ROOT/'followup_receipts'/f'grid_{b}_s{s}.json').is_file():
                    raise ValueError('A planned grid receipt is absent')
        while True:
            if time.time()>=DEADLINE:
                raise TimeoutError('No safe headroom before finalization deadline')
            capacity = headroom()
            if capacity['idle_cpu_equivalents']>=2 and capacity['available_gib']>=4:
                break
            state('WAITING_FOR_HEADROOM', capacity=capacity)
            time.sleep(30)
        state('CHECKING_RAW_GRID_PREDICTIONS', capacity=capacity)
        if not (ROOT/'grid_prediction_check.json').exists():
            (ROOT/'grid_check_final.lock').mkdir()
            run([PYTHON,'-B',str(ROOT/'check_completed_grids.py'),'--root',str(ROOT),
                 '--data','/home/yitz/nmkc/data/jpl_oco2','--seeds','0,1,2',
                 '--out',str(ROOT/'grid_prediction_check.json')], 'grid_check_final.log', 1200)
        validate_grid_check()
        state('COLLECTING_EVIDENCE')
        archive = ROOT/'paper1_completed_evidence_20260905.zip'
        run([PYTHON,'-B',str(ROOT/'collect_sensitivity.py'),'--root',str(ROOT),
             '--out',str(archive)], 'collect_final.log', 600)
        state('COMPLETE', archive=archive.name, archive_sha256=sha(archive),
              grid_check_sha256=sha(ROOT/'grid_prediction_check.json'))
    except Exception as exc:
        state('FAILED', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':
    main()
