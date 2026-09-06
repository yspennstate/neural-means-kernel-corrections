"""Exercise the driver's real command construction without training or data.

Only the orchestrator runs. Its preparation, cache checks and subprocess boundary
are replaced by recorders, and execution stops at OOF verification. These checks
establish argument routing; they do not establish numerical reproducibility.
"""
import contextlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
DRIVER=HERE/'seed_pipeline.py'

class RoutingComplete(Exception):pass

@contextlib.contextmanager
def fake_posix_lock_module():
    missing=object();previous=sys.modules.get('fcntl',missing)
    sys.modules['fcntl']=types.ModuleType('fcntl')
    try:yield
    finally:
        if previous is missing:sys.modules.pop('fcntl',None)
        else:sys.modules['fcntl']=previous

def trace(root, mode=None, source=None):
    source=DRIVER.read_text(encoding='utf-8') if source is None else source
    environment=dict(os.environ, NMKC_ROOT=str(root), NMKC_SEED='7', NMKC_THREADS='1')
    environment.pop('TASK_ID',None)
    environment.pop('NMKC_TARGET_CENTERING',None)
    if mode is not None:environment['NMKC_TARGET_CENTERING']=mode
    namespace={'__name__':'routing_dry_run','__file__':str(DRIVER)}
    with patch.dict(os.environ,environment,clear=True),fake_posix_lock_module():
        exec(compile(source,str(DRIVER),'exec'),namespace)
        calls=[];seen={}
        def make(settings,files):
            seen['contract_settings']=settings
            return {'files':{'code/krr_oof.py':'dry-run-producer-identity'}}
        def verify(runs,mode,seed,producer):
            seen['verification']={'mode':mode,'seed':seed,'producer':producer}
            raise RoutingComplete()
        namespace.update(prep_data_once=lambda:None,gram_lock=contextlib.nullcontext,
            make_contract=make,require_contract=lambda *args:None,
            version=lambda name:'not-loaded-in-command-routing-check',
            step=lambda outputs,argv,log:calls.append({'stage':log,'argv':list(map(str,argv))}),
            verify_oof=verify)
        try:namespace['main']()
        except RoutingComplete:pass
        else:raise AssertionError('Did not reach the expected stopping boundary')
        seen.update(calls=calls,seed_dir=str(namespace['SEED_DIR']),task_id=namespace['TASK_ID'])
        return seen

def assert_route(result,mode):
    assert [c['stage'] for c in result['calls']]==['krr','krr_oof']
    command=result['calls'][1]['argv']
    assert Path(command[1]).name=='krr_oof.py'
    assert command[2:]==['--target-centering',mode]
    assert result['contract_settings']['target_centering']==mode
    assert result['verification']['mode']==mode and result['verification']['seed']==7

class DriverRouting(unittest.TestCase):
    def setUp(self):
        base=HERE.parent/'.local-verification/pipeline-routing-tests'
        base.mkdir(parents=True,exist_ok=True)
        self.directory=tempfile.TemporaryDirectory(dir=base)
        self.root=Path(self.directory.name).resolve()
        assert self.root.is_relative_to(base.resolve())
        self.addCleanup(self.directory.cleanup)

    def test_historical_default_reaches_oof_as_pooled(self):
        assert_route(trace(self.root),'pooled')

    def test_both_explicit_recipes_route_and_remain_disjoint(self):
        pooled=trace(self.root,'pooled');local=trace(self.root,'fold-local')
        assert_route(pooled,'pooled');assert_route(local,'fold-local')
        self.assertNotEqual(pooled['seed_dir'],local['seed_dir'])
        self.assertNotEqual(pooled['task_id'],local['task_id'])

    def test_invalid_mode_stops_before_creating_runs(self):
        with self.assertRaisesRegex(ValueError,'NMKC_TARGET_CENTERING'):
            trace(self.root,'ambiguous')
        self.assertEqual(list(self.root.iterdir()),[])

    def test_missing_forwarded_argument_is_detected(self):
        source=DRIVER.read_text(encoding='utf-8')
        original='[PY, CODE / "krr_oof.py", "--target-centering", TARGET_CENTERING]'
        self.assertEqual(source.count(original),1)
        broken=source.replace(original,'[PY, CODE / "krr_oof.py"]')
        with self.assertRaises(AssertionError):assert_route(trace(self.root,source=broken),'pooled')

if __name__=='__main__':unittest.main()
