"""Bounded regression checks for mixing historical and revised campaign outputs."""
import json
from pathlib import Path
import tempfile
import unittest

from pipeline_contract import file_sha, make_contract, require_contract, verify_oof


class ResumeBoundary(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.runs = self.root / 'runs'
        self.runs.mkdir()
        self.source = self.root / 'source.py'
        self.source.write_text('version = 1\n', encoding='utf-8')
        self.data = self.root / 'data.bin'
        self.data.write_bytes(b'observed input fixture')
        self.settings = {'target_centering': 'pooled', 'seed': 3, 'epochs': 100}

    def contract(self):
        return make_contract(self.settings, {'code': self.source, 'data': self.data})

    def test_matching_resume_preserves_contract(self):
        expected = self.contract()
        require_contract(self.runs, expected)
        receipt = self.runs / 'pipeline_contract.json'
        original = receipt.read_bytes()
        (self.runs / 'model.json').write_text('{}', encoding='utf-8')
        require_contract(self.runs, expected)
        self.assertEqual(receipt.read_bytes(), original)

    def test_unreceipted_legacy_outputs_are_never_adopted(self):
        artifact = self.runs / 'model.json'
        artifact.write_text('{"historical":true}', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'no pipeline contract'):
            require_contract(self.runs, self.contract())
        self.assertEqual(artifact.read_text(encoding='utf-8'), '{"historical":true}')
        self.assertFalse((self.runs / 'pipeline_contract.json').exists())

    def test_changed_centering_or_schedule_refuses_resume(self):
        require_contract(self.runs, self.contract())
        for key, value in [('target_centering', 'fold-local'), ('epochs', 200), ('seed', 4)]:
            changed = self.contract()
            changed['settings'] = dict(self.settings, **{key: value})
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'Resume refused'):
                require_contract(self.runs, changed)

    def test_changed_bytes_at_same_path_refuse_resume(self):
        require_contract(self.runs, self.contract())
        for path in [self.source, self.data]:
            previous = path.read_bytes()
            path.write_bytes(previous + b'changed')
            with self.subTest(path=path.name), self.assertRaisesRegex(RuntimeError, 'Resume refused'):
                require_contract(self.runs, self.contract())
            path.write_bytes(previous)

    def test_truncated_contract_is_not_silently_recreated(self):
        receipt = self.runs / 'pipeline_contract.json'
        receipt.write_text('{', encoding='utf-8')
        with self.assertRaises(json.JSONDecodeError):
            require_contract(self.runs, self.contract())
        self.assertEqual(receipt.read_text(encoding='utf-8'), '{')

    def test_oof_receipt_checks_mode_split_and_field_bytes(self):
        field = self.runs / 'krr_oof_train.npy'
        field.write_bytes(b'opaque field fixture; no numerical result')
        receipt = self.runs / 'krr_oof_train.json'
        producer = file_sha(self.source)
        receipt.write_text(json.dumps(dict(target_centering='pooled', split_seed=3,
                                          driver_sha256=producer,
                                          field_sha256=file_sha(field))), encoding='utf-8')
        verify_oof(self.runs, 'pooled', 3, producer)
        for mode, seed in [('fold-local', 3), ('pooled', 4)]:
            with self.subTest(mode=mode, seed=seed), self.assertRaises(RuntimeError):
                verify_oof(self.runs, mode, seed, producer)
        with self.assertRaises(RuntimeError):
            verify_oof(self.runs, 'pooled', 3, '0' * 64)
        field.write_bytes(b'changed output')
        with self.assertRaises(RuntimeError):
            verify_oof(self.runs, 'pooled', 3, producer)


if __name__ == '__main__':
    unittest.main()
