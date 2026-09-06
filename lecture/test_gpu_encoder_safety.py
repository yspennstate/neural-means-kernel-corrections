"""Driver-fault and lost-observer controls; no actual GPU work."""
from datetime import datetime, timezone
from unittest import TestCase, main
from ffmpeg_writer import gpu_observation_reasons


class FaultObservationTests(TestCase):
    def setUp(self):
        self.now = 1788679600.0
        self.healthy = dict(observed_at=datetime.fromtimestamp(self.now-5, timezone.utc).isoformat(),
                            event_query_complete=True, gpu_reset_latched=False,
                            guard_health={'status': 'healthy'})

    def test_fresh_complete_recovery_is_accepted(self):
        self.assertEqual(gpu_observation_reasons(self.healthy, self.now), [])

    def test_low_utilization_never_overrides_a_fault(self):
        record = dict(self.healthy, gpu_reset_latched=True, gpu={'utilization': 0})
        self.assertIn('GPU driver fault is latched or unknown', gpu_observation_reasons(record, self.now))

    def test_old_observer_is_rejected_even_with_free_memory(self):
        record = dict(self.healthy, observed_at=datetime.fromtimestamp(self.now-121, timezone.utc).isoformat(),
                      free_vram_bytes=8*1024**3)
        self.assertIn('GPU driver observation is stale or invalid', gpu_observation_reasons(record, self.now))

    def test_missing_unknown_incomplete_or_future_evidence_is_rejected(self):
        for patch in ({'gpu_reset_latched': None}, {'event_query_complete': False},
                      {'guard_health': None}, {'observed_at': 'bad'},
                      {'observed_at': datetime.fromtimestamp(self.now+1, timezone.utc).isoformat()},
                      {'observed_at': '2026-09-06T07:00:00'}):
            with self.subTest(patch=patch):
                self.assertTrue(gpu_observation_reasons(dict(self.healthy, **patch), self.now))
        self.assertTrue(gpu_observation_reasons(None, self.now))


if __name__ == '__main__':
    main()
