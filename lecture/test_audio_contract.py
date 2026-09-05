"""Meaningful controls: stale voice, text and overwritten audio must fail."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave

from audio_contract import identity, audio_info, validate


class AudioIdentityTests(unittest.TestCase):
    def test_current_recording_and_three_independent_stale_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "segment.wav"
            with wave.open(str(path), "wb") as f:
                f.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                f.writeframes(struct.pack("<h", 100) * 240)
            path.with_suffix(".json").write_text(json.dumps({
                "identity": identity("A kernel projection."),
                "audio": audio_info(path)}), encoding="utf-8")
            self.assertEqual(validate(path, " A  kernel projection. "), .01)
            with self.assertRaisesRegex(ValueError, "Stale"):
                validate(path, "A different projection.")
            with self.assertRaisesRegex(ValueError, "Stale"):
                validate(path, "A kernel projection.", voice="en-US-BrianMultilingualNeural")
            with self.assertRaisesRegex(ValueError, "Stale"):
                validate(path, "A kernel projection.", rate="+0%")
            data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "Audio changed"):
                validate(path, "A kernel projection.")


if __name__ == "__main__":
    unittest.main()
