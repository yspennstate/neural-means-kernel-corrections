"""Catch missing chapters, cumulative padding drift and clipped narration."""
from fractions import Fraction
import io
from pathlib import Path
import struct
import tempfile
import unittest
import wave

from assembly_contract import CHAPTERS, append_recording, pcm_frames, require_complete_order, voice_sample


class AssemblyBoundaries(unittest.TestCase):
    def test_missing_duplicate_reordered_or_unreviewed_chapter_is_rejected(self):
        rows = [dict(chapter=c, author_frame_review='PASS') for c in CHAPTERS]
        require_complete_order(rows)
        for invalid in [rows[:-1], rows+[rows[-1]], rows[:3]+rows[4:5]+rows[3:4]+rows[5:],
                        [dict(r, author_frame_review='PENDING') for r in rows]]:
            with self.subTest(chapters=[r['chapter'] for r in invalid]), self.assertRaises(ValueError):
                require_complete_order(invalid)

    def test_frame_boundaries_do_not_accumulate_rounded_milliseconds(self):
        # Three one-frame chapters end at exactly 0.1 seconds, not 0.099.
        self.assertEqual(sum(pcm_frames(1) for _ in range(3)), 2400)
        self.assertEqual(voice_sample(15050, .4666666666666666), 12051184)
        self.assertEqual(pcm_frames(18780), 15024000)
        with self.assertRaises(ValueError):
            pcm_frames(1, Fraction(30000, 1001))

    def test_original_pcm_and_exact_padding_survive_a_chapter_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'recording.wav'
            with wave.open(str(path), 'wb') as writer:
                writer.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
                writer.writeframes(struct.pack('<4h', 1000, -2000, 3000, -4000))
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as writer:
                writer.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
                first = append_recording(writer, path, 7)
                second = append_recording(writer, path, 4)
            with wave.open(io.BytesIO(buffer.getvalue()), 'rb') as reader:
                values = struct.unpack('<11h', reader.readframes(11))
            self.assertEqual(values, (1000, -2000, 3000, -4000, 0, 0, 0, 1000, -2000, 3000, -4000))
            self.assertEqual(first['padded_frames'], 3)
            self.assertEqual(second['padded_frames'], 0)

    def test_nonzero_truncation_is_rejected_before_appending(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'recording.wav'
            for tail, permitted in [(0, True), (1, False), (-1, False)]:
                with wave.open(str(path), 'wb') as writer:
                    writer.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
                    writer.writeframes(struct.pack('<3h', 1000, -1000, tail))
                buffer = io.BytesIO()
                with wave.open(buffer, 'wb') as writer:
                    writer.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
                    if permitted:
                        self.assertEqual(append_recording(writer, path, 2)['removed_silent_frames'], 1)
                    else:
                        with self.assertRaises(ValueError):
                            append_recording(writer, path, 2)
                        self.assertEqual(writer.getnframes(), 0)


if __name__ == '__main__':
    unittest.main()
