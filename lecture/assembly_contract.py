"""Exact chapter order and PCM boundaries for the complete lecture."""
from fractions import Fraction
from pathlib import Path
import wave


CHAPTERS = [f'{number:02}' for number in range(1, 13)]
RATE = 24000


def require_complete_order(rows):
    if [row['chapter'] for row in rows] != CHAPTERS:
        raise ValueError('The lecture requires chapters 01 through 12, exactly once and in order')
    if any(row.get('author_frame_review') != 'PASS' for row in rows):
        raise ValueError('Every selected chapter needs an explicit author frame-review disposition')


def pcm_frames(video_frames, fps=30):
    if type(video_frames) is not int or video_frames <= 0:
        raise ValueError('Positive integer video frame count required')
    count = Fraction(video_frames * RATE) / Fraction(fps)
    if count.denominator != 1:
        raise ValueError('Video boundary is not on the PCM sample grid')
    return int(count)


def append_recording(destination, source, target_frames):
    """Append original PCM and exact silent padding, never AAC packet padding.

    A recording longer than its chapter is accepted only when the discarded
    tail is digital silence. Nonzero discarded samples fail before any append.
    """
    source = Path(source)
    if type(target_frames) is not int or target_frames <= 0:
        raise ValueError('Positive target PCM length required')
    with wave.open(str(source), 'rb') as recording:
        if (recording.getnchannels(), recording.getsampwidth(), recording.getframerate(),
                recording.getcomptype()) != (1, 2, RATE, 'NONE'):
            raise ValueError('Expected uncompressed mono 24 kHz signed 16-bit PCM')
        original = recording.getnframes()
        if original > target_frames:
            recording.setpos(target_frames)
            while block := recording.readframes(65536):
                if any(block):
                    raise ValueError('Chapter boundary would cut nonzero narration samples')
            recording.rewind()
        remaining = min(original, target_frames)
        while remaining:
            requested = min(remaining, 65536)
            block = recording.readframes(requested)
            if len(block) != requested * 2:
                raise ValueError('Truncated source WAV')
            destination.writeframesraw(block)
            remaining -= requested
        remaining = max(0, target_frames - original)
        while remaining:
            count = min(remaining, 65536)
            destination.writeframesraw(bytes(count * 2))
            remaining -= count
    return dict(source_frames=original, output_frames=target_frames,
                padded_frames=max(0, target_frames-original),
                removed_silent_frames=max(0, original-target_frames))


def voice_sample(chapter_start_frame, voice_start_seconds, fps=30):
    if type(chapter_start_frame) is not int or chapter_start_frame < 0:
        raise ValueError('Nonnegative chapter start frame required')
    offset = Fraction(chapter_start_frame * RATE) / Fraction(fps)
    if offset.denominator != 1 or voice_start_seconds < 0:
        raise ValueError('Invalid narration offset')
    # Match Manim's integer-millisecond placement within each chapter.
    return int(offset) + int(1000 * voice_start_seconds) * (RATE // 1000)
