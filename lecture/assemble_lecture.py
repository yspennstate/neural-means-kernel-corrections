"""Join twelve explicitly reviewed chapters and encode their original PCM once.

No chapter selection by filename ordering, modification time, or silent fallback.
The manifest pins each media, waveform and visual-review receipt by SHA-256.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import wave

import psutil

from assembly_contract import RATE, append_recording, pcm_frames, require_complete_order, voice_sample
from audio_contract import validate
from compute_admission import health
from ffmpeg_writer import FFMPEG, FFMPEG_SHA256

HERE = Path(__file__).resolve().parent
FLAGS = 0x08000000 | 0x4000 if os.name == 'nt' else 0


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pinned(spec):
    path = Path(spec['path']).resolve()
    if digest(path) != spec['sha256']:
        raise ValueError(f'Changed pinned file: {path}')
    return path, read_json(path)


def probe(path, count=False):
    command = [shutil.which('ffprobe'), '-v', 'error', '-show_streams', '-show_format',
               '-show_chapters', '-of', 'json']
    if count:
        command += ['-count_frames']
    result = subprocess.run(command + [str(path)], check=True, capture_output=True,
                            text=True, encoding='utf-8', creationflags=FLAGS)
    return json.loads(result.stdout)


def video_stream(info):
    streams = [s for s in info['streams'] if s['codec_type'] == 'video']
    if len(streams) != 1:
        raise ValueError('Expected one video stream')
    return streams[0]


def stream_signature(stream):
    return {key: stream[key] for key in ('codec_name', 'profile', 'width', 'height',
            'pix_fmt', 'r_frame_rate', 'avg_frame_rate', 'time_base')}


def load_chapter(row):
    chapter_id = row['chapter']
    build = Path(row['build']).resolve()
    if build.parent != (HERE / 'builds').resolve():
        raise ValueError('Select a frozen local lecture build')
    media_path, media = pinned(row['media_receipt'])
    _, audio = pinned(row['audio_receipt'])
    visual_path, visual = pinned(row['visual_receipt'])
    if media['chapter'] != chapter_id or audio['chapter'] != chapter_id:
        raise ValueError('Receipt belongs to another chapter')
    if Path(media['build_directory']).resolve() != build or audio['build'] != build.name:
        raise ValueError('Receipt belongs to another build')
    if media['selected_board'] is not None or audio['selected_board'] is not None:
        raise ValueError('A board preview cannot replace a chapter')
    if not audio['all_passed'] or visual.get('status') != 'PASS':
        raise ValueError('Unapproved chapter media or frame inspection')
    video = Path(media['video_path']).resolve()
    video.relative_to(build)
    video_hash = digest(video)
    if any(r['video_sha256'] != video_hash for r in (media, audio, visual)):
        raise ValueError('Reviews do not bind the selected movie')
    if digest(build / 'input_manifest.json') != media['input_manifest_sha256']:
        raise ValueError('Changed frozen input manifest')
    inputs = read_json(build / 'input_manifest.json')
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError('Empty or malformed frozen input manifest')
    for relative, expected in inputs.items():
        path = (build / relative).resolve()
        path.relative_to(build)
        if digest(path) != expected:
            raise ValueError(f'Changed frozen input: {relative}')
    script = build / 'chapters' / f'{chapter_id}.json'
    if digest(script) != digest(HERE / 'chapters' / f'{chapter_id}.json'):
        raise ValueError('Selected chapter script is no longer current')
    chapter = read_json(script)
    timing = read_json(build / 'timing' / f'chapter{chapter_id}.json')
    if timing['script_sha256'] != digest(script) or timing['selected_board'] is not None:
        raise ValueError('Timing/script mismatch')
    entries = [(f'{b["key"]}_{i:02}', b, s) for b in chapter['boards']
               for i, s in enumerate(b['segments'], 1)]
    keys = [key for key, _, _ in entries]
    if any([r['key'] for r in rows] != keys for rows in
           (timing['segments'], audio['rows'], media['frames'], visual['frames_inspected'])):
        raise ValueError('Incomplete, reordered or duplicate segment review')
    for frame, inspected in zip(media['frames'], visual['frames_inspected']):
        if frame != inspected or digest(media_path.parent / frame['path']) != frame['sha256']:
            raise ValueError('Frame inspection no longer matches the decoded frame')
    for (key, _, segment), moment, sound in zip(entries, timing['segments'], audio['rows']):
        recording = build / 'audio' / chapter_id / (key + '.wav')
        if validate(recording, segment['say']) != moment['voice_seconds']:
            raise ValueError('Frozen narration duration mismatch')
        if digest(recording) != sound['source_sha256'] or not sound['passed']:
            raise ValueError('Narration waveform check does not bind the source')
        validate(HERE / 'audio' / chapter_id / (key + '.wav'), segment['say'])
    silent = video.with_suffix('.silent.mp4')
    original = video.with_suffix('.wav')
    mux = read_json(video.with_suffix('.audio_mux.json'))
    frames = media['frame_count']
    duration = Fraction(frames, 30)
    if mux['returncode'] or Fraction(mux['requested_seconds']) != duration:
        raise ValueError('Chapter mux duration or completion mismatch')
    if digest(silent) != mux['source_video_sha256'] or digest(original) != mux['source_audio_sha256']:
        raise ValueError('Original video or PCM differs from reviewed chapter mux')
    stream = video_stream(probe(silent, count=True))
    if int(stream['nb_read_frames']) != frames or Fraction(stream['avg_frame_rate']) != 30:
        raise ValueError('Silent video count/rate differs from reviewed chapter')
    if Fraction(stream['duration_ts']) * Fraction(stream['time_base']) != duration:
        raise ValueError('Silent video timestamp duration mismatch')
    return dict(row=row, chapter=chapter, entries=entries, timing=timing, silent=silent,
                original=original, frames=frames, duration=duration, signature=stream_signature(stream),
                media_path=media_path, visual_path=visual_path)


def metadata_value(value):
    return ''.join('\\' + char if char in '=;#\\\n' else char for char in value)


def timestamp(samples):
    seconds = samples // RATE
    return f'{seconds//3600:02}:{seconds//60%60:02}:{seconds%60:02}'


def admit():
    state = health()
    if not state['allow']:
        raise RuntimeError(f'Assembly paused before the next operation: {state}')
    return state


def command_log(command, path):
    admit()
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                            errors='replace', creationflags=FLAGS)
    path.write_text(json.dumps(dict(command=command, returncode=result.returncode,
                    stdout=result.stdout, stderr=result.stderr), indent=2), encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'Assembly command failed; retained log: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--agent', required=True)
    parser.add_argument('--cpu', type=int, default=15)
    args = parser.parse_args()
    if os.name == 'nt':
        if args.cpu not in (4, 5, 6, 7, 8, 9, 14, 15):
            raise ValueError('Select an allowed background CPU')
        me = psutil.Process()
        me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        me.cpu_affinity([args.cpu])
    admission = admit()
    manifest = read_json(args.manifest)
    rows = manifest['chapters']
    require_complete_order(rows)
    if digest(FFMPEG) != FFMPEG_SHA256:
        raise ValueError('Verified native FFmpeg executable changed')
    output = args.output.resolve()
    output.relative_to((HERE / 'out').resolve())
    if output.exists():
        raise FileExistsError(output)
    mesh = Path('C:/Users/owner/ai-memories-and-functionality/12_cognitive_architecture/agent_mesh/agent_mesh.py')
    subprocess.run(['C:/Python314/python.exe', '-B', str(mesh), 'assert', '--agent', args.agent,
                    '--resource', 'file:' + output.as_posix()], check=True,
                   capture_output=True, creationflags=FLAGS)
    output.mkdir()
    (output / 'selection.json').write_bytes(args.manifest.read_bytes())
    chapters = []
    for row in rows:
        admit()
        chapter = load_chapter(row)
        if chapters and chapter['signature'] != chapters[0]['signature']:
            raise ValueError('Chapter video streams cannot be concatenated losslessly')
        chapters.append(chapter)
        print(f'Validated chapter {row["chapter"]}', flush=True)
    if sum(len(c['entries']) for c in chapters) != 252:
        raise ValueError('Expected all 252 narration segments')
    video_frames = sum(c['frames'] for c in chapters)
    total_samples = pcm_frames(video_frames)
    total_seconds = Fraction(video_frames, 30)
    listing = ['ffconcat version 1.0\n']
    metadata = [';FFMETADATA1\n', 'title=Neural means and kernel corrections\n']
    transcript = ['# Neural means and kernel corrections\n\n',
                  'Narration transcript for the assembled twelve-chapter lecture.\n\n']
    timeline = []
    boundaries = []
    start_frame = 0
    pcm_path = output / 'lecture.wav'
    with wave.open(str(pcm_path), 'wb') as pcm:
        pcm.setnchannels(1); pcm.setsampwidth(2); pcm.setframerate(RATE)
        for chapter in chapters:
            admit()
            row = chapter['row']
            chapter_id = row['chapter']
            path = chapter['silent'].as_posix()
            if "'" in path or '\n' in path or '\r' in path:
                raise ValueError('Unsupported concat path')
            listing += [f"file '{path}'\n", f'duration {float(chapter["duration"]):.12f}\n']
            start = start_frame * 800
            end = start + pcm_frames(chapter['frames'])
            title = chapter['chapter']['title']
            metadata += [f'[CHAPTER]\nTIMEBASE=1/{RATE}\nSTART={start}\nEND={end}\n',
                         f'title={metadata_value(chapter_id + ": " + title)}\n']
            pcm_receipt = append_recording(pcm, chapter['original'], end-start)
            boundaries.append(dict(chapter=chapter_id, title=title, start_sample=start,
                end_sample=end, start_frame=start_frame, frame_count=chapter['frames'], pcm=pcm_receipt))
            transcript.append(f'## {chapter_id}. {title} — {timestamp(start)}\n\n')
            for (key, board, segment), moment in zip(chapter['entries'], chapter['timing']['segments']):
                recording = Path(row['build']) / 'audio' / chapter_id / (key + '.wav')
                voice_start = voice_sample(start_frame, moment['voice_start'])
                with wave.open(str(recording), 'rb') as source:
                    voice_frames = source.getnframes()
                if voice_start + voice_frames > end:
                    raise ValueError('Chapter boundary truncates a spoken segment')
                timeline.append(dict(chapter=chapter_id, key=key, board=board['key'],
                    source_path=str(recording.resolve()), source_sha256=digest(recording),
                    start_sample=voice_start, frames=voice_frames, say=segment['say']))
                transcript += [f'### {timestamp(voice_start)} — {board["title"]}\n\n',
                               segment['say'] + '\n\n', '$$\n' + segment['math'] + '\n$$\n\n']
            start_frame += chapter['frames']
    with wave.open(str(pcm_path), 'rb') as pcm:
        if pcm.getnframes() != total_samples:
            raise ValueError('Whole-film PCM length mismatch')
    (output / 'chapters.ffconcat').write_text(''.join(listing), encoding='utf-8')
    (output / 'chapters.ffmetadata').write_text(''.join(metadata), encoding='utf-8')
    (output / 'transcript.md').write_text(''.join(transcript), encoding='utf-8')
    (output / 'narration_timeline.json').write_text(json.dumps(timeline, indent=2), encoding='utf-8')
    silent = output / 'lecture.silent.mp4'
    command_log([str(FFMPEG), '-hide_banner', '-loglevel', 'warning', '-nostdin', '-n',
        '-f', 'concat', '-safe', '0', '-i', str(output / 'chapters.ffconcat'),
        '-map', '0:v:0', '-an', '-c:v', 'copy', '-movflags', '+faststart', str(silent)],
        output / 'concat_receipt.json')
    movie = output / 'lecture.mp4'
    command_log([str(FFMPEG), '-hide_banner', '-loglevel', 'warning', '-nostdin', '-n',
        '-threads', '1', '-i', str(silent), '-i', str(pcm_path), '-i', str(output / 'chapters.ffmetadata'),
        '-map', '0:v:0', '-map', '1:a:0', '-map_metadata', '2', '-map_chapters', '2',
        '-c:v', 'copy', '-c:a', 'aac', '-ar', str(RATE), '-ac', '1', '-b:a', '128k',
        '-t', f'{float(total_seconds):.12f}', '-movflags', '+faststart', str(movie)],
        output / 'mux_receipt.json')
    observed = probe(movie, count=True)
    (output / 'ffprobe.json').write_text(json.dumps(observed, indent=2), encoding='utf-8')
    stream = video_stream(observed)
    if int(stream['nb_read_frames']) != video_frames or Fraction(stream['avg_frame_rate']) != 30:
        raise ValueError('Assembled frame count or rate mismatch')
    if Fraction(stream['duration_ts']) * Fraction(stream['time_base']) != total_seconds:
        raise ValueError('Assembled exact video duration mismatch')
    if len(observed['chapters']) != 12:
        raise ValueError('Missing assembled chapter markers')
    for expected, actual in zip(boundaries, observed['chapters']):
        for side in ('start', 'end'):
            error = abs(Fraction(actual[side]) * Fraction(actual['time_base']) -
                        Fraction(expected[side + '_sample'], RATE))
            if error > Fraction(1, 1000):
                raise ValueError('Assembled chapter marker differs by more than one millisecond')
    receipt = dict(kind='lecture_assembly_integrity', video_path=str(movie), video_sha256=digest(movie),
        selection_sha256=digest(output/'selection.json'), source_pcm_sha256=digest(pcm_path),
        narration_timeline_sha256=digest(output/'narration_timeline.json'),
        transcript_sha256=digest(output/'transcript.md'), frame_count=video_frames,
        exact_duration=str(total_seconds), sample_count=total_samples, chapters=boundaries,
        narration_segments=len(timeline), admission=admission, ffmpeg_sha256=digest(FFMPEG),
        video_encoding='Original reviewed NVENC frames, stream copied',
        audio_encoding='One native AAC encode of the exact assembled original mono PCM',
        whole_film_waveform_review='PENDING', auditory_review='NOT_PERFORMED',
        transition_review='PENDING', human_visual_review='PENDING')
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
