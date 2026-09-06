"""Publish an explicitly provisional full cut without weakening final review gates.

This cut preserves the selected frozen narration, including documented material
awaiting correction. It cannot produce a final-review PASS receipt.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import wave

import psutil

from assembly_contract import append_recording, pcm_frames
from assemble_lecture import metadata_value, probe, stream_signature, video_stream
from compute_admission import health
from ffmpeg_writer import FFMPEG, FFMPEG_SHA256

HERE = Path(__file__).resolve().parent
MESH = Path.home()/'ai-memories-and-functionality/12_cognitive_architecture/agent_mesh/agent_mesh.py'
FLAGS = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
_mesh_api = None


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def owned(agent, path):
    # Use the same exact-target Mesh assertion as the driver observer. Spawning
    # a fresh interpreter on a starved background CPU can time out before the
    # CLI even checks the lease. An in-process assertion still fails closed.
    global _mesh_api
    if _mesh_api is None:
        spec = importlib.util.spec_from_file_location('snapshot_mesh_api', MESH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _mesh_api = module.Mesh(module.DEFAULT_DB, busy_timeout_ms=2500)
    _mesh_api.assert_claim(agent, [str(path)])


def write(agent, path, value):
    owned(agent, path)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def admitted(agent, out):
    deadline = time.monotonic()+900
    while True:
        state = health()
        write(agent, out/'health_latest.json', state)
        if state['allow']:
            return
        if time.monotonic() > deadline:
            raise TimeoutError('Fresh pressure check did not admit snapshot assembly')
        print('Pressure hold: '+str(state['hold_reasons']), flush=True)
        time.sleep(20)


def run(agent, out, name, command):
    admitted(agent, out)
    owned(agent, out)
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                            errors='replace', creationflags=FLAGS)
    write(agent, out/(name+'.json'), dict(command=command, returncode=result.returncode,
                                         stdout=result.stdout, stderr=result.stderr))
    if result.returncode:
        raise RuntimeError('Failed '+name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--agent', required=True)
    parser.add_argument('--cpu', type=int, default=6)
    args = parser.parse_args()
    if args.cpu not in (4, 5, 6, 7, 8, 9, 14, 15):
        raise ValueError('Outside background CPU partition')
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity([args.cpu])
    manifest = read(args.manifest)
    if manifest.get('status') != 'PROVISIONAL_FULL_CUT_CORRECTIONS_PENDING':
        raise ValueError('Progress cut requires explicit provisional status')
    if not manifest.get('known_corrections'):
        raise ValueError('A progress cut must disclose its pending corrections')
    rows = manifest['chapters']
    if [row['chapter'] for row in rows] != [f'{i:02}' for i in range(1, 13)]:
        raise ValueError('Require all twelve chapters exactly once in order')
    out = args.output.resolve()
    owned(args.agent, out)
    out.mkdir(parents=True, exist_ok=False)
    write(args.agent, out/'selection.json', manifest)
    admitted(args.agent, out)
    selected = []
    signature = None
    total_frames = 0
    transcript = ['# NMKC complete progress cut: corrections pending', '',
                  'This transcript preserves the selected frozen recordings. It is not the corrected final lecture.', '']
    for row in rows:
        chapter_id = row['chapter']
        build = Path(row['build']).resolve()
        if build.parent != (HERE/'builds').resolve():
            raise ValueError('Build outside the frozen lecture directory')
        mux_path = Path(row['mux_receipt']).resolve()
        if sha(mux_path) != row['mux_sha256']:
            raise ValueError('Changed mux receipt')
        mux = read(mux_path)
        if mux['returncode'] != 0:
            raise ValueError('Incomplete chapter mux')
        inputs = [Path(mux['command'][i+1]) for i, arg in enumerate(mux['command']) if arg == '-i']
        if len(inputs) != 2:
            raise ValueError('Expected exactly original video and original PCM')
        source_video, audio = inputs
        if sha(source_video) != mux['source_video_sha256'] or sha(audio) != mux['source_audio_sha256']:
            raise ValueError('Changed original video or PCM')
        video = Path(row['video']).resolve()
        if video != Path(mux['command'][-1]).resolve() or sha(video) != row['video_sha256']:
            raise ValueError('Changed explicitly selected completed mux output')
        stream = video_stream(probe(video))
        frames = int(stream['nb_frames'])
        duration = Fraction(frames, 30)
        if Fraction(stream['avg_frame_rate']) != 30 or Fraction(stream['duration_ts'])*Fraction(stream['time_base']) != duration:
            raise ValueError('Frame count, rate or exact timestamp mismatch')
        if Fraction(mux['requested_seconds']) != duration:
            raise ValueError('Mux duration differs from video')
        current_signature = stream_signature(stream)
        if signature is None:
            signature = current_signature
        if current_signature != signature:
            raise ValueError('Incompatible chapter video streams')
        frozen = read(build/'input_manifest.json')
        for rel, expected in frozen.items():
            path = (build/rel).resolve()
            path.relative_to(build)
            if sha(path) != expected:
                raise ValueError('Changed frozen source: '+str(path))
        chapter = read(build/'chapters'/f'{chapter_id}.json')
        timing = read(build/'timing'/f'chapter{chapter_id}.json')
        if timing['selected_board'] is not None or timing['script_sha256'] != sha(build/'chapters'/f'{chapter_id}.json'):
            raise ValueError('Partial or mismatched chapter timing')
        keys = [f'{b["key"]}_{i:02}' for b in chapter['boards'] for i, _ in enumerate(b['segments'], 1)]
        if keys != [s['key'] for s in timing['segments']]:
            raise ValueError('Incomplete narration timing')
        transcript += [f'## Chapter {chapter_id}: '+chapter['title'], '']
        for board in chapter['boards']:
            transcript += ['### '+board['title'], '']
            transcript += [s['say']+'\n' for s in board['segments']]
        selected.append(dict(row, video=str(video), audio=str(audio), frames=frames,
                             start_frame=total_frames, title=chapter['title'],
                             frozen_source_hashes_checked=len(frozen), narration_segments=len(keys)))
        total_frames += frames
        print('Pinned chapter '+chapter_id+': '+str(frames)+' frames', flush=True)
    owned(args.agent, out/'complete_original_pcm.wav')
    with wave.open(str(out/'complete_original_pcm.wav'), 'wb') as dest:
        dest.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        for row in selected:
            row['pcm_boundary'] = append_recording(dest, row['audio'], pcm_frames(row['frames']))
    owned(args.agent, out/'video.ffconcat')
    (out/'video.ffconcat').write_text('ffconcat version 1.0\n'+''.join(
        "file '"+Path(row['video']).as_posix().replace("'", "'\\''")+"'\n" for row in selected), encoding='utf-8')
    metadata = [';FFMETADATA1', 'title=NMKC full progress cut - corrections pending',
                'comment=Provisional full lecture; see accompanying correction list.']
    for row in selected:
        metadata += ['[CHAPTER]', 'TIMEBASE=1/30', 'START='+str(row['start_frame']),
                     'END='+str(row['start_frame']+row['frames']),
                     'title='+metadata_value(row['chapter']+' - '+row['title'])]
    owned(args.agent, out/'chapters.ffmeta')
    (out/'chapters.ffmeta').write_text('\n'.join(metadata)+'\n', encoding='utf-8')
    owned(args.agent, out/'TRANSCRIPT.md')
    (out/'TRANSCRIPT.md').write_text('\n'.join(transcript), encoding='utf-8')
    final = out/'NMKC_FULL_PROGRESS_CUT_CORRECTIONS_PENDING_20260906.mp4'
    duration = Fraction(total_frames, 30)
    command = [str(FFMPEG), '-hide_banner', '-loglevel', 'warning', '-nostdin', '-n',
               '-threads', '1', '-f', 'concat', '-safe', '0', '-i', str(out/'video.ffconcat'),
               '-i', str(out/'complete_original_pcm.wav'), '-i', str(out/'chapters.ffmeta'),
               '-map', '0:v:0', '-map', '1:a:0', '-map_metadata', '2', '-map_chapters', '2',
               '-c:v', 'copy', '-c:a', 'aac', '-threads', '1', '-ar', '24000', '-ac', '1',
               '-b:a', '128k', '-t', f'{float(duration):.9f}', '-movflags', '+faststart', str(final)]
    run(args.agent, out, 'mux', command)
    info = probe(final)
    stream = video_stream(info)
    if int(stream['nb_frames']) != total_frames or len(info['chapters']) != 12:
        raise ValueError('Assembled video lost frames or chapter markers')
    audio_stream = [s for s in info['streams'] if s['codec_type'] == 'audio']
    if len(audio_stream) != 1 or audio_stream[0]['channels'] != 1 or int(audio_stream[0]['sample_rate']) != 24000:
        raise ValueError('Assembled audio format differs')
    receipt = dict(status=manifest['status'], created_at=datetime.now(timezone.utc).isoformat(),
                   video=str(final), video_sha256=sha(final), video_bytes=final.stat().st_size,
                   total_frames=total_frames, exact_seconds=str(duration), seconds=float(duration),
                   chapters=selected, known_corrections=manifest['known_corrections'],
                   ffmpeg_sha256=FFMPEG_SHA256, producer_sha256=sha(__file__),
                   final_review_pass=False, full_movie_watch_performed=False,
                   scope='Frozen input and stream integrity checked; corrective renders and perceptual final review remain pending.')
    write(args.agent, out/'receipt.json', receipt)
    write(args.agent, out/'probe.json', info)
    print(json.dumps({k: receipt[k] for k in ['status', 'video', 'video_sha256', 'total_frames', 'seconds']}), flush=True)


if __name__ == '__main__':
    main()
