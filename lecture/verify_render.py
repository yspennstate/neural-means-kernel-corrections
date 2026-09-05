"""Verify frozen inputs, narration coverage and media metadata; extract review frames.

These are author production checks. They do not claim listening, human approval,
or a successful <=50 ms console-window observation.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import psutil

from audio_contract import validate

HERE=Path(__file__).resolve().parent
FLAGS=0x08000000 | 0x4000 if os.name=='nt' else 0


def run(args):
    return subprocess.run(args,check=True,capture_output=True,text=True,
                          encoding='utf-8',creationflags=FLAGS)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--build',type=Path,required=True)
    ap.add_argument('--chapter',required=True)
    ap.add_argument('--cpu',type=int,default=14)
    args=ap.parse_args()
    if os.name=='nt':
        process=psutil.Process()
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        if args.cpu not in [4,5,6,7,8,9,14,15]:
            raise ValueError('Choose an allowed background CPU')
        process.cpu_affinity([args.cpu])
    root=args.build.resolve()
    chapter=json.loads((root/'chapters'/f'{args.chapter}.json').read_text(encoding='utf-8'))
    timing=json.loads((root/'timing'/f'chapter{args.chapter}.json').read_text(encoding='utf-8'))
    if timing['script_sha256']!=digest(root/'chapters'/f'{args.chapter}.json'):
        raise ValueError('Timing belongs to a different script')
    manifest=json.loads((root/'input_manifest.json').read_text(encoding='utf-8'))
    # The manifest's schema is checked explicitly; do not accept an empty loop.
    inputs=manifest
    if not isinstance(inputs,dict) or not inputs or any(
        not isinstance(k,str) or not isinstance(v,str) or len(v)!=64 for k,v in inputs.items()):
        raise ValueError('Invalid input manifest')
    for rel,expected in inputs.items():
        if digest(root/rel)!=expected:raise ValueError(f'Changed frozen input: {rel}')
    selected=timing.get('selected_board')
    pattern=f'*/{selected}.mp4' if selected else f'*/chapter{args.chapter}_*.mp4'
    videos=list((root/'media/videos/scenes').glob(pattern))
    if len(videos)!=1:raise ValueError('Expected exactly one completed chapter video')
    video=videos[0]
    probe=json.loads(run([shutil.which('ffprobe'),'-v','error','-count_frames','-show_format','-show_streams',
                          '-of','json',str(video)]).stdout)
    vs=[s for s in probe['streams'] if s['codec_type']=='video']
    aus=[s for s in probe['streams'] if s['codec_type']=='audio']
    if len(vs)!=1 or len(aus)!=1:raise ValueError('Missing or duplicate media stream')
    duration=float(vs[0]['duration'])
    exact_duration=Fraction(vs[0]['duration_ts'])*Fraction(vs[0]['time_base'])
    frame_count=int(vs[0]['nb_read_frames'])
    if Fraction(frame_count)/Fraction(vs[0]['avg_frame_rate'])!=exact_duration:
        raise ValueError('Frame count and exact stream duration disagree')
    if Fraction(vs[0]['avg_frame_rate'])!=Fraction(vs[0]['r_frame_rate']):
        raise ValueError('Output is not constant-frame-rate')
    if abs(duration-timing['seconds'])>.1:raise ValueError('Rendered duration differs from timing')
    boards=[b for b in chapter['boards'] if selected is None or b['key']==selected]
    entries=[(f'{b["key"]}_{i:02}',s['say']) for b in boards
             for i,s in enumerate(b['segments'],1)]
    if [k for k,_ in entries]!=[r['key'] for r in timing['segments']]:
        raise ValueError('Missing, reordered or duplicated narration segment')
    out=HERE/'out'/f'{root.name}_author_check'
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    previous_end=0.
    for (key,say),row in zip(entries,timing['segments']):
        seconds=validate(root/'audio'/args.chapter/(key+'.wav'),say)
        if abs(seconds-row['voice_seconds'])>1e-6:raise ValueError('Audio/timing mismatch')
        if abs(row['start']-previous_end)>.05:raise ValueError('Unaccounted timeline gap or overlap')
        if row['voice_start']<row['start'] or row['voice_start']+seconds>row['end']+.01:
            raise ValueError('Narration truncated by segment boundary')
        at=min(row['end']-.3,duration-.1)
        path=out/(key+'.png')
        run([shutil.which('ffmpeg'),'-y','-nostdin','-loglevel','error','-threads','1',
             '-ss',str(at),'-i',str(video),'-frames:v','1','-threads','1',str(path)])
        rows.append(dict(key=key,at_seconds=at,path=path.name,sha256=digest(path)))
        previous_end=row['end']
    last_voice_end=timing['segments'][-1]['voice_start']+timing['segments'][-1]['voice_seconds']
    if float(aus[0]['duration'])+.05<last_voice_end:
        raise ValueError('Muxed audio ends before the final spoken segment')
    result=dict(kind='author_media_integrity_check',chapter=args.chapter,build_directory=str(root),
                selected_board=selected,frame_count=frame_count,exact_duration=str(exact_duration),
                video_sha256=digest(video),input_manifest_sha256=digest(root/'input_manifest.json'),
                duration_seconds=duration,spoken_seconds=sum(r['voice_seconds'] for r in timing['segments']),
                width=vs[0]['width'],height=vs[0]['height'],frozen_inputs_checked=len(inputs),
                narration_segments_checked=len(entries),frames=rows,
                auditory_review='NOT_PERFORMED',human_visual_review='PENDING')
    (out/'receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='frames'},indent=2))


if __name__=='__main__':main()
