"""Compare every muxed spoken segment with its text-bound source waveform.

This checks audio identity and alignment, not pronunciation or listening quality.
"""
import argparse, hashlib, json, os, subprocess, wave
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import psutil
from audio_contract import validate

HERE=Path(__file__).resolve().parent
FFMPEG=Path(r'C:\Users\owner\tools\ffmpeg-7.1.1\bin\ffmpeg.exe')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--chapter',required=True)
    p.add_argument('--cpu',type=int,default=14)
    p.add_argument('--video',type=Path,help='Verify an explicitly recorded audio remux against the original frozen narration')
    a=p.parse_args()
    if a.cpu not in [4,5,6,7,8,9,14,15]:raise ValueError('CPU is outside background partition')
    me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);me.cpu_affinity([a.cpu])
    root=a.build.resolve()
    timing=json.loads((root/'timing'/f'chapter{a.chapter}.json').read_text(encoding='utf-8'))
    chapter=json.loads((root/'chapters'/f'{a.chapter}.json').read_text(encoding='utf-8'))
    texts={f'{b["key"]}_{i:02}':s['say'] for b in chapter['boards'] for i,s in enumerate(b['segments'],1)}
    selected=timing.get('selected_board')
    pattern=f'*/{selected}.mp4' if selected else f'*/chapter{a.chapter}_*.mp4'
    videos=[x for x in (root/'media/videos/scenes').glob(pattern) if not x.name.endswith(('.silent.mp4','_temp.mp4'))]
    if len(videos)!=1:raise ValueError('Expected one completed video')
    video=a.video.resolve() if a.video else videos[0]
    suffix='_remux_'+hashlib.sha256(video.read_bytes()).hexdigest()[:12] if a.video else ''
    out=HERE/'out'/f'{root.name}_audio_alignment{suffix}'
    out.mkdir(parents=True,exist_ok=False)
    pcm=out/'muxed.pcm'
    command=[str(FFMPEG),'-hide_banner','-loglevel','error','-nostdin','-n','-threads','1',
        '-i',str(video),'-map','0:a:0','-ac','1','-ar','24000','-c:a','pcm_s16le','-f','s16le',str(pcm)]
    subprocess.run(command,check=True,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    actual=np.memmap(pcm,dtype='<i2',mode='r')
    rows=[]
    for row in timing['segments']:
        key=row['key'];path=root/'audio'/a.chapter/(key+'.wav')
        validate(path,texts[key])
        with wave.open(str(path),'rb') as w:
            if (w.getnchannels(),w.getsampwidth(),w.getframerate())!=(1,2,24000):
                raise ValueError('Source audio format changed')
            expected=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.float64)
        # Manim places recordings on integer milliseconds; 24 kHz has exactly
        # 24 samples per millisecond. Search only a two-millisecond mux offset.
        nominal=int(1000*row['voice_start'])*24
        norm=np.linalg.norm(expected)
        if norm==0:raise ValueError('Entire spoken source is silent')
        probe=expected[::8]
        candidates=[]
        for shift in range(-48,49):
            start=nominal+shift
            if start<0 or start+len(expected)>len(actual):continue
            observed=np.asarray(actual[start:start+len(expected):8],dtype=np.float64)
            denominator=np.linalg.norm(probe)*np.linalg.norm(observed)
            candidates.append((float(np.dot(probe,observed)/denominator) if denominator else 0.,shift))
        if not candidates:raise ValueError('Recording lies outside decoded audio')
        _,shift=max(candidates)
        observed=np.asarray(actual[nominal+shift:nominal+shift+len(expected)],dtype=np.float64)
        correlation=float(np.dot(expected,observed)/(norm*np.linalg.norm(observed)))
        gain=float(np.linalg.norm(observed)/norm)
        passed=correlation>=.98 and .9<=gain<=1.1
        rows.append(dict(key=key,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            frames=len(expected),nominal_sample=nominal,offset_samples=shift,
            correlation=correlation,rms_ratio=gain,passed=passed))
    result=dict(kind='author_audio_waveform_alignment',video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        build=root.name,chapter=a.chapter,selected_board=selected,decoded_frames=len(actual),
        rows=rows,all_passed=all(r['passed'] for r in rows),
        scope='Whole-recording waveform similarity and timing; no pronunciation or auditory approval')
    (out/'receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not result['all_passed']:raise SystemExit(1)

if __name__=='__main__':main()
