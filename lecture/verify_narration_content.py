"""Transcribe text-bound WAVs with an unprompted local ASR model.

This is a content-screening record, not human listening or pronunciation approval.
Every disagreement is retained; an author must distinguish ASR errors from speech
errors. No reference words, initial prompt or hotwords enter the recognizer.
"""
import argparse, hashlib, importlib.metadata, json, os, re, subprocess, sys, time, unicodedata
from datetime import datetime, timezone
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
import psutil
from rapidfuzz.distance import Levenshtein
from audio_contract import validate, identity
from compute_admission import health

HERE = Path(__file__).resolve().parent
MODEL = Path.home()/'.cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def tokens(text):
    text=unicodedata.normalize('NFKC', text).casefold().replace('\u2019', "'")
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)

def compare(expected, heard):
    ref, hyp=tokens(expected), tokens(heard)
    if not ref: raise ValueError('Empty reference')
    edits=[]
    for tag, a,b,c,d in Levenshtein.opcodes(ref,hyp):
        if tag!='equal': edits.append(dict(kind=tag,reference=ref[a:b],heard=hyp[c:d],
                                           reference_span=[a,b],heard_span=[c,d]))
    distance=Levenshtein.distance(ref,hyp)
    assert sum(max(e['reference_span'][1]-e['reference_span'][0],
                   e['heard_span'][1]-e['heard_span'][0]) for e in edits)==distance
    return dict(reference_words=len(ref), heard_words=len(hyp), edit_distance=distance,
                word_error_rate=distance/len(ref), edits=edits)

def controls():
    exact=compare('The radius is two.', 'The radius is two.')
    repeat=compare('The radius is two.', 'The radius is two two.')
    negation=compare('The bound is not an estimate.', 'The bound is an estimate.')
    number=compare('There are sixty predictors.', 'There are sixteen predictors.')
    swap=compare('The residual lies in a Hilbert space.', 'A boundary load produces a stress field.')
    assert exact['edit_distance']==0 and repeat['edit_distance']==1
    assert negation['edits'][0]['reference']==['not']
    assert number['edits'][0]['reference']==['sixty'] and number['edits'][0]['heard']==['sixteen']
    assert swap['word_error_rate']>.5
    return dict(status='PASS',exact=exact,repetition=repeat,negation=negation,number=number,wrong_script=swap)

def write(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    temp.replace(path)

def admitted(out, device='cpu'):
    deadline=time.monotonic()+600
    while True:
        state=health()
        with (out/'health.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(state)+'\n')
        if state['allow'] and device=='cuda':
            probe=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.free',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,
                timeout=20,creationflags=subprocess.CREATE_NO_WINDOW)
            values=[float(x.strip()) for x in probe.stdout.strip().split(',')]
            if len(values)!=2:raise ValueError('Expected one GPU')
            write(out/'gpu_latest.json',dict(at=datetime.now(timezone.utc).isoformat(),
                utilization_pct=values[0],free_mib=values[1]))
            if values[0]>=80 or values[1]<2048:
                state['allow']=False
                state['hold_reasons'].append('GPU headroom')
        if state['allow']:return
        if time.monotonic()>deadline:raise TimeoutError('Headroom unavailable; resume from saved transcriptions')
        time.sleep(15)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--chapter',action='append',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cpu',type=int,default=6)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--cuda-library',type=Path)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.cpu not in [4,5,6,7,8,9,14,15]:raise ValueError('Outside background partition')
    own=psutil.Process();own.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);own.cpu_affinity([args.cpu])
    dll_handle=None;libraries={}
    if args.device=='cuda':
        if args.cuda_library is None:raise ValueError('CUDA requires an explicit DLL directory')
        library=args.cuda_library.resolve()
        dlls=sorted(set(library.glob('cublas*64_12.dll'))|set(library.glob('cudnn*64_9.dll')))
        if not (library/'cublas64_12.dll').exists() or not (library/'cudnn64_9.dll').exists():
            raise ValueError('CUDA 12 cuBLAS and cuDNN 9 are required')
        libraries={str(p):sha(p) for p in dlls}
        dll_handle=os.add_dll_directory(str(library))
    compute_type='float16' if args.device=='cuda' else 'int8'
    rows=[]
    for chapter_id in args.chapter:
        path=HERE/'chapters'/f'{chapter_id}.json'
        chapter=json.loads(path.read_text(encoding='utf-8'))
        for board in chapter['boards']:
            for i,segment in enumerate(board['segments'],1):
                key=f'{board["key"]}_{i:02}'
                audio=HERE/'audio'/chapter_id/(key+'.wav')
                seconds=validate(audio,segment['say'])
                rows.append(dict(chapter=chapter_id,key=key,script=segment['say'],
                    chapter_sha256=sha(path),audio=str(audio),audio_sha256=sha(audio),
                    synthesis_identity=identity(segment['say']),seconds=seconds))
    if args.limit is not None:
        if args.limit<1:raise ValueError('Limit must be positive')
        rows=rows[:args.limit]
    if not rows or len({x['key'] for x in rows})!=len(rows):raise ValueError('Empty or duplicate input list')
    model_files={name:sha(MODEL/name) for name in ['config.json','model.bin','tokenizer.json','vocabulary.txt']}
    contract=dict(kind='unprompted_local_ASR_content_screen',producer_sha256=sha(__file__),
        model_path=str(MODEL),model_files=model_files,
        packages={name:importlib.metadata.version(name) for name in ['faster-whisper','ctranslate2','rapidfuzz','av','numpy']},
        device=args.device,compute_type=compute_type,cuda_libraries=libraries,
        cpu_threads=1,num_workers=1,beam_size=5,
        language='en',vad_filter=False,condition_on_previous_text=False,initial_prompt=None,hotwords=None,
        rows=rows)
    out=args.out.resolve()
    if args.resume:
        if json.loads((out/'contract.json').read_text(encoding='utf-8'))!=contract:
            raise ValueError('Resume contract differs')
    else:
        out.mkdir(parents=True,exist_ok=False)
        write(out/'contract.json',contract)
    write(out/'alignment_controls.json',controls())
    admitted(out,args.device)
    print('Loading unprompted ASR model on '+args.device,flush=True)
    from faster_whisper import WhisperModel
    model=WhisperModel(str(MODEL),device=args.device,compute_type=compute_type,cpu_threads=1,num_workers=1,
                       local_files_only=True)
    completed=[]
    for row in rows:
        target=out/(row['key']+'.json')
        if target.exists():
            result=json.loads(target.read_text(encoding='utf-8'))
            if result['input']!=row:raise ValueError('Cached transcription differs')
        else:
            admitted(out,args.device)
            print('Transcribing '+row['key'],flush=True)
            if sha(row['audio'])!=row['audio_sha256']:raise ValueError('WAV changed before recognition')
            started=time.monotonic()
            segments,info=model.transcribe(row['audio'],language='en',beam_size=5,
                condition_on_previous_text=False,vad_filter=False,initial_prompt=None,hotwords=None)
            heard=[]
            for segment in segments:
                heard.append(dict(start=segment.start,end=segment.end,text=segment.text,
                    avg_logprob=segment.avg_logprob,no_speech_prob=segment.no_speech_prob))
            transcript=' '.join(x['text'].strip() for x in heard)
            result=dict(input=row,transcript=transcript,segments=heard,
                comparison=compare(row['script'],transcript),elapsed_seconds=time.monotonic()-started,
                observed_at=datetime.now(timezone.utc).isoformat(),
                review_status='UNREVIEWED',scope='ASR screening; no human listening or pronunciation approval')
            if sha(row['audio'])!=row['audio_sha256']:raise ValueError('WAV changed during recognition')
            write(target,result)
        completed.append(dict(key=row['key'],receipt_sha256=sha(target),
            edits=result['comparison']['edit_distance'],wer=result['comparison']['word_error_rate']))
        print(row['key'],result['comparison']['edit_distance'],round(result['comparison']['word_error_rate'],4),flush=True)
        write(out/'progress.json',dict(completed=len(completed),total=len(rows),last=row['key']))
    write(out/'receipt.json',dict(status='TRANSCRIBED_REQUIRES_AUTHOR_REVIEW',
        contract_sha256=sha(out/'contract.json'),controls_sha256=sha(out/'alignment_controls.json'),
        rows=completed,spoken_seconds=sum(x['seconds'] for x in rows),
        scope='All transcript disagreements retained; this is not auditory approval'))

if __name__=='__main__':main()
