"""Bind every chapter's text to its WAV and count duration two ways."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
from audio_contract import validate, VOICE, RATE

HERE=Path(__file__).resolve().parent

def manifest():
    chapters=[];total=Fraction(0);validated_total=0.0
    for source in sorted((HERE/'chapters').glob('*.json')):
        raw=source.read_bytes();chapter=json.loads(raw.decode('utf-8'))
        rows=[];chapter_time=Fraction(0)
        for board in chapter['boards']:
            for number,segment in enumerate(board['segments'],1):
                key=f'{board["key"]}_{number:02}'
                wav=HERE/'audio'/chapter['id']/(key+'.wav')
                seconds=validate(wav,segment['say'])
                # Count stored PCM frames independently of the duration in
                # the narration receipt or the rounded recording log.
                # Deliberately do not use audio_contract's wave parser here.
                # Walk RIFF chunks directly and count the PCM data bytes.
                raw_wav=wav.read_bytes()
                magic,size,kind=struct.unpack_from('<4sI4s',raw_wav)
                assert magic==b'RIFF' and kind==b'WAVE' and size+8==len(raw_wav)
                offset=12;chunks={}
                while offset<len(raw_wav):
                    tag,length=struct.unpack_from('<4sI',raw_wav,offset)
                    assert offset+8+length<=len(raw_wav)
                    assert tag not in chunks,'Repeated WAV chunk'
                    chunks[tag]=raw_wav[offset+8:offset+8+length]
                    offset+=8+length+(length%2)
                assert offset==len(raw_wav)
                codec,channels,rate,byte_rate,alignment,bits=struct.unpack_from('<HHIIHH',chunks[b'fmt '])
                assert (codec,channels,rate,byte_rate,alignment,bits)==(1,1,24000,48000,2,16)
                assert len(chunks[b'data'])%alignment==0
                frames=len(chunks[b'data'])//alignment
                exact=Fraction(frames,rate)
                assert abs(float(exact)-seconds)<1e-9
                rows.append(dict(key=key,frames=frames,sample_rate=rate,seconds=float(exact),
                    wav_sha256=hashlib.sha256(raw_wav).hexdigest(),
                    receipt_sha256=hashlib.sha256(wav.with_suffix('.json').read_bytes()).hexdigest()))
                chapter_time+=exact;validated_total+=seconds
        total+=chapter_time
        chapters.append(dict(chapter=chapter['id'],source_sha256=hashlib.sha256(raw).hexdigest(),
            segment_count=len(rows),spoken_seconds=float(chapter_time),segments=rows))
    assert abs(float(total)-validated_total)<1e-8
    return dict(voice=VOICE,rate=RATE,generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),chapter_count=len(chapters),
        segment_count=sum(c['segment_count'] for c in chapters),spoken_seconds=float(total),
        exact_spoken_seconds=str(total),
        validation='Current text/settings/WAV identity plus independent PCM frame counts',
        limits='Duration and provenance checks do not certify pronunciation, audible quality, or final audiovisual timing.',
        chapters=chapters)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=manifest()
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='chapters'},indent=2))
