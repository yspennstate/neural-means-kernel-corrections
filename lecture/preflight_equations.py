"""Batch-compile every authored equation and flag likely layout overflow.

Box widths are a conservative preflight, not a substitute for the final
Manim glyph bounds or visual inspection of the rendered frame.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import psutil
from tex_style import PREAMBLE

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--chapter', help='Check one chapter instead of the entire lecture')
    parser.add_argument('--out', type=Path, default=HERE.parent/'.local-verification/lecture_equations')
    args=parser.parse_args()
    if os.name=='nt':
        proc=psutil.Process();proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);proc.cpu_affinity([6])
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    rows={};sources={}
    tex=[r'\documentclass{article}',PREAMBLE,
         r'\newwrite\measureout',r'\immediate\openout\measureout=dimensions.txt',r'\begin{document}']
    chapters=([HERE/'chapters'/(args.chapter+'.json')] if args.chapter else
              sorted((HERE/'chapters').glob('*.json')))
    for p in chapters:
        sources[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        data=json.loads(p.read_text(encoding='utf-8'))
        for b in data['boards']:
            for i,s in enumerate(b['segments'],1):
                key=f'{b["key"]}_{i:02}'
                if key in rows:raise ValueError('Duplicate equation identity')
                if any(ord(c)<32 and c not in '\n\r' for c in s['math']):
                    raise ValueError('Invalid JSON escape in equation')
                rows[key]={'math':s['math']}
                tex.append(r'\setbox0=\hbox{$\displaystyle '+s['math']+r'$}')
                tex.append(r'\immediate\write\measureout{'+key+r'|\the\wd0|\the\ht0|\the\dp0}')
    tex += [r'\immediate\closeout\measureout',r'\null\end{document}']
    source=out/'equations.tex';source.write_text('\n'.join(tex),encoding='utf-8')
    run=subprocess.run([shutil.which('pdflatex'),'-interaction=nonstopmode','-halt-on-error',source.name],
                       cwd=out,capture_output=True,text=True,encoding='utf-8',errors='replace',
                       creationflags=0x08000000|0x4000 if os.name=='nt' else 0)
    (out/'compiler_output.txt').write_text(run.stdout+run.stderr,encoding='utf-8')
    if run.returncode:raise RuntimeError(run.stdout[-2400:])
    measurements=(out/'dimensions.txt').read_text(encoding='utf-8').splitlines()
    if len(measurements)!=len(rows):raise ValueError('Incomplete equation measurements')
    for line in measurements:
        key,w,h,d=line.split('|')
        rows[key].update(width_pt=float(w[:-2]),height_pt=float(h[:-2])+float(d[:-2]))
    risky={k:v for k,v in rows.items() if v['width_pt']>170 or v['height_pt']>65}
    result=dict(kind='author_batch_latex_preflight',compiled_equations=len(rows),
                source_sha256=sources,rows=rows,likely_layout_overflow=list(risky),
                limitation='TeX boxes only. Final glyph bounding boxes and board placement still require rendered inspection.')
    (out/'receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(compiled_equations=len(rows),likely_layout_overflow=risky),indent=2))


if __name__=='__main__':main()
