"""Correct one PDF title while checking every pixel outside it is unchanged.

The archived curve data are preserved as vector objects. This is an editorial
repair of the existing figure, not a regenerated spectral computation.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import fitz
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def texts(page):
    return [s for b in page.get_text('dict')['blocks'] if 'lines' in b
            for line in b['lines'] for s in line['spans']]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    old = 'effective dimension (Lemma 4.4 identity)'
    new = 'effective dimension (Lemma S5.1 identity)'
    doc = fitz.open(a.input)
    if len(doc) != 1:
        raise ValueError('Expected the one-page spectra figure')
    page = doc[0]
    spans = texts(page)
    matches = [s for s in spans if s['text'] == old]
    if len(matches) != 1:
        raise ValueError('Expected exactly the original title')
    span = matches[0]
    if span['font'] != 'DejaVuSans' or span['size'] != 9:
        raise ValueError('Original title typography changed')
    rect = fitz.Rect(span['bbox'])
    font_path = Path(importlib.util.find_spec('matplotlib').origin).parent / 'mpl-data/fonts/ttf/DejaVuSans.ttf'
    font = fitz.Font(fontfile=str(font_path))
    width = font.text_length(new, fontsize=span['size'])
    x = (rect.x0 + rect.x1 - width) / 2
    page.add_redact_annot(rect, fill=(1, 1, 1))
    page.apply_redactions(images=0, graphics=0)
    page.insert_text((x, span['origin'][1]), new, fontsize=span['size'],
                     fontname='CorrectedTitle', fontfile=str(font_path), color=(0, 0, 0))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(a.output, garbage=4, deflate=True)
    doc.close()
    original = fitz.open(a.input)
    repaired = fitz.open(a.output)
    expected = Counter(s['text'] for s in spans)
    expected.subtract([old]); expected.update([new])
    expected = +expected
    if Counter(s['text'] for s in texts(repaired[0])) != expected:
        raise ValueError('A text span outside the title changed')
    before = original[0].get_pixmap(matrix=fitz.Matrix(4, 4), alpha=False)
    after = repaired[0].get_pixmap(matrix=fitz.Matrix(4, 4), alpha=False)
    if (before.width, before.height, before.n) != (after.width, after.height, after.n):
        raise ValueError('Figure dimensions changed')
    left = np.frombuffer(before.samples, dtype=np.uint8).reshape(before.height, before.width, before.n)
    right = np.frombuffer(after.samples, dtype=np.uint8).reshape(after.height, after.width, after.n)
    different = np.any(left != right, axis=2)
    title = repaired[0].search_for(new)
    if len(title) != 1:
        raise ValueError('New title is not uniquely readable')
    area = (rect | title[0]) + (-1, -1, 1, 1)
    outside = different.copy()
    outside[max(0, int(area.y0 * 4)):int(np.ceil(area.y1 * 4)),
            max(0, int(area.x0 * 4)):int(np.ceil(area.x1 * 4))] = False
    if np.any(outside):
        raise ValueError('Pixels outside the title changed')
    after.save(a.output.with_suffix('.png'))
    receipt = dict(kind='editorial_vector_pdf_reference_repair', input_sha256=sha(a.input),
                   output_sha256=sha(a.output), font_sha256=sha(font_path), old=old, new=new,
                   changed_pixels=int(different.sum()), changed_pixels_outside_title=int(outside.sum()),
                   compared_dpi=288, title_rectangle=list(area), all_other_text_spans_unchanged=True,
                   scope='One internal lemma reference; all scientific curves retained')
    a.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
