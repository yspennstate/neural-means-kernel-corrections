"""Regression checks for the fixed 1920x1080 NotationProbe scene.

The pixel controls inspect the actual Cairo-rendered PNG, not SVG metadata.
They cover missing rules only; equation meaning and other glyphs still need
native-resolution visual inspection. Deliberately hide each rule in memory
and require the pixel check to reject that defective image.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from tex_rules import normalize

REGIONS = {
    'fraction': ((680,270,830,290), 115, 1),
    'text_fraction': ((940,270,985,290), 18, 1),
    'display_fraction': ((1090,270,1240,290), 110, 1),
    'radical_overline': ((660,415,850,435), 155, 1),
    'overline': ((950,421,1050,440), 70, 1),
    'underline': ((1150,483,1300,503), 110, 1),
    'table_horizontal': ((695,855,900,875), 160, 1),
    'table_vertical': ((790,785,805,945), 125, 0),
}


def pixel_checks(pixels):
    if pixels.shape != (1080,1920,3):
        raise ValueError('Notation probe must be native 1920x1080 RGB')
    mask = ((pixels[:,:,0] > 170) & (pixels[:,:,1] > 120)
            & (pixels[:,:,2] < 140))
    result = {}
    for name, (box, minimum, axis) in REGIONS.items():
        x0,y0,x1,y1 = box
        measured = int(mask[y0:y1,x0:x1].sum(axis=axis).max())
        result[name] = dict(minimum=minimum, measured=measured, present=measured>=minimum)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Refuse to overwrite an earlier check')
    source = b'<svg xmlns="http://www.w3.org/2000/svg"><path d="M1 2H5" stroke="#000" fill="none" stroke-width=".5"/></svg>'
    fixed,count = normalize(source)
    rule = next(iter(ET.fromstring(fixed)))
    if count != 1 or rule.attrib['d'] != 'M1 2.25L5 2.25L5 1.75L1 1.75Z':
        raise ValueError('Known rule outline differs from its analytical rectangle')
    if normalize(fixed) != (fixed,0):
        raise ValueError('Normalization is not idempotent')
    unsupported = False
    try:
        normalize(source.replace(b'M1 2H5',b'M1 2C3 4 5 6 7 8'))
    except ValueError:
        unsupported = True
    if not unsupported:
        raise ValueError('Unsupported curve was silently changed')
    pixels = np.asarray(Image.open(args.probe).convert('RGB'))
    checks = pixel_checks(pixels)
    if not all(r['present'] for r in checks.values()):
        raise ValueError('Rendered notation has a missing rule: '+json.dumps(checks))
    rejected = []
    for name,(box,_,_) in REGIONS.items():
        damaged = pixels.copy()
        x0,y0,x1,y1 = box
        damaged[y0:y1,x0:x1] = [17,26,35]
        if pixel_checks(damaged)[name]['present']:
            raise ValueError('Pixel check missed an intentionally removed rule: '+name)
        rejected.append(name)
    result = dict(kind='notation_rule_regression',
        probe_sha256=hashlib.sha256(args.probe.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        repair_sha256=hashlib.sha256(Path(__file__).with_name('tex_rules.py').read_bytes()).hexdigest(),
        actual_pixel_checks=checks, removed_rule_controls_rejected=rejected,
        known_rectangle=True, idempotence=True, unsupported_curve_rejected=True,
        scope='Author regression for rule presence; not independent or full slide-quality approval')
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
