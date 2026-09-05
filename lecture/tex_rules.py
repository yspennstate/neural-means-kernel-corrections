"""Preserve PDF-derived LaTeX rules when Manim zeroes glyph stroke widths.

The PDF converter represents fraction bars and radical overbars as stroked
line segments. MathTex deliberately sets all strokes to zero. Replace only
these straight, butt-capped rules with geometrically identical filled paths
before MathTex reads the SVG. Unrecognized stroke geometry fails closed.
The wrapper also visits cache hits; frozen cache donors remain unchanged.
"""
import functools
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

TOKEN = re.compile(r'[A-Za-z]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?')
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')


def endpoints(d):
    tokens = TOKEN.findall(d)
    if re.sub(r'[\s,]', '', TOKEN.sub('', d)):
        raise ValueError('Unrecognized TeX rule path syntax')
    if len(tokens) not in (5, 6) or tokens[0] != 'M':
        raise ValueError('TeX rule must be one explicit straight segment')
    x, y = map(float, tokens[1:3])
    command = tokens[3]
    if command in ('H', 'h') and len(tokens) == 5:
        end = float(tokens[4])
        return x, y, end if command == 'H' else x+end, y
    if command in ('V', 'v') and len(tokens) == 5:
        end = float(tokens[4])
        return x, y, x, end if command == 'V' else y+end
    if command in ('L', 'l') and len(tokens) == 6:
        a, b = map(float, tokens[4:])
        return x, y, a if command == 'L' else x+a, b if command == 'L' else y+b
    raise ValueError('Unsupported TeX rule path command')


def normalize(payload):
    root = ET.fromstring(payload)
    changed = 0
    for element in root.iter():
        attrs = element.attrib
        stroke = attrs.get('stroke', 'none')
        if stroke == 'none':
            continue
        if attrs.get('fill') != 'none' or element.tag != '{'+NS+'}path':
            raise ValueError('Unexpected stroked element in generated TeX SVG')
        if attrs.get('stroke-linecap', 'butt') != 'butt' or attrs.get('stroke-dasharray', 'none') != 'none':
            raise ValueError('Unsupported TeX rule caps or dashes')
        x, y, a, b = endpoints(attrs['d'])
        width = float(attrs['stroke-width'])
        length = math.hypot(a-x, b-y)
        if width <= 0 or length <= 0 or not all(map(math.isfinite, (x,y,a,b,width,length))):
            raise ValueError('Degenerate TeX rule geometry')
        nx, ny = -(b-y)*width/(2*length), (a-x)*width/(2*length)
        vertices = [(x+nx,y+ny),(a+nx,b+ny),(a-nx,b-ny),(x-nx,y-ny)]
        attrs['d'] = 'M'+'L'.join(f'{u:.12g} {v:.12g}' for u,v in vertices)+'Z'
        attrs['fill'] = stroke
        attrs['fill-opacity'] = attrs.get('stroke-opacity', '1')
        for key in tuple(attrs):
            if key == 'stroke' or key.startswith('stroke-'):
                del attrs[key]
        attrs['data-nmkc-rule'] = 'filled-v1'
        changed += 1
    return (ET.tostring(root, encoding='utf-8', xml_declaration=True) if changed else payload), changed


def install(here):
    from manim import config
    from manim.mobject.text import tex_mobject
    from manim.utils import tex_file_writing
    original = tex_file_writing.tex_to_svg_file
    if getattr(original, '_nmkc_rules', False):
        raise RuntimeError('TeX rule wrapper was installed twice')
    receipts = Path(here) / 'tex_rule_conversions.jsonl'

    @functools.wraps(original)
    def checked_svg(*args, **kwargs):
        path = Path(original(*args, **kwargs)).resolve()
        if path.parent != Path(config.tex_dir).resolve():
            raise ValueError('TeX converter returned a file outside this render cache')
        before = path.read_bytes()
        after, count = normalize(before)
        if count:
            path.write_bytes(after)
            with receipts.open('a', encoding='utf-8') as f:
                f.write(json.dumps(dict(path=path.name, rules=count,
                    original_sha256=hashlib.sha256(before).hexdigest(),
                    normalized_sha256=hashlib.sha256(after).hexdigest()))+'\n')
        return path

    checked_svg._nmkc_rules = True
    tex_file_writing.tex_to_svg_file = checked_svg
    tex_mobject.tex_to_svg_file = checked_svg

