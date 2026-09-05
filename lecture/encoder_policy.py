"""Bound this lecture's CPU or NVIDIA encoder without editing shared Manim.

Manim 0.21's serial encoder otherwise uses an unbounded frame queue and the
codec's automatic thread count. Each lecture child has only two admitted CPUs.
"""
import ast
import __future__
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap

WRITER_SHA256 = '66ffdee7bc6d2b62e3b135c8d0a7e66dff607c5ea6994ad1e413cde683f6ba7f'
NVENC_OPTIONS = dict(an='1', preset='p5', tune='hq', rc='vbr', cq='18', b='0')


def install_nvenc(writer):
    """Replace only the opaque H.264 branch in a verified method, in memory."""
    original = writer.SceneFileWriter.open_partial_movie_stream
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    changed = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id == 'partial_movie_file_codec' and isinstance(node.value, ast.Constant) and node.value.value == 'libx264':
            node.value = ast.Constant('h264_nvenc'); changed.append('codec')
        if target.id == 'av_options':
            if ast.literal_eval(node.value) != {'an': '1', 'crf': '23'}:
                raise RuntimeError('Manim encoder option branch changed')
            node.value = ast.parse(repr(NVENC_OPTIONS), mode='eval').body
            changed.append('options')
    if sorted(changed) != ['codec', 'options']:
        raise RuntimeError('Expected exactly two NVENC substitutions')
    ast.fix_missing_locations(tree)
    scope = dict(writer.__dict__)
    exec(compile(tree, '<nmkc-process-local-nvenc>', 'exec', flags=__future__.annotations.compiler_flag), scope)
    replacement = scope['open_partial_movie_stream']
    if inspect.signature(replacement) != inspect.signature(original):
        raise RuntimeError('Patched encoder signature changed')
    writer.SceneFileWriter.open_partial_movie_stream = replacement
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def install():
    import manim
    from manim.scene import scene_file_writer as writer
    if manim.__version__!='0.21.0':
        raise RuntimeError('Recheck the process-local encoder policy for this Manim version')
    codec = os.environ.get('NMKC_ENCODER', 'libx264')
    if codec not in ('libx264', 'h264_nvenc'):
        raise RuntimeError('Unsupported lecture encoder')
    if os.environ.get('NMKC_WRITER', 'pyav') == 'ffmpeg71':
        from ffmpeg_writer import install as install_native
        return install_native(codec)
    source=Path(inspect.getfile(writer))
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    patch_hash=None
    if codec=='h264_nvenc':
        if source_hash != WRITER_SHA256:
            raise RuntimeError('Inspect the changed Manim writer before adapting NVENC')
        if manim.config.transparent or manim.config.movie_file_extension!='.mp4':
            raise RuntimeError('NVENC policy supports opaque MP4 output only')
        patch_hash=install_nvenc(writer)
    cls=writer._PartialMovieEncodeJob
    original=cls.__init__
    parameters=inspect.signature(original).parameters
    if not {'stream','frame_queue_size'}.issubset(parameters):
        raise RuntimeError('Encoder interface changed')
    @functools.wraps(original)
    def bounded_init(self,*,path,animation_index,container,stream,frame_queue_size):
        context=stream.codec_context
        if context.name!=codec:
            raise RuntimeError(f'Expected {codec}, got {context.name}')
        context.thread_count=1
        options=dict(context.options)
        options.update(dict(preset='veryfast',crf='20',threads='1') if codec=='libx264' else NVENC_OPTIONS)
        context.options=options
        original(self,path=path,animation_index=animation_index,container=container,
                 stream=stream,frame_queue_size=8)
    cls.__init__=bounded_init
    return dict(manim_version=manim.__version__,codec=codec,threads=1,
                options=dict(preset='veryfast',crf='20') if codec=='libx264' else NVENC_OPTIONS,
                frame_queue_capacity=8,max_inflight_encoders=1,
                installed_writer_sha256=source_hash,patch_ast_sha256=patch_hash,
                scope='This render process only; shared environment files are read-only')


def record(here):
    result=install()
    (Path(here)/'encoder_policy_receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
