"""Bound the lecture's own CPU encoder; never change the shared Manim install.

Manim 0.21's serial encoder otherwise uses an unbounded frame queue and the
codec's automatic thread count. Each lecture child has only two admitted CPUs.
"""
import functools
import hashlib
import inspect
import json
from pathlib import Path


def install():
    import manim
    from manim.scene import scene_file_writer as writer
    if manim.__version__!='0.21.0':
        raise RuntimeError('Recheck the process-local encoder policy for this Manim version')
    cls=writer._PartialMovieEncodeJob
    original=cls.__init__
    parameters=inspect.signature(original).parameters
    if not {'stream','frame_queue_size'}.issubset(parameters):
        raise RuntimeError('Encoder interface changed')
    @functools.wraps(original)
    def bounded_init(self,*,path,animation_index,container,stream,frame_queue_size):
        context=stream.codec_context
        if context.name!='libx264':
            raise RuntimeError('Lecture encoder policy expects CPU libx264')
        context.thread_count=1
        options=dict(context.options)
        options.update(preset='veryfast',crf='20',threads='1')
        context.options=options
        original(self,path=path,animation_index=animation_index,container=container,
                 stream=stream,frame_queue_size=8)
    cls.__init__=bounded_init
    source=Path(inspect.getfile(writer))
    return dict(manim_version=manim.__version__,codec='libx264',threads=1,preset='veryfast',crf=20,
                frame_queue_capacity=8,max_inflight_encoders=1,
                installed_writer_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                scope='This render process only; shared environment files are read-only')


def record(here):
    result=install()
    (Path(here)/'encoder_policy_receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
