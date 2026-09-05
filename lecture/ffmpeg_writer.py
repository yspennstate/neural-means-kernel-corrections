"""Synchronous, process-local Manim writer using the owner's FFmpeg 7.1.

Frozen waits send one RGBA frame and an exact repetition count to FFmpeg.
Animations send every distinct frame. No shared Manim installation is edited.
"""
from fractions import Fraction
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time

import numpy as np

FFMPEG = Path(r'C:\Users\owner\tools\ffmpeg-7.1.1\bin\ffmpeg.exe')
FFMPEG_SHA256 = '2ce797a0f88d7f067180338fb227f7b1928ea727bd9a4d7a1d022f7c52af71a3'
WRITER_SHA256 = '66ffdee7bc6d2b62e3b135c8d0a7e66dff607c5ea6994ad1e413cde683f6ba7f'


def mux_narration(video, audio, output, seconds):
    """Keep mono level and AAC priming metadata in the final MP4 container."""
    video, audio, output = map(Path, (video, audio, output))
    if output.exists():raise FileExistsError(output)
    command=[str(FFMPEG),'-hide_banner','-loglevel','warning','-nostdin','-n',
        '-threads','1','-i',str(video),'-i',str(audio),'-map','0:v:0','-map','1:a:0',
        '-c:v','copy','-c:a','aac','-ar','24000','-ac','1','-b:a','128k',
        '-af','apad','-t',f'{float(seconds):.9f}','-movflags','+faststart',str(output)]
    result=subprocess.run(command,capture_output=True,text=True,
        creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS,timeout=900)
    receipt=dict(command=command,returncode=result.returncode,stderr=result.stderr,
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        source_audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
        requested_seconds=str(seconds),channels=1,sample_rate=24000,
        purpose='Encode directly into MP4; retain priming metadata and mono level')
    output.with_suffix('.audio_mux.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    if result.returncode:raise RuntimeError(result.stderr)
    return receipt


class Partial:
    def __init__(self, path, width, height, fps, codec):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.width, self.height = width, height
        self.fps, self.codec = Fraction(str(fps)), codec
        self.process = None
        self.frames = 0
        self.frozen = False
        self.log = None
        self.started = time.perf_counter()

    def put(self, frame, count):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise ValueError('Expected uint8 Cairo RGBA frame')
        if frame.shape != (self.height, self.width, 4) or count < 1 or int(count) != count:
            raise ValueError('Frame dimensions or repetition count changed')
        if self.frozen or (self.process is not None and count != 1):
            raise ValueError('A frozen wait must contain exactly one repeated-frame submission')
        if self.process is None:
            self.frozen = count > 1
            self.command = [str(FFMPEG), '-hide_banner', '-loglevel', 'warning', '-nostdin', '-n',
                '-threads', '1', '-filter_threads', '1', '-f', 'rawvideo', '-pixel_format', 'rgba',
                '-video_size', f'{self.width}x{self.height}', '-framerate', str(self.fps), '-i', 'pipe:0', '-an']
            if self.frozen:
                self.command += ['-vf', f'format=yuv420p,loop=loop=-1:size=1:start=0,setpts=N/({self.fps}*TB)',
                                 '-frames:v', str(count)]
            else:
                self.command += ['-pix_fmt', 'yuv420p']
            self.command += ['-r', str(self.fps), '-fps_mode', 'cfr', '-c:v', self.codec, '-threads', '1']
            if self.codec == 'h264_nvenc':
                self.command += ['-preset', 'p5', '-tune', 'hq', '-rc', 'vbr', '-cq', '18', '-b:v', '0']
            elif self.codec == 'libx264':
                self.command += ['-preset', 'veryfast', '-crf', '20']
            else:
                raise ValueError('Unsupported encoder')
            self.command += [str(self.path)]
            self.log = self.path.with_suffix('.ffmpeg.log').open('xb')
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=self.log,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        # A blocking pipe bounds outstanding memory; FFmpeg inherits the
        # admitted two-CPU affinity. There is no accumulating frame queue.
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames += count

    def close(self, aborted=False):
        if self.process is None:
            if not aborted:
                raise RuntimeError('Empty partial movie')
            return
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        code = self.process.wait(timeout=900)
        self.log.close()
        receipt = dict(command=self.command, returncode=code, expected_frames=self.frames,
            fps=str(self.fps), frozen=self.frozen, aborted=aborted,
            elapsed_seconds=time.perf_counter()-self.started,
            source_frame_submissions=1 if self.frozen else self.frames,
            movie_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest() if self.path.exists() else None)
        self.path.with_suffix('.writer.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        self.process = None
        if code or aborted:
            raise RuntimeError(f'Incomplete partial movie, retained with receipt: {self.path}')


def install(codec):
    import manim
    from manim.scene import scene_file_writer as writer
    source = Path(inspect.getfile(writer))
    if manim.__version__ != '0.21.0' or hashlib.sha256(source.read_bytes()).hexdigest() != WRITER_SHA256:
        raise RuntimeError('Inspect the changed Manim writer before adapting it')
    if manim.config.transparent or manim.config.movie_file_extension != '.mp4':
        raise RuntimeError('The native writer supports opaque Cairo MP4 only')
    if manim.config.renderer != manim.RendererType.CAIRO:
        raise RuntimeError('The native writer requires Cairo RGBA frames')
    version = subprocess.run([str(FFMPEG), '-version'], capture_output=True, check=True,
        creationflags=subprocess.CREATE_NO_WINDOW, text=True).stdout.splitlines()[0]
    executable_hash = hashlib.sha256(FFMPEG.read_bytes()).hexdigest()
    # The directory says 7.1.1; the owner's installed binary identifies as
    # 7.1-essentials. Bind the measured executable bytes, not its folder name.
    if not version.startswith('ffmpeg version 7.1-essentials_build-') or executable_hash != FFMPEG_SHA256:
        raise RuntimeError('The verified FFmpeg 7.1 binary changed')
    cls = writer.SceneFileWriter

    def open_stream(self, file_path=None):
        if getattr(self, '_nmkc_partial', None) is not None:
            raise RuntimeError('Previous partial was not closed')
        if file_path is None:
            file_path = self.partial_movie_files[self.renderer.num_plays]
        if file_path is None:
            raise RuntimeError('Missing partial movie path')
        self.partial_movie_file_path = file_path
        self._nmkc_partial = Partial(file_path, manim.config.pixel_width,
            manim.config.pixel_height, manim.config.frame_rate, codec)

    def write_frame(self, frame_or_renderer, num_frames=1):
        if writer.write_to_movie():
            part = getattr(self, '_nmkc_partial', None)
            if part is None:
                raise RuntimeError('Frame outside an open partial')
            part.put(frame_or_renderer, num_frames)

    def close_stream(self):
        part = getattr(self, '_nmkc_partial', None)
        if part is None:
            raise RuntimeError('No partial to close')
        part.close()
        self._nmkc_partial = None

    original_abort = cls.abort_encode_jobs
    def abort(self, reraise_encoder_failures=False):
        part = getattr(self, '_nmkc_partial', None)
        if part is not None:
            try:
                part.close(aborted=True)
            finally:
                self._nmkc_partial = None
        original_abort(self, reraise_encoder_failures=reraise_encoder_failures)

    cls.open_partial_movie_stream = open_stream
    cls.write_frame = write_frame
    cls.close_partial_movie_stream = close_stream
    cls.abort_encode_jobs = abort

    def combine(self, input_files, output_file, create_gif=False, includes_sound=False):
        if create_gif:
            raise RuntimeError('The native lecture writer supports MP4 only')
        output_file = Path(output_file)
        if output_file.exists():
            raise FileExistsError(output_file)
        listing = output_file.with_suffix('.concat.txt')
        lines = []
        expected_frames = 0
        for name in input_files:
            path = Path(name).resolve()
            if "'" in str(path) or '\n' in str(path):
                raise ValueError('Unsupported concat-list path')
            receipt = json.loads(path.with_suffix('.writer.json').read_text(encoding='utf-8'))
            if receipt['returncode'] or receipt['aborted'] or hashlib.sha256(path.read_bytes()).hexdigest() != receipt['movie_sha256']:
                raise ValueError('Partial movie receipt failed')
            expected_frames += receipt['expected_frames']
            lines.append(f"file '{path.as_posix()}'\n")
        listing.write_text(''.join(lines), encoding='utf-8')
        # Preserve the encoder's decode timestamps. Manim's generic PyAV
        # concatenator clears DTS, which rejects reordered H.264 packets here.
        command = [str(FFMPEG), '-hide_banner', '-loglevel', 'warning', '-nostdin', '-n',
            '-f', 'concat', '-safe', '0', '-i', str(listing), '-map', '0:v:0',
            '-an', '-c:v', 'copy', '-movflags', '+faststart', str(output_file)]
        result = subprocess.run(command, capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS, timeout=900)
        output_file.with_suffix('.concat.json').write_text(json.dumps(dict(command=command,
            returncode=result.returncode, stderr=result.stderr, expected_frames=expected_frames,
            input_files=len(input_files)), indent=2), encoding='utf-8')
        if result.returncode:
            raise RuntimeError(result.stderr)
    cls.combine_files = combine

    def combine_movie(self):
        files=[p for p in self.partial_movie_files if p is not None]
        if not files:raise RuntimeError('No lecture video fragments')
        movie=Path(self.movie_file_path)
        silent=movie.with_suffix('.silent.mp4') if self.includes_sound else movie
        self.combine_files(files,silent)
        if self.includes_sound:
            # Avoid the generic ADTS intermediate: it changes mono/24 kHz to
            # stereo/48 kHz and loses the encoder-delay metadata on this stack.
            audio=movie.with_suffix('.wav')
            self.audio_segment.export(audio,format='wav')
            frames=sum(json.loads(Path(p).with_suffix('.writer.json').read_text(encoding='utf-8'))['expected_frames'] for p in files)
            mux_narration(silent,audio,movie,Fraction(frames)/Fraction(str(manim.config.frame_rate)))
        self.print_file_ready_message(str(movie))
    cls.combine_to_movie=combine_movie
    return dict(backend='ffmpeg71', executable=str(FFMPEG), version=version,
        executable_sha256=executable_hash,
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        codec=codec, threads=1, max_inflight_encoders=1, installed_writer_sha256=WRITER_SHA256,
        frozen_waits='One source frame, exact native repetition count',
        scope='This render process only; shared environment files are read-only')
