"""Encode an actual lecture board through Manim's adapted writer.

One codec per process. Compare encoder throughput, not end-to-end scene speed.
Run only within the owner's current compute policy and the NMKC GPU lease for NVENC.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--encoder', choices=('libx264', 'h264_nvenc'), required=True)
    p.add_argument('--frames', type=int, default=600)
    args = p.parse_args()
    if args.out.exists() or args.frames < 1:
        raise ValueError('Require a new output directory and positive frame count')
    if os.name != 'nt':
        raise RuntimeError('Windows background policy required')
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    handle = kernel.GetCurrentProcess()
    if not kernel.SetPriorityClass(handle, 0x4000) or not kernel.SetProcessAffinityMask(handle, 1 << 5):
        raise ctypes.WinError()
    if args.encoder == 'h264_nvenc':
        subprocess.run([r'C:\Python314\python.exe', '-B',
                        r'C:\Users\owner\ai-memories-and-functionality\12_cognitive_architecture\agent_mesh\agent_mesh.py',
                        'assert', '--agent', 'codex-nmkc-resume-20260905', '--resource', 'topic:gpu-workload/MATH-ROSS20/codex-nmkc-resume-20260905'],
                       check=True, creationflags=0x08000000)
    os.environ['NMKC_ENCODER'] = args.encoder
    import av
    import numpy as np
    from PIL import Image
    from manim import config
    from manim.scene.scene_file_writer import SceneFileWriter
    from encoder_policy import install
    policy = install()
    frame = np.asarray(Image.open(args.image).convert('RGBA'))
    h, w, _ = frame.shape
    if w % 2 or h % 2:
        raise ValueError('Opaque 4:2:0 input requires even dimensions')
    config.pixel_width, config.pixel_height, config.frame_rate = w, h, 30
    config.max_inflight_encoders = 1
    args.out.mkdir(parents=True)
    output = args.out / 'board.mp4'
    from gpu_telemetry import Telemetry
    telemetry=Telemetry()
    samples, errors, stop = [telemetry.snapshot()], [], threading.Event()
    def poll_gpu():
        while not stop.is_set():
            try:samples.append(telemetry.snapshot())
            except Exception as error:
                errors.append({'at':time.time(),'error':str(error)});break
            stop.wait(.1)
    watcher = threading.Thread(target=poll_gpu)
    watcher.start()
    try:
        writer = object.__new__(SceneFileWriter)
        writer._inflight_by_path = {}
        writer.renderer = SimpleNamespace(num_plays=0)
        encode_started_at=time.time()
        start = time.perf_counter()
        writer.open_partial_movie_stream(str(output))
        job = writer._current_encode_job
        job.put(args.frames, frame)
        job.seal(); job.join()
        elapsed = time.perf_counter()-start
        encode_ended_at=time.time()
    finally:
        stop.set(); watcher.join()
        try:samples.append(telemetry.snapshot())
        finally:telemetry.close()
    decoded = 0
    with av.open(output) as container:
        stream = container.streams.video[0]
        codec = stream.codec_context.name
        for f in container.decode(stream):
            if decoded == 0:
                image = f.to_image()
                image.save(args.out / 'decoded_first_frame.png')
                error = np.abs(np.asarray(image).astype(float)-frame[:, :, :3]).mean()
            decoded += 1
    if decoded != args.frames:
        raise ValueError(f'Wrong decoded frame count: {decoded}')
    report = dict(pid=os.getpid(), source_image=str(args.image.resolve()),
                  source_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
                  policy=policy, frames=decoded, width=w, height=h, fps=30,
                  encoder_seconds=elapsed, decoded_codec=codec,
                  mean_absolute_rgb_difference=float(error),
                  output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  gpu_samples=samples,gpu_sampling_errors=errors,
                  encode_started_at=encode_started_at,encode_ended_at=encode_ended_at,
                  gpu_telemetry_status='INCOMPLETE' if errors else 'SAMPLED_DEVICE_WIDE',
                  human_visual_review='PENDING',
                  scope='Measured encoding and decoding of a fixed lecture board; excludes scene construction and typesetting')
    (args.out / 'receipt.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
