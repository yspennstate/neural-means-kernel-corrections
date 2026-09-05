"""One bounded render, with a measured window-visibility observation.

Launched hidden through Start-Process. The owned child is awaited to completion.
No scheduler, queue, daemon, or automatic duplicate retry is installed.
"""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import psutil

HERE = Path(__file__).resolve().parent
PYTHON = Path(r"C:\Users\owner\lecture\venv\Scripts\python.exe")
GUARD = Path(r"C:\Users\owner\.claude\compute\interactive_guard\state.json")
BACKGROUND = [4, 5, 6, 7, 8, 9, 14, 15]
RESERVED = [0, 1, 2, 3, 10, 11, 12, 13]


def windows():
    user = ctypes.windll.user32
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user.IsWindowVisible.argtypes = [wintypes.HWND]
    user.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    result = {}
    def callback(hwnd, _):
        if not user.IsWindowVisible(hwnd):
            return True
        name = ctypes.create_unicode_buffer(256)
        user.GetClassNameW(hwnd, name, 256)
        if name.value not in ("ConsoleWindowClass", "PseudoConsoleWindow"):
            return True
        rect = wintypes.RECT()
        user.GetWindowRect(hwnd, ctypes.byref(rect))
        if rect.right <= rect.left or rect.bottom <= rect.top:
            return True
        pid = wintypes.DWORD()
        user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        result[int(hwnd)] = {"pid": pid.value, "class": name.value,
                             "width": rect.right-rect.left, "height": rect.bottom-rect.top}
        return True
    user.EnumWindows(callback_type(callback), 0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter", required=True)
    parser.add_argument("--board")
    parser.add_argument("--segment", type=int, default=0)
    parser.add_argument("--reuse-tex-from", type=Path,
                        help="Reuse exact TeX/SVG asset pairs from a completed owned build")
    parser.add_argument("--quality", choices=("preview", "samples", "draft", "final"), required=True)
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("This launcher implements the Windows environment only")
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                             ctypes.POINTER(ctypes.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    handle = kernel.GetCurrentProcess()
    kernel.SetPriorityClass(handle, 0x4000)  # BELOW_NORMAL_PRIORITY_CLASS
    guard = json.loads(GUARD.read_text(encoding="utf-8-sig"))
    if not 0 <= time.time()-GUARD.stat().st_mtime <= 15:
        raise RuntimeError("CPU guard is stale")
    if guard.get("background_cpus") != BACKGROUND or guard.get("reserved_cpus") != RESERVED:
        raise RuntimeError("CPU partition changed")
    mask = (1 << 4) | (1 << 5)
    # The observer stays on the background pool; the render child gets two
    # explicitly designated background CPUs. Lowest available is NOT safe:
    # a full inherited mask includes the owner's protected terminal CPUs.
    if not kernel.SetProcessAffinityMask(handle, ctypes.c_size_t(sum(1 << c for c in BACKGROUND))):
        raise ctypes.WinError()
    env = dict(os.environ, NMKC_CHAPTER=args.chapter, NMKC_SEGMENT=str(args.segment),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    if args.board:
        env["NMKC_BOARD"] = args.board
    log_dir = HERE / "logs"; log_dir.mkdir(exist_ok=True)
    tag = args.board or f"chapter{args.chapter}_{args.quality}"
    # A long render reads a frozen source/audio/asset snapshot. Continued
    # authoring of other chapters cannot change its inputs halfway through.
    build = HERE / "builds" / (time.strftime("%Y%m%dT%H%M%S")+"_"+tag)
    build.mkdir(parents=True, exist_ok=False)
    (build / "chapters").mkdir()
    sources = list(HERE.glob("*.py"))+[HERE / "chapters" / (args.chapter+".json")]
    sources += [p for p in (HERE/"assets").glob("*") if p.is_file()]
    if args.quality in ("draft", "final"):
        sources += list((HERE/"audio"/args.chapter).glob("*.wav"))
        sources += list((HERE/"audio"/args.chapter).glob("*.json"))
    inputs = {}
    for source in sources:
        relative = source.relative_to(HERE)
        destination = build / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = source.read_bytes()
        destination.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed while freezing: {relative}")
        inputs[relative.as_posix()] = digest
    if args.reuse_tex_from:
        old = args.reuse_tex_from.resolve()
        if old.parent != (HERE / "builds").resolve() or not (old / "input_manifest.json").is_file():
            raise ValueError("TeX reuse requires an owned build in this lecture directory")
        # A complete sample manifest proves the render reached its final board;
        # a running or failed build is never used as a cache donor.
        manifests = list((old / "media/board_samples").glob("*/manifest.json"))
        if len(manifests) != 1:
            raise ValueError("TeX donor has no unique completed board manifest")
        old_manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        if not old_manifest.get("rows"):
            raise ValueError("TeX donor has no completed boards")
        for row in old_manifest["rows"]:
            frame = manifests[0].parent / row["path"]
            if hashlib.sha256(frame.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("TeX donor board identity has changed")
        for tex in sorted((old / "media/Tex").glob("p*/*.tex")):
            svg = tex.with_suffix(".svg")
            if not svg.is_file():
                continue
            # Manim keys its TeX source, including the full template, by this
            # digest. Color and placement are applied after loading the SVG.
            source_text = tex.read_text(encoding="utf-8")
            if hashlib.sha256(source_text.encode("utf-8")).hexdigest()[:16] != tex.stem:
                raise ValueError("TeX donor source no longer matches its content key")
            for asset in (tex, svg):
                relative = Path("tex_seed") / asset.name
                destination = build / relative
                destination.parent.mkdir(exist_ok=True)
                payload = asset.read_bytes()
                if destination.exists() and destination.read_bytes() != payload:
                    raise ValueError("Conflicting TeX cache keys")
                destination.write_bytes(payload)
                inputs[relative.as_posix()] = hashlib.sha256(payload).hexdigest()
        provenance = dict(source_build=old.name,
                          source_input_manifest_sha256=hashlib.sha256((old/"input_manifest.json").read_bytes()).hexdigest(),
                          pairs=sum(name.endswith(".svg") for name in inputs if name.startswith("tex_seed/")),
                          scope="Exact previously typeset vector assets; new or changed TeX compiles normally")
        cache_record = build / "tex_cache_provenance.json"
        cache_record.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        inputs[cache_record.name] = hashlib.sha256(cache_record.read_bytes()).hexdigest()
    (build/"input_manifest.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
    quality = ["-s", "-r", "1920,1080"] if args.quality in ("preview", "samples") else (
        ["-r", "1280,720", "--fps", "30"] if args.quality == "draft" else ["-r", "1920,1080", "--fps", "30"])
    target = ("BoardPreview" if args.quality == "preview" else
              "BoardSamples" if args.quality == "samples" else "LectureChapter")
    command = [str(PYTHON), "-B", "-m", "manim", "render", *quality, "--disable_caching",
               "--media_dir", str(build / "media"), "-o", tag, "scenes.py", target]
    stop = threading.Event()
    samples = []; observed = []
    baseline = windows()
    def poll():
        previous = time.perf_counter()
        while not stop.is_set():
            now = time.perf_counter(); samples.append(now-previous); previous = now
            for hwnd, info in windows().items():
                if hwnd not in baseline and not any(x["hwnd"] == hwnd for x in observed):
                    observed.append({"hwnd": hwnd, "elapsed": now-start, **info})
            stop.wait(.015)
    start = time.perf_counter()
    watcher = threading.Thread(target=poll, daemon=True); watcher.start()
    try:
        with (log_dir / (tag + "_render.log")).open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(command, cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=0x08000000 | 0x4000 | 0x4)
            resumed = False
            try:
                child = psutil.Process(proc.pid)
                child.cpu_affinity([4, 5])
                if child.cpu_affinity() != [4, 5] or child.nice() != psutil.BELOW_NORMAL_PRIORITY_CLASS:
                    raise RuntimeError("Pre-start affinity or priority verification failed")
                child.resume()
                resumed = True
            finally:
                if not resumed:
                    proc.kill()
                    proc.wait()
            proc.wait()
    finally:
        stop.set(); watcher.join()
    intervals = samples[1:]
    receipt = {"command": command, "returncode": proc.returncode, "affinity_mask": mask,
               "build_directory": str(build), "input_manifest_sha256": hashlib.sha256(
                   (build/"input_manifest.json").read_bytes()).hexdigest(),
               "elapsed_seconds": time.perf_counter()-start, "visibility_samples": len(samples),
               "max_poll_interval_ms": max(intervals, default=0)*1000,
               "mean_poll_interval_ms": sum(intervals)/max(1, len(intervals))*1000,
               "new_visible_console_windows": observed,
               "visibility_scope": "New nonzero-area console windows systemwide during this render; unrelated windows are not attributed to this task.",
               "visibility_conclusion": "INCONCLUSIVE" if observed or max(intervals, default=0) > .05 else "No new visible console observed at intervals <=50ms"}
    (log_dir / (tag + "_render_receipt.json")).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
