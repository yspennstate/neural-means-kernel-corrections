"""Record an explicit chapter script, resumably, without visible windows.

Example: python narrate.py chapters/01.json --concurrency 2
Authorizes no extra jobs: chapter and bounded concurrency are required arguments.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from audio_contract import VOICE, RATE, normalized, identity, audio_info, validate

HERE = Path(__file__).resolve().parent
NOWIN = 0x08000000 if os.name == "nt" else 0


def atomic_json(path, obj):
    part = path.with_suffix(".json.part")
    part.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(part, path)


async def main(args):
    import edge_tts
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required on PATH")
    data = json.loads(Path(args.chapter).read_text(encoding="utf-8"))
    out = HERE / "audio" / data["id"]
    out.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    entries = [(f'{board["key"]}_{i:02}', segment["say"])
               for board in data["boards"] for i, segment in enumerate(board["segments"], 1)]
    if len({k for k, _ in entries}) != len(entries):
        raise ValueError("Duplicate segment key")

    async def one(key, text):
        wav = out / (key + ".wav")
        try:
            seconds = validate(wav, text)
            print(f"CURRENT {key} {seconds:.2f}s", flush=True)
            return
        except (OSError, ValueError, KeyError):
            pass
        async with semaphore:
            part = out / (key + ".part.mp3")
            wav_part = out / (key + ".part.wav")
            for attempt in range(5):
                try:
                    await edge_tts.Communicate(normalized(text), VOICE, rate=RATE).save(str(part))
                    proc = await asyncio.create_subprocess_exec(
                        ffmpeg, "-y", "-nostdin", "-loglevel", "error", "-threads", "1",
                        "-i", str(part), "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le",
                        "-threads", "1", str(wav_part), creationflags=NOWIN,
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                    _, err = await proc.communicate()
                    if proc.returncode:
                        raise RuntimeError(err.decode(errors="replace")[-300:])
                    info = audio_info(wav_part)
                    os.replace(wav_part, wav)
                    atomic_json(wav.with_suffix(".json"), {"identity": identity(text), "audio": info,
                                                           "recorded_at_unix": time.time()})
                    validate(wav, text)
                    part.unlink(missing_ok=True)
                    print(f"RECORDED {key} {info['seconds']:.2f}s", flush=True)
                    return
                except Exception as exc:
                    print(f"RETRY {key} {attempt+1}: {type(exc).__name__}: {str(exc)[:180]}", flush=True)
                    if attempt == 4:
                        raise
                    await asyncio.sleep(3 * (attempt + 1))

    await asyncio.gather(*(one(key, text) for key, text in entries))
    total = sum(validate(out / (key + ".wav"), text) for key, text in entries)
    print(f"COMPLETE {data['id']}: {len(entries)} segments; {total/60:.2f} spoken minutes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chapter")
    parser.add_argument("--concurrency", type=int, choices=(1, 2), required=True)
    asyncio.run(main(parser.parse_args()))
