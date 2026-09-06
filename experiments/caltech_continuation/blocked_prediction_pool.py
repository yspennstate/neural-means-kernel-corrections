"""Read pinned prediction arrays in bounded blocks without stacking the full pool.

This module performs file verification and array reads only. It neither trains
nor launches work. A caller still needs host admission and protected immutable
inputs; verify_integrity() provides the final full-content recheck before output.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np


class PoolError(ValueError):
    pass


def positive_integer(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise PoolError(f"{label} must be a positive integer")
    return int(value)


def signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Member:
    name: str
    path: Path
    sha256: str
    shape: tuple[int, ...]
    dtype: str


class PredictionPool:
    def __init__(self, members, *, max_block_bytes=64 * 1024 * 1024):
        self.max_block_bytes = positive_integer(max_block_bytes, "Block allocation budget")
        self.members = tuple(members)
        if not self.members:
            raise PoolError("A predictor pool must contain at least one member")
        if any(not isinstance(m, Member) or not isinstance(m.name, str) or not m.name for m in self.members):
            raise PoolError("Each pool member requires a named Member record")
        if len({m.name for m in self.members}) != len(self.members):
            raise PoolError("Duplicate predictor names")
        if len({Path(m.path).resolve() for m in self.members}) != len(self.members):
            raise PoolError("Duplicate predictor files")
        self._signatures = []
        dimensions = []
        for member in self.members:
            if not re.fullmatch(r"[0-9a-f]{64}", member.sha256):
                raise PoolError("Each predictor requires a lowercase SHA-256 pin")
            path = Path(member.path)
            before = signature(path)
            if file_hash(path) != member.sha256 or signature(path) != before:
                raise PoolError(f"Predictor content does not match its pin: {member.name}")
            self._signatures.append(before)
            with self._mapped(len(self._signatures) - 1) as array:
                if array.ndim < 2 or min(array.shape) < 1:
                    raise PoolError("Predictors require case and pixel dimensions")
                dimensions.append((array.shape[0], int(np.prod(array.shape[1:]))))
        if len(set(dimensions)) != 1:
            raise PoolError("Predictors have inconsistent case/pixel dimensions")
        self.n_cases, self.n_pixels = dimensions[0]

    @contextmanager
    def _mapped(self, index):
        member = self.members[index]
        path = Path(member.path)
        expected = self._signatures[index]
        if signature(path) != expected:
            raise PoolError(f"Predictor file changed: {member.name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            if not isinstance(array, np.memmap):
                raise PoolError("Each member must be one standalone NPY array")
            if tuple(array.shape) != tuple(member.shape) or array.dtype.str != member.dtype:
                raise PoolError(f"Predictor header differs from its manifest: {member.name}")
            if array.dtype.kind != "f" or array.dtype.itemsize not in (4, 8):
                raise PoolError("Only float32/float64 prediction arrays are supported")
            if not array.flags.c_contiguous:
                raise PoolError("Non-C-contiguous input needs an explicit separate conversion")
            yield array
        finally:
            if isinstance(array, np.memmap):
                array._mmap.close()
            elif hasattr(array, "close"):
                array.close()
        if signature(path) != expected:
            raise PoolError(f"Predictor changed while mapped: {member.name}")

    def block(self, rows, pixel_start, pixel_stop, *, dtype=np.float32):
        """Return member x selected-case x contiguous-pixel data with no pool-sized copy."""
        rows = np.asarray(rows)
        if rows.ndim != 1 or len(rows) == 0 or rows.dtype.kind not in "iu":
            raise PoolError("Rows must be a nonempty one-dimensional integer array")
        if np.any(rows < 0) or np.any(rows >= self.n_cases) or len(np.unique(rows)) != len(rows):
            raise PoolError("Rows must be unique valid case positions")
        for value in (pixel_start, pixel_stop):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise PoolError("Pixel bounds must be integers")
        if not 0 <= pixel_start < pixel_stop <= self.n_pixels:
            raise PoolError("Pixel bounds are empty or outside the input")
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise PoolError("Output must be native float32 or float64")
        width = int(pixel_stop - pixel_start)
        # Output plus the largest advanced-indexing temporary, before allocation.
        output_bytes = len(self.members) * len(rows) * width * dtype.itemsize
        temporary_bytes = len(rows) * width * max(np.dtype(m.dtype).itemsize for m in self.members)
        if output_bytes + temporary_bytes > self.max_block_bytes:
            raise PoolError("Requested block exceeds the explicit allocation budget")
        out = np.empty((len(self.members), len(rows), width), dtype=dtype)
        pixels = np.arange(pixel_start, pixel_stop)
        for index in range(len(self.members)):
            with self._mapped(index) as array:
                flat = array.reshape(self.n_cases, self.n_pixels)
                out[index] = flat[rows[:, None], pixels[None, :]]
            if not np.isfinite(out[index]).all():
                raise PoolError(f"Nonfinite predictor values: {self.members[index].name}")
        return out

    def verify_integrity(self):
        """Recheck every full hash before adopting output from a completed run."""
        for index, member in enumerate(self.members):
            path = Path(member.path)
            before = signature(path)
            if before != self._signatures[index] or file_hash(path) != member.sha256 or signature(path) != before:
                raise PoolError(f"Predictor integrity changed during the run: {member.name}")
        return {member.name: member.sha256 for member in self.members}
