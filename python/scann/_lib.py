"""ctypes bridge for the standalone Mojo scoring library."""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIB = Path(os.environ.get("MOJO_SCANN_LIB", ROOT / "dist" / "libmojo-scann.so"))
I = ctypes.c_int64

_SIGNATURES = {
    "msc_l2_scores": ([I, I, I, I, I], None),
    "msc_dot_scores": ([I, I, I, I, I], None),
    "msc_l2_scores_batched": ([I, I, I, I, I, I], None),
    "msc_dot_scores_batched": ([I, I, I, I, I, I], None),
    "msc_exact_candidates_batched": ([I, I, I, I, I, I, I, I], None),
    "msc_ah_scores": ([I, I, I, I, I, I], None),
    "msc_ah_scores_batched": ([I, I, I, I, I, I, I], None),
    "msc_ah_top_batched": ([I, I, I, I, I, I, I, I, I, I], None),
}
_loaded: ctypes.CDLL | None = None


def build(force: bool = False) -> Path:
    if not force and LIB.exists() and LIB.stat().st_mtime >= (ROOT / "src" / "capi.mojo").stat().st_mtime:
        return LIB
    proc = subprocess.run(["bash", str(ROOT / "build" / "build.sh")], cwd=ROOT, text=True,
                          capture_output=True, timeout=1800)
    if proc.returncode or not LIB.exists():
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        loaded = ctypes.CDLL(str(build()))
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(loaded, name)
            fn.argtypes, fn.restype = argtypes, restype
        _loaded = loaded
    return _loaded


def f64(value: object) -> np.ndarray:
    """Return a C-contiguous Float64 array without lossy dtype conversion.

    The shared library dereferences every address as ``Float64``.  Float32 is
    deliberately widened here; integers, complex values, and extended-precision
    floats are rejected instead of being silently reinterpreted or narrowed.
    """
    array = np.asarray(value)
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("vectors must have dtype float32 or float64")
    return np.ascontiguousarray(array, dtype=np.float64)


def i64(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise TypeError("integer values must have an integer dtype")
    if array.dtype.kind == "u" and array.size and array.max() > np.iinfo(np.int64).max:
        raise OverflowError("integer values must fit in int64")
    return np.ascontiguousarray(array, dtype=np.int64)


def addr(value: np.ndarray) -> int:
    if not value.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    return int(value.ctypes.data)
