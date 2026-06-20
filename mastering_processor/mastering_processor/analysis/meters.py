"""Stereo analysis meters: correlation, goniometer, LU history.

Pure analysis — these don't touch the signal. GUI reads them at ~30 fps.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

__all__ = ["CorrelationMeter", "Goniometer", "LUHistory"]


class CorrelationMeter:
    """Smoothed stereo correlation: -1 (out of phase) .. +1 (mono)."""

    def __init__(self, fs: float, window_ms: float = 200.0) -> None:
        self.fs = float(fs)
        self.window_ms = float(window_ms)
        self._alpha = math.exp(-1.0 / max(window_ms * 1e-3 * fs, 1e-6))
        self._corr: float = 1.0

    def reset(self) -> None:
        self._corr = 1.0

    def process_stereo(self, left: np.ndarray, right: np.ndarray) -> float:
        if left.size == 0:
            return self._corr
        left = np.ascontiguousarray(left, dtype=np.float64)
        right = np.ascontiguousarray(right, dtype=np.float64)
        lr = float(np.sum(left * right))
        ll = float(np.sum(left * left))
        rr = float(np.sum(right * right))
        denom = math.sqrt(max(ll * rr, 1e-12))
        instant = lr / denom if denom > 1e-12 else 1.0
        self._corr = self._alpha * self._corr + (1.0 - self._alpha) * instant
        return self._corr

    def value(self) -> float:
        return self._corr


class Goniometer:
    """Lissajous points for the goniometer (phase scope).

    Stores up to `max_points` (x, y) pairs:
      x = (L - R) / sqrt(2)   (side, rotated 45°)
      y = (L + R) / sqrt(2)   (mid, rotated 45°)

    The 45° rotation gives the classic goniometer look: mono is a vertical
    line, out-of-phase is a horizontal line.
    """

    def __init__(self, max_points: int = 4096) -> None:
        self.max_points = int(max_points)
        self._x = np.zeros(self.max_points, dtype=np.float32)
        self._y = np.zeros(self.max_points, dtype=np.float32)
        self._pos = 0
        self._full = False

    def reset(self) -> None:
        self._x.fill(0)
        self._y.fill(0)
        self._pos = 0
        self._full = False

    def process_stereo(self, left: np.ndarray, right: np.ndarray) -> None:
        if left.size == 0:
            return
        left = np.ascontiguousarray(left, dtype=np.float32)
        right = np.ascontiguousarray(right, dtype=np.float32)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        x = (left - right) * inv_sqrt2
        y = (left + right) * inv_sqrt2

        n = x.size
        buf_len = self.max_points
        end = self._pos + n
        if end <= buf_len:
            self._x[self._pos:end] = x
            self._y[self._pos:end] = y
        else:
            first = buf_len - self._pos
            self._x[self._pos:] = x[:first]
            self._y[self._pos:] = y[:first]
            self._x[:end - buf_len] = x[first:]
            self._y[:end - buf_len] = y[first:]
        self._pos = end % buf_len
        if self._pos == 0:
            self._full = True

    def get_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (x, y) as 1-D arrays, oldest first."""
        if self._full:
            return (np.roll(self._x, -self._pos),
                    np.roll(self._y, -self._pos))
        return self._x[:self._pos], self._y[:self._pos]


class LUHistory:
    """Rolling LU history (momentary + short-term + integrated)."""

    def __init__(self, history_seconds: float = 60.0, fs_block: float = 43.0) -> None:
        # ~43 blocks/sec at 1024/44k
        n = int(history_seconds * fs_block)
        self._momentary = deque(maxlen=n)
        self._short_term = deque(maxlen=n)
        self._integrated = deque(maxlen=n)

    def reset(self) -> None:
        self._momentary.clear()
        self._short_term.clear()
        self._integrated.clear()

    def push(self, momentary_lufs: float, short_term_lufs: float, integrated_lufs: float) -> None:
        self._momentary.append(momentary_lufs)
        self._short_term.append(short_term_lufs)
        self._integrated.append(integrated_lufs)

    def get_momentary(self) -> np.ndarray:
        return np.asarray(self._momentary, dtype=np.float32)

    def get_short_term(self) -> np.ndarray:
        return np.asarray(self._short_term, dtype=np.float32)

    def get_integrated(self) -> np.ndarray:
        return np.asarray(self._integrated, dtype=np.float32)
