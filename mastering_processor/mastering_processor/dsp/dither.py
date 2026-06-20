"""TPDF dither with first-order noise shaping.

Apply to the final output just before quantization (e.g., 16-bit export).
For real-time float32 playback dither is unnecessary (no quantization
step), but it's essential for 16-bit PCM export to avoid quantization
distortion on quiet tails.

  - TPDF: two uniform random samples scaled to 1 LSB.
  - First-order noise shaping: subtract the previous quantization error
    from the current sample before dithering.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

__all__ = ["Dither", "DitherType"]


class DitherType(str, Enum):
    OFF = "off"
    TPDF = "tpdf"
    TPDF_NS1 = "tpdf_ns1"  # TPDF + first-order noise shaping


class Dither:
    """Bit-depth dither with optional noise shaping."""

    def __init__(self, target_bits: int = 16, dtype: DitherType = DitherType.OFF) -> None:
        self.target_bits = int(target_bits)
        self.dtype = dtype
        self._lsb = 1.0 / (2.0 ** (self.target_bits - 1))
        self._prev_error: float = 0.0

    def reset(self) -> None:
        self._prev_error = 0.0

    def set_bits(self, bits: int) -> None:
        self.target_bits = int(bits)
        self._lsb = 1.0 / (2.0 ** (self.target_bits - 1))

    def set_type(self, dtype: DitherType) -> None:
        self.dtype = dtype

    def process(self, x: np.ndarray) -> np.ndarray:
        """Apply dither + quantization to a mono block."""
        if self.dtype == DitherType.OFF or x.size == 0:
            return x

        x = np.ascontiguousarray(x, dtype=np.float64)
        n = x.size
        lsb = self._lsb

        if self.dtype == DitherType.TPDF:
            # TPDF = (rand1 + rand2 - 1) * lsb → triangular, ±lsb
            r = (np.random.random(n) + np.random.random(n) - 1.0) * lsb
            quantized = np.round((x + r) / lsb) * lsb
            return quantized

        if self.dtype == DitherType.TPDF_NS1:
            # First-order noise shaping: subtract previous quantization error
            out = np.empty(n, dtype=np.float64)
            prev_err = self._prev_error
            r_arr = (np.random.random(n) + np.random.random(n) - 1.0) * lsb
            for i in range(n):
                shaped = x[i] - prev_err
                dithered = shaped + r_arr[i]
                q = round(dithered / lsb) * lsb
                out[i] = q
                prev_err = q - shaped
            self._prev_error = prev_err
            return out

        return x
