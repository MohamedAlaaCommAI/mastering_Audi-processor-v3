"""Feed-forward compressor with soft knee.

  - Peak or RMS detection (configurable window)
  - Soft-knee gain computer (quadratic transition)
  - Smoothed gain follower with separate attack / release
  - Manual + auto makeup (auto estimated from threshold/ratio)
  - Denormal-safe state
  - Stereo link — process L/R with linked detector to preserve image

The gain follower still runs a per-sample Python loop (it has to, to switch
attack/release per sample). Everything else is vectorized NumPy. At
1024 samples/block/48k the loop is ~0.2 ms — fine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter, lfilter_zi

__all__ = ["DynamicCompressor", "CompressorParams"]

_DENORMAL_OFFSET: float = 1e-20


@dataclass
class CompressorParams:
    """User-tunable compressor parameters."""

    threshold_db: float = -20.0       # Above this, compression engages
    ratio: float = 4.0                # N:1 (>=1; 1 = bypass)
    knee_db: float = 6.0              # Soft-knee width (0 = hard knee)
    attack_ms: float = 10.0           # Time-constant when gain is decreasing
    release_ms: float = 120.0         # Time-constant when gain is recovering
    makeup_db: float = 0.0            # Static makeup gain
    auto_makeup: bool = True          # Auto makeup from threshold/ratio
    rms_window_ms: float = 0.0        # 0 = peak detection, >0 = RMS averaging
    stereo_link: bool = True          # Link L+R detector (stereo mode only)


class DynamicCompressor:
    """Feed-forward compressor with persistent state across blocks."""

    def __init__(self, fs: float, params: CompressorParams | None = None) -> None:
        self.fs = float(fs)
        self.params = params or CompressorParams()

        # Gain follower state (dB domain, 0 = no reduction)
        self._gain_db: float = 0.0

        self._rms_zi: np.ndarray | None = None

        self._auto_makeup_db: float = self._compute_auto_makeup()
        self._attack_coef: float = self._coef(self.params.attack_ms)
        self._release_coef: float = self._coef(self.params.release_ms)
        self._rms_coef: float = (
            self._coef(self.params.rms_window_ms) if self.params.rms_window_ms > 0 else 0.0
        )

    # ----- parameter setters ----- #

    def set_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self.params, k, v)
        if "attack_ms" in kwargs:
            self._attack_coef = self._coef(self.params.attack_ms)
        if "release_ms" in kwargs:
            self._release_coef = self._coef(self.params.release_ms)
        if "rms_window_ms" in kwargs:
            self._rms_coef = (
                self._coef(self.params.rms_window_ms)
                if self.params.rms_window_ms > 0 else 0.0
            )
            self._rms_zi = None
        if any(k in kwargs for k in
               ("threshold_db", "ratio", "auto_makeup", "makeup_db")):
            self._auto_makeup_db = self._compute_auto_makeup()

    def reset(self) -> None:
        self._gain_db = 0.0
        self._rms_zi = None

    # ----- processing ----- #

    def process(self, x: np.ndarray) -> np.ndarray:
        """Compress a mono block. Returns float64."""
        if x.size == 0:
            return x
        x = np.ascontiguousarray(x, dtype=np.float64)

        detector = self._detect_level(x)
        det_db = 20.0 * np.log10(np.maximum(detector, 1e-9))
        target_gain_db = self._gain_computer(det_db)
        smoothed_gain = self._apply_ballistics(target_gain_db)
        gain_lin = np.power(10.0, smoothed_gain / 20.0)
        return x * gain_lin

    def process_stereo(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compress a stereo block with linked detector (preserves image)."""
        if left.size == 0:
            return left, right
        left = np.ascontiguousarray(left, dtype=np.float64)
        right = np.ascontiguousarray(right, dtype=np.float64)

        if self.params.stereo_link:
            if self.params.rms_window_ms > 0:
                detector = self._detect_level_rms(np.sqrt((left * left + right * right) * 0.5))
            else:
                detector = np.maximum(np.abs(left), np.abs(right))
        else:
            # Independent detectors — can shift stereo image
            det_l = self._detect_level(left)
            det_r = self._detect_level(right)
            detector = np.maximum(det_l, det_r)

        det_db = 20.0 * np.log10(np.maximum(detector, 1e-9))
        target_gain_db = self._gain_computer(det_db)
        smoothed_gain = self._apply_ballistics(target_gain_db)
        gain_lin = np.power(10.0, smoothed_gain / 20.0)
        return left * gain_lin, right * gain_lin

    def get_gain_reduction_db(self) -> float:
        """Most recent gain reduction (dB, <=0)."""
        return self._gain_db

    # ----- internals ----- #

    def _detect_level(self, x: np.ndarray) -> np.ndarray:
        if self.params.rms_window_ms > 0:
            return self._detect_level_rms(x)
        return np.abs(x)

    def _detect_level_rms(self, x: np.ndarray) -> np.ndarray:
        rect = x * x
        a = [1.0, -self._rms_coef]
        b = [1.0 - self._rms_coef]
        if self._rms_zi is None:
            self._rms_zi = lfilter_zi(b, a) * float(rect[0])
        smoothed, self._rms_zi = lfilter(b, a, rect, zi=self._rms_zi)
        self._rms_zi += _DENORMAL_OFFSET
        return np.sqrt(np.maximum(smoothed, 1e-30))

    def _coef(self, time_ms: float) -> float:
        if time_ms <= 0:
            return 0.0
        t_samples = time_ms * 1e-3 * self.fs
        return math.exp(-1.0 / max(t_samples, 1e-6))

    def _compute_auto_makeup(self) -> float:
        """Auto makeup: ~50% of the GR we'd see at +12 dB above threshold."""
        if not self.params.auto_makeup:
            return self.params.makeup_db
        over = 12.0
        ratio = max(self.params.ratio, 1.0)
        gr = over * (1.0 - 1.0 / ratio)
        return 0.5 * gr + self.params.makeup_db

    def _gain_computer(self, det_db: np.ndarray) -> np.ndarray:
        """Static gain reduction curve (dB). Returns gain to apply (typically <=0)."""
        thr = self.params.threshold_db
        ratio = max(self.params.ratio, 1.0)
        knee = max(self.params.knee_db, 0.0)
        out = np.zeros_like(det_db)

        if knee <= 1e-6:
            mask = det_db > thr
            out[mask] = -(det_db[mask] - thr) * (1.0 - 1.0 / ratio)
        else:
            knee_low = thr - knee / 2.0
            knee_high = thr + knee / 2.0
            below = det_db <= knee_low
            above = det_db >= knee_high
            in_knee = ~below & ~above
            out[above] = -(det_db[above] - thr) * (1.0 - 1.0 / ratio)
            delta = det_db[in_knee] - knee_low
            out[in_knee] = -delta * delta * (1.0 - 1.0 / ratio) / (2.0 * knee)

        out += self._auto_makeup_db
        return out

    def _apply_ballistics(self, target_gain_db: np.ndarray) -> np.ndarray:
        """Smooth gain with separate attack/release coefficients (per-sample)."""
        a_coef = self._attack_coef
        r_coef = self._release_coef
        cur = self._gain_db
        out = np.empty_like(target_gain_db)

        # Hot loop — bind locals for speed
        t_arr = target_gain_db
        n = t_arr.size
        for i in range(n):
            t = t_arr[i]
            if t < cur:
                cur = a_coef * cur + (1.0 - a_coef) * t
            else:
                cur = r_coef * cur + (1.0 - r_coef) * t
            out[i] = cur
        if abs(cur) < _DENORMAL_OFFSET:
            cur = 0.0
        self._gain_db = cur
        return out
