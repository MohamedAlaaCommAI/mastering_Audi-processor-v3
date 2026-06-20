"""De-esser — sibilance reducer via sidechain bandpass detection.

  1. Sidechain: bandpass around the sibilance frequency (default 7 kHz,
     range 4-12 kHz).
  2. Envelope follower on the sidechain (fast attack, slow release).
  3. When the envelope exceeds threshold, compute gain reduction.
  4. Apply smoothed gain to the full-band signal — or in "split" mode, only
     to the sibilance band.

Split mode is gentler: splits into sibilance band + "everything else",
applies reduction only to the sibilance band, then re-sums. Preserves
clarity of non-sibilant content.

Detection band uses a 4th-order Butterworth bandpass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

from .filters import LinkwitzRileyCrossover

__all__ = ["DeEsser", "DeEsserParams"]

_DENORMAL_OFFSET: float = 1e-20


@dataclass
class DeEsserParams:
    frequency_hz: float = 7000.0    # Center of sibilance detector
    q_factor: float = 3.0           # Q of the bandpass detector
    threshold_db: float = -25.0     # Sidechain threshold
    reduction_db: float = -6.0      # Max gain reduction
    attack_ms: float = 1.0
    release_ms: float = 100.0
    split_mode: bool = True         # True = reduce only sibilance band
    enabled: bool = True


class DeEsser:
    """Sidechain bandpass de-esser."""

    def __init__(self, fs: float, params: DeEsserParams | None = None) -> None:
        self.fs = float(fs)
        self.params = params or DeEsserParams()
        self._rebuild_filters()

        self._env_lin: float = 0.0
        self._attack_coef = self._coef(self.params.attack_ms)
        self._release_coef = self._coef(self.params.release_ms)

        self._gain_lin: float = 1.0
        self._gain_attack_coef = self._coef(5.0)
        self._gain_release_coef = self._coef(50.0)

        self._current_reduction_db: float = 0.0

    def _rebuild_filters(self) -> None:
        # 4th-order bandpass = Butterworth order 2 with btype='bandpass'
        nyq = self.fs * 0.5
        bw = self.params.frequency_hz / max(self.params.q_factor, 0.1)
        low = max(20.0, self.params.frequency_hz - bw) / nyq
        high = min(self.params.frequency_hz + bw, nyq - 1.0) / nyq
        low = max(1e-3, min(low, 0.999))
        high = max(low + 1e-3, min(high, 0.999))
        self._bp_b, self._bp_a = butter(2, [low, high], btype="bandpass")
        self._bp_zi: np.ndarray | None = None

        # Split mode: separate LR4 crossovers per channel (state isolation)
        if self.params.split_mode:
            f_low = max(80.0, self.params.frequency_hz - bw * 0.5)
            f_high = min(nyq - 100.0, self.params.frequency_hz + bw * 0.5)
            self._xover_low_l = LinkwitzRileyCrossover(self.fs, f_low, order=4)
            self._xover_high_l = LinkwitzRileyCrossover(self.fs, f_high, order=4)
            self._xover_low_r = LinkwitzRileyCrossover(self.fs, f_low, order=4)
            self._xover_high_r = LinkwitzRileyCrossover(self.fs, f_high, order=4)

    def set_params(self, **kwargs) -> None:
        rebuild = False
        for k, v in kwargs.items():
            setattr(self.params, k, v)
        if any(k in kwargs for k in ("frequency_hz", "q_factor", "split_mode")):
            rebuild = True
        if "attack_ms" in kwargs:
            self._attack_coef = self._coef(self.params.attack_ms)
        if "release_ms" in kwargs:
            self._release_coef = self._coef(self.params.release_ms)
        if rebuild:
            self._rebuild_filters()

    def reset(self) -> None:
        self._bp_zi = None
        self._env_lin = 0.0
        self._gain_lin = 1.0
        self._current_reduction_db = 0.0
        if self.params.split_mode:
            self._xover_low_l.reset()
            self._xover_high_l.reset()
            self._xover_low_r.reset()
            self._xover_high_r.reset()

    def get_reduction_db(self) -> float:
        return self._current_reduction_db

    def process_stereo(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process a stereo block. Returns (left_out, right_out)."""
        if not self.params.enabled or left.size == 0:
            return left, right

        left = np.ascontiguousarray(left, dtype=np.float64)
        right = np.ascontiguousarray(right, dtype=np.float64)

        # Sidechain: downmix to mono, bandpass
        mono = (left + right) * 0.5
        if self._bp_zi is None:
            self._bp_zi = lfilter_zi(self._bp_b, self._bp_a) * float(mono[0])
        sidechain, self._bp_zi = lfilter(self._bp_b, self._bp_a, mono, zi=self._bp_zi)
        self._bp_zi += _DENORMAL_OFFSET

        # Envelope follower
        side_abs = np.abs(sidechain)
        env = self._follow_envelope(side_abs)

        # Target gain reduction per sample.
        # Implicit 1:∞ ratio — we scale linearly from threshold (gain=1.0)
        # to threshold + 20 dB (gain=max_reduction_lin).
        threshold_lin = 10.0 ** (self.params.threshold_db / 20.0)
        max_reduction_lin = 10.0 ** (self.params.reduction_db / 20.0)

        over_db = 20.0 * np.log10(np.maximum(env / threshold_lin, 1.0))
        target_gain = np.where(
            env > threshold_lin,
            10.0 ** (np.maximum(over_db, 0.0) / 20.0 * math.log10(max(max_reduction_lin, 1e-6))),
            1.0,
        )
        np.minimum(target_gain, 1.0, out=target_gain)
        np.maximum(target_gain, max_reduction_lin, out=target_gain)

        smoothed_gain = self._smooth_gain(target_gain)

        if self.params.split_mode:
            # Reduce only the sibilance band
            low_l, rest_l = self._xover_low_l.process(left)
            band_l, high_l = self._xover_high_l.process(rest_l)
            low_r, rest_r = self._xover_low_r.process(right)
            band_r, high_r = self._xover_high_r.process(rest_r)

            out_l = low_l + band_l * smoothed_gain + high_l
            out_r = low_r + band_r * smoothed_gain + high_r
        else:
            out_l = left * smoothed_gain
            out_r = right * smoothed_gain

        self._current_reduction_db = 20.0 * math.log10(max(smoothed_gain[-1], 1e-6))

        return out_l, out_r

    # ----- internals ----- #

    def _coef(self, time_ms: float) -> float:
        if time_ms <= 0:
            return 0.0
        t_samples = time_ms * 1e-3 * self.fs
        return math.exp(-1.0 / max(t_samples, 1e-6))

    def _follow_envelope(self, side_abs: np.ndarray) -> np.ndarray:
        """Per-sample envelope follower with attack/release."""
        a = self._attack_coef
        r = self._release_coef
        cur = self._env_lin
        out = np.empty_like(side_abs)
        for i in range(side_abs.size):
            s = side_abs[i]
            if s > cur:
                cur = a * cur + (1.0 - a) * s
            else:
                cur = r * cur + (1.0 - r) * s
            out[i] = cur
        if abs(cur) < _DENORMAL_OFFSET:
            cur = 0.0
        self._env_lin = cur
        return out

    def _smooth_gain(self, target: np.ndarray) -> np.ndarray:
        a = self._gain_attack_coef
        r = self._gain_release_coef
        cur = self._gain_lin
        out = np.empty_like(target)
        for i in range(target.size):
            t = target[i]
            if t < cur:
                cur = a * cur + (1.0 - a) * t
            else:
                cur = r * cur + (1.0 - r) * t
            out[i] = cur
        if abs(cur) < _DENORMAL_OFFSET:
            cur = 1.0
        self._gain_lin = cur
        return out
