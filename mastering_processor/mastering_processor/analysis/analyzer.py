"""Per-block signal analyzer.

Cheap stats for the adaptive controller (loudness, crest, band energy) and
the GUI level meters (RMS, peak). Computes:

  - short/long-term RMS (dB, one-pole smoothed)
  - peak with fast attack / slow release
  - per-band energy via 10 bandpass filters (cheaper than running a full
    FFT every block)
  - spectral centroid proxy from those band energies
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter, lfilter_zi

from ..dsp.filters import design_biquad

__all__ = ["SignalAnalyzer", "AnalysisResult"]

_DENORMAL_OFFSET: float = 1e-20


@dataclass
class AnalysisResult:
    """Analyzer snapshot at the end of a block."""

    rms_short_db: float
    rms_long_db: float
    peak_db: float
    crest_db: float
    band_db: np.ndarray
    centroid_hz: float
    is_silent: bool


class SignalAnalyzer:
    """One-pole-smoothed level + band-energy analyzer (stereo)."""

    def __init__(
        self,
        fs: float,
        band_freqs: np.ndarray,
        short_window_ms: float = 50.0,
        long_window_ms: float = 1000.0,
    ) -> None:
        self.fs = float(fs)
        self.band_freqs = np.asarray(band_freqs, dtype=np.float64)

        self._alpha_short = self._alpha(short_window_ms)
        self._alpha_long = self._alpha(long_window_ms)
        self._alpha_band = self._alpha(150.0)

        self._rms_short_sq = 0.0
        self._rms_long_sq = 0.0
        self._peak = 0.0

        self._band_env = np.zeros(len(band_freqs), dtype=np.float64)
        self._band_coeffs, self._band_zi = self._build_band_filters()

        self._centroid_hz = 1000.0

    def reset(self) -> None:
        self._rms_short_sq = 0.0
        self._rms_long_sq = 0.0
        self._peak = 0.0
        self._band_env[:] = 0.0
        self._band_zi[:] = 0.0
        self._centroid_hz = 1000.0

    def analyze_stereo(self, left: np.ndarray, right: np.ndarray) -> AnalysisResult:
        """Update state from a stereo block, return a snapshot."""
        if left.size == 0:
            return AnalysisResult(
                rms_short_db=-120.0, rms_long_db=-120.0, peak_db=-120.0,
                crest_db=0.0, band_db=np.full(len(self.band_freqs), -120.0),
                centroid_hz=self._centroid_hz, is_silent=True,
            )

        left = np.ascontiguousarray(left, dtype=np.float64)
        right = np.ascontiguousarray(right, dtype=np.float64)
        mono = (left + right) * 0.5

        # RMS one-pole on the squared signal
        sq = float(np.mean(mono * mono))
        a_s = self._alpha_short
        a_l = self._alpha_long
        self._rms_short_sq = a_s * self._rms_short_sq + (1.0 - a_s) * sq
        self._rms_long_sq = a_l * self._rms_long_sq + (1.0 - a_l) * sq

        # Peak: fast attack, slow release
        block_peak = float(np.max(np.abs(mono)))
        if block_peak > self._peak:
            self._peak += (block_peak - self._peak) * (1.0 - self._alpha(5.0))
        else:
            self._peak *= self._alpha(500.0)

        # Per-band: bandpass, mean(y²)
        band_env_new = np.empty(len(self.band_freqs), dtype=np.float64)
        for i in range(len(self.band_freqs)):
            b, a = self._band_coeffs[i]
            y, zi = lfilter(b, a, mono, zi=self._band_zi[i])
            self._band_zi[i] = zi + _DENORMAL_OFFSET
            band_env_new[i] = float(np.mean(y * y))

        a_b = self._alpha_band
        self._band_env = a_b * self._band_env + (1.0 - a_b) * band_env_new

        # Centroid from band energies
        eps = 1e-12
        total_energy = float(np.sum(self._band_env) + eps)
        if total_energy > eps * 10:
            new_centroid = float(np.sum(self.band_freqs * self._band_env) / total_energy)
            a_c = self._alpha(400.0)
            self._centroid_hz = a_c * self._centroid_hz + (1.0 - a_c) * new_centroid

        rms_short_db = 10.0 * math.log10(max(self._rms_short_sq, 1e-12))
        rms_long_db = 10.0 * math.log10(max(self._rms_long_sq, 1e-12))
        peak_db = 20.0 * math.log10(max(self._peak, 1e-9))
        crest_db = peak_db - rms_short_db
        band_db = 10.0 * np.log10(np.maximum(self._band_env, 1e-12))
        is_silent = rms_short_db < -75.0

        return AnalysisResult(
            rms_short_db=rms_short_db, rms_long_db=rms_long_db,
            peak_db=peak_db, crest_db=crest_db, band_db=band_db,
            centroid_hz=self._centroid_hz, is_silent=is_silent,
        )

    def _alpha(self, window_ms: float) -> float:
        if window_ms <= 0:
            return 0.0
        t_samples = window_ms * 1e-3 * self.fs
        return math.exp(-1.0 / max(t_samples, 1e-6))

    def _build_band_filters(self):
        coeffs = []
        zi_init = np.zeros((len(self.band_freqs), 2), dtype=np.float64)
        for fc in self.band_freqs:
            b, a = design_biquad("bandpass", float(fc), self.fs, Q=2.0, gain_db=0.0)
            coeffs.append((b, a))
        return coeffs, zi_init
