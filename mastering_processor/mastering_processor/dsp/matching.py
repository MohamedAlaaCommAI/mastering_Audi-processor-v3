"""Reference-track spectrum matching (DSP only, no AI).

  1. Compute Long-Term Average Spectrum (LTAS) of both target and
     reference: Hann-windowed FFT over 4096-sample blocks, averaged in dB.
  2. Correction curve = ref_db - target_db (pointwise).
  3. 1/3-octave smoothing to avoid over-fitting narrow spectral features.
  4. Sample the smoothed curve at the EQ band center frequencies.
  5. Soft constraint (±6 dB by default).
  6. Return per-band gain adjustments.

Runs offline on whole files, not in the real-time callback. Result gets
applied to the live EQ via ParametricEqualizer.update_band.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["ReferenceMatcher"]


class ReferenceMatcher:
    """Compute EQ-band gains to match a target signal to a reference."""

    def __init__(
        self,
        fs: float,
        fft_size: int = 4096,
        max_correction_db: float = 6.0,
        smoothing_octaves: float = 1.0 / 3.0,
    ) -> None:
        self.fs = float(fs)
        self.fft_size = int(fft_size)
        self.max_correction_db = float(max_correction_db)
        self.smoothing_octaves = float(smoothing_octaves)

    def compute_ltas(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Long-Term Average Spectrum in dB. Returns (freqs_hz, magnitude_db)."""
        signal = np.ascontiguousarray(signal, dtype=np.float64)
        if signal.ndim == 2:
            signal = signal.mean(axis=1)

        n = self.fft_size
        hop = n // 2
        window = np.hanning(n).astype(np.float64)

        if signal.size < n:
            signal = np.pad(signal, (0, n - signal.size))

        n_frames = max(1, (signal.size - n) // hop + 1)
        accum = np.zeros(n // 2 + 1, dtype=np.float64)

        for i in range(n_frames):
            frame = signal[i * hop:i * hop + n]
            if frame.size < n:
                frame = np.pad(frame, (0, n - frame.size))
            spec = np.fft.rfft(frame * window)
            mag = np.abs(spec)
            accum += mag

        avg_mag = accum / n_frames
        mag_db = 20.0 * np.log10(np.maximum(avg_mag, 1e-9))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)
        return freqs, mag_db

    def compute_correction(
        self,
        target: np.ndarray,
        reference: np.ndarray,
        band_freqs: Sequence[float],
    ) -> np.ndarray:
        """Per-band gain corrections (dB), clamped to ±max_correction_db."""
        ref_f, ref_db = self.compute_ltas(reference)
        tgt_f, tgt_db = self.compute_ltas(target)

        # Interpolate onto the reference's frequency grid
        tgt_db_aligned = np.interp(ref_f, tgt_f, tgt_db)

        correction = ref_db - tgt_db_aligned

        smoothed = self._fractional_octave_smooth(ref_f, correction)

        gains = np.interp(band_freqs, ref_f, smoothed)

        gains = np.clip(gains, -self.max_correction_db, self.max_correction_db)
        return gains

    def _fractional_octave_smooth(
        self, freqs: np.ndarray, curve: np.ndarray
    ) -> np.ndarray:
        """Fractional-octave moving-average smoothing in log-frequency space."""
        if freqs.size < 4:
            return curve.copy()

        log_f = np.log10(np.maximum(freqs, 1.0))
        dlog = float(np.median(np.diff(log_f)))
        if dlog <= 0:
            return curve.copy()

        # 1/3 octave in log-freq samples (log10(2^(1/3)) ≈ 0.1)
        window_octaves = self.smoothing_octaves
        window_log = np.log10(2.0 ** window_octaves)
        window_samples = max(1, int(round(window_log / dlog)))

        kernel = np.ones(window_samples, dtype=np.float64) / window_samples
        pad = window_samples // 2
        padded = np.pad(curve, (pad, pad), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        # 'valid' returns len = len(curve) + 1 - window_samples — trim/pad
        if smoothed.size < curve.size:
            smoothed = np.pad(smoothed, (0, curve.size - smoothed.size), mode="edge")
        elif smoothed.size > curve.size:
            smoothed = smoothed[:curve.size]
        return smoothed
