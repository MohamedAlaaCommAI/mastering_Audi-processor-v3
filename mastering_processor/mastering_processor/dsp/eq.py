"""Parametric EQ built from biquad sections.

Standard mastering EQ controls: peak, lowshelf, highshelf, lowpass,
highpass, notch, bandpass. Each band keeps a separate user gain and an
adaptive-gain field so the AdaptiveController can apply content-aware cuts
without touching the user's manual curve.

compute_response_curve() returns the full transfer function over a
log-frequency grid. The GUI overlays this on the live spectrum
(FabFilter Pro-Q / iZotope Ozone style).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.signal import freqz

from .filters import BiquadFilter

__all__ = ["Band", "ParametricEqualizer", "DEFAULT_EQ_BANDS"]


@dataclass
class Band:
    """A single EQ band spec."""

    fc: float
    Q: float = 0.707
    gain_db: float = 0.0
    filter_type: str = "peak"
    adaptive_gain_db: float = 0.0
    enabled: bool = True


# Standard 10-band ISO frequencies
DEFAULT_EQ_BANDS: List[Band] = [
    Band(31.0, 1.41, 0.0, "peak"),
    Band(63.0, 1.41, 0.0, "peak"),
    Band(125.0, 1.41, 0.0, "peak"),
    Band(250.0, 1.41, 0.0, "peak"),
    Band(500.0, 1.41, 0.0, "peak"),
    Band(1000.0, 1.41, 0.0, "peak"),
    Band(2000.0, 1.41, 0.0, "peak"),
    Band(4000.0, 1.41, 0.0, "peak"),
    Band(8000.0, 1.41, 0.0, "peak"),
    Band(16000.0, 1.41, 0.0, "peak"),
]


# Factory presets (dB per band, 10 bands)
EQ_PRESETS: dict[str, List[float]] = {
    "Flat":         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Bass Boost":   [6, 5, 3, 1, 0, 0, 0, 0, 0, 0],
    "Treble Boost": [0, 0, 0, 0, 0, 0, 1, 3, 5, 6],
    "V-Shape":      [5, 4, 2, 0, -2, -2, 0, 2, 4, 5],
    "Loudness":     [4, 3, 1, 0, -1, -1, 0, 1, 3, 4],
}


class ParametricEqualizer:
    """Chain of biquad filters forming a parametric EQ.

    Separate filter state per stereo channel so L and R don't
    cross-contaminate their IIR states.
    """

    def __init__(self, fs: float, bands: Sequence[Band] | None = None) -> None:
        self.fs = float(fs)
        self.bands: List[Band] = [Band(**vars(b)) for b in (bands or DEFAULT_EQ_BANDS)]
        self._filters_l: List[BiquadFilter] = []
        self._filters_r: List[BiquadFilter] = []
        self._rebuild_filters()
        self._curve_cache: Tuple[np.ndarray, np.ndarray] | None = None
        self._last_channel: int = 0

    # ----- band management ----- #

    def add_band(self, band: Band) -> None:
        self.bands.append(band)
        self._rebuild_filters()
        self._curve_cache = None

    def remove_band(self, index: int) -> None:
        if 0 <= index < len(self.bands):
            del self.bands[index]
            self._rebuild_filters()
            self._curve_cache = None

    def update_band(
        self,
        index: int,
        fc: float | None = None,
        Q: float | None = None,
        gain_db: float | None = None,
        filter_type: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        if not (0 <= index < len(self.bands)):
            return
        band = self.bands[index]
        if fc is not None: band.fc = float(fc)
        if Q is not None: band.Q = float(Q)
        if gain_db is not None: band.gain_db = float(gain_db)
        if filter_type is not None: band.filter_type = filter_type
        if enabled is not None: band.enabled = bool(enabled)
        self._filters_l[index].update(
            self.fs,
            fc=band.fc, Q=band.Q,
            gain_db=band.gain_db + band.adaptive_gain_db,
            filter_type=band.filter_type,
        )
        self._filters_r[index].update(
            self.fs,
            fc=band.fc, Q=band.Q,
            gain_db=band.gain_db + band.adaptive_gain_db,
            filter_type=band.filter_type,
        )
        self._curve_cache = None

    def set_adaptive_gain(self, index: int, gain_db: float) -> None:
        if not (0 <= index < len(self.bands)):
            return
        band = self.bands[index]
        if abs(gain_db - band.adaptive_gain_db) < 1e-6:
            return
        band.adaptive_gain_db = gain_db
        self._filters_l[index].update(
            self.fs, gain_db=band.gain_db + band.adaptive_gain_db,
        )
        self._filters_r[index].update(
            self.fs, gain_db=band.gain_db + band.adaptive_gain_db,
        )
        self._curve_cache = None

    def reset_adaptive(self) -> None:
        for i, band in enumerate(self.bands):
            if abs(band.adaptive_gain_db) > 1e-6:
                band.adaptive_gain_db = 0.0
                self._filters_l[i].update(self.fs, gain_db=band.gain_db)
                self._filters_r[i].update(self.fs, gain_db=band.gain_db)
        self._curve_cache = None

    def load_preset(self, name: str) -> bool:
        if name not in EQ_PRESETS:
            return False
        for i, g in enumerate(EQ_PRESETS[name]):
            self.update_band(i, gain_db=float(g))
        return True

    # ----- processing ----- #

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process a mono block. State is shared with whichever channel was
        processed last — for stereo use ``process_stereo``.
        """
        y = x
        for band, f in zip(self.bands, self._filters_l):
            if band.enabled:
                y = f.process(y)
        return y

    def process_stereo(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process a stereo block with independent per-channel state."""
        yl = left
        yr = right
        for band, fl, fr in zip(self.bands, self._filters_l, self._filters_r):
            if band.enabled:
                yl = fl.process(yl)
                yr = fr.process(yr)
        return yl, yr

    def reset(self) -> None:
        for f in self._filters_l:
            f.reset()
        for f in self._filters_r:
            f.reset()

    # ----- response curve (GUI overlay) ----- #

    def compute_response_curve(
        self, n_points: int = 512
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._curve_cache is not None and len(self._curve_cache[0]) == n_points:
            return self._curve_cache
        freqs = np.logspace(np.log10(20.0), np.log10(20000.0), n_points)
        total_db = np.zeros(n_points, dtype=np.float64)
        for band, f in zip(self.bands, self._filters_l):
            if not band.enabled:
                continue
            _, h = freqz(f.b, f.a, worN=freqs, fs=self.fs)
            total_db += 20.0 * np.log10(np.maximum(np.abs(h), 1e-9))
        self._curve_cache = (freqs, total_db)
        return self._curve_cache

    # ----- internals ----- #

    def _rebuild_filters(self) -> None:
        self._filters_l = [
            BiquadFilter(b.filter_type, b.fc, self.fs, b.Q,
                         b.gain_db + b.adaptive_gain_db)
            for b in self.bands
        ]
        self._filters_r = [
            BiquadFilter(b.filter_type, b.fc, self.fs, b.Q,
                         b.gain_db + b.adaptive_gain_db)
            for b in self.bands
        ]
        self._curve_cache = None
