"""Multiband compressor with Linkwitz-Riley crossovers.

Splits the signal into N bands (default 4: low / low-mid / high-mid / high)
with LR4 crossovers, compresses each band independently, then re-sums.
LR4 has identical phase response in low and high paths, so recombination
is phase-safe (no cancellation notches at crossover frequencies).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .compressor import CompressorParams, DynamicCompressor
from .filters import LinkwitzRileyCrossover

__all__ = ["MultibandBandParams", "MultibandCompressor"]


@dataclass
class MultibandBandParams:
    """Per-band compressor parameters."""

    label: str
    crossover_hz: float       # Upper edge of this band (ignored for top band)
    threshold_db: float = -20.0
    ratio: float = 3.0
    attack_ms: float = 15.0
    release_ms: float = 200.0
    knee_db: float = 6.0
    makeup_db: float = 0.0
    enabled: bool = True


# Sensible 4-band mastering defaults
DEFAULT_MULTIBAND_BANDS: List[MultibandBandParams] = [
    MultibandBandParams("Low",     120.0,  threshold_db=-25, ratio=3.0,
                        attack_ms=20,  release_ms=250, makeup_db=2.0),
    MultibandBandParams("Low-Mid", 1500.0, threshold_db=-22, ratio=2.5,
                        attack_ms=15,  release_ms=200, makeup_db=1.5),
    MultibandBandParams("High-Mid", 6000.0, threshold_db=-20, ratio=2.0,
                        attack_ms=10,  release_ms=150, makeup_db=1.0),
    MultibandBandParams("High",    20000.0, threshold_db=-18, ratio=2.0,
                        attack_ms=5,   release_ms=100, makeup_db=0.5),
]


class MultibandCompressor:
    """N-band multiband compressor with LR4 crossovers."""

    def __init__(
        self,
        fs: float,
        bands: List[MultibandBandParams] | None = None,
    ) -> None:
        self.fs = float(fs)
        self.bands: List[MultibandBandParams] = list(bands or DEFAULT_MULTIBAND_BANDS)
        self._build_chain()
        self._enabled = True
        self._band_gr_db: List[float] = [0.0] * len(self.bands)

    def _build_chain(self) -> None:
        # One LR4 per split point per channel — state isolation.
        self._crossovers_l: List[LinkwitzRileyCrossover] = [
            LinkwitzRileyCrossover(self.fs, b.crossover_hz, order=4)
            for b in self.bands[:-1]
        ]
        self._crossovers_r: List[LinkwitzRileyCrossover] = [
            LinkwitzRileyCrossover(self.fs, b.crossover_hz, order=4)
            for b in self.bands[:-1]
        ]
        self._compressors: List[DynamicCompressor] = [
            DynamicCompressor(
                self.fs,
                CompressorParams(
                    threshold_db=b.threshold_db,
                    ratio=b.ratio,
                    knee_db=b.knee_db,
                    attack_ms=b.attack_ms,
                    release_ms=b.release_ms,
                    makeup_db=b.makeup_db,
                    auto_makeup=False,
                    stereo_link=True,
                ),
            )
            for b in self.bands
        ]

    def set_band_params(self, index: int, **kwargs) -> None:
        if not (0 <= index < len(self.bands)):
            return
        band = self.bands[index]
        for k, v in kwargs.items():
            setattr(band, k, v)
        # Rebuild just this band's compressor (cheap)
        self._compressors[index] = DynamicCompressor(
            self.fs,
            CompressorParams(
                threshold_db=band.threshold_db,
                ratio=band.ratio,
                knee_db=band.knee_db,
                attack_ms=band.attack_ms,
                release_ms=band.release_ms,
                makeup_db=band.makeup_db,
                auto_makeup=False,
                stereo_link=True,
            ),
        )

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def reset(self) -> None:
        for x in self._crossovers_l:
            x.reset()
        for x in self._crossovers_r:
            x.reset()
        for c in self._compressors:
            c.reset()
        self._band_gr_db = [0.0] * len(self.bands)

    def get_band_gain_reduction_db(self) -> List[float]:
        return list(self._band_gr_db)

    def process_stereo(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process a stereo block. Returns (left_out, right_out)."""
        if not self._enabled or left.size == 0:
            return left, right

        # Split into N bands via cascaded LR4 crossovers
        band_l: List[np.ndarray] = [None] * len(self.bands)  # type: ignore[list-item]
        band_r: List[np.ndarray] = [None] * len(self.bands)  # type: ignore[list-item]

        cur_l, cur_r = left, right
        for i, (xl, xr) in enumerate(zip(self._crossovers_l, self._crossovers_r)):
            band_l[i], cur_l = xl.process(cur_l)
            band_r[i], cur_r = xr.process(cur_r)
        band_l[-1] = cur_l
        band_r[-1] = cur_r

        out_l = np.zeros_like(left, dtype=np.float64)
        out_r = np.zeros_like(right, dtype=np.float64)
        for i, (comp, band_p) in enumerate(zip(self._compressors, self.bands)):
            if not band_p.enabled:
                out_l += band_l[i]
                out_r += band_r[i]
                self._band_gr_db[i] = 0.0
                continue
            cl, cr = comp.process_stereo(band_l[i], band_r[i])
            out_l += cl
            out_r += cr
            self._band_gr_db[i] = comp.get_gain_reduction_db()

        return out_l, out_r
