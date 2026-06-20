"""Mid/Side stereo processor with per-band width control and bass mono.

  M = (L + R) / 2
  S = (L - R) / 2

  L = M + S
  R = M - S

Operations:
  - Width: scale S relative to M (0 = mono, 1 = original, 2 = super-wide).
    Applied per-band via a crossover so the user can narrow lows (bass
    should be mono for vinyl/cutting and small-speaker compatibility) while
    keeping highs wide.
  - Bass mono: force S = 0 below bass_mono_hz (default 200 Hz). EBU R128
    broadcast standard for low-frequency mono.
  - Phase correlation: (M² − S²) / (M² + S²), -1 (out of phase) .. +1 (mono).

Single LR4 split → independent width for the low band (below bass_mono_hz)
and the high band (above).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .filters import LinkwitzRileyCrossover

__all__ = ["StereoImager", "StereoWidthParams"]


@dataclass
class StereoWidthParams:
    width_low_pct: float = 0.0      # Bass region (default: mono bass)
    width_high_pct: float = 100.0   # Highs region (default: original)
    bass_mono_hz: float = 200.0     # Crossover frequency
    enabled: bool = True


class StereoImager:
    """Mid/Side stereo imager with bass-mono and per-band width."""

    def __init__(self, fs: float, params: StereoWidthParams | None = None) -> None:
        self.fs = float(fs)
        self.params = params or StereoWidthParams()
        # Separate crossover instances for M and S so their state doesn't
        # bleed into each other.
        self._xover_m = LinkwitzRileyCrossover(
            fs, self.params.bass_mono_hz, order=4
        )
        self._xover_s = LinkwitzRileyCrossover(
            fs, self.params.bass_mono_hz, order=4
        )
        self._width_low_lin: float = self._pct_to_lin(self.params.width_low_pct)
        self._width_high_lin: float = self._pct_to_lin(self.params.width_high_pct)

        self._correlation: float = 1.0

    def set_params(self, **kwargs) -> None:
        rebuild = False
        for k, v in kwargs.items():
            setattr(self.params, k, v)
        if "bass_mono_hz" in kwargs:
            rebuild = True
        if rebuild:
            self._xover_m = LinkwitzRileyCrossover(
                self.fs, self.params.bass_mono_hz, order=4
            )
            self._xover_s = LinkwitzRileyCrossover(
                self.fs, self.params.bass_mono_hz, order=4
            )
        self._width_low_lin = self._pct_to_lin(self.params.width_low_pct)
        self._width_high_lin = self._pct_to_lin(self.params.width_high_pct)

    def reset(self) -> None:
        self._xover_m.reset()
        self._xover_s.reset()
        self._correlation = 1.0

    def get_correlation(self) -> float:
        return self._correlation

    def process_stereo(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process a stereo block. Returns (left_out, right_out)."""
        if not self.params.enabled or left.size == 0:
            return left, right

        left = np.ascontiguousarray(left, dtype=np.float64)
        right = np.ascontiguousarray(right, dtype=np.float64)

        m = (left + right) * 0.5
        s = (left - right) * 0.5

        m_low, m_high = self._xover_m.process(m)
        s_low, s_high = self._xover_s.process(s)

        # Per-band width = scale S
        s_low_out = s_low * self._width_low_lin
        s_high_out = s_high * self._width_high_lin

        m_out = m_low + m_high
        s_out = s_low_out + s_high_out

        out_l = m_out + s_out
        out_r = m_out - s_out

        # Correlation (slow-smoothed)
        m2 = float(np.mean(m_out * m_out))
        s2 = float(np.mean(s_out * s_out))
        ms = float(np.mean(m_out * s_out))
        denom = m2 + s2
        if denom > 1e-12:
            corr = (m2 - s2) / denom
        else:
            corr = 1.0
        self._correlation = 0.95 * self._correlation + 0.05 * corr

        return out_l, out_r

    @staticmethod
    def _pct_to_lin(pct: float) -> float:
        """Width percentage → linear S scale.

        0% → 0.0 (mono), 100% → 1.0 (original), 200% → 2.0 (super-wide).
        """
        return float(pct) / 100.0
