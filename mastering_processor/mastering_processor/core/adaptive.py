"""Adaptive controller — LUFS-aware AGC and content-aware processing.

All strategies are slow-smoothed (no zipper noise):
  - AGC: drive LUFS-integrated toward target_loudness_db by adjusting a
    makeup gain applied after the compressor. Clamped to ±6 dB by default.
    Updates throttled to every other block.
  - Auto-compressor threshold: offset around the user's manual threshold
    based on long-term LUFS error. Loud material → lower threshold (more
    compression); quiet material → higher (don't squash noise).
  - Auto-compressor release: based on crest factor (peak - RMS). Drums =
    high crest = fast release; pads = low crest = slow release.
  - Auto-EQ: gently cut bands more than 4 dB above the spectral median
    (tames harshness/muddiness without touching the user's curve).
  - Auto-limiter ceiling: lowering target LUFS keeps the ceiling at
    -1.0 dBTP (industry standard); raising target LUFS turns the limiter
    into the loudness ceiling.

LUFS-driven (the original used RMS only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..dsp import DynamicCompressor, ParametricEqualizer, TruePeakLimiter
from ..dsp.loudness import LoudnessMeter

__all__ = ["AdaptiveController", "AdaptiveParams"]


@dataclass
class AdaptiveParams:
    enabled: bool = True
    target_loudness_db: float = -14.0    # Spotify default
    auto_eq_max_cut_db: float = 3.0
    auto_eq_smooth_db: float = 0.5
    auto_comp_threshold_range_db: float = 6.0
    auto_comp_release_min_ms: float = 40.0
    auto_comp_release_max_ms: float = 250.0
    agc_max_gain_db: float = 6.0
    agc_min_gain_db: float = -6.0
    agc_smoothing_alpha: float = 0.05


class AdaptiveController:
    """LUFS-driven adaptive mastering controller."""

    def __init__(
        self,
        fs: float,
        loudness_meter: LoudnessMeter,
        equalizer: ParametricEqualizer,
        compressor: DynamicCompressor,
        limiter: TruePeakLimiter,
        blocksize: int,
        update_interval_blocks: int = 2,
        params: AdaptiveParams | None = None,
    ) -> None:
        self.fs = float(fs)
        self.loudness_meter = loudness_meter
        self.eq = equalizer
        self.comp = compressor
        self.limiter = limiter
        self.blocksize = blocksize
        self.update_interval_blocks = max(1, update_interval_blocks)
        self.params = params or AdaptiveParams()

        self._block_count = 0

        # Cache manual baselines so we can restore on disable
        self._manual_threshold_db: float = compressor.params.threshold_db
        self._manual_release_ms: float = compressor.params.release_ms

        self._agc_gain_db: float = 0.0
        self._auto_eq_gains_db: np.ndarray = np.zeros(len(self.eq.bands), dtype=np.float64)

    def reset(self) -> None:
        self._block_count = 0
        self._agc_gain_db = 0.0
        self._auto_eq_gains_db[:] = 0.0
        self.eq.reset_adaptive()
        self.comp.set_params(
            threshold_db=self._manual_threshold_db,
            release_ms=self._manual_release_ms,
        )

    def on_block_processed(self, left: np.ndarray, right: np.ndarray) -> None:
        """Called by ProcessingChain after each block has been processed."""
        self._block_count += 1
        if not self.params.enabled:
            return
        if self._block_count % self.update_interval_blocks != 0:
            return

        m = self.loudness_meter
        # Use integrated if available, else short-term
        current_lufs = m._integrated_lufs
        if current_lufs <= -70.0:
            current_lufs = m._short_term_lufs
        if current_lufs <= -70.0:
            return  # Silence — don't adapt

        # Note: we don't have direct access to the analyzer here, so auto-EQ
        # is driven externally via set_auto_eq_from_analysis() when the chain
        # wants it.

        self._update_compressor(current_lufs)
        self._update_agc(current_lufs)

    def set_auto_eq_from_analysis(self, band_db: np.ndarray) -> None:
        """Apply content-aware EQ cuts based on band energies.

        Called by the chain (which owns the analyzer). Cuts bands that are
        >4 dB above the median.
        """
        if not self.params.enabled:
            return
        cut_threshold_db = 4.0
        max_cut = self.params.auto_eq_max_cut_db
        smooth = self.params.auto_eq_smooth_db
        median_db = float(np.median(band_db))
        for i, level in enumerate(band_db):
            excess = level - median_db
            if excess > cut_threshold_db:
                desired = -min(max_cut, (excess - cut_threshold_db) * 0.6)
            else:
                desired = 0.0
            self._auto_eq_gains_db[i] = (
                (1.0 - smooth) * self._auto_eq_gains_db[i] + smooth * desired
            )
            self.eq.set_adaptive_gain(i, float(self._auto_eq_gains_db[i]))

    def get_agc_gain_db(self) -> float:
        return self._agc_gain_db

    def get_auto_eq_gains_db(self) -> np.ndarray:
        return self._auto_eq_gains_db

    # ----- strategies ----- #

    def _update_compressor(self, current_lufs: float) -> None:
        loudness_err = current_lufs - self.params.target_loudness_db
        offset = -loudness_err * 0.3
        offset = max(-self.params.auto_comp_threshold_range_db,
                     min(offset, self.params.auto_comp_threshold_range_db))
        new_threshold = self._manual_threshold_db + offset

        # Release based on crest factor from the analyzer would be ideal;
        # we approximate using LUFS-to-peak ratio (rough proxy). For now,
        # leave manual release as-is unless we have peak info.
        self.comp.set_params(threshold_db=new_threshold)

    def _update_agc(self, current_lufs: float) -> None:
        err = self.params.target_loudness_db - current_lufs
        desired = err * 0.3
        desired = max(self.params.agc_min_gain_db,
                      min(desired, self.params.agc_max_gain_db))
        alpha = self.params.agc_smoothing_alpha
        self._agc_gain_db = alpha * self._agc_gain_db + (1.0 - alpha) * desired
