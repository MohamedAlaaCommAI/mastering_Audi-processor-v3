"""Mastering processing chain.

Owns the ordered DSP modules and runs them on each stereo block:

  Input → EQ → Multiband → Stereo Imager → Compressor → De-Esser
       → Effects → Limiter → AGC → Output

Each module can be enabled/disabled independently. The chain keeps
references to all per-module state so the GUI can read meters, GR, etc.

Stereo-native throughout. Mono collapse happens only inside modules that
need it (e.g., LUFS meter, which is mono by spec).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..dsp import (
    CompressorParams, DEFAULT_EQ_BANDS,
    DeEsser, DeEsserParams, DynamicCompressor, FlangerEffect, LimiterParams,
    LoudnessMeter, MultibandCompressor, ParametricEqualizer,
    ReferenceMatcher, StereoImager, StereoWidthParams, TruePeakLimiter,
    VibratoEffect,
)
from .adaptive import AdaptiveController, AdaptiveParams
from ..analysis import (
    CorrelationMeter, Goniometer, LUHistory, SignalAnalyzer,
    WaterfallSpectrogram,
)


@dataclass
class ChainState:
    """Per-stage enable toggles."""

    eq_enabled: bool = True
    multiband_enabled: bool = False  # Off by default (mastering choice)
    stereo_enabled: bool = True
    compressor_enabled: bool = True
    deesser_enabled: bool = False    # Off by default (only for vocals)
    effects_enabled: bool = False
    limiter_enabled: bool = True     # Always on by default for safety
    adaptive_enabled: bool = True


class ProcessingChain:
    """The full mastering DSP chain, owned by the GUI."""

    def __init__(
        self,
        fs: float,
        blocksize: int = 1024,
        target_lufs: float = -14.0,
    ) -> None:
        self.fs = float(fs)
        self.blocksize = int(blocksize)
        self.target_lufs = float(target_lufs)
        self.state = ChainState()

        self._out_left = np.zeros(self.blocksize, dtype=np.float64)
        self._out_right = np.zeros(self.blocksize, dtype=np.float64)

        # Latest AnalysisResult (read by GUI for adaptive control)
        self.last_analysis = None

        self.eq = ParametricEqualizer(self.fs)
        self.multiband = MultibandCompressor(self.fs)
        self.stereo = StereoImager(self.fs, StereoWidthParams())
        self.compressor = DynamicCompressor(
            self.fs,
            CompressorParams(
                threshold_db=-18.0, ratio=2.5, knee_db=6.0,
                attack_ms=10.0, release_ms=120.0,
                makeup_db=0.0, auto_makeup=True, stereo_link=True,
            ),
        )
        self.deesser = DeEsser(self.fs, DeEsserParams())
        self.vibrato = VibratoEffect(self.fs, frequency_hz=5.0, depth_ms=5.0)
        self.flanger = FlangerEffect(self.fs, frequency_hz=0.5, depth_ms=5.0)
        self.active_effect: str = ""  # "", "vibrato", "flanger"
        self.limiter = TruePeakLimiter(self.fs, LimiterParams())

        # Analysis modules
        self.analyzer = SignalAnalyzer(
            self.fs, band_freqs=np.array([b.fc for b in self.eq.bands])
        )
        self.loudness_meter = LoudnessMeter(
            self.fs, channels=2, block_size_samples=self.blocksize
        )
        self.correlation_meter = CorrelationMeter(self.fs, window_ms=200.0)
        self.goniometer = Goniometer(max_points=4096)
        self.lu_history = LUHistory(history_seconds=60.0)
        self.spectrogram = WaterfallSpectrogram(
            self.fs, n_mels=64, fft_size=1024, hop=512, history_rows=200
        )

        self.adaptive_params = AdaptiveParams(target_loudness_db=self.target_lufs)
        self.adaptive = AdaptiveController(
            self.fs, self.loudness_meter, self.eq, self.compressor,
            self.limiter, self.blocksize, params=self.adaptive_params,
        )

        # AGC state (LUFS-based, smoothed)
        self._agc_gain_db: float = 0.0

    # ----- main processing entry point ----- #

    def process(self, left_in: np.ndarray, right_in: np.ndarray,
                frames: int) -> tuple[np.ndarray, np.ndarray]:
        """Run the full chain. Returns (left_out, right_out) float64."""
        L = np.ascontiguousarray(left_in, dtype=np.float64)
        R = np.ascontiguousarray(right_in, dtype=np.float64)

        if self.state.eq_enabled:
            L, R = self.eq.process_stereo(L, R)

        if self.state.multiband_enabled:
            L, R = self.multiband.process_stereo(L, R)

        if self.state.stereo_enabled:
            L, R = self.stereo.process_stereo(L, R)

        if self.state.compressor_enabled:
            L, R = self.compressor.process_stereo(L, R)

        if self.state.deesser_enabled:
            L, R = self.deesser.process_stereo(L, R)

        if self.state.effects_enabled and self.active_effect:
            if self.active_effect == "vibrato":
                L, R = self.vibrato.process_stereo(L, R)
            elif self.active_effect == "flanger":
                L, R = self.flanger.process_stereo(L, R)

        # Adaptive AGC (LUFS-driven)
        if self.state.adaptive_enabled:
            self.adaptive.on_block_processed(L, R)
            agc_target = self.adaptive.get_agc_gain_db()
            self._agc_gain_db = 0.95 * self._agc_gain_db + 0.05 * agc_target
            if abs(self._agc_gain_db) > 0.001:
                gain_lin = 10.0 ** (self._agc_gain_db / 20.0)
                L *= gain_lin
                R *= gain_lin

        # Limiter always last before output
        if self.state.limiter_enabled:
            L, R = self.limiter.process_stereo(L, R)

        # Analysis (post-processing — meters reflect what user hears)
        self.last_analysis = self.analyzer.analyze_stereo(L, R)
        self.loudness_meter.process_stereo(L, R)
        self.correlation_meter.process_stereo(L, R)
        self.goniometer.process_stereo(L, R)
        self.spectrogram.push_block(L, R)
        self.lu_history.push(
            self.loudness_meter._momentary_lufs,
            self.loudness_meter._short_term_lufs,
            self.loudness_meter._integrated_lufs,
        )

        return L, R

    def reset_all(self) -> None:
        """Reset all DSP state (e.g., when loading a new file)."""
        self.eq.reset()
        self.multiband.reset()
        self.stereo.reset()
        self.compressor.reset()
        self.deesser.reset()
        self.vibrato.reset()
        self.flanger.reset()
        self.limiter.reset()
        self.analyzer.reset()
        self.loudness_meter.reset()
        self.correlation_meter.reset()
        self.goniometer.reset()
        self.spectrogram.reset()
        self.lu_history.reset()
        self.adaptive.reset()
        self._agc_gain_db = 0.0

    def get_agc_gain_db(self) -> float:
        return self._agc_gain_db
