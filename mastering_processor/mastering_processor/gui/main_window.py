"""Main application window for the mastering processor.

Wires ProcessingChain to AudioEngine and exposes everything in a tabbed
GUI:
  - Source: file/mic loader, A/B toggle, preset save/load
  - EQ: 10 vertical sliders + preset dropdown + curve overlay
  - Dynamics: single-band compressor + multiband + limiter + de-esser
  - Stereo: M/S width controls + correlation meter + goniometer
  - Loudness: LUFS meter (M/S/I/LRA) + LU history graph
  - Effects: vibrato / flanger
  - Analysis: waterfall spectrogram + reference matching
"""

from __future__ import annotations

import json
import math
import os
import traceback
from typing import List

import numpy as np
import pyqtgraph as pg
import soundfile as sf
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QProgressBar, QPushButton, QSizePolicy, QTabWidget, QVBoxLayout, QWidget,
    QComboBox, QCheckBox, QFormLayout, QLineEdit, QDoubleSpinBox,
)

from ..core import (
    ABComparator, AdaptiveParams, AudioEngine, AudioEngineError,
    AudioSource, PresetManager, ProcessingChain,
)
from ..dsp import (
    CompressorParams, DEFAULT_EQ_BANDS, DeEsserParams, FlangerEffect,
    LimiterParams, MultibandCompressor, ParametricEqualizer,
    ReferenceMatcher, StereoImager, StereoWidthParams, TruePeakLimiter,
    VibratoEffect,
)
from ..dsp.eq import EQ_PRESETS
from .spectrum import SpectrumWithEQOverlay
from .style import STYLE_SHEET
from .widgets import EQSlider, MeterBar, RotaryControl, ToggleGroup

EQ_FREQ_LABELS = ["31", "63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"]


class MasteringGUI(QWidget):
    """Top-level mastering processor window."""

    def __init__(self, samplerate: int = 44100, blocksize: int = 1024) -> None:
        super().__init__()
        self.setWindowTitle("Mastering Processor v3.0")
        self.resize(1400, 900)
        self.setStyleSheet(STYLE_SHEET)

        self.samplerate = int(samplerate)
        self.blocksize = int(blocksize)

        self.chain = ProcessingChain(self.samplerate, self.blocksize, target_lufs=-14.0)

        # A/B comparator (gets dry vs processed per block)
        self.ab = ABComparator(self.samplerate)

        self.engine = AudioEngine(samplerate=self.samplerate,
                                  blocksize=self.blocksize, channels=2)
        self.engine.set_processor(self._process_block)

        self._build_ui()

        # Visual refresh at ~30 fps
        self._vis_timer = QTimer(self)
        self._vis_timer.setTimerType(Qt.PreciseTimer)
        self._vis_timer.timeout.connect(self._refresh_visuals)
        self._vis_timer.start(33)

        self.destroyed.connect(self._on_destroyed)

        # Start audio
        try:
            self.engine.start()
        except AudioEngineError as e:
            QMessageBox.critical(self, "Audio Error", str(e))

    # ------------------------------------------------------------------ #
    # Real-time audio callback
    # ------------------------------------------------------------------ #

    def _process_block(self, left_in: np.ndarray, right_in: np.ndarray,
                        frames: int) -> tuple[np.ndarray, np.ndarray]:
        """Audio engine calls this with stereo float32 blocks."""
        try:
            # Full chain (also runs the analyzer and stashes the latest
            # AnalysisResult on chain.last_analysis)
            proc_l, proc_r = self.chain.process(left_in, right_in, frames)

            # A/B comparison (loudness-matched bypass)
            out_l, out_r = self.ab.process(
                left_in.astype(np.float64), right_in.astype(np.float64),
                proc_l, proc_r,
            )

            # Feed the adaptive controller band energies for auto-EQ
            if self.chain.state.adaptive_enabled and self.chain.last_analysis is not None:
                self.chain.adaptive.set_auto_eq_from_analysis(
                    self.chain.last_analysis.band_db
                )

            return out_l.astype(np.float32), out_r.astype(np.float32)
        except Exception:
            traceback.print_exc()
            return left_in, right_in

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        root.addWidget(self._build_source_bar())

        middle = QHBoxLayout()
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_eq_tab(), "EQ")
        self.tabs.addTab(self._build_dynamics_tab(), "Dynamics")
        self.tabs.addTab(self._build_stereo_tab(), "Stereo")
        self.tabs.addTab(self._build_effects_tab(), "FX")
        self.tabs.addTab(self._build_loudness_tab(), "Loudness")
        self.tabs.addTab(self._build_analysis_tab(), "Analysis")
        middle.addWidget(self.tabs, stretch=1)
        middle.addWidget(self._build_visuals_panel(), stretch=2)
        root.addLayout(middle, stretch=1)

        root.addWidget(self._build_status_bar())

    # ----- source bar ----- #

    def _build_source_bar(self) -> QWidget:
        box = QGroupBox("Source")
        layout = QHBoxLayout(box)

        self.load_btn = QPushButton("Load File")
        self.load_btn.clicked.connect(self._on_load_file)
        self.mic_btn = QPushButton("Microphone")
        self.mic_btn.setCheckable(True)
        self.mic_btn.clicked.connect(self._on_toggle_mic)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._on_stop_source)

        self.file_progress = QProgressBar()
        self.file_progress.setRange(0, 1000)
        self.file_progress.setTextVisible(False)
        self.file_progress.setMinimumWidth(150)

        self.ab_btn = QPushButton("A/B: A (Processed)")
        self.ab_btn.setCheckable(True)
        self.ab_btn.clicked.connect(self._on_toggle_ab)

        self.preset_save_btn = QPushButton("Save Preset")
        self.preset_save_btn.clicked.connect(self._on_save_preset)
        self.preset_load_btn = QPushButton("Load Preset")
        self.preset_load_btn.clicked.connect(self._on_load_preset)

        layout.addWidget(self.load_btn)
        layout.addWidget(self.mic_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.file_progress, stretch=1)
        layout.addWidget(self.ab_btn)
        layout.addWidget(self.preset_save_btn)
        layout.addWidget(self.preset_load_btn)
        return box

    # ----- EQ tab ----- #

    def _build_eq_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        # Enable + preset row
        top = QHBoxLayout()
        self.eq_enable_btn = QPushButton("EQ: ON")
        self.eq_enable_btn.setCheckable(True)
        self.eq_enable_btn.setChecked(True)
        self.eq_enable_btn.clicked.connect(self._on_toggle_eq)
        top.addWidget(self.eq_enable_btn)
        top.addWidget(QLabel("Preset:"))
        self.eq_preset = QComboBox()
        self.eq_preset.addItems(list(EQ_PRESETS.keys()) + ["Custom"])
        self.eq_preset.currentTextChanged.connect(self._on_eq_preset)
        top.addWidget(self.eq_preset)
        top.addStretch()
        v.addLayout(top)

        # Slider row
        row = QHBoxLayout()
        self.eq_sliders: List[EQSlider] = []
        for i, lbl in enumerate(EQ_FREQ_LABELS):
            s = EQSlider(lbl, -12.0, 12.0, 0.0,
                          callback=lambda val, idx=i: self._on_eq_slider(idx, val))
            self.eq_sliders.append(s)
            row.addWidget(s)
        v.addLayout(row)
        v.addStretch()
        return w

    # ----- Dynamics tab ----- #

    def _build_dynamics_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        # Compressor
        comp_box = QGroupBox("Single-Band Compressor")
        cl = QVBoxLayout(comp_box)
        ctop = QHBoxLayout()
        self.comp_enable_btn = QPushButton("Compressor: ON")
        self.comp_enable_btn.setCheckable(True)
        self.comp_enable_btn.setChecked(True)
        self.comp_enable_btn.clicked.connect(self._on_toggle_compressor)
        ctop.addWidget(self.comp_enable_btn)
        ctop.addStretch()
        self.comp_gr_label = QLabel("GR: 0.0 dB")
        ctop.addWidget(self.comp_gr_label)
        cl.addLayout(ctop)
        cgrid = QGridLayout()
        self.comp_threshold = RotaryControl("Threshold", -60, 0, -18, "dB", 1,
            callback=lambda val: self.chain.compressor.set_params(threshold_db=val))
        self.comp_ratio = RotaryControl("Ratio", 1.0, 20.0, 2.5, ":1", 1,
            callback=lambda val: self.chain.compressor.set_params(ratio=val))
        self.comp_knee = RotaryControl("Knee", 0, 24, 6, "dB", 1,
            callback=lambda val: self.chain.compressor.set_params(knee_db=val))
        self.comp_attack = RotaryControl("Attack", 0.1, 200, 10, "ms", 1,
            callback=lambda val: self.chain.compressor.set_params(attack_ms=val))
        self.comp_release = RotaryControl("Release", 10, 1000, 120, "ms", 0,
            callback=lambda val: self.chain.compressor.set_params(release_ms=val))
        self.comp_makeup = RotaryControl("Makeup", 0, 24, 0, "dB", 1,
            callback=lambda val: self.chain.compressor.set_params(makeup_db=val, auto_makeup=False))
        for i, ctl in enumerate([self.comp_threshold, self.comp_ratio, self.comp_knee,
                                  self.comp_attack, self.comp_release, self.comp_makeup]):
            cgrid.addWidget(ctl, 0, i)
        cl.addLayout(cgrid)
        v.addWidget(comp_box)

        # Multiband
        mb_box = QGroupBox("Multiband Compressor (4-band LR4)")
        ml = QVBoxLayout(mb_box)
        mtop = QHBoxLayout()
        self.mb_enable_btn = QPushButton("Multiband: OFF")
        self.mb_enable_btn.setCheckable(True)
        self.mb_enable_btn.setChecked(False)
        self.mb_enable_btn.clicked.connect(self._on_toggle_multiband)
        mtop.addWidget(self.mb_enable_btn)
        mtop.addStretch()
        self.mb_gr_label = QLabel("GR: 0.0/0.0/0.0/0.0 dB")
        mtop.addWidget(self.mb_gr_label)
        ml.addLayout(mtop)
        v.addWidget(mb_box)

        # Limiter
        lim_box = QGroupBox("True-Peak Limiter")
        ll = QVBoxLayout(lim_box)
        ltop = QHBoxLayout()
        self.lim_enable_btn = QPushButton("Limiter: ON")
        self.lim_enable_btn.setCheckable(True)
        self.lim_enable_btn.setChecked(True)
        self.lim_enable_btn.clicked.connect(self._on_toggle_limiter)
        ltop.addWidget(self.lim_enable_btn)
        ltop.addStretch()
        self.lim_gr_label = QLabel("GR: 0.0 dB")
        ltop.addWidget(self.lim_gr_label)
        ll.addLayout(ltop)
        lgrid = QGridLayout()
        self.lim_ceiling = RotaryControl("Ceiling", -6, 0, -1.0, "dBTP", 1,
            callback=lambda val: self.chain.limiter.set_params(ceiling_db=val))
        self.lim_lookahead = RotaryControl("Lookahead", 1, 20, 5, "ms", 1,
            callback=lambda val: self.chain.limiter.set_params(lookahead_ms=val))
        self.lim_attack = RotaryControl("Attack", 0.1, 5, 0.5, "ms", 2,
            callback=lambda val: self.chain.limiter.set_params(attack_ms=val))
        self.lim_release = RotaryControl("Release", 10, 500, 50, "ms", 0,
            callback=lambda val: self.chain.limiter.set_params(release_ms=val))
        for i, ctl in enumerate([self.lim_ceiling, self.lim_lookahead,
                                  self.lim_attack, self.lim_release]):
            lgrid.addWidget(ctl, 0, i)
        ll.addLayout(lgrid)
        v.addWidget(lim_box)

        # De-esser
        de_box = QGroupBox("De-Esser")
        dl = QVBoxLayout(de_box)
        dtop = QHBoxLayout()
        self.de_enable_btn = QPushButton("De-Esser: OFF")
        self.de_enable_btn.setCheckable(True)
        self.de_enable_btn.setChecked(False)
        self.de_enable_btn.clicked.connect(self._on_toggle_deesser)
        dtop.addWidget(self.de_enable_btn)
        dtop.addStretch()
        self.de_gr_label = QLabel("Red: 0.0 dB")
        dtop.addWidget(self.de_gr_label)
        dl.addLayout(dtop)
        dgrid = QGridLayout()
        self.de_freq = RotaryControl("Freq", 2000, 12000, 7000, "Hz", 0,
            callback=lambda val: self.chain.deesser.set_params(frequency_hz=val))
        self.de_threshold = RotaryControl("Threshold", -50, -10, -25, "dB", 1,
            callback=lambda val: self.chain.deesser.set_params(threshold_db=val))
        self.de_reduction = RotaryControl("Reduction", -12, 0, -6, "dB", 1,
            callback=lambda val: self.chain.deesser.set_params(reduction_db=val))
        self.de_q = RotaryControl("Q", 1.0, 8.0, 3.0, "", 2,
            callback=lambda val: self.chain.deesser.set_params(q_factor=val))
        for i, ctl in enumerate([self.de_freq, self.de_threshold, self.de_reduction, self.de_q]):
            dgrid.addWidget(ctl, 0, i)
        dl.addLayout(dgrid)
        v.addWidget(de_box)

        v.addStretch()
        return w

    # ----- Stereo tab ----- #

    def _build_stereo_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        st_box = QGroupBox("Mid/Side Stereo Imager")
        sl = QVBoxLayout(st_box)
        stop = QHBoxLayout()
        self.st_enable_btn = QPushButton("Stereo Imager: ON")
        self.st_enable_btn.setCheckable(True)
        self.st_enable_btn.setChecked(True)
        self.st_enable_btn.clicked.connect(self._on_toggle_stereo)
        stop.addWidget(self.st_enable_btn)
        stop.addStretch()
        self.st_corr_label = QLabel("Correlation: +1.00")
        stop.addWidget(self.st_corr_label)
        sl.addLayout(stop)

        sgrid = QGridLayout()
        self.st_width_low = RotaryControl("Bass Width", 0, 200, 0, "%", 0,
            callback=lambda val: self.chain.stereo.set_params(width_low_pct=val))
        self.st_width_high = RotaryControl("High Width", 0, 200, 100, "%", 0,
            callback=lambda val: self.chain.stereo.set_params(width_high_pct=val))
        self.st_bass_mono = RotaryControl("Bass Mono", 50, 500, 200, "Hz", 0,
            callback=lambda val: self.chain.stereo.set_params(bass_mono_hz=val))
        for i, ctl in enumerate([self.st_width_low, self.st_width_high, self.st_bass_mono]):
            sgrid.addWidget(ctl, 0, i)
        sl.addLayout(sgrid)
        v.addWidget(st_box)

        # Goniometer
        gon_box = QGroupBox("Goniometer (Phase Scope)")
        gl = QVBoxLayout(gon_box)
        self.goniometer_plot = pg.PlotWidget()
        self.goniometer_plot.setBackground("#1e1e2e")
        self.goniometer_plot.setXRange(-1, 1)
        self.goniometer_plot.setYRange(-1, 1)
        self.goniometer_plot.setAspectLocked(1.0)
        self.goniometer_plot.showGrid(x=True, y=True, alpha=0.3)
        self.goniometer_curve = self.goniometer_plot.plot(
            pen=pg.mkPen("#89b4fa", width=1), symbol=None
        )
        gl.addWidget(self.goniometer_plot)
        v.addWidget(gon_box)

        v.addStretch()
        return w

    # ----- Effects tab ----- #

    def _build_effects_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        box = QGroupBox("Modulation Effects")
        bl = QVBoxLayout(box)
        self.effect_group = ToggleGroup(
            items=[("vibrato", "Vibrato"), ("flanger", "Flanger")]
        )
        self.effect_group.selectionChanged.connect(self._on_effect_changed)
        bl.addWidget(self.effect_group)

        fgrid = QGridLayout()
        self.vib_freq = RotaryControl("V.Rate", 0.1, 20, 5, "Hz", 2,
            callback=lambda val: self.chain.vibrato.set_frequency(val))
        self.vib_depth = RotaryControl("V.Depth", 0, 20, 5, "ms", 2,
            callback=lambda val: self.chain.vibrato.set_depth(val))
        self.fla_freq = RotaryControl("F.Rate", 0.05, 5, 0.5, "Hz", 2,
            callback=lambda val: self.chain.flanger.set_frequency(val))
        self.fla_depth = RotaryControl("F.Depth", 0, 20, 5, "ms", 2,
            callback=lambda val: self.chain.flanger.set_depth_ms(val))
        self.fla_mix = RotaryControl("F.Mix", 0, 1, 0.7, "", 2,
            callback=lambda val: self.chain.flanger.set_depth(val))
        self.fla_fb = RotaryControl("F.FB", 0, 0.95, 0.4, "", 2,
            callback=lambda val: self.chain.flanger.set_feedback(val))
        for i, ctl in enumerate([self.vib_freq, self.vib_depth, self.fla_freq,
                                  self.fla_depth, self.fla_mix, self.fla_fb]):
            fgrid.addWidget(ctl, 0, i)
        bl.addLayout(fgrid)

        self.fx_enable_btn = QPushButton("Effects: OFF")
        self.fx_enable_btn.setCheckable(True)
        self.fx_enable_btn.setChecked(False)
        self.fx_enable_btn.clicked.connect(self._on_toggle_effects)
        bl.addWidget(self.fx_enable_btn)

        v.addWidget(box)
        v.addStretch()
        return w

    # ----- Loudness tab ----- #

    def _build_loudness_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        box = QGroupBox("ITU-R BS.1770-4 LUFS Meter")
        bl = QVBoxLayout(box)

        # Big readouts
        readout_grid = QGridLayout()
        self.lufs_m_label = QLabel("M: -70.0")
        self.lufs_m_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #89b4fa;")
        self.lufs_s_label = QLabel("S: -70.0")
        self.lufs_s_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #89b4fa;")
        self.lufs_i_label = QLabel("I: -70.0")
        self.lufs_i_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #a6e3a1;")
        self.lufs_lra_label = QLabel("LRA: 0.0 LU")
        self.lufs_lra_label.setStyleSheet("font-size: 14px; color: #f9e2af;")
        self.lufs_tp_label = QLabel("TP: -inf dBTP")
        self.lufs_tp_label.setStyleSheet("font-size: 14px; color: #f38ba8;")

        readout_grid.addWidget(QLabel("Momentary (400ms):"), 0, 0)
        readout_grid.addWidget(self.lufs_m_label, 0, 1)
        readout_grid.addWidget(QLabel("Short-Term (3s):"), 1, 0)
        readout_grid.addWidget(self.lufs_s_label, 1, 1)
        readout_grid.addWidget(QLabel("Integrated:"), 2, 0)
        readout_grid.addWidget(self.lufs_i_label, 2, 1)
        readout_grid.addWidget(QLabel("LRA:"), 3, 0)
        readout_grid.addWidget(self.lufs_lra_label, 3, 1)
        readout_grid.addWidget(QLabel("True Peak:"), 4, 0)
        readout_grid.addWidget(self.lufs_tp_label, 4, 1)
        bl.addLayout(readout_grid)

        # Target LUFS control
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target LUFS:"))
        self.target_combo = QComboBox()
        self.target_combo.addItems([
            "Spotify (-14 LUFS)",
            "Apple Music (-16 LUFS)",
            "YouTube (-14 LUFS)",
            "EBU R128 (-23 LUFS)",
            "Podcast (-16 LUFS)",
            "Audiobook (-18 LUFS)",
        ])
        self.target_combo.currentIndexChanged.connect(self._on_target_lufs)
        target_row.addWidget(self.target_combo)
        target_row.addStretch()
        bl.addLayout(target_row)

        # Adaptive toggle
        self.adaptive_btn = QPushButton("Adaptive: ON")
        self.adaptive_btn.setCheckable(True)
        self.adaptive_btn.setChecked(True)
        self.adaptive_btn.clicked.connect(self._on_toggle_adaptive)
        bl.addWidget(self.adaptive_btn)

        self.agc_label = QLabel("AGC: 0.0 dB")
        self.agc_label.setStyleSheet("font-size: 14px; color: #a6e3a1;")
        bl.addWidget(self.agc_label)

        v.addWidget(box)

        # LU history graph
        hist_box = QGroupBox("LU History (60 s)")
        hl = QVBoxLayout(hist_box)
        self.lu_history_plot = pg.PlotWidget()
        self.lu_history_plot.setBackground("#1e1e2e")
        self.lu_history_plot.setYRange(-40, 0)
        self.lu_history_plot.setLabel("left", "Loudness", units="LUFS")
        self.lu_history_plot.setLabel("bottom", "Time", units="blocks")
        self.lu_history_plot.showGrid(x=True, y=True, alpha=0.3)
        self.lu_m_curve = self.lu_history_plot.plot(pen=pg.mkPen("#89b4fa", width=1))
        self.lu_s_curve = self.lu_history_plot.plot(pen=pg.mkPen("#a6e3a1", width=2))
        self.lu_i_curve = self.lu_history_plot.plot(pen=pg.mkPen("#f9e2af", width=2))
        hl.addWidget(self.lu_history_plot)
        v.addWidget(hist_box)

        v.addStretch()
        return w

    # ----- Analysis tab ----- #

    def _build_analysis_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        # Waterfall spectrogram
        spec_box = QGroupBox("Waterfall Spectrogram (Mel)")
        sl = QVBoxLayout(spec_box)
        self.spectrogram_plot = pg.PlotWidget()
        self.spectrogram_plot.setBackground("#1e1e2e")
        self.spectrogram_plot.setLabel("left", "Mel Band")
        self.spectrogram_plot.setLabel("bottom", "Time (frames)")
        # Image item for the heatmap
        self.spectrogram_image = pg.ImageItem()
        self.spectrogram_image.setColorMap(pg.colormap.get("viridis"))
        self.spectrogram_plot.addItem(self.spectrogram_image)
        sl.addWidget(self.spectrogram_plot)
        v.addWidget(spec_box)

        # Reference matching
        ref_box = QGroupBox("Reference Track Matching")
        rl = QVBoxLayout(ref_box)
        ref_row = QHBoxLayout()
        self.ref_load_btn = QPushButton("Load Reference Track")
        self.ref_load_btn.clicked.connect(self._on_load_reference)
        self.ref_match_btn = QPushButton("Apply Match")
        self.ref_match_btn.clicked.connect(self._on_apply_match)
        ref_row.addWidget(self.ref_load_btn)
        ref_row.addWidget(self.ref_match_btn)
        ref_row.addStretch()
        self.ref_label = QLabel("No reference loaded.")
        ref_row.addWidget(self.ref_label)
        rl.addLayout(ref_row)
        v.addWidget(ref_box)

        v.addStretch()
        return w

    # ----- Visuals panel (right side) ----- #

    def _build_visuals_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(4)

        # Spectrum with EQ overlay
        self.spectrum = SpectrumWithEQOverlay()
        v.addWidget(self.spectrum, stretch=2)

        # Oscilloscope (stereo)
        self.scope = pg.PlotWidget()
        self.scope.setBackground("#1e1e2e")
        self.scope.setYRange(-1, 1, padding=0.05)
        self.scope.setXRange(0, self.blocksize, padding=0.0)
        self.scope.setLabel("left", "Amplitude")
        self.scope.setLabel("bottom", "Sample")
        self.scope.showGrid(x=True, y=True, alpha=0.25)
        for axis_name in ("bottom", "left"):
            axis = self.scope.getAxis(axis_name)
            axis.setPen(color="#cdd6f4")
            axis.setTextPen(color="#cdd6f4")
        self.scope_l_curve = self.scope.plot(pen=pg.mkPen("#89b4fa", width=1))
        self.scope_r_curve = self.scope.plot(pen=pg.mkPen("#f38ba8", width=1))
        v.addWidget(self.scope, stretch=1)

        # Level meters
        meter_row = QHBoxLayout()
        meter_row.addWidget(QLabel("L:"))
        self.meter_l = MeterBar(min_db=-60, max_db=0, orientation=Qt.Horizontal)
        meter_row.addWidget(self.meter_l, stretch=1)
        meter_row.addWidget(QLabel("R:"))
        self.meter_r = MeterBar(min_db=-60, max_db=0, orientation=Qt.Horizontal)
        meter_row.addWidget(self.meter_r, stretch=1)
        v.addLayout(meter_row)

        return w

    # ----- Status bar ----- #

    def _build_status_bar(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        self.status_label = QLabel("Ready.")
        self.xrun_label = QLabel("Xruns: 0")
        layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addWidget(self.xrun_label)
        return w

    # ------------------------------------------------------------------ #
    # Event handlers
    # ------------------------------------------------------------------ #

    def _on_load_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio Files (*.wav *.flac *.aiff *.ogg *.mp3);;All Files (*)",
        )
        if not path: return
        try:
            data, sr = sf.read(path, always_2d=True, dtype="float32")
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"Could not load file:\n{e}")
            return
        if sr != self.samplerate:
            QMessageBox.warning(
                self, "Sample Rate Mismatch",
                f"File is {sr} Hz; engine runs at {self.samplerate} Hz. "
                f"The file will play at the wrong pitch. Resample externally.",
            )
        self.mic_btn.setChecked(False)
        # Reset DSP state for clean playback
        self.chain.reset_all()
        self.ab.reset()
        self.engine.load_file(data, loop=False)
        self.status_label.setText(f"Loaded: {os.path.basename(path)}")

    def _on_toggle_mic(self, checked: bool) -> None:
        if checked:
            self.chain.reset_all()
            self.ab.reset()
            self.engine.enable_microphone()
        else:
            self._on_stop_source()

    def _on_stop_source(self) -> None:
        self.mic_btn.setChecked(False)
        self.engine.stop_source()
        self.status_label.setText("Stopped.")

    def _on_toggle_ab(self, checked: bool) -> None:
        self.ab.set_state(checked)
        self.ab_btn.setText("A/B: B (Bypass)" if checked else "A/B: A (Processed)")

    def _on_toggle_eq(self, checked: bool) -> None:
        self.chain.state.eq_enabled = checked
        self.eq_enable_btn.setText("EQ: ON" if checked else "EQ: OFF")

    def _on_eq_slider(self, idx: int, gain_db: float) -> None:
        self.chain.eq.update_band(idx, gain_db=gain_db)
        # Switch preset dropdown to Custom
        if self.eq_preset.currentText() != "Custom":
            self.eq_preset.blockSignals(True)
            self.eq_preset.setCurrentText("Custom")
            self.eq_preset.blockSignals(False)

    def _on_eq_preset(self, name: str) -> None:
        if name not in EQ_PRESETS: return
        for i, g in enumerate(EQ_PRESETS[name]):
            self.eq_sliders[i].setValue(float(g))
            self.chain.eq.update_band(i, gain_db=float(g))

    def _on_toggle_compressor(self, checked: bool) -> None:
        self.chain.state.compressor_enabled = checked
        self.comp_enable_btn.setText("Compressor: ON" if checked else "Compressor: OFF")

    def _on_toggle_multiband(self, checked: bool) -> None:
        self.chain.state.multiband_enabled = checked
        self.mb_enable_btn.setText("Multiband: ON" if checked else "Multiband: OFF")

    def _on_toggle_limiter(self, checked: bool) -> None:
        self.chain.state.limiter_enabled = checked
        self.lim_enable_btn.setText("Limiter: ON" if checked else "Limiter: OFF")

    def _on_toggle_deesser(self, checked: bool) -> None:
        self.chain.state.deesser_enabled = checked
        self.de_enable_btn.setText("De-Esser: ON" if checked else "De-Esser: OFF")

    def _on_toggle_stereo(self, checked: bool) -> None:
        self.chain.state.stereo_enabled = checked
        self.st_enable_btn.setText("Stereo Imager: ON" if checked else "Stereo Imager: OFF")

    def _on_toggle_effects(self, checked: bool) -> None:
        self.chain.state.effects_enabled = checked
        self.fx_enable_btn.setText("Effects: ON" if checked else "Effects: OFF")

    def _on_effect_changed(self, key: str) -> None:
        self.chain.active_effect = key

    def _on_toggle_adaptive(self, checked: bool) -> None:
        self.chain.state.adaptive_enabled = checked
        self.chain.adaptive.params.enabled = checked
        self.adaptive_btn.setText("Adaptive: ON" if checked else "Adaptive: OFF")
        if not checked:
            self.chain.adaptive.reset()

    def _on_target_lufs(self, idx: int) -> None:
        targets = [-14.0, -16.0, -14.0, -23.0, -16.0, -18.0]
        target = targets[idx]
        self.chain.target_lufs = target
        self.chain.adaptive_params.target_loudness_db = target
        self.chain.adaptive.params.target_loudness_db = target

    def _on_load_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Track", "",
            "Audio Files (*.wav *.flac *.aiff *.ogg *.mp3);;All Files (*)",
        )
        if not path: return
        try:
            data, sr = sf.read(path, always_2d=False, dtype="float32")
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"Could not load reference:\n{e}")
            return
        if data.ndim == 2: data = data.mean(axis=1)
        self._reference_signal = data
        self._reference_sr = sr
        self.ref_label.setText(f"Reference: {os.path.basename(path)} ({len(data)/sr:.1f}s)")

    def _on_apply_match(self) -> None:
        if not hasattr(self, "_reference_signal"):
            QMessageBox.warning(self, "No Reference", "Load a reference track first.")
            return
        if self.engine._file_audio is None:
            QMessageBox.warning(self, "No Target", "Load a target audio file first.")
            return
        target = self.engine._file_audio.mean(axis=1)
        matcher = ReferenceMatcher(self.samplerate)
        band_freqs = [b.fc for b in self.chain.eq.bands]
        gains = matcher.compute_correction(target, self._reference_signal, band_freqs)
        for i, g in enumerate(gains):
            self.eq_sliders[i].setValue(float(g))
            self.chain.eq.update_band(i, gain_db=float(g))
        self.eq_preset.blockSignals(True)
        self.eq_preset.setCurrentText("Custom")
        self.eq_preset.blockSignals(False)
        self.status_label.setText(
            f"Applied reference match: gains {gains.min():+.1f}..{gains.max():+.1f} dB"
        )

    def _on_save_preset(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Preset", "preset.json", "JSON (*.json)")
        if not path: return
        try:
            PresetManager.save(
                path,
                self.chain.eq.bands,
                self.chain.compressor.params,
                self.chain.multiband.bands,
                self.chain.stereo.params,
                self.chain.deesser.params,
                self.chain.limiter.params,
                {"frequency_hz": self.chain.vibrato.frequency_hz,
                 "depth_ms": self.chain.vibrato.depth_ms},
                {"frequency_hz": self.chain.flanger.frequency_hz,
                 "depth_ms": self.chain.flanger.depth_ms,
                 "depth": self.chain.flanger.depth,
                 "feedback": self.chain.flanger.feedback},
                self.chain.active_effect,
                self.chain.state,
                self.chain.adaptive_params,
            )
            self.status_label.setText(f"Saved preset: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))

    def _on_load_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Preset", "", "JSON (*.json);;All Files (*)")
        if not path: return
        try:
            data = PresetManager.load(path)
            for i, band_data in enumerate(data.get("eq_bands", [])[:len(self.chain.eq.bands)]):
                self.chain.eq.update_band(
                    i, fc=band_data.get("fc"),
                    Q=band_data.get("Q"),
                    gain_db=band_data.get("gain_db"),
                    filter_type=band_data.get("filter_type"),
                    enabled=band_data.get("enabled", True),
                )
                if "gain_db" in band_data:
                    self.eq_sliders[i].setValue(float(band_data["gain_db"]))
            comp = data.get("compressor", {})
            if comp:
                self.chain.compressor.set_params(**comp)
            lim = data.get("limiter", {})
            if lim:
                self.chain.limiter.set_params(**lim)
            st = data.get("stereo", {})
            if st:
                self.chain.stereo.set_params(**st)
            de = data.get("deesser", {})
            if de:
                self.chain.deesser.set_params(**de)
            self.status_label.setText(f"Loaded preset: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", str(e))

    # ------------------------------------------------------------------ #
    # Visual refresh
    # ------------------------------------------------------------------ #

    def _refresh_visuals(self) -> None:
        left, right = self.engine.get_last_block()
        if left is None or left.size == 0: return

        # Oscilloscope
        self.scope_l_curve.setData(left)
        self.scope_r_curve.setData(right)

        # Spectrum (mono downmix for display)
        mono = (left + right) * 0.5
        self.spectrum.update_spectrum(mono, self.samplerate)

        # EQ overlay
        freqs, gain_db = self.chain.eq.compute_response_curve()
        band_freqs = np.array([b.fc for b in self.chain.eq.bands])
        band_gains = np.array([b.gain_db + b.adaptive_gain_db for b in self.chain.eq.bands])
        self.spectrum.update_eq_curve(freqs, gain_db, band_freqs, band_gains)

        # Level meters
        rms_l = float(np.sqrt(np.mean(left * left)) + 1e-12)
        rms_r = float(np.sqrt(np.mean(right * right)) + 1e-12)
        rms_l_db = 20.0 * math.log10(max(rms_l, 1e-9))
        rms_r_db = 20.0 * math.log10(max(rms_r, 1e-9))
        self.meter_l.set_level_db(rms_l_db)
        self.meter_r.set_level_db(rms_r_db)

        # LUFS readouts
        lm = self.chain.loudness_meter
        self.lufs_m_label.setText(f"M: {lm._momentary_lufs:+6.1f}")
        self.lufs_s_label.setText(f"S: {lm._short_term_lufs:+6.1f}")
        self.lufs_i_label.setText(f"I: {lm._integrated_lufs:+6.1f}")
        self.lufs_lra_label.setText(f"LRA: {lm._lra:.1f} LU")
        tp = max(lm._true_peak_l_lin, lm._true_peak_r_lin)
        tp_db = 20.0 * math.log10(max(tp, 1e-9))
        self.lufs_tp_label.setText(f"TP: {tp_db:+6.1f} dBTP")

        # LU history
        m_arr = self.chain.lu_history.get_momentary()
        s_arr = self.chain.lu_history.get_short_term()
        i_arr = self.chain.lu_history.get_integrated()
        if m_arr.size > 0:
            self.lu_m_curve.setData(m_arr)
        if s_arr.size > 0:
            self.lu_s_curve.setData(s_arr)
        if i_arr.size > 0:
            self.lu_i_curve.setData(i_arr)

        # GR readouts
        gr = self.chain.compressor.get_gain_reduction_db()
        self.comp_gr_label.setText(f"GR: {gr:+.1f} dB")

        gr_lim = self.chain.limiter.get_gain_reduction_db()
        self.lim_gr_label.setText(f"GR: {gr_lim:+.1f} dB")

        de_red = self.chain.deesser.get_reduction_db()
        self.de_gr_label.setText(f"Red: {de_red:+.1f} dB")

        mb_grs = self.chain.multiband.get_band_gain_reduction_db()
        if mb_grs:
            txt = "/".join(f"{g:+.1f}" for g in mb_grs)
            self.mb_gr_label.setText(f"GR: {txt} dB")

        # Stereo correlation
        corr = self.chain.correlation_meter.value()
        self.st_corr_label.setText(f"Correlation: {corr:+.2f}")

        # Goniometer
        x, y = self.chain.goniometer.get_points()
        if x.size > 0:
            self.goniometer_curve.setData(x, y)

        # Spectrogram (transpose: x = time, y = mel band)
        history = self.chain.spectrogram.get_history()
        if history.size > 0:
            self.spectrogram_image.setImage(history.T)

        # AGC
        agc = self.chain.get_agc_gain_db()
        self.agc_label.setText(f"AGC: {agc:+.2f} dB")

        # File progress
        if self.engine.is_playing_file():
            self.file_progress.setValue(int(self.engine.file_progress() * 1000))
        else:
            self.file_progress.setValue(0)

        self.xrun_label.setText(f"Xruns: {self.engine.xrun_count}")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _on_destroyed(self, *args) -> None: self._shutdown()

    def closeEvent(self, event) -> None:
        self._shutdown()
        super().closeEvent(event)

    def _shutdown(self) -> None:
        try: self._vis_timer.stop()
        except Exception: pass
        try: self.engine.stop()
        except Exception: traceback.print_exc()
