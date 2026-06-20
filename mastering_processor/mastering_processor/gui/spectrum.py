"""Spectrum analyzer widget with EQ response-curve overlay.

Built on pyqtgraph. Uses np.fft.rfft (about 2x faster than fft for real
signals). Three curves:
  - Live spectrum (blue, filled)
  - EQ response (green, transparent — shows what the EQ is doing)
  - Peak-hold (yellow, thin)

Log-frequency x-axis, 20 Hz..20 kHz.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt


class SpectrumWithEQOverlay(pg.PlotWidget):
    """Log-frequency spectrum with EQ response-curve overlay."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setBackground("#1e1e2e")
        self.setLogMode(x=True, y=False)
        self.setXRange(np.log10(20), np.log10(20000), padding=0.0)
        self.setYRange(-90, 3, padding=0.0)
        self.setLabel("left", "Amplitude", units="dB")
        self.setLabel("bottom", "Frequency", units="Hz")
        self.showGrid(x=True, y=True, alpha=0.25)
        for axis_name in ("bottom", "left"):
            axis = self.getAxis(axis_name)
            axis.setPen(color="#cdd6f4")
            axis.setTextPen(color="#cdd6f4")

        self.curve = self.plot(pen=pg.mkPen("#89b4fa", width=2))
        self.curve_fill = self.plot(fillLevel=-90, brush=(137, 180, 250, 40))
        self.eq_curve = self.plot(pen=pg.mkPen("#a6e3a1", width=2))
        self.eq_band_dots = self.plot(symbol="o", symbolSize=8,
                                       symbolPen=pg.mkPen(None),
                                       symbolBrush=pg.mkBrush("#a6e3a1"),
                                       pen=pg.mkPen(None))
        self.peak_curve = self.plot(pen=pg.mkPen("#f9e2af", width=1))

        self._peak_db: Optional[np.ndarray] = None
        self._window_cache: dict[int, np.ndarray] = {}

    def update_spectrum(self, data: np.ndarray, fs: float) -> None:
        if data is None or data.size < 16:
            return
        data = np.ascontiguousarray(data, dtype=np.float32)
        n = data.size
        win = self._window_cache.get(n)
        if win is None:
            win = np.hanning(n).astype(np.float32)
            self._window_cache[n] = win
        spectrum = np.fft.rfft(data * win)
        mag = np.abs(spectrum) / (n / 2.0 + 1e-12)
        mag_db = 20.0 * np.log10(np.maximum(mag, 1e-9))
        freq = np.fft.rfftfreq(n, d=1.0 / fs)

        if self._peak_db is None or self._peak_db.size != mag_db.size:
            self._peak_db = mag_db.copy()
        else:
            self._peak_db = np.maximum(self._peak_db - 0.2, mag_db)

        f_plot = np.maximum(freq, 1.0)
        self.curve.setData(f_plot, mag_db)
        self.curve_fill.setData(f_plot, mag_db)
        self.peak_curve.setData(f_plot, self._peak_db)

    def update_eq_curve(self, freqs: np.ndarray, gain_db: np.ndarray,
                        band_freqs: np.ndarray, band_gains_db: np.ndarray) -> None:
        f_plot = np.maximum(freqs, 1.0)
        self.eq_curve.setData(f_plot, gain_db)
        self.eq_band_dots.setData(np.maximum(band_freqs, 1.0), band_gains_db)

    def reset_peaks(self) -> None:
        self._peak_db = None
