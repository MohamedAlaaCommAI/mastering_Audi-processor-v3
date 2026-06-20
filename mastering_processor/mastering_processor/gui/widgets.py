"""Reusable Qt widgets.

RotaryControl — labelled dial with decimals + unit in the readout.
EQSlider — vertical dB slider at 0.1 dB resolution.
ToggleGroup — mutually-exclusive toggle buttons (Off / Vibrato / Flanger).
MeterBar — level meter with peak hold, horizontal or vertical.
"""

from __future__ import annotations

from typing import Callable, List

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (
    QDial, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QSlider, QVBoxLayout, QWidget,
)


class RotaryControl(QWidget):
    """Labelled dial with live value readout (decimals + unit)."""

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        min_val: float,
        max_val: float,
        default_val: float,
        unit: str = "",
        decimals: int = 1,
        callback: Callable[[float], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._min = float(min_val)
        self._max = float(max_val)
        self._decimals = int(decimals)
        self._unit = unit
        self._callback = callback

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.value_label = QLabel(self._format(default_val))
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: #89b4fa; background: transparent;"
        )

        self.dial = QDial(self)
        self.dial.setRange(0, 1000)
        self.dial.setValue(self._value_to_int(default_val))
        self.dial.setNotchesVisible(True)
        self.dial.setNotchTarget(40)
        self.dial.setWrapping(False)
        self.dial.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.dial.setMinimumSize(54, 54)

        self.name_label = QLabel(label, self)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet(
            "font-size: 10px; color: #a6adc8; background: transparent;"
        )

        layout.addWidget(self.value_label)
        layout.addWidget(self.dial)
        layout.addWidget(self.name_label)
        self.dial.valueChanged.connect(self._on_dial_changed)

    def _on_dial_changed(self, raw: int) -> None:
        v = self._int_to_value(raw)
        self.value_label.setText(self._format(v))
        self.valueChanged.emit(v)
        if self._callback is not None:
            self._callback(v)

    def value(self) -> float: return self._int_to_value(self.dial.value())
    def setValue(self, v: float) -> None: self.dial.setValue(self._value_to_int(v))

    def _value_to_int(self, v: float) -> int:
        v = max(self._min, min(v, self._max))
        return int(round((v - self._min) / (self._max - self._min) * 1000.0))

    def _int_to_value(self, raw: int) -> float:
        return self._min + (raw / 1000.0) * (self._max - self._min)

    def _format(self, v: float) -> str:
        return f"{v:.{self._decimals}f}{self._unit}"


class EQSlider(QWidget):
    """Vertical EQ band slider with dB readout (0.1 dB resolution)."""

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        freq_label: str,
        min_db: float = -12.0,
        max_db: float = 12.0,
        default_db: float = 0.0,
        callback: Callable[[float], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._min_db = float(min_db)
        self._max_db = float(max_db)
        self._callback = callback

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.value_label = QLabel(f"{default_db:+.1f}")
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setStyleSheet(
            "font-size: 10px; color: #89b4fa; background: transparent;"
        )

        self.slider = QSlider(Qt.Vertical, self)
        self.slider.setRange(int(round(self._min_db * 10)),
                              int(round(self._max_db * 10)))
        self.slider.setValue(int(round(default_db * 10)))
        self.slider.setSingleStep(1)
        self.slider.setPageStep(10)
        self.slider.setMinimumHeight(110)

        self.freq_label = QLabel(freq_label, self)
        self.freq_label.setAlignment(Qt.AlignCenter)
        self.freq_label.setStyleSheet(
            "font-size: 9px; color: #a6adc8; background: transparent;"
        )

        layout.addWidget(self.value_label)
        layout.addWidget(self.slider, alignment=Qt.AlignHCenter)
        layout.addWidget(self.freq_label)
        self.slider.valueChanged.connect(self._on_slider_changed)

    def _on_slider_changed(self, raw: int) -> None:
        v = raw / 10.0
        self.value_label.setText(f"{v:+.1f}")
        self.valueChanged.emit(v)
        if self._callback is not None:
            self._callback(v)

    def value(self) -> float: return self.slider.value() / 10.0
    def setValue(self, v: float) -> None: self.slider.setValue(int(round(v * 10)))


class ToggleGroup(QWidget):
    """Mutually-exclusive toggle buttons; first is always 'Off'."""
    selectionChanged = pyqtSignal(str)

    def __init__(self, items: List[tuple[str, str]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buttons: dict[str, QPushButton] = {}
        self._current = ""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        none_btn = QPushButton("Off")
        none_btn.setCheckable(True)
        none_btn.setChecked(True)
        none_btn.clicked.connect(lambda _=False: self._select(""))
        self._buttons[""] = none_btn
        layout.addWidget(none_btn)

        for key, label in items:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _=False, k=key: self._select(k))
            self._buttons[key] = btn
            layout.addWidget(btn)

    def _select(self, key: str) -> None:
        if key == self._current and key != "":
            key = ""
        self._current = key
        for k, btn in self._buttons.items():
            btn.setChecked(k == key)
        self.selectionChanged.emit(key)

    def current(self) -> str: return self._current
    def set_current(self, key: str) -> None:
        if key in self._buttons: self._select(key)


class MeterBar(QWidget):
    """Level meter with peak-hold marker.

    Range: min_db..max_db. Green (low) → yellow → red (peak).
    """

    def __init__(
        self,
        min_db: float = -60.0,
        max_db: float = 0.0,
        orientation: Qt.Orientation = Qt.Horizontal,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._min_db = float(min_db)
        self._max_db = float(max_db)
        self._orientation = orientation
        self._level_db: float = min_db
        self._peak_db: float = min_db
        self._peak_decay_rate: float = 0.5  # dB per refresh
        self.setMinimumHeight(18 if orientation == Qt.Horizontal else 100)
        self.setMinimumWidth(100 if orientation == Qt.Horizontal else 18)

    def set_level_db(self, level_db: float) -> None:
        self._level_db = max(self._min_db, min(level_db, self._max_db))
        if level_db > self._peak_db:
            self._peak_db = level_db
        else:
            self._peak_db = max(self._min_db, self._peak_db - self._peak_decay_rate)
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()

        p.fillRect(0, 0, w, h, QColor("#181825"))

        level_norm = (self._level_db - self._min_db) / (self._max_db - self._min_db)
        level_norm = max(0.0, min(level_norm, 1.0))

        if self._orientation == Qt.Horizontal:
            level_w = int(w * level_norm)
            for x in range(level_w):
                t = x / max(w - 1, 1)
                if t < 0.6:
                    c = QColor("#a6e3a1")  # green
                elif t < 0.85:
                    c = QColor("#f9e2af")  # yellow
                else:
                    c = QColor("#f38ba8")  # red
                p.setPen(QPen(c, 1))
                p.drawLine(x, 1, x, h - 2)
            peak_norm = (self._peak_db - self._min_db) / (self._max_db - self._min_db)
            peak_x = int(w * max(0.0, min(peak_norm, 1.0)))
            p.setPen(QPen(QColor("#f5e0dc"), 2))
            p.drawLine(peak_x, 0, peak_x, h)
        else:
            level_h = int(h * level_norm)
            for y in range(h - level_h, h):
                t = 1.0 - (y / max(h - 1, 1))
                if t < 0.6:
                    c = QColor("#a6e3a1")
                elif t < 0.85:
                    c = QColor("#f9e2af")
                else:
                    c = QColor("#f38ba8")
                p.setPen(QPen(c, 1))
                p.drawLine(1, y, w - 2, y)
            peak_norm = (self._peak_db - self._min_db) / (self._max_db - self._min_db)
            peak_y = h - int(h * max(0.0, min(peak_norm, 1.0)))
            p.setPen(QPen(QColor("#f5e0dc"), 2))
            p.drawLine(0, peak_y, w, peak_y)

        p.end()
