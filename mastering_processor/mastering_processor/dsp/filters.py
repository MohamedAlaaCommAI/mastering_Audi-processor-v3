"""Biquad IIR filters with denormal-safe state.

A few notes on the design choices:

1. scipy.signal.lfilter uses Direct Form II Transposed internally, which is
   more numerically stable than DF1 at low frequencies / high sample rates
   (the near-unity-pole gain gets absorbed into the zero section).

2. Every persistent filter state gets a tiny DC offset (~1e-20) added on
   write to break denormal propagation. Denormals can slow Intel CPUs by
   10-100x when the input goes silent. The offset is inaudible (~-380 dBFS).

3. Coefficients are only recomputed when parameters actually change
   (compared by value, not by reference).

4. Audio-EQ-Cookbook (RBJ) formulas cover peak, lowshelf, highshelf,
   lowpass, highpass, notch, bandpass, allpass.

5. Linkwitz-Riley crossovers are cascaded Butterworth sections
   (LR4 = 2× Butterworth-2; LR8 = 2× Butterworth-4). Low/high outputs
   sum to unity at DC and have identical phase, so multiband recombination
   is phase-safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.signal import lfilter, lfilter_zi, butter

__all__ = ["BiquadFilter", "design_biquad", "LinkwitzRileyCrossover"]

# Tiny DC offset for IIR state. Inaudible (~-380 dBFS) but kills denormals.
_DENORMAL_OFFSET: float = 1e-20

_VALID_TYPES = {
    "peak", "lowshelf", "highshelf",
    "lowpass", "highpass", "notch", "bandpass", "allpass",
}


def design_biquad(
    filter_type: str,
    fc: float,
    fs: float,
    Q: float = 0.707,
    gain_db: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (b, a) coefficients for a second-order IIR.

    Audio EQ Cookbook (RBJ) formulas.
    """
    if filter_type not in _VALID_TYPES:
        raise ValueError(f"Unsupported filter type: {filter_type!r}")
    if fs <= 0:
        raise ValueError("Sample rate must be positive")
    if Q <= 0:
        raise ValueError("Q must be positive")

    fc = float(max(1.0, min(fc, fs * 0.5 - 1.0)))
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * fc / fs
    cos_w = math.cos(w0)
    sin_w = math.sin(w0)
    alpha = sin_w / (2.0 * Q)

    if filter_type == "peak":
        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_w
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cos_w
        a2 = 1.0 - alpha / A
    elif filter_type == "lowshelf":
        sqA = math.sqrt(A)
        b0 = A * ((A + 1) - (A - 1) * cos_w + 2.0 * sqA * alpha)
        b1 = 2.0 * A * ((A - 1) - (A + 1) * cos_w)
        b2 = A * ((A + 1) - (A - 1) * cos_w - 2.0 * sqA * alpha)
        a0 = (A + 1) + (A - 1) * cos_w + 2.0 * sqA * alpha
        a1 = -2.0 * ((A - 1) + (A + 1) * cos_w)
        a2 = (A + 1) + (A - 1) * cos_w - 2.0 * sqA * alpha
    elif filter_type == "highshelf":
        sqA = math.sqrt(A)
        b0 = A * ((A + 1) + (A - 1) * cos_w + 2.0 * sqA * alpha)
        b1 = -2.0 * A * ((A - 1) + (A + 1) * cos_w)
        b2 = A * ((A + 1) + (A - 1) * cos_w - 2.0 * sqA * alpha)
        a0 = (A + 1) - (A - 1) * cos_w + 2.0 * sqA * alpha
        a1 = 2.0 * ((A - 1) - (A + 1) * cos_w)
        a2 = (A + 1) - (A - 1) * cos_w - 2.0 * sqA * alpha
    elif filter_type == "lowpass":
        b0 = (1.0 - cos_w) / 2.0
        b1 = 1.0 - cos_w
        b2 = (1.0 - cos_w) / 2.0
        a0 = 1.0 + alpha; a1 = -2.0 * cos_w; a2 = 1.0 - alpha
    elif filter_type == "highpass":
        b0 = (1.0 + cos_w) / 2.0
        b1 = -(1.0 + cos_w)
        b2 = (1.0 + cos_w) / 2.0
        a0 = 1.0 + alpha; a1 = -2.0 * cos_w; a2 = 1.0 - alpha
    elif filter_type == "notch":
        b0 = 1.0; b1 = -2.0 * cos_w; b2 = 1.0
        a0 = 1.0 + alpha; a1 = -2.0 * cos_w; a2 = 1.0 - alpha
    elif filter_type == "bandpass":
        b0 = alpha; b1 = 0.0; b2 = -alpha
        a0 = 1.0 + alpha; a1 = -2.0 * cos_w; a2 = 1.0 - alpha
    else:  # allpass
        b0 = 1.0 - alpha; b1 = -2.0 * cos_w; b2 = 1.0 + alpha
        a0 = 1.0 + alpha; a1 = -2.0 * cos_w; a2 = 1.0 - alpha

    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b, a


@dataclass
class _FilterParams:
    filter_type: str
    fc: float
    Q: float
    gain_db: float


class BiquadFilter:
    """Single second-order IIR with persistent, denormal-safe state."""

    __slots__ = ("_params", "_b", "_a", "_zi")

    def __init__(
        self,
        filter_type: str,
        fc: float,
        fs: float,
        Q: float = 0.707,
        gain_db: float = 0.0,
    ) -> None:
        self._params = _FilterParams(filter_type, float(fc), float(Q), float(gain_db))
        self._b, self._a = design_biquad(filter_type, fc, fs, Q, gain_db)
        self._zi: np.ndarray | None = None

    # ----- introspection ----- #

    @property
    def filter_type(self) -> str: return self._params.filter_type
    @property
    def fc(self) -> float: return self._params.fc
    @property
    def Q(self) -> float: return self._params.Q
    @property
    def gain_db(self) -> float: return self._params.gain_db
    @property
    def b(self) -> np.ndarray: return self._b
    @property
    def a(self) -> np.ndarray: return self._a

    # ----- parameter updates ----- #

    def update(
        self,
        fs: float,
        fc: float | None = None,
        Q: float | None = None,
        gain_db: float | None = None,
        filter_type: str | None = None,
    ) -> bool:
        """Recompute coefficients if any parameter changed. Returns True if changed."""
        new_type = filter_type if filter_type is not None else self._params.filter_type
        new_fc = float(fc) if fc is not None else self._params.fc
        new_Q = float(Q) if Q is not None else self._params.Q
        new_gain = float(gain_db) if gain_db is not None else self._params.gain_db
        if (new_type == self._params.filter_type and new_fc == self._params.fc
                and new_Q == self._params.Q and new_gain == self._params.gain_db):
            return False
        self._params = _FilterParams(new_type, new_fc, new_Q, new_gain)
        self._b, self._a = design_biquad(new_type, new_fc, fs, new_Q, new_gain)
        # Keep state — DF2T tolerates coefficient changes with minimal transient.
        return True

    def reset(self) -> None:
        self._zi = None

    # ----- processing ----- #

    def process(self, x: np.ndarray) -> np.ndarray:
        """Filter a mono block."""
        if x.size == 0:
            return x
        x = np.ascontiguousarray(x, dtype=np.float64)
        if self._zi is None:
            self._zi = lfilter_zi(self._b, self._a) * float(x[0])
        y, self._zi = lfilter(self._b, self._a, x, zi=self._zi)
        self._zi += _DENORMAL_OFFSET
        return y


# --------------------------------------------------------------------------- #
# Linkwitz-Riley crossovers
# --------------------------------------------------------------------------- #


class LinkwitzRileyCrossover:
    """LR4 or LR8 crossover with low + high outputs.

    LR4 = 2 cascaded Butterworth-2 sections (24 dB/oct).
    LR8 = 2 cascaded Butterworth-4 sections (48 dB/oct).

    Low and high outputs have identical phase (same number of Butterworth
    stages on each path), so summing them reconstructs the input with no
    phase cancellation. Standard crossover for multiband work.

    Each output keeps its own state for block-by-block processing.
    """

    def __init__(self, fs: float, fc: float, order: int = 4) -> None:
        if order not in (4, 8):
            raise ValueError("LR crossover order must be 4 or 8")
        self.fs = float(fs)
        self.fc = float(fc)
        self.order = int(order)

        butter_order = order // 2
        self._lp_b, self._lp_a = butter(butter_order, fc / (fs * 0.5), btype="low")
        self._hp_b, self._hp_a = butter(butter_order, fc / (fs * 0.5), btype="high")

        # Each LR output cascades the Butterworth section twice.
        self._lp_zi1: np.ndarray | None = None
        self._lp_zi2: np.ndarray | None = None
        self._hp_zi1: np.ndarray | None = None
        self._hp_zi2: np.ndarray | None = None

    def reset(self) -> None:
        self._lp_zi1 = self._lp_zi2 = None
        self._hp_zi1 = self._hp_zi2 = None

    def process(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (low, high) outputs."""
        if x.size == 0:
            return x, x
        x = np.ascontiguousarray(x, dtype=np.float64)

        if self._lp_zi1 is None:
            self._lp_zi1 = lfilter_zi(self._lp_b, self._lp_a) * float(x[0])
            self._lp_zi2 = lfilter_zi(self._lp_b, self._lp_a) * float(x[0])
            self._hp_zi1 = lfilter_zi(self._hp_b, self._hp_a) * float(x[0])
            self._hp_zi2 = lfilter_zi(self._hp_b, self._hp_a) * float(x[0])

        y_lp, self._lp_zi1 = lfilter(self._lp_b, self._lp_a, x, zi=self._lp_zi1)
        y_lp, self._lp_zi2 = lfilter(self._lp_b, self._lp_a, y_lp, zi=self._lp_zi2)

        y_hp, self._hp_zi1 = lfilter(self._hp_b, self._hp_a, x, zi=self._hp_zi1)
        y_hp, self._hp_zi2 = lfilter(self._hp_b, self._hp_a, y_hp, zi=self._hp_zi2)

        self._lp_zi1 += _DENORMAL_OFFSET
        self._lp_zi2 += _DENORMAL_OFFSET
        self._hp_zi1 += _DENORMAL_OFFSET
        self._hp_zi2 += _DENORMAL_OFFSET

        return y_lp, y_hp
