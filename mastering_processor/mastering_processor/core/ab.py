"""A/B comparison with loudness matching.

Keeps a copy of the bypassed (dry) signal's short-term LUFS alongside the
processed signal's. When the user toggles to B (bypass), the comparator
applies a gain offset so loudness matches the processed signal (and vice
versa) — kills the "louder is better" bias.

Switching is crossfaded over 20 ms to avoid clicks.
"""

from __future__ import annotations

import math

import numpy as np

from ..dsp.loudness import LoudnessMeter

__all__ = ["ABComparator"]


class ABComparator:
    """Gain-matched A/B switcher."""

    def __init__(self, fs: float, crossfade_ms: float = 20.0) -> None:
        self.fs = float(fs)
        self.crossfade_samples = max(1, int(crossfade_ms * 1e-3 * fs))

        # One LUFS meter per side
        self._lufs_a: LoudnessMeter = LoudnessMeter(fs, channels=2)
        self._lufs_b: LoudnessMeter = LoudnessMeter(fs, channels=2)

        self._is_b: bool = False  # False = A (processed), True = B (bypass)
        self._crossfade_pos: int = 0
        self._in_crossfade: bool = False
        self._crossfade_from_b: bool = False

        # Gain offset (dB) to apply when in B
        self._b_gain_db: float = 0.0

    def reset(self) -> None:
        self._lufs_a.reset()
        self._lufs_b.reset()
        self._is_b = False
        self._crossfade_pos = 0
        self._in_crossfade = False
        self._b_gain_db = 0.0

    def toggle(self) -> None:
        """Toggle between A (processed) and B (bypass)."""
        self._in_crossfade = True
        self._crossfade_pos = 0
        self._crossfade_from_b = self._is_b
        self._is_b = not self._is_b

    def set_state(self, is_b: bool) -> None:
        if is_b != self._is_b:
            self.toggle()

    def is_bypass(self) -> bool:
        return self._is_b

    def process(
        self,
        dry_left: np.ndarray,
        dry_right: np.ndarray,
        processed_left: np.ndarray,
        processed_right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the selected signal (A or B), crossfaded during transitions.

        Both LUFS meters are always updated so the gain offset is ready
        when the user toggles.
        """
        self._lufs_a.process_stereo(processed_left, processed_right)
        self._lufs_b.process_stereo(dry_left, dry_right)

        # Gain offset so B matches A's loudness
        a_lufs = self._lufs_a._integrated_lufs
        b_lufs = self._lufs_b._integrated_lufs
        if a_lufs > -70.0 and b_lufs > -70.0:
            self._b_gain_db = a_lufs - b_lufs  # boost B by this much

        b_gain_lin = 10.0 ** (self._b_gain_db / 20.0) if self._is_b else 1.0

        if not self._in_crossfade:
            if self._is_b:
                return dry_left * b_gain_lin, dry_right * b_gain_lin
            return processed_left, processed_right

        # Equal-power crossfade
        n = dry_left.size
        cf_len = min(self.crossfade_samples, n)
        t = np.linspace(0.0, 1.0, cf_len, dtype=np.float64)
        gain_in = np.sin(t * math.pi * 0.5)
        gain_out = np.cos(t * math.pi * 0.5)

        if self._crossfade_from_b:
            from_l = dry_left * b_gain_lin
            from_r = dry_right * b_gain_lin
            to_l = processed_left
            to_r = processed_right
        else:
            from_l = processed_left
            from_r = processed_right
            to_l = dry_left * b_gain_lin
            to_r = dry_right * b_gain_lin

        out_l = np.empty(n, dtype=np.float64)
        out_r = np.empty(n, dtype=np.float64)
        out_l[:cf_len] = from_l[:cf_len] * gain_out + to_l[:cf_len] * gain_in
        out_r[:cf_len] = from_r[:cf_len] * gain_out + to_r[:cf_len] * gain_in
        if n > cf_len:
            out_l[cf_len:] = to_l[cf_len:]
            out_r[cf_len:] = to_r[cf_len:]

        self._crossfade_pos += n
        if self._crossfade_pos >= self.crossfade_samples:
            self._in_crossfade = False

        return out_l, out_r

    def get_gain_offset_db(self) -> float:
        return self._b_gain_db
