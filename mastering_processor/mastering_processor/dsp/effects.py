"""Modulated delay-line effects: vibrato and flanger.

Vibrato is fully vectorized (LFO + read indices + writes all computed
up-front). Flanger has a feedback loop so it's inherently sequential, but
the loop body is tight with all state bound to locals.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["VibratoEffect", "FlangerEffect"]

_DENORMAL_OFFSET: float = 1e-20


class VibratoEffect:
    """Pure pitch-modulation vibrato (no dry blend).

    Separate delay buffers per channel so L and R don't cross-contaminate.
    LFO phase is shared so both channels modulate in sync (preserves image).
    """

    def __init__(
        self,
        fs: float,
        frequency_hz: float = 5.0,
        depth_ms: float = 5.0,
    ) -> None:
        self.fs = float(fs)
        self.frequency_hz = float(frequency_hz)
        self.depth_ms = float(depth_ms)
        max_samples = int(math.ceil(0.05 * self.fs))
        self._buf_len = max_samples + 4
        self._buffer_l = np.zeros(self._buf_len, dtype=np.float64)
        self._buffer_r = np.zeros(self._buf_len, dtype=np.float64)
        self._wp = 0
        self._phase = 0.0

    def set_frequency(self, f: float) -> None:
        self.frequency_hz = float(max(0.01, min(f, 20.0)))

    def set_depth(self, depth_ms: float) -> None:
        self.depth_ms = float(max(0.0, min(depth_ms, 50.0)))

    def reset(self) -> None:
        self._buffer_l.fill(0.0)
        self._buffer_r.fill(0.0)
        self._wp = 0
        self._phase = 0.0

    def process_stereo(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Process both channels with shared LFO (preserves image)."""
        if left.size == 0:
            return left, right
        left = np.ascontiguousarray(left, dtype=np.float64)
        right = np.ascontiguousarray(right, dtype=np.float64)
        n = left.size
        buf_len = self._buf_len

        phases = self._phase + np.arange(n, dtype=np.float64) * (self.frequency_hz / self.fs)
        phases -= np.floor(phases)
        lfo = 0.5 - 0.5 * np.cos(2.0 * np.pi * phases)
        depth_samples = (self.depth_ms * 1e-3) * self.fs * lfo

        wp_arr = (self._wp + np.arange(n)) % buf_len
        self._buffer_l[wp_arr] = left
        self._buffer_r[wp_arr] = right

        read_pos = (wp_arr - depth_samples) % buf_len
        prev_idx = np.floor(read_pos).astype(np.int64) % buf_len
        next_idx = (prev_idx + 1) % buf_len
        frac = read_pos - np.floor(read_pos)

        out_l = (1.0 - frac) * self._buffer_l[prev_idx] + frac * self._buffer_l[next_idx]
        out_r = (1.0 - frac) * self._buffer_r[prev_idx] + frac * self._buffer_r[next_idx]

        self._wp = int((self._wp + n) % buf_len)
        self._phase = float(phases[-1])
        self._phase -= math.floor(self._phase)
        return out_l, out_r

    def process(self, x: np.ndarray) -> np.ndarray:
        """Mono process — uses the left buffer."""
        if x.size == 0:
            return x
        x = np.ascontiguousarray(x, dtype=np.float64)
        n = x.size
        buf_len = self._buf_len

        phases = self._phase + np.arange(n, dtype=np.float64) * (self.frequency_hz / self.fs)
        phases -= np.floor(phases)
        lfo = 0.5 - 0.5 * np.cos(2.0 * np.pi * phases)
        depth_samples = (self.depth_ms * 1e-3) * self.fs * lfo

        wp_arr = (self._wp + np.arange(n)) % buf_len
        self._buffer_l[wp_arr] = x

        read_pos = (wp_arr - depth_samples) % buf_len
        prev_idx = np.floor(read_pos).astype(np.int64) % buf_len
        next_idx = (prev_idx + 1) % buf_len
        frac = read_pos - np.floor(read_pos)
        out = (1.0 - frac) * self._buffer_l[prev_idx] + frac * self._buffer_l[next_idx]

        self._wp = int((self._wp + n) % buf_len)
        self._phase = float(phases[-1])
        self._phase -= math.floor(self._phase)
        return out


class FlangerEffect:
    """Flanger with feedback and dry/wet blend.

    Separate delay buffers per channel, shared LFO.
    """

    def __init__(
        self,
        fs: float,
        frequency_hz: float = 0.5,
        depth_ms: float = 5.0,
        depth: float = 0.7,
        feedback: float = 0.4,
    ) -> None:
        self.fs = float(fs)
        self.frequency_hz = float(frequency_hz)
        self.depth_ms = float(depth_ms)
        self.depth = float(depth)
        self.feedback = float(feedback)
        max_samples = int(math.ceil(0.05 * self.fs))
        self._buf_len = max_samples + 4
        self._buffer_l = np.zeros(self._buf_len, dtype=np.float64)
        self._buffer_r = np.zeros(self._buf_len, dtype=np.float64)
        self._wp = 0
        self._phase = 0.0

    def set_frequency(self, f: float) -> None:
        self.frequency_hz = float(max(0.01, min(f, 10.0)))

    def set_depth_ms(self, depth_ms: float) -> None:
        self.depth_ms = float(max(0.0, min(depth_ms, 50.0)))

    def set_depth(self, depth: float) -> None:
        self.depth = float(max(0.0, min(depth, 1.0)))

    def set_feedback(self, fb: float) -> None:
        self.feedback = float(max(0.0, min(fb, 0.95)))

    def reset(self) -> None:
        self._buffer_l.fill(0.0)
        self._buffer_r.fill(0.0)
        self._wp = 0
        self._phase = 0.0

    def process_stereo(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if left.size == 0:
            return left, right
        left = np.ascontiguousarray(left, dtype=np.float64)
        right = np.ascontiguousarray(right, dtype=np.float64)
        n = left.size
        buf_len = self._buf_len
        buf_l = self._buffer_l
        buf_r = self._buffer_r
        fb = self.feedback
        depth = self.depth
        freq = self.frequency_hz
        fs = self.fs
        out_l = np.empty(n, dtype=np.float64)
        out_r = np.empty(n, dtype=np.float64)
        wp = self._wp
        phase = self._phase
        phase_inc = freq / fs
        depth_samples_max = (self.depth_ms * 1e-3) * fs

        two_pi = 2.0 * math.pi
        for i in range(n):
            lfo = 0.5 - 0.5 * math.cos(two_pi * phase)
            delay = depth_samples_max * lfo
            read_pos = (wp - delay) % buf_len
            prev_idx = int(read_pos)
            next_idx = (prev_idx + 1) % buf_len
            frac = read_pos - prev_idx
            delayed_l = (1.0 - frac) * buf_l[prev_idx] + frac * buf_l[next_idx]
            delayed_r = (1.0 - frac) * buf_r[prev_idx] + frac * buf_r[next_idx]
            buf_l[wp] = left[i] + fb * delayed_l
            buf_r[wp] = right[i] + fb * delayed_r
            out_l[i] = left[i] + depth * delayed_l
            out_r[i] = right[i] + depth * delayed_r
            wp = (wp + 1) % buf_len
            phase += phase_inc
            if phase >= 1.0:
                phase -= 1.0

        self._wp = wp
        self._phase = phase
        return out_l, out_r

    def process(self, x: np.ndarray) -> np.ndarray:
        """Mono process — uses the left buffer."""
        if x.size == 0:
            return x
        x = np.ascontiguousarray(x, dtype=np.float64)
        n = x.size
        buf_len = self._buf_len
        buf = self._buffer_l
        fb = self.feedback
        depth = self.depth
        freq = self.frequency_hz
        fs = self.fs
        out = np.empty(n, dtype=np.float64)
        wp = self._wp
        phase = self._phase
        phase_inc = freq / fs
        depth_samples_max = (self.depth_ms * 1e-3) * fs

        two_pi = 2.0 * math.pi
        for i in range(n):
            lfo = 0.5 - 0.5 * math.cos(two_pi * phase)
            delay = depth_samples_max * lfo
            read_pos = (wp - delay) % buf_len
            prev_idx = int(read_pos)
            next_idx = (prev_idx + 1) % buf_len
            frac = read_pos - prev_idx
            delayed = (1.0 - frac) * buf[prev_idx] + frac * buf[next_idx]
            buf[wp] = x[i] + fb * delayed
            out[i] = x[i] + depth * delayed
            wp = (wp + 1) % buf_len
            phase += phase_inc
            if phase >= 1.0:
                phase -= 1.0

        self._wp = wp
        self._phase = phase
        return out
