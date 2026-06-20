"""Mel-scale waterfall spectrogram.

STFTs run on the audio thread (fft 1024, hop 512). The mel-banded result
gets pushed into a rolling buffer that the GUI renders as a heatmap.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.signal import stft

__all__ = ["WaterfallSpectrogram"]


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    """Slaney/HTK mel scale."""
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _build_mel_filterbank(
    n_fft: int, fs: float, n_mels: int, fmin: float = 30.0, fmax: float = 16000.0
) -> np.ndarray:
    """Build a (n_mels, n_fft//2+1) triangular mel filterbank."""
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    mel_min = _hz_to_mel(np.array([fmin]))[0]
    mel_max = _hz_to_mel(np.array([fmax]))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    bank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        left = hz_points[m]
        center = hz_points[m + 1]
        right = hz_points[m + 2]
        for k, f in enumerate(fft_freqs):
            if f < left or f > right:
                continue
            if f <= center:
                bank[m, k] = (f - left) / max(center - left, 1e-9)
            else:
                bank[m, k] = (right - f) / max(right - center, 1e-9)
    return bank


class WaterfallSpectrogram:
    """Rolling mel spectrogram for the GUI heatmap."""

    def __init__(
        self,
        fs: float,
        n_mels: int = 64,
        fft_size: int = 1024,
        hop: int = 512,
        history_rows: int = 200,
        fmin: float = 30.0,
        fmax: float = 16000.0,
    ) -> None:
        self.fs = float(fs)
        self.n_mels = int(n_mels)
        self.fft_size = int(fft_size)
        self.hop = int(hop)
        self.history_rows = int(history_rows)

        self._mel_bank = _build_mel_filterbank(
            self.fft_size, self.fs, self.n_mels, fmin=fmin, fmax=fmax
        )
        self._window = np.hanning(self.fft_size).astype(np.float32)

        # Mono downmix ring buffer
        self._input_buf = np.zeros(self.fft_size + self.hop, dtype=np.float32)
        self._input_pos = 0

        # (rows × mels) rolling history, newest at the end
        self._history = np.zeros((self.history_rows, self.n_mels), dtype=np.float32)
        self._history_pos = 0
        self._history_full = False

        self._min_db = -80.0
        self._max_db = -20.0

    def reset(self) -> None:
        self._input_buf.fill(0)
        self._input_pos = 0
        self._history.fill(0)
        self._history_pos = 0
        self._history_full = False

    def push_block(self, left: np.ndarray, right: np.ndarray) -> None:
        """Push a stereo block; emit one STFT frame per call."""
        if left.size == 0:
            return
        mono = (left.astype(np.float32) + right.astype(np.float32)) * 0.5
        n = mono.size
        buf = self._input_buf
        buf_len = buf.size

        end = self._input_pos + n
        if end <= buf_len:
            buf[self._input_pos:end] = mono
        else:
            first = buf_len - self._input_pos
            buf[self._input_pos:] = mono[:first]
            buf[:end - buf_len] = mono[first:]
        self._input_pos = (self._input_pos + n) % buf_len

        # Compute a STFT every `hop` samples
        while True:
            # Grab the last fft_size samples (wrapping if needed). Approximate but
            # visually fine for a scrolling heatmap.
            if self._input_pos >= self.fft_size:
                frame = buf[self._input_pos - self.fft_size:self._input_pos]
            else:
                tail = buf[:self._input_pos]
                head = buf[-(self.fft_size - self._input_pos):]
                frame = np.concatenate([head, tail])

            spec = np.fft.rfft(frame.astype(np.float64) * self._window)
            mag = np.abs(spec).astype(np.float32)
            mel = self._mel_bank @ mag
            mel_db = 10.0 * np.log10(np.maximum(mel, 1e-9)).astype(np.float32)

            self._history[self._history_pos] = mel_db
            self._history_pos = (self._history_pos + 1) % self.history_rows
            if self._history_pos == 0:
                self._history_full = True

            # One STFT per call (block_size_samples ≈ hop)
            break

    def get_history(self) -> np.ndarray:
        """Return history as (rows, mels), oldest→newest."""
        if self._history_full:
            return np.roll(self._history, -self._history_pos, axis=0)
        return self._history[:self._history_pos]

    @property
    def min_db(self) -> float: return self._min_db
    @property
    def max_db(self) -> float: return self._max_db
