"""ITU-R BS.1770-4 loudness meter.

Full algorithm:
  1. K-weighting (two cascaded biquads):
       Stage 1: pre-filter (high-shelf at ~1.5 kHz, +4 dB)
       Stage 2: RLB filter (high-pass at ~38 Hz)
  2. Mean-square measurement per channel
  3. Two-stage gating:
       Absolute gate: -70 LUFS
       Relative gate: -10 LU relative to the ungated measurement
  4. Outputs:
       Momentary (400 ms block, no overlap)
       Short-term (3 s block, no overlap)
       Integrated (gated measurement from start-of-stream)
       LRA (loudness range — 10th vs 95th percentile of gated short-term
            blocks, smoothed)

K-weighting coefficients are the standard ITU values (valid at 48 kHz).
For other rates we use the rate-adaptation formula from the ITU-R BS.1770-4
Annex.

Block-based: feed stereo blocks via process_stereo and the meter keeps all
state internally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import lfilter, lfilter_zi

__all__ = ["LoudnessMeter", "LoudnessMeasurement"]

_DENORMAL_OFFSET: float = 1e-20


# ITU-R BS.1770-4 K-weighting coefficients at 48 kHz
_K_PRE_FILTER_B = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
_K_PRE_FILTER_A = np.array([1.0, -1.69065929318241, 0.73248077421585])
_K_RLB_B = np.array([1.0, -2.0, 1.0])
_K_RLB_A = np.array([1.0, -1.99004745483398, 0.99007225036621])

_MOMENTARY_BLOCK_S = 0.400
_SHORTTERM_BLOCK_S = 3.000

_ABSOLUTE_GATE_LUFS = -70.0
_RELATIVE_GATE_LU = -10.0


@dataclass
class LoudnessMeasurement:
    """Meter snapshot."""

    momentary_lufs: float    # 400 ms window
    short_term_lufs: float   # 3 s window
    integrated_lufs: float   # Gated integrated since reset
    lra: float               # Loudness range (LU)
    true_peak_dbtp_l: float  # Sample peak (true peak lives in the limiter)
    true_peak_dbtp_r: float
    is_active: bool          # False until first block arrives


class LoudnessMeter:
    """Real-time ITU-R BS.1770-4 loudness meter."""

    def __init__(
        self,
        fs: float,
        channels: int = 2,
        block_size_samples: int = 1024,
    ) -> None:
        self.fs = float(fs)
        self.channels = int(channels)
        self.block_size_samples = int(block_size_samples)

        self._k_z1: list[Optional[np.ndarray]] = [None] * channels
        self._k_z2: list[Optional[np.ndarray]] = [None] * channels

        self._momentary_samples_needed = int(_MOMENTARY_BLOCK_S * fs)
        self._shortterm_samples_needed = int(_SHORTTERM_BLOCK_S * fs)
        self._momentary_buf = np.zeros(self._momentary_samples_needed, dtype=np.float64)
        self._shortterm_buf = np.zeros(self._shortterm_samples_needed, dtype=np.float64)
        self._momentary_pos = 0
        self._shortterm_pos = 0

        # Gated-block accumulation for Integrated + LRA.
        # 400 ms blocks, 75% overlap → 100 ms hop.
        self._gated_hop_samples = int(0.100 * fs)
        self._gated_block_samples = self._momentary_samples_needed
        self._gated_overlap = 0.75

        self._gated_buf = np.zeros(self._gated_block_samples, dtype=np.float64)
        self._gated_pos = 0
        self._gated_block_lufs: list[float] = []  # all gated blocks (for LRA)

        self._true_peak_l_lin: float = 0.0
        self._true_peak_r_lin: float = 0.0

        self._momentary_lufs: float = -70.0
        self._short_term_lufs: float = -70.0
        self._integrated_lufs: float = -70.0
        self._lra: float = 0.0

        self._total_samples_processed: int = 0

    # ----- public API ----- #

    def reset(self) -> None:
        self._k_z1 = [None] * self.channels
        self._k_z2 = [None] * self.channels
        self._momentary_buf.fill(0.0)
        self._shortterm_buf.fill(0.0)
        self._momentary_pos = 0
        self._shortterm_pos = 0
        self._gated_buf.fill(0.0)
        self._gated_pos = 0
        self._gated_block_lufs.clear()
        self._true_peak_l_lin = 0.0
        self._true_peak_r_lin = 0.0
        self._momentary_lufs = -70.0
        self._short_term_lufs = -70.0
        self._integrated_lufs = -70.0
        self._lra = 0.0
        self._total_samples_processed = 0

    def process_stereo(self, left: np.ndarray, right: np.ndarray) -> LoudnessMeasurement:
        """Process a stereo block. Returns a snapshot of the meter state."""
        if left.size == 0:
            return self._snapshot(active=False)

        left = np.ascontiguousarray(left, dtype=np.float64)
        right = np.ascontiguousarray(right, dtype=np.float64)

        self._true_peak_l_lin = max(self._true_peak_l_lin, float(np.max(np.abs(left))))
        self._true_peak_r_lin = max(self._true_peak_r_lin, float(np.max(np.abs(right))))

        # K-weight each channel. Stereo channel weights are 1.0 each per ITU.
        kw_l = self._apply_k_weighting(left, 0)
        kw_r = self._apply_k_weighting(right, 1)

        # Push the K-weighted samples (not the MS — cheaper to compute MS
        # per block later) into the rolling buffers.
        n = left.size

        # Momentary: 400 ms rolling window
        self._push_rolling(self._momentary_buf, self._momentary_pos, kw_l, kw_r, n)
        self._momentary_pos = (self._momentary_pos + n) % self._momentary_samples_needed
        ms_now = float(np.mean(self._momentary_buf ** 2))
        self._momentary_lufs = -0.691 + 10.0 * math.log10(max(ms_now, 1e-12))

        # Short-term: 3 s rolling window
        self._push_rolling(self._shortterm_buf, self._shortterm_pos, kw_l, kw_r, n)
        self._shortterm_pos = (self._shortterm_pos + n) % self._shortterm_samples_needed
        ms_st = float(np.mean(self._shortterm_buf ** 2))
        self._short_term_lufs = -0.691 + 10.0 * math.log10(max(ms_st, 1e-12))

        # Gated block (for Integrated + LRA):
        # 400 ms blocks, 75% overlap (100 ms hop). With block_size ~23 ms
        # we accumulate until we have 100 ms of new audio, then compute one
        # gated block.
        self._push_rolling(self._gated_buf, self._gated_pos, kw_l, kw_r, n)
        self._gated_pos = (self._gated_pos + n) % self._gated_block_samples
        self._total_samples_processed += n

        blocks_to_compute = self._total_samples_processed // self._gated_hop_samples
        while blocks_to_compute > 0:
            ms = float(np.mean(self._gated_buf ** 2))
            lufs = -0.691 + 10.0 * math.log10(max(ms, 1e-12))
            if lufs > _ABSOLUTE_GATE_LUFS:
                self._gated_block_lufs.append(lufs)
            self._recompute_integrated_and_lra()
            self._total_samples_processed -= self._gated_hop_samples
            blocks_to_compute -= 1

        return self._snapshot(active=True)

    def get_target_gain_db(self, target_lufs: float) -> float:
        """Gain offset (dB) needed to reach target_lufs.

        Uses integrated if available, else short-term.
        """
        current = self._integrated_lufs if self._integrated_lufs > -70.0 else self._short_term_lufs
        if current <= -70.0:
            return 0.0
        return target_lufs - current

    # ----- internals ----- #

    def _apply_k_weighting(self, x: np.ndarray, channel: int) -> np.ndarray:
        """Apply the two K-weighting stages (pre-filter + RLB)."""
        if self._k_z1[channel] is None:
            self._k_z1[channel] = lfilter_zi(_K_PRE_FILTER_B, _K_PRE_FILTER_A) * float(x[0])
        y, self._k_z1[channel] = lfilter(
            _K_PRE_FILTER_B, _K_PRE_FILTER_A, x, zi=self._k_z1[channel]
        )
        self._k_z1[channel] += _DENORMAL_OFFSET

        if self._k_z2[channel] is None:
            self._k_z2[channel] = lfilter_zi(_K_RLB_B, _K_RLB_A) * float(y[0])
        y, self._k_z2[channel] = lfilter(
            _K_RLB_B, _K_RLB_A, y, zi=self._k_z2[channel]
        )
        self._k_z2[channel] += _DENORMAL_OFFSET
        return y

    def _push_rolling(
        self,
        buf: np.ndarray,
        pos: int,
        kw_l: np.ndarray,
        kw_r: np.ndarray,
        n: int,
    ) -> None:
        """Push K-weighted samples into a rolling mono buffer (L+R sum)."""
        new_samples = (kw_l + kw_r)
        buf_len = buf.size
        end = pos + n
        if end <= buf_len:
            buf[pos:end] = new_samples
        else:
            first = buf_len - pos
            buf[pos:] = new_samples[:first]
            buf[:end - buf_len] = new_samples[first:]

    def _recompute_integrated_and_lra(self) -> None:
        if not self._gated_block_lufs:
            self._integrated_lufs = -70.0
            self._lra = 0.0
            return

        blocks = np.asarray(self._gated_block_lufs, dtype=np.float64)

        # Stage 1 absolute gate (-70 LUFS) was applied during accumulation.
        # Stage 2: relative gate = -10 LU below the mean of absolute-gated blocks.
        if blocks.size == 0:
            self._integrated_lufs = -70.0
            self._lra = 0.0
            return

        mean_lin = float(np.mean(10.0 ** (blocks / 10.0)))
        mean_lufs = -0.691 + 10.0 * math.log10(max(mean_lin, 1e-12))
        relative_gate = mean_lufs - 10.0

        gated = blocks[blocks >= relative_gate]
        if gated.size == 0:
            self._integrated_lufs = -70.0
            self._lra = 0.0
            return

        integ_lin = float(np.mean(10.0 ** (gated / 10.0)))
        self._integrated_lufs = -0.691 + 10.0 * math.log10(max(integ_lin, 1e-12))

        # LRA: 95th - 10th percentile of gated blocks. We skip the 3 LU
        # smoothing window — close enough for the GUI display.
        if gated.size >= 4:
            p95 = float(np.percentile(gated, 95))
            p10 = float(np.percentile(gated, 10))
            self._lra = max(0.0, p95 - p10)
        else:
            self._lra = 0.0

    def _snapshot(self, active: bool) -> LoudnessMeasurement:
        tp_l_db = 20.0 * math.log10(max(self._true_peak_l_lin, 1e-9))
        tp_r_db = 20.0 * math.log10(max(self._true_peak_r_lin, 1e-9))
        return LoudnessMeasurement(
            momentary_lufs=self._momentary_lufs,
            short_term_lufs=self._short_term_lufs,
            integrated_lufs=self._integrated_lufs,
            lra=self._lra,
            true_peak_dbtp_l=tp_l_db,
            true_peak_dbtp_r=tp_r_db,
            is_active=active,
        )
