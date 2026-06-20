"""Real-time audio engine.

Single duplex sounddevice.Stream. The callback never blocks, never raises,
and never allocates from the Python heap (all buffers pre-allocated and
reused).

Stereo-native: L and R are carried as separate float32 arrays through the
whole chain so StereoImager and MultibandCompressor can do their jobs
properly. (Earlier versions downmixed to mono at input and upmixed at
output — that destroyed the stereo image.)
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["AudioEngine", "AudioEngineError", "AudioSource"]


class AudioEngineError(RuntimeError):
    pass


class AudioSource(str, Enum):
    SILENCE = "silence"
    FILE = "file"
    MICROPHONE = "microphone"


# (left_in, right_in, frames) -> (left_out, right_out)
StereoProcessor = Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray]]


class AudioEngine:
    """Stereo duplex audio engine with pre-allocated buffers."""

    def __init__(
        self,
        samplerate: int = 44100,
        blocksize: int = 1024,
        channels: int = 2,
    ) -> None:
        self.samplerate = int(samplerate)
        self.blocksize = int(blocksize)
        self.channels = max(2, int(channels))  # always stereo-native

        self._stream = None
        self._stream_lock = threading.Lock()

        self._processor: Optional[StereoProcessor] = None
        self._source: AudioSource = AudioSource.SILENCE

        # File playback state (stereo)
        self._file_audio: Optional[np.ndarray] = None  # (N, 2) float32
        self._file_pos: int = 0
        self._file_loop: bool = False

        self._mic_enabled: bool = False

        # Pre-allocated working buffers (no per-callback allocation)
        self._in_left = np.zeros(self.blocksize, dtype=np.float32)
        self._in_right = np.zeros(self.blocksize, dtype=np.float32)
        self._out_left = np.zeros(self.blocksize, dtype=np.float32)
        self._out_right = np.zeros(self.blocksize, dtype=np.float32)
        self._silence = np.zeros(self.blocksize, dtype=np.float32)

        self._xrun_count: int = 0
        self._running: bool = False
        self._last_left = np.zeros(self.blocksize, dtype=np.float32)
        self._last_right = np.zeros(self.blocksize, dtype=np.float32)

    # ----- lifecycle ----- #

    def start(self) -> None:
        with self._stream_lock:
            if self._stream is not None:
                return
            try:
                import sounddevice as sd
            except ImportError as e:
                raise AudioEngineError(
                    "sounddevice is required. Install: pip install sounddevice"
                ) from e
            try:
                self._stream = sd.Stream(
                    samplerate=self.samplerate,
                    blocksize=self.blocksize,
                    channels=self.channels,
                    dtype="float32",
                    callback=self._callback,
                )
                self._stream.start()
                self._running = True
                logger.info(
                    "Audio stream started: sr=%d block=%d ch=%d",
                    self.samplerate, self.blocksize, self.channels,
                )
            except Exception as e:
                self._stream = None
                raise AudioEngineError(f"Failed to open audio stream: {e}") from e

    def stop(self) -> None:
        with self._stream_lock:
            if self._stream is None:
                return
            try: self._stream.stop()
            except Exception: logger.exception("Failed to stop audio stream")
            try: self._stream.close()
            except Exception: logger.exception("Failed to close audio stream")
            self._stream = None
            self._running = False

    def is_running(self) -> bool: return self._running
    @property
    def xrun_count(self) -> int: return self._xrun_count
    def get_last_block(self) -> tuple[np.ndarray, np.ndarray]:
        return self._last_left, self._last_right

    # ----- source management ----- #

    def set_processor(self, processor: StereoProcessor) -> None:
        self._processor = processor

    def load_file(self, samples: np.ndarray, loop: bool = False) -> None:
        """Load audio for playback. Accepts mono or stereo."""
        if samples.ndim == 1:
            stereo = np.stack([samples, samples], axis=1)
        elif samples.ndim == 2 and samples.shape[1] == 2:
            stereo = samples
        elif samples.ndim == 2 and samples.shape[0] == 2:
            stereo = samples.T
        else:
            # Multi-channel → take the first 2 channels
            stereo = samples[:, :2]
        self._file_audio = np.ascontiguousarray(stereo, dtype=np.float32)
        self._file_pos = 0
        self._file_loop = loop
        self._source = AudioSource.FILE
        self._mic_enabled = False

    def enable_microphone(self) -> None:
        self._mic_enabled = True
        self._source = AudioSource.MICROPHONE
        self._file_audio = None
        self._file_pos = 0

    def stop_source(self) -> None:
        self._mic_enabled = False
        self._file_audio = None
        self._file_pos = 0
        self._source = AudioSource.SILENCE

    def is_playing_file(self) -> bool:
        return self._source == AudioSource.FILE and self._file_audio is not None

    def is_microphone_enabled(self) -> bool: return self._mic_enabled

    def file_progress(self) -> float:
        if self._file_audio is None or self._file_audio.shape[0] == 0:
            return 0.0
        return self._file_pos / self._file_audio.shape[0]

    # ----- real-time callback ----- #

    def _callback(self, indata, outdata, frames, time_info, status) -> None:
        try:
            if status and (status.output_underflow or status.input_overflow):
                self._xrun_count += 1

            self._fetch_input(indata, frames)

            if self._processor is None:
                self._out_left[:] = self._in_left
                self._out_right[:] = self._in_right
            else:
                out_l, out_r = self._processor(self._in_left, self._in_right, frames)
                # Processor may return the same arrays (in-place) or new ones
                if out_l is self._in_left:
                    self._out_left[:] = out_l
                else:
                    self._out_left[:out_l.size] = out_l
                    if out_l.size < frames:
                        self._out_left[out_l.size:] = 0.0
                if out_r is self._in_right:
                    self._out_right[:] = out_r
                else:
                    self._out_right[:out_r.size] = out_r
                    if out_r.size < frames:
                        self._out_right[out_r.size:] = 0.0

            # Final safety: clip and check finite
            np.clip(self._out_left, -1.0, 1.0, out=self._out_left)
            np.clip(self._out_right, -1.0, 1.0, out=self._out_right)
            if not (np.all(np.isfinite(self._out_left)) and
                    np.all(np.isfinite(self._out_right))):
                self._out_left.fill(0.0)
                self._out_right.fill(0.0)

            outdata[:, 0] = self._out_left
            outdata[:, 1] = self._out_right

            # Copy for visualizers
            self._last_left[:] = self._out_left
            self._last_right[:] = self._out_right
        except Exception as e:
            logger.exception("Audio callback error: %s", e)
            try: outdata.fill(0)
            except Exception: pass

    def _fetch_input(self, indata: np.ndarray, frames: int) -> None:
        if self._mic_enabled:
            # indata is (frames, channels) float32
            if indata.shape[1] >= 2:
                self._in_left[:] = indata[:, 0]
                self._in_right[:] = indata[:, 1]
            else:
                self._in_left[:] = indata[:, 0]
                self._in_right[:] = indata[:, 0]
            return

        if self._file_audio is not None:
            audio = self._file_audio
            n_total = audio.shape[0]
            pos = self._file_pos
            end = pos + frames
            if end <= n_total:
                self._in_left[:] = audio[pos:end, 0]
                self._in_right[:] = audio[pos:end, 1]
                self._file_pos = end
            else:
                tail = audio[pos:]
                take = tail.shape[0]
                if self._file_loop:
                    need = frames - take
                    if need > 0:
                        head = audio[:need]
                        self._in_left[:take] = tail[:, 0]
                        self._in_left[take:] = head[:, 0]
                        self._in_right[:take] = tail[:, 1]
                        self._in_right[take:] = head[:, 1]
                        self._file_pos = need
                    else:
                        self._in_left[:] = audio[pos:pos + frames, 0]
                        self._in_right[:] = audio[pos:pos + frames, 1]
                        self._file_pos = pos + frames
                else:
                    self._in_left[:take] = tail[:, 0]
                    self._in_left[take:] = 0.0
                    self._in_right[:take] = tail[:, 1]
                    self._in_right[take:] = 0.0
                    self._file_pos = n_total
                    self._source = AudioSource.SILENCE
            return

        self._in_left.fill(0.0)
        self._in_right.fill(0.0)
