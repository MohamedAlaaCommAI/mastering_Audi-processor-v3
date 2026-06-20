"""Core engine: audio I/O, processing chain, adaptive control, presets, A/B."""
from .engine import AudioEngine, AudioEngineError, AudioSource
from .chain import ProcessingChain, ChainState
from .adaptive import AdaptiveController, AdaptiveParams
from .ab import ABComparator
from .preset import PresetManager

__all__ = [
    "AudioEngine",
    "AudioEngineError",
    "AudioSource",
    "ProcessingChain",
    "ChainState",
    "AdaptiveController",
    "AdaptiveParams",
    "ABComparator",
    "PresetManager",
]
