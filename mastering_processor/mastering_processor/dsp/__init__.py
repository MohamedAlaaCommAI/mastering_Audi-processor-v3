"""DSP blocks: filters, EQ, dynamics, limiter, loudness, stereo, effects."""

from .filters import BiquadFilter, design_biquad, LinkwitzRileyCrossover
from .eq import ParametricEqualizer, Band, DEFAULT_EQ_BANDS
from .compressor import DynamicCompressor, CompressorParams
from .multiband import MultibandCompressor, MultibandBandParams
from .limiter import TruePeakLimiter, LimiterParams
from .loudness import LoudnessMeter, LoudnessMeasurement
from .stereo import StereoImager, StereoWidthParams
from .deesser import DeEsser, DeEsserParams
from .effects import VibratoEffect, FlangerEffect
from .matching import ReferenceMatcher
from .dither import Dither, DitherType

__all__ = [
    "BiquadFilter",
    "design_biquad",
    "LinkwitzRileyCrossover",
    "ParametricEqualizer",
    "Band",
    "DEFAULT_EQ_BANDS",
    "DynamicCompressor",
    "CompressorParams",
    "MultibandCompressor",
    "MultibandBandParams",
    "TruePeakLimiter",
    "LimiterParams",
    "LoudnessMeter",
    "LoudnessMeasurement",
    "StereoImager",
    "StereoWidthParams",
    "DeEsser",
    "DeEsserParams",
    "VibratoEffect",
    "FlangerEffect",
    "ReferenceMatcher",
    "Dither",
    "DitherType",
]
