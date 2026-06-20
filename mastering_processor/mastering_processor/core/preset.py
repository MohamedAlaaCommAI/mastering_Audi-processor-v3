"""Preset manager — JSON serialization of all user-tunable parameters.

Saves EQ bands, compressor params, multiband bands, stereo params,
de-esser params, limiter params, effects settings, chain state, and
adaptive params.

File format: JSON, versioned, forward-compatible.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any

from ..dsp import (
    CompressorParams, DeEsserParams, LimiterParams, StereoWidthParams,
)
from ..dsp.eq import Band, EQ_PRESETS
from ..dsp.multiband import MultibandBandParams
from .chain import ChainState
from .adaptive import AdaptiveParams

PRESET_VERSION = 1


class PresetManager:
    """Save/load all processor parameters to JSON."""

    @staticmethod
    def save(
        path: str,
        eq_bands: list,
        compressor_params: CompressorParams,
        multiband_bands: list,
        stereo_params: StereoWidthParams,
        deesser_params: DeEsserParams,
        limiter_params: LimiterParams,
        vibrato_settings: dict,
        flanger_settings: dict,
        active_effect: str,
        chain_state: ChainState,
        adaptive_params: AdaptiveParams,
    ) -> None:
        data = {
            "version": PRESET_VERSION,
            "eq_bands": [asdict(b) if is_dataclass(b) else dict(b) for b in eq_bands],
            "compressor": asdict(compressor_params),
            "multiband_bands": [asdict(b) for b in multiband_bands],
            "stereo": asdict(stereo_params),
            "deesser": asdict(deesser_params),
            "limiter": asdict(limiter_params),
            "vibrato": vibrato_settings,
            "flanger": flanger_settings,
            "active_effect": active_effect,
            "chain_state": asdict(chain_state),
            "adaptive": asdict(adaptive_params),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preset file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != PRESET_VERSION:
            # Future: handle migrations
            pass
        return data

    @staticmethod
    def list_factory_presets() -> list[str]:
        """Names of built-in EQ presets."""
        return list(EQ_PRESETS.keys())
