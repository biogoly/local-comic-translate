"""Pure contracts and image helpers for the optional Artistic Edit feature.

This package must remain importable without Torch, Diffusers, CUDA, or Qt.
"""

from .contracts import (
    ArtisticEditRequest,
    ArtisticEditResult,
    ContractError,
    DevicePolicy,
    FluxModelSpec,
    LoraSpec,
    ModelKey,
    SelectionMode,
)

__all__ = [
    "ArtisticEditRequest",
    "ArtisticEditResult",
    "ContractError",
    "DevicePolicy",
    "FluxModelSpec",
    "LoraSpec",
    "ModelKey",
    "SelectionMode",
]
