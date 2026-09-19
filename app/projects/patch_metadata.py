"""Shared data contract for typed image patches.

Patch records (``ComicTranslate.image_patches``) are plain dictionaries that
flow through undo commands, project persistence, export, and webtoon
compositing. Beyond the original ``bbox``/``png_path``/``hash`` fields a
patch record can carry:

* ``kind`` — which subsystem produced the patch. ``"inpaint"`` for cleanup
  patches (legacy patches without a kind behave exactly like inpaint) and
  ``"flux2_edit"`` for accepted FLUX.2 Klein artistic edits.
* ``metadata`` — a small msgpack-safe dictionary describing the edit
  (prompt, model key, seed, checksums, ...). Only primitive values are
  persisted; unknown keys survive so future producers stay compatible.

This module deliberately has no Qt or Diffusers dependency so both the UI
command layer and the project persistence layers can share one contract.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

PATCH_KIND_INPAINT = "inpaint"
PATCH_KIND_FLUX2_EDIT = "flux2_edit"

#: Maximum persisted length of a single metadata string value.
MAX_METADATA_STRING_LENGTH = 2048
#: Maximum number of metadata keys kept per patch record.
MAX_METADATA_KEYS = 32
#: Maximum number of entries kept for a list-valued metadata entry.
MAX_METADATA_LIST_LENGTH = 32
#: Maximum length of a persisted metadata key.
MAX_METADATA_KEY_LENGTH = 64


def normalize_patch_kind(value) -> str:
    """Return a usable patch kind for ``value``.

    Missing, empty, or non-string kinds collapse to the inpaint default so
    every legacy patch record keeps behaving like an inpaint patch.
    """
    if isinstance(value, str) and value:
        return value
    return PATCH_KIND_INPAINT


def patch_kind(patch) -> str:
    """Return the effective kind of a patch record."""
    if not isinstance(patch, Mapping):
        return PATCH_KIND_INPAINT
    return normalize_patch_kind(patch.get("kind"))


def patch_matches_kind(patch, kind) -> bool:
    """True when ``patch`` belongs to ``kind``.

    Legacy patches without a ``kind`` field match inpaint. Patches carrying
    an unknown kind match nothing, so kind-scoped reverts leave them alone.
    """
    return patch_kind(patch) == normalize_patch_kind(kind)


def _sanitize_scalar(value):
    """Return ``value`` if it is a persistable primitive, else ``None``."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if abs(value) < 2 ** 63 else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:MAX_METADATA_STRING_LENGTH]
    return None


def sanitize_patch_metadata(metadata) -> dict:
    """Coerce an arbitrary mapping into a small msgpack-safe dictionary.

    Only primitive values survive: ``None``, ``bool``, finite ``int`` /
    ``float``, strings (truncated to :data:`MAX_METADATA_STRING_LENGTH`),
    and flat lists of those. Unknown keys are preserved for forward
    compatibility; nested mappings, binary payloads, NaN/Infinity floats,
    and out-of-range ints are dropped.
    """
    if not isinstance(metadata, Mapping):
        return {}

    sanitized: dict = {}
    for key, value in metadata.items():
        if len(sanitized) >= MAX_METADATA_KEYS:
            break
        if not isinstance(key, str) or not key:
            continue
        safe_key = key[:MAX_METADATA_KEY_LENGTH]

        if value is None or isinstance(value, (bool, int, float, str)):
            scalar = _sanitize_scalar(value)
            if scalar is None and value is not None:
                continue
            sanitized[safe_key] = scalar
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            items = []
            for entry in value[:MAX_METADATA_LIST_LENGTH]:
                if isinstance(entry, (bool, int, float, str)):
                    scalar = _sanitize_scalar(entry)
                    if scalar is not None:
                        items.append(scalar)
            sanitized[safe_key] = items

    return sanitized
