"""Portable protected-pixel masks and the final RGB write boundary.

Protection is binary and applies after an operation's soft edges or other
blending. It preserves current composed background pixels, not text layers.
Invalid stored protection fails closed rather than silently permitting edits.
"""
from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from numbers import Integral
import zlib

import numpy as np


SCHEMA_VERSION = 1
MAX_PIXELS = 100_000_000
MAX_DIMENSION = 100_000


class PixelProtectionError(ValueError):
    """Stored protection or a proposed pixel write cannot be validated."""


def _dimension(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise PixelProtectionError("Protection dimensions must be integers.")
    value = int(value)
    if not 0 < value <= MAX_DIMENSION:
        raise PixelProtectionError("Protection dimensions are outside the supported limits.")
    return value


def _shape(shape):
    if not isinstance(shape, (tuple, list)) or len(shape) not in (2, 3):
        raise PixelProtectionError("Protection requires a two-dimensional page shape.")
    height, width = _dimension(shape[0]), _dimension(shape[1])
    if height * width > MAX_PIXELS:
        raise PixelProtectionError("Protection exceeds the supported page size.")
    return height, width


def _binary_mask(mask, shape=None):
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise PixelProtectionError("Protection must be a two-dimensional pixel mask.")
    dimensions = _shape(mask.shape)
    if shape is not None and dimensions != _shape(shape):
        raise PixelProtectionError("Protection dimensions do not match this page.")
    if mask.dtype.kind not in "bui":
        raise PixelProtectionError("Protection pixels must be boolean or integer values.")
    return mask != 0


def encode_protection(mask):
    """Encode an HxW integer/bool mask; return None when nothing is protected."""
    if mask is None:
        return None
    binary = _binary_mask(mask)
    if not binary.any():
        return None
    height, width = binary.shape
    packed = np.packbits(binary, bitorder="big").tobytes()
    return {
        "schema_version": SCHEMA_VERSION,
        "width": width,
        "height": height,
        "data": base64.b64encode(zlib.compress(packed)).decode("ascii"),
    }


def decode_protection(value, shape=None):
    """Decode a portable record to uint8 0/255, validating an optional page shape.

    None and an empty mapping represent no protection. Corrupt, oversized, or
    unknown-version records raise PixelProtectionError before any image write.
    """
    if value is None or (isinstance(value, Mapping) and not value):
        return None
    if not isinstance(value, Mapping):
        raise PixelProtectionError("Protection must be a stored mask record.")
    version = value.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, Integral) or version != SCHEMA_VERSION:
        raise PixelProtectionError("Unsupported pixel protection version.")
    height, width = _shape((value.get("height"), value.get("width")))
    if shape is not None and (height, width) != _shape(shape):
        raise PixelProtectionError("Protection dimensions do not match this page.")
    expected = (height * width + 7) // 8
    encoded = value.get("data")
    # Bound both the input and decompressed output; dimensions alone must not
    # let a corrupt project request an arbitrarily large allocation.
    if not isinstance(encoded, str) or len(encoded) > 4 * (expected + 1024):
        raise PixelProtectionError("Invalid pixel protection data size.")
    try:
        compressed = base64.b64decode(encoded, validate=True)
        decoder = zlib.decompressobj()
        packed = decoder.decompress(compressed, expected + 1)
    except (ValueError, binascii.Error, zlib.error) as error:
        raise PixelProtectionError("Corrupt pixel protection data.") from error
    if len(packed) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise PixelProtectionError("Pixel protection data does not match its dimensions.")
    binary = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="big")
    if binary[height * width:].any():
        raise PixelProtectionError("Pixel protection contains invalid padding bits.")
    mask = binary[:height * width].reshape(height, width) * np.uint8(255)
    return mask if mask.any() else None


def normalize_protection(value):
    """Validate and return a canonical JSON-safe record with known fields only."""
    return encode_protection(decode_protection(value))


def get_protection(main, path, shape=None):
    """Read per-page protection without requiring an image or a Qt application."""
    states = getattr(main, "image_states", {})
    state = states.get(path, {})
    return decode_protection(state.get("pixel_protection"), shape)


def _rgb(image):
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise PixelProtectionError("Pixel protection requires an RGB uint8 image.")
    return image


def protect_image(before, after, mask):
    """Copy generated RGB pixels, restoring protected pixels exactly from before."""
    before, after = _rgb(before), _rgb(after)
    if before.shape != after.shape:
        raise PixelProtectionError("The repair changed the page dimensions.")
    result = after.copy()
    if mask is not None:
        binary = _binary_mask(mask, before.shape)
        result[binary] = before[binary]
    return result


def protect_patches(current, patches, mask):
    """Clip final opaque RGB patches against the current composed page.

    Patch positions are page-relative x/y/width/height. Pixels outside the page
    are left alone. Metadata and caller-owned image arrays are not modified.
    With no mask, return the original patches without requiring a page image.
    """
    if mask is None:
        return patches
    current = _rgb(current)
    binary = _binary_mask(mask, current.shape)
    page_height, page_width = current.shape[:2]
    result = []
    for patch in patches:
        bbox = patch.get("bbox")
        if not isinstance(bbox, (tuple, list, np.ndarray)) or len(bbox) != 4 or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) for value in bbox
        ):
            raise PixelProtectionError("Protected patches require integer pixel bounds.")
        x, y, width, height = map(int, bbox)
        patch_image = _rgb(patch.get("image"))
        if width <= 0 or height <= 0 or patch_image.shape[:2] != (height, width):
            raise PixelProtectionError("Protected patch dimensions do not match its pixels.")
        image = patch_image.copy()
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(page_width, x + width), min(page_height, y + height)
        if x1 < x2 and y1 < y2:
            protected = binary[y1:y2, x1:x2]
            target = image[y1 - y:y2 - y, x1 - x:x2 - x]
            target[protected] = current[y1:y2, x1:x2][protected]
        clipped = dict(patch)
        clipped["image"] = image
        result.append(clipped)
    return result
