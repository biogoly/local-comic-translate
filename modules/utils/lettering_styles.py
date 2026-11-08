"""Versioned project lettering presets containing portable plain data only."""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
import re
from uuid import UUID


STYLE_SCHEMA_VERSION = 1
MAX_STYLE_PRESETS = 100
MAX_STYLE_NAME = 80
STYLE_PROPERTY_KEYS = frozenset({
    'font_family', 'font_size', 'text_color', 'outline', 'outline_color',
    'outline_width', 'line_spacing', 'alignment', 'bold', 'italic', 'underline',
})
_BOOL_KEYS = {'outline', 'bold', 'italic', 'underline'}
_NUMBER_BOUNDS = {'font_size': (1, 500), 'outline_width': (0, 50), 'line_spacing': (0.1, 10)}


def empty_style_library() -> dict:
    return {'schema_version': STYLE_SCHEMA_VERSION, 'presets': []}


def _text(value, limit):
    if not isinstance(value, str):
        raise ValueError('Style text must be a string.')
    value = value.strip()
    if not value or len(value) > limit or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError('Style text is empty, too long, or contains control characters.')
    return value


def _color(value):
    if not isinstance(value, str) or not re.fullmatch(r'#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{5})?', value):
        raise ValueError('Style colors must be hexadecimal RGB or ARGB values.')
    return ('#' + ''.join(char * 2 for char in value[1:]) if len(value) == 4 else value).lower()


def validate_style_name(value) -> str:
    """Validate and trim a project or portable preset name."""
    return _text(value, MAX_STYLE_NAME)


def _property(key, value):
    if key in _BOOL_KEYS:
        if not isinstance(value, bool):
            raise ValueError(f'{key} must be boolean.')
        return value
    if key in _NUMBER_BOUNDS:
        low, high = _NUMBER_BOUNDS[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key} is outside the supported range.')
        return float(value)
    if key in {'text_color', 'outline_color'}:
        return _color(value)
    if key == 'font_family':
        return _text(value, 128)
    if key == 'alignment':
        if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2, 4, 8):
            raise ValueError('Alignment must be a supported horizontal alignment.')
        return int(value)
    raise ValueError('Unknown style property.')


def normalize_style_properties(value, *, strict=False) -> dict:
    """Return independent, partial properties; never fill unselected defaults.

    Unknown fields are omitted. The strict mode rejects invalid recognized
    values; the tolerant mode omits them for safe legacy project normalization.
    """
    if not isinstance(value, Mapping):
        if strict:
            raise ValueError('Style properties must be an object.')
        return {}
    result = {}
    for key in STYLE_PROPERTY_KEYS:
        if key not in value:
            continue
        try:
            result[key] = _property(key, value[key])
        except ValueError:
            if strict:
                raise
    return result


def _preset(value, *, strict):
    if not isinstance(value, Mapping):
        raise ValueError('Each preset must be an object.')
    identity = value.get('id')
    if not isinstance(identity, str):
        raise ValueError('Each preset must have a UUID.')
    try:
        identity = str(UUID(identity))
    except (ValueError, AttributeError):
        raise ValueError('Each preset must have a valid UUID.') from None
    properties = normalize_style_properties(value.get('properties'), strict=strict)
    if not properties:
        raise ValueError('A preset must contain at least one supported style property.')
    return {'id': identity, 'name': validate_style_name(value.get('name')), 'properties': properties}


def _library(value, *, strict):
    if value is None or value == {}:
        if strict:
            raise ValueError('A style library must declare its schema version and presets.')
        return empty_style_library()
    if not isinstance(value, Mapping):
        if strict:
            raise ValueError('A style library must be an object.')
        return empty_style_library()
    version = value.get('schema_version', STYLE_SCHEMA_VERSION if not strict else None)
    if type(version) is not int or version != STYLE_SCHEMA_VERSION:
        raise ValueError('Unsupported lettering style schema version.')
    presets = value.get('presets', [])
    if not isinstance(presets, list):
        if strict:
            raise ValueError('Style presets must be a list.')
        return empty_style_library()
    if strict and ('presets' not in value or len(presets) > MAX_STYLE_PRESETS):
        raise ValueError('Style preset list is missing or too large.')
    result, identities = empty_style_library(), set()
    for value in presets[:MAX_STYLE_PRESETS]:
        try:
            preset = _preset(value, strict=strict)
            if preset['id'] in identities:
                raise ValueError('Style preset UUIDs must be unique.')
        except ValueError:
            if strict:
                raise
            continue
        identities.add(preset['id'])
        result['presets'].append(preset)
    return result


def normalize_style_library(value) -> dict:
    """Normalize old/current project data without retaining invalid settings."""
    return _library(value, strict=False)


def validate_style_library(value) -> dict:
    """Validate imported/user-edited library data before replacing live presets."""
    return _library(value, strict=True)
