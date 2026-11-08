"""Portable, bounded parameters for editable lettering transforms."""
from __future__ import annotations

import math
from numbers import Real


DEFAULTS = {'scale_x': 1.0, 'scale_y': 1.0, 'skew_x': 0.0,
            'perspective_x': 0.0, 'perspective_y': 0.0}
LIMITS = {'scale_x': (0.25, 4.0), 'scale_y': (0.25, 4.0),
          'skew_x': (-45.0, 45.0), 'perspective_x': (-0.6, 0.6),
          'perspective_y': (-0.6, 0.6), 'arc_angle': (-120.0, 120.0)}


def normalize_lettering_transform(value) -> dict:
    """Canonical identity is {}; invalid supplied values never silently reset."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError('Lettering transform must be an object.')
    if set(value) - LIMITS.keys():
        raise ValueError('Unknown lettering transform property.')
    result = dict(DEFAULTS)
    for key, number in value.items():
        if isinstance(number, bool) or not isinstance(number, Real):
            raise ValueError(f'Invalid lettering transform value: {key}')
        number = float(number)
        lower, upper = LIMITS[key]
        if not math.isfinite(number) or not lower <= number <= upper:
            raise ValueError(f'Lettering transform value outside limits: {key}')
        result[key] = number
    # Preserve the pre-curve canonical representation for old project snapshots
    # and layout fingerprints. Arc geometry lives in the text document layout.
    if result.get('arc_angle', 0) == 0:
        result.pop('arc_angle', None)
    return {} if result == DEFAULTS else result
