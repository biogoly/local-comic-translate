"""Native Qt projective geometry shared by text painting and interaction."""
from __future__ import annotations

import math
from PySide6.QtGui import QTransform

from modules.utils.lettering_transform import normalize_lettering_transform


def lettering_qtransform(rect, settings):
    """Map centered unit coordinates with a strictly positive denominator.

    Parameter bounds keep d=1+px*u+py*v >= 0.4 throughout the text rectangle.
    The positive stretch and one-axis skew cannot fold or mirror the quad.
    arc_angle is applied by the text document layout before this matrix.
    """
    settings = normalize_lettering_transform(settings)
    if not settings:
        return QTransform()
    w, h = float(rect.width()), float(rect.height())
    cx, cy = float(rect.center().x()), float(rect.center().y())
    if not all(math.isfinite(v) for v in (w, h, cx, cy)) or w <= 0 or h <= 0:
        raise ValueError('Lettering bounds must have finite positive dimensions.')
    sx, sy = settings['scale_x'], settings['scale_y']
    skew = math.tan(math.radians(settings['skew_x']))
    px, py = settings['perspective_x'], settings['perspective_y']
    # x' = cx + (w*sx*u + h*skew*v)/d; y' = cy + h*sy*v/d.
    dx, dy, dc = px / w, py / h, 1.0 - px * cx / w - py * cy / h
    result = QTransform(
        sx + cx * dx, cy * dx, dx,
        skew + cx * dy, sy + cy * dy, dy,
        -sx * cx - skew * cy + cx * dc,
        -sy * cy + cy * dc, dc,
    )
    if not result.isInvertible():
        raise ValueError('Lettering transform must be invertible.')
    return result
