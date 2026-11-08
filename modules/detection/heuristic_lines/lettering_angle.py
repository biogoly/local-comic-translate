"""Estimate the source baseline using the glyph components already measured by Detect."""
import math

import numpy as np


def _points_in_line(points, line):
    bounds = np.asarray(line, dtype=float)
    if bounds.shape == (4,):
        x1, y1, x2, y2 = bounds
        return ((points[:, 0] >= x1) & (points[:, 0] <= x2)
                & (points[:, 1] >= y1) & (points[:, 1] <= y2))

    # Detector quads may overlap neighboring rows in their axis-aligned bounds.
    corners = bounds.reshape(4, 2)
    edges = np.roll(corners, -1, axis=0) - corners
    offsets = points[:, None, :] - corners
    cross = edges[:, 0] * offsets[:, :, 1] - edges[:, 1] * offsets[:, :, 0]
    return np.all(cross >= 0, axis=1) | np.all(cross <= 0, axis=1)


def _fit_row(points, heights, glyph_height):
    """Return the angle and supported glyphs for a substantially straight row."""
    if len(points) < 4 or np.ptp(points[:, 0]) < 2.8 * glyph_height:
        return None
    a, b = np.triu_indices(len(points), k=1)
    dx, dy = (points[b] - points[a]).T
    good_pairs = np.abs(dx) >= glyph_height
    if np.count_nonzero(good_pairs) < 3:
        return None
    slope = np.median(dy[good_pairs] / dx[good_pairs])
    intercepts = points[:, 1] - slope * points[:, 0]
    # Descenders and punctuation need not have their centers on the baseline.
    supported = np.abs(intercepts - np.median(intercepts)) <= 0.20 * glyph_height
    if np.count_nonzero(supported) < max(4, 0.75 * len(points)):
        return None
    span = np.ptp(points[supported, 0])
    if span < 2.8 * glyph_height:
        return None
    long_pairs = supported[a] & supported[b] & (np.abs(dx) >= 0.5 * span)
    slopes = dy[long_pairs] / dx[long_pairs]
    angle = float(np.degrees(np.arctan(np.median(slopes))))
    # Independent long spans should describe the same tilt. Reject curved or
    # scattered lettering instead of assigning every word one dubious angle.
    if np.percentile(np.abs(np.degrees(np.arctan(slopes)) - angle), 75) > 3:
        return None
    # A transition from capitals to lowercase or descenders can make centers
    # drift across a perfectly level word. For small slopes, also require the
    # upper and lower edges of the glyphs to support the same direction.
    if abs(angle) < 8:
        if np.count_nonzero(supported) < 6:
            return None
        dh = (heights[b] - heights[a]) / 2
        for edge_dy in (dy - dh, dy + dh):
            edge_angle = np.degrees(np.arctan(np.median(edge_dy[long_pairs] / dx[long_pairs])))
            if np.sign(angle) * edge_angle < max(1, abs(angle) - 3):
                return None
    return angle, supported


def _parallel_rows_angle(points, heights, glyph_height):
    """Recover multiple parallel baselines when OCR supplied merged/split rows."""
    if len(points) < 8:
        return None
    # Find a candidate from distant pairs of glyphs. A decisive peak over
    # competing directions is needed before trying to regroup their rows.
    a, b = np.triu_indices(len(points), k=1)
    dx, dy = (points[b] - points[a]).T
    distant = np.abs(dx) >= 2.8 * glyph_height
    directions = np.degrees(np.arctan(dy[distant] / dx[distant]))
    counts, _ = np.histogram(directions, bins=np.arange(-40.5, 41.5))
    votes = np.convolve(counts, [1, 2, 3, 2, 1], mode="same")
    peak = int(np.argmax(votes))
    candidate = peak - 40
    competing = votes[np.abs(np.arange(len(votes)) - peak) >= 7]
    if not 3 <= abs(candidate) <= 35 or votes[peak] < max(18, 1.6 * competing.max()):
        return None

    # Check actual row membership and refit each row independently. An angled
    # border, a chance alignment, or one isolated word cannot earn acceptance.
    intercepts = points[:, 1] - math.tan(math.radians(candidate)) * points[:, 0]
    order = np.argsort(intercepts)
    breaks = np.flatnonzero(np.diff(intercepts[order]) > 0.45 * glyph_height) + 1
    angles = []
    total_supported = 0
    for row in np.split(order, breaks):
        fit = _fit_row(points[row], heights[row], glyph_height)
        if fit is not None:
            angle, supported = fit
            if abs(angle - candidate) > 3:
                return None
            angles.append(angle)
            total_supported += np.count_nonzero(supported)
    if len(angles) < 2 or max(angles) - min(angles) > 3 or total_supported < 0.8 * len(points):
        return None
    return float(np.median(angles))


def estimate_lettering_angle(lines, components) -> float | None:
    """Return clockwise degrees in page coordinates, or None for weak evidence."""
    if not (4 <= len(components) <= 256 and 1 <= len(lines) <= 32):
        return None
    glyph_height = float(np.median([c["height"] for c in components]))
    glyphs = [c for c in components if 0.55 * glyph_height <= c["height"] <= 1.8 * glyph_height
              and c["width"] <= 2.2 * glyph_height]
    if len(glyphs) < max(4, 0.6 * len(components)):
        return None
    points = np.array([((c["x1"] + c["x2"]) / 2, (c["y1"] + c["y2"]) / 2) for c in glyphs])
    heights = np.array([c["height"] for c in glyphs])
    seen = np.zeros(len(glyphs), dtype=bool)
    angles = []
    for line in lines:
        indices = np.flatnonzero(_points_in_line(points, line))
        fit = _fit_row(points[indices], heights[indices], glyph_height)
        if fit is not None:
            angle, supported = fit
            angles.append(angle)
            seen[indices[supported]] = True
    if angles and max(angles) - min(angles) > 4:
        return None
    if angles and seen.mean() >= 0.75:
        angle = float(np.median(angles))
    else:
        angle = _parallel_rows_angle(points, heights, glyph_height)
        if angle is None or any(abs(row_angle - angle) > 4 for row_angle in angles):
            return None
    # Qt uses positive clockwise angles in Y-down page coordinates.
    if not math.isfinite(angle) or not 3 <= abs(angle) <= 35:
        return None
    return round(angle, 1)
