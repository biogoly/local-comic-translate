"""Fit a smooth RGB surface using only explicitly selected background pixels.

This is color interpolation, not texture reconstruction. There is deliberately
no neighborhood search: a nearby panel border cannot become a sample merely
because it is close to the repair mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np


MAX_SAMPLES = 64
MIN_SAMPLES = 4
# RGB root-mean-square distance in 8-bit color levels. Scanned backgrounds need
# a little tolerance for noise/rounding, but strongly different pixels are not
# allowed to bend the surface. This is not a confidence/quality guarantee.
RESIDUAL_LIMIT = 6.0
MAX_CONDITION = 1000.0
MAX_CANDIDATES = 512


class GradientFitError(ValueError):
    """A stable reason code that callers can translate into UI instructions."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class GradientResult:
    replacement: np.ndarray
    sample_count: int
    inlier_count: int
    rms_error: float


@dataclass(frozen=True)
class GradientSampleLine:
    start: tuple[int, int]
    end: tuple[int, int]


def sample_center(sample):
    if isinstance(sample, GradientSampleLine):
        return tuple((int(a) + int(b)) / 2 for a, b in zip(sample.start, sample.end))
    return tuple(map(int, sample))


def sample_coordinates(sample, mask):
    """Return the traced pixel centers, inclusive, in the same order for either drag direction."""
    is_line = isinstance(sample, GradientSampleLine)
    try:
        ends = np.asarray((sample.start, sample.end) if is_line else (sample,))
    except (TypeError, ValueError):
        raise GradientFitError('invalid_samples') from None
    if (ends.shape != (2 if is_line else 1, 2) or ends.dtype.kind not in 'iuf'
            or not np.all(np.isfinite(ends)) or np.any(ends != np.floor(ends))):
        raise GradientFitError('invalid_samples')
    height, width = mask.shape
    if np.any(ends < 0) or np.any(ends[:, 0] >= width) or np.any(ends[:, 1] >= height):
        raise GradientFitError('sample_outside_image')
    if is_line:
        start, end = sorted(map(tuple, ends.astype(np.int64)))
        length = max(abs(b-a) for a, b in zip(start, end))
        pixels = np.rint(np.linspace(start, end, length + 1)).astype(np.intp)
    else:
        pixels = ends.astype(np.intp)
    xs, ys = pixels.T
    if np.any(mask[ys, xs]):
        raise GradientFitError('sample_inside_mask')
    return xs, ys


def _observations(source, mask, samples):
    try:
        samples = iter(samples)
    except TypeError:
        raise GradientFitError('invalid_samples') from None
    # Give each distinct location one vote in the fit. Drag length, direction
    # and repeated clicks must not change its weight or the coverage check.
    colors_by_center = {}
    seen = set()
    for sample in samples:
        xs, ys = sample_coordinates(sample, mask)
        endpoints = (xs[0], ys[0], xs[-1], ys[-1])
        if endpoints in seen:
            continue
        seen.add(endpoints)
        center = sample_center(sample)
        color = (source[ys[0], xs[0]] if len(xs) == 1
                 else np.median(source[ys, xs], axis=0))
        colors_by_center.setdefault(center, []).append(color)
        if len(colors_by_center) > MAX_SAMPLES:
            raise GradientFitError('too_many_samples')
    if len(colors_by_center) < MIN_SAMPLES:
        raise GradientFitError('too_few_samples')
    centers = sorted(colors_by_center)
    colors = [np.median(colors_by_center[center], axis=0) for center in centers]
    return np.asarray(centers, dtype=np.float64), np.asarray(colors, dtype=np.float64)


def _hull(points):
    """Counterclockwise convex hull of two-dimensional pixel coordinates."""
    ordered = sorted(set(map(tuple, points.tolist())))
    if len(ordered) <= 1:
        return np.asarray(ordered, dtype=np.float64).reshape(-1, 2)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _covers(points, vertices):
    hull = _hull(points)
    if len(hull) < 3:
        return False
    for start, end in zip(hull, np.roll(hull, -1, axis=0)):
        edge = end - start
        offset = vertices - start
        cross = edge[0] * offset[:, 1] - edge[1] * offset[:, 0]
        # Permit only subpixel rounding at the boundary, never broad
        # extrapolation beyond the explicitly approved sample region.
        if np.any(cross < -np.linalg.norm(edge)):
            return False
    return True


def _mask_hull(xs, ys):
    """Hull of painted pixels, reduced to at most two points per occupied row.

    ``nonzero`` returns row-major coordinates, so each row's first/last x
    contains every painted pixel on that row between them. Their hull is
    therefore exactly the mask's hull without sorting millions of pixels.
    """
    counts = np.bincount(ys)
    rows = np.flatnonzero(counts)
    ends = np.cumsum(counts)[rows] - 1
    starts = ends - counts[rows] + 1
    outline = np.concatenate((np.column_stack((xs[starts], rows)),
                              np.column_stack((xs[ends], rows))))
    return _hull(outline)


def _well_conditioned(design):
    values = np.linalg.svd(design, compute_uv=False)
    return len(values) >= 3 and values[-1] > 0 and values[0] / values[-1] <= MAX_CONDITION


def _residuals(design, colors, coefficients):
    return np.sqrt(np.mean((design @ coefficients - colors) ** 2, axis=1))


def _fit(design, colors):
    count = len(design)
    minimum = max(MIN_SAMPLES, math.ceil(0.75 * count))
    # Four observations cannot reliably identify one bad sample for a plane
    # with three parameters. Accept agreement, otherwise request more samples.
    if count == MIN_SAMPLES:
        coefficients = np.linalg.lstsq(design, colors, rcond=None)[0]
        if np.max(_residuals(design, colors, coefficients)) > RESIDUAL_LIMIT:
            raise GradientFitError('inconsistent_samples')
        return coefficients, np.ones(count, dtype=bool)

    triples = list(combinations(range(count), 3))
    if len(triples) > MAX_CANDIDATES:
        indexes = np.linspace(0, len(triples) - 1, MAX_CANDIDATES, dtype=int)
        triples = [triples[index] for index in indexes]
    best, best_score = None, None
    for indices in [tuple(range(count)), *triples]:
        subset = design[list(indices)]
        if not _well_conditioned(subset):
            continue
        coefficients = np.linalg.lstsq(subset, colors[list(indices)], rcond=None)[0]
        residuals = _residuals(design, colors, coefficients)
        inliers = residuals <= RESIDUAL_LIMIT
        number = int(np.count_nonzero(inliers))
        if number < minimum or not _well_conditioned(design[inliers]):
            continue
        score = (number, -float(np.median(residuals[inliers])),
                 -float(np.mean(residuals[inliers] ** 2)))
        if best_score is None or score > best_score:
            best, best_score = inliers, score
    if best is None:
        raise GradientFitError('inconsistent_samples')

    # Refit the consensus and remove any pixels whose disagreement only became
    # apparent after fitting all accepted samples. Never grow a rejected set.
    for _ in range(4):
        coefficients = np.linalg.lstsq(design[best], colors[best], rcond=None)[0]
        inliers = best & (_residuals(design, colors, coefficients) <= RESIDUAL_LIMIT)
        if int(np.count_nonzero(inliers)) < minimum or not _well_conditioned(design[inliers]):
            raise GradientFitError('inconsistent_samples')
        if np.array_equal(inliers, best):
            return coefficients, best
        best = inliers
    raise GradientFitError('inconsistent_samples')


def fit_gradient(source: np.ndarray, mask: np.ndarray, sample_points, *,
                 protection: np.ndarray | None = None) -> GradientResult:
    """Return a fixed RGB plane suitable for compositing through ``mask``.

    Each sample is an integer ``(x, y)`` point or a GradientSampleLine. A
    line supplies the median RGB of the pixels on its trace at its midpoint.
    At least four distinct sample centers must surround the writable mask;
    six or more allow more useful robustness checks. Up to 25 percent of
    samples may be rejected when enough consistent, covering samples remain.
    Every sample must stay outside the full painted mask, including protected
    paint. Protection restricts destination coverage only. Inputs are unchanged.
    Returned pixels outside the writable bounding box equal the source; callers
    must still composite through the exact mask and preserve protected pixels.
    """
    if (not isinstance(source, np.ndarray) or source.ndim != 3 or source.shape[2] != 3
            or source.dtype != np.uint8 or not source.shape[0] or not source.shape[1]):
        raise GradientFitError('invalid_image')
    if (not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.uint8
            or mask.shape != source.shape[:2]):
        raise GradientFitError('invalid_mask')
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise GradientFitError('empty_mask')

    points, colors = _observations(source, mask, sample_points)
    if protection is not None:
        if (not isinstance(protection, np.ndarray) or protection.shape != mask.shape
                or protection.dtype.kind not in "bui"):
            raise GradientFitError('invalid_mask')
        writable = protection[ys, xs] == 0
        ys, xs = ys[writable], xs[writable]
        if not len(xs):
            raise GradientFitError('fully_protected')

    minimum = points.min(axis=0).astype(np.float64)
    maximum = points.max(axis=0).astype(np.float64)
    center = (minimum + maximum) / 2
    scale = np.maximum((maximum - minimum) / 2, 1)
    local = (points - center) / scale
    design = np.column_stack((np.ones(len(points)), local))
    if not _well_conditioned(design):
        raise GradientFitError('degenerate_samples')
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    covered_pixels = _mask_hull(xs, ys)
    if not _covers(points, covered_pixels):
        raise GradientFitError('insufficient_coverage')

    coefficients, inliers = _fit(design, colors)
    if not _covers(points[inliers], covered_pixels):
        raise GradientFitError('insufficient_coverage')
    rms = float(np.sqrt(np.mean((design[inliers] @ coefficients - colors[inliers]) ** 2)))
    # Render in bounded row batches rather than allocating an HxWx3 float64
    # coordinate array for a long webtoon page.
    replacement = source.copy()
    local_x = (np.arange(x1, x2 + 1, dtype=np.float64) - center[0]) / scale[0]
    x_colors = local_x[:, None] * coefficients[1]
    for start in range(y1, y2 + 1, 128):
        end = min(start + 128, y2 + 1)
        local_y = (np.arange(start, end, dtype=np.float64) - center[1]) / scale[1]
        colors = coefficients[0] + x_colors[None, :, :] + local_y[:, None, None] * coefficients[2]
        replacement[start:end, x1:x2 + 1] = np.clip(np.rint(colors), 0, 255).astype(np.uint8)
    replacement.setflags(write=False)
    return GradientResult(replacement, len(points), int(np.count_nonzero(inliers)), rms)
