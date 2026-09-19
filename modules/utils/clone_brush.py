"""Fixed-patch clone stamps in image coordinates; no Qt or model dependency."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CloneSample:
    pixels: np.ndarray
    valid: np.ndarray
    size: int


def capture_sample(image, x, y, size):
    """Freeze only the chosen patch. Missing edge pixels stay invalid, not black."""
    size = max(1, int(size))
    half = math.ceil(size / 2)
    side = 2 * half + 1
    pixels = np.zeros((side, side, 3), np.uint8)
    valid = np.zeros((side, side), bool)
    h, w = image.shape[:2]
    left, top = math.floor(x) - half, math.floor(y) - half
    x1, y1 = max(0, left), max(0, top)
    x2, y2 = min(w, left + side), min(h, top + side)
    if x2 > x1 and y2 > y1:
        pixels[y1-top:y2-top, x1-left:x2-left] = image[y1:y2, x1:x2, :3]
        valid[y1-top:y2-top, x1-left:x2-left] = True
    yy, xx = np.ogrid[-half:half+1, -half:half+1]
    valid &= xx**2 + yy**2 < (size / 2)**2
    pixels[~valid] = 0
    pixels.setflags(write=False)
    valid.setflags(write=False)
    return CloneSample(pixels, valid, size)


class CloneStroke:
    def __init__(self, current, sample, softness):
        self.current = current
        self.result = current.copy()
        self.sample = sample
        self.radius = sample.size / 2
        self.half = sample.pixels.shape[0] // 2
        self.mask = np.zeros(current.shape[:2], np.uint8)
        self.last = None
        self.last_dab = None
        self.bounds = None
        yy, xx = np.ogrid[-self.half:self.half+1, -self.half:self.half+1]
        distance = np.sqrt(xx**2 + yy**2)
        feather = max(0.5, self.radius * max(0, min(1, softness)))
        self.alpha = np.clip((self.radius - distance) / feather, 0, 1) * sample.valid

    def move(self, x, y):
        if self.last is None:
            self._dab(x, y)
        else:
            lx, ly = self.last
            distance = math.hypot(x - lx, y - ly)
            steps = max(1, math.ceil(distance / max(1, self.radius * 0.2)))
            for i in range(1, steps + 1):
                self._dab(lx + (x-lx) * i/steps, ly + (y-ly) * i/steps)
        self.last = (x, y)

    def _dab(self, x, y):
        center = (math.floor(x), math.floor(y))
        if center == self.last_dab:
            return
        self.last_dab = center
        h, w = self.mask.shape
        left, top = center[0] - self.half, center[1] - self.half
        side = self.sample.pixels.shape[0]
        x1, y1 = max(0, left), max(0, top)
        x2, y2 = min(w, left + side), min(h, top + side)
        if x2 <= x1 or y2 <= y1:
            return
        sy, sx = slice(y1-top, y2-top), slice(x1-left, x2-left)
        alpha = self.alpha[sy, sx, None]
        if not np.any(alpha):
            return
        target = self.result[y1:y2, x1:x2]
        source = self.sample.pixels[sy, sx]
        target[:] = np.rint(target * (1-alpha) + source * alpha).astype(np.uint8)
        mask = self.mask[y1:y2, x1:x2]
        np.maximum(mask, np.rint(alpha[..., 0] * 255).astype(np.uint8), out=mask)
        if self.bounds is None:
            self.bounds = (x1, y1, x2, y2)
        else:
            a, b, c, d = self.bounds
            self.bounds = (min(a, x1), min(b, y1), max(c, x2), max(d, y2))

    def patch(self):
        if self.bounds is None:
            return None
        x1, y1, x2, y2 = self.bounds
        after = self.result[y1:y2, x1:x2]
        if np.array_equal(self.current[y1:y2, x1:x2], after):
            return None
        return {"bbox": [x1, y1, x2-x1, y2-y1], "image": np.ascontiguousarray(after)}
