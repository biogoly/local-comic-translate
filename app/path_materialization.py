from __future__ import annotations

import logging

import imkit as imk
import numpy as np

from app.projects.project_state import ensure_project_blob_materialized
from modules.utils.file_handler import ensure_prepared_path_materialized


def ensure_path_materialized(path: str) -> bool:
    return ensure_project_blob_materialized(path) or ensure_prepared_path_materialized(path)


def load_original_rgb(path, expected_shape=None):
    """Read the source page (including embedded project images), independently of edits."""
    try:
        ensure_path_materialized(path)
        image = imk.read_image(path)
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3 or not image.size):
            return None
        if expected_shape is not None and tuple(image.shape[:2]) != tuple(expected_shape[:2]):
            return None
        pixels = image.copy()
        pixels.setflags(write=False)
        return pixels
    except (OSError, ValueError, RuntimeError, TypeError):
        logging.getLogger(__name__).debug('Original page image unavailable', exc_info=True)
        return None
