"""Preserve user-owned text regions during the page-wide manual workflow."""
from modules.detection.utils.geometry import calculate_iou, is_mostly_contained
from modules.utils.translator_utils import is_renderable_translation


def is_manual_region(block) -> bool:
    # Older projects had no provenance flag: drawn rectangles were the only
    # regions with no detector class. Keep those regions compatible as well.
    return bool(getattr(block, 'is_manual', False)) or not getattr(block, 'text_class', '')


def has_manual_translation(block) -> bool:
    return is_manual_region(block) and is_renderable_translation(
        getattr(block, 'translation', '') or ''
    )


def merge_manual_regions(previous, detected, *, retain_ids=()):
    """Keep manual/protected boxes intact and suppress duplicate detections.

    New detections still replace unprotected automatic boxes. A slight overlap alone
    must not suppress neighbouring text or a much larger detected paragraph.
    """
    from .region_policy import should_preserve_region
    retain_ids = set(retain_ids)
    manual = [block for block in previous if should_preserve_region(block)
              or getattr(block, 'region_id', '') in retain_ids]
    additions = []
    for block in detected:
        duplicate = False
        for existing in manual:
            if existing.xyxy is None or block.xyxy is None:
                continue
            if (calculate_iou(existing.xyxy, block.xyxy) >= 0.5
                    or is_mostly_contained(existing.xyxy, block.xyxy, 0.8)):
                duplicate = True
                break
        if not duplicate:
            additions.append(block)
    return manual + additions
