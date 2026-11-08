"""Small, serializable identity helpers shared by projects, canvas and undo.

An ID describes a logical region, not its current position or detection index.
Snapshots retain IDs; an intentional duplicate must explicitly request a new one.
"""
from __future__ import annotations

import copy
import math
import re
import uuid


REVIEW_STAGES = ("source", "translation", "layout", "cleanup")
REVIEW_ISSUE_KINDS = ("missing_ocr", "uncertain_ocr", "missing_translation", "overflow", "small_text",
                      "awaiting_review", "failed_operation")
MAX_DISMISSED_FINGERPRINTS = 8
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


def normalized_review_fingerprint(value) -> str:
    """Persist only fixed-size SHA256 values, never arbitrary review payloads."""
    return value.lower() if isinstance(value, str) and _SHA256.fullmatch(value) else ""


def normalize_review_metadata(value) -> dict:
    value = value if isinstance(value, dict) else {}
    approved = value.get("approved_fingerprints", {})
    approved = approved if isinstance(approved, dict) else {}
    dismissed = value.get("dismissed_issue_fingerprints", {})
    dismissed = dismissed if isinstance(dismissed, dict) else {}
    result = {"approved_fingerprints": {}, "dismissed_issue_fingerprints": {}}
    for stage in REVIEW_STAGES:
        fingerprint = normalized_review_fingerprint(approved.get(stage))
        if fingerprint:
            result["approved_fingerprints"][stage] = fingerprint
    for kind in REVIEW_ISSUE_KINDS:
        values = dismissed.get(kind, [])
        # Accept a single hash from early project writers as well as the bounded
        # list needed to dismiss different awaiting-review stages independently.
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, (list, tuple)):
            continue
        hashes = []
        for value in reversed(values[-MAX_DISMISSED_FINGERPRINTS:]):
            fingerprint = normalized_review_fingerprint(value)
            if fingerprint and fingerprint not in hashes:
                hashes.append(fingerprint)
        if hashes:
            result["dismissed_issue_fingerprints"][kind] = list(reversed(hashes))
    return result


def new_id() -> str:
    return uuid.uuid4().hex


def normalized_id(value) -> str:
    if not isinstance(value, str):
        return ""
    try:
        return uuid.UUID(value).hex
    except (ValueError, AttributeError):
        return ""


def ensure_region_id(block) -> str:
    identity = normalized_id(getattr(block, "region_id", None)) or new_id()
    block.region_id = identity
    return identity


def normalize_layout_preferences(value) -> dict:
    """Only store explicit fit preferences; old items keep their original layout."""
    if not isinstance(value, dict):
        return {}
    result = {}
    for key in ("bound_width", "bound_height", "min_font_size", "padding"):
        number = value.get(key)
        if isinstance(number, (float, int)) and not isinstance(number, bool):
            if math.isfinite(number) and (number >= 0 if key == "padding" else number > 0):
                result[key] = float(number)
    if isinstance(value.get("keep_manual_breaks"), bool):
        result["keep_manual_breaks"] = value["keep_manual_breaks"]
    if value.get("mode") in {"box", "bubble"}:
        result["mode"] = value["mode"]
    if value.get("boundary_shape") in {"auto", "rectangle", "ellipse"}:
        result["boundary_shape"] = value["boundary_shape"]
    bounds = value.get("boundary_box")
    if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in bounds):
            if bounds[2] > bounds[0] and bounds[3] > bounds[1]:
                result["boundary_box"] = [float(v) for v in bounds]
    return result


def normalize_region_metadata(value) -> dict:
    """Normalize independent protection, fitting and content-bound review data."""
    value = value if isinstance(value, dict) else {}
    result = {"schema_version": 1,
              "layout_preferences": normalize_layout_preferences(value.get("layout_preferences"))}
    protections = value.get("protections", {})
    result["protections"] = {
        key: protections.get(key, False) is True if isinstance(protections, dict) else False
        for key in ("keep_source", "keep_translation", "keep_layout", "skip_auto_cleanup")
    }
    result["review"] = normalize_review_metadata(value.get("review"))
    return result


def find_region(blocks, *, region_id="", x=None, y=None, angle=0, tolerance=5):
    """ID first; a missing ID may use a unique legacy position match only.

    An explicit but stale ID never reattaches to a neighbouring/replacement box.
    """
    if region_id:
        matches = [block for block in blocks if getattr(block, "region_id", None) == region_id]
    else:
        matches = []
        for block in blocks:
            try:
                bx, by = float(block.xyxy[0]), float(block.xyxy[1])
                if (abs(bx - float(x)) < tolerance and abs(by - float(y)) < tolerance
                        and abs(float(block.angle) - float(angle)) < 1):
                    matches.append(block)
            except (TypeError, ValueError, IndexError, AttributeError):
                continue
    return matches[0] if len(matches) == 1 else None


def duplicate_region(block):
    """Duplicate content, but not identity or prior approval/protection intent."""
    duplicate = copy.deepcopy(block)
    duplicate.region_id = new_id()
    duplicate.region_metadata = normalize_region_metadata({
        "layout_preferences": getattr(block, "region_metadata", {}).get("layout_preferences", {})
    })
    return duplicate
