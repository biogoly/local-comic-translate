"""Content-bound review issues without image access, models or Qt creation.

The controller supplies page-local properties and optional measured layout
diagnostics. Unknown measurements never establish a successful fit. Cleanup
review requires an explicit workflow cleanup_fingerprint token; visual cleanup
defects cannot be inferred by this evaluator.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
from html.parser import HTMLParser
import json
import math
from numbers import Real
import re
import unicodedata

from modules.utils.region_metadata import (
    MAX_DISMISSED_FINGERPRINTS, REVIEW_ISSUE_KINDS, REVIEW_STAGES,
    normalize_region_metadata, normalized_review_fingerprint,
)
from modules.ocr.evidence import uncertain_ocr_evidence


@dataclass(frozen=True)
class ReviewIssue:
    page_id: str
    region_id: str
    kind: str
    stage: str
    fingerprint: str
    detail: str = ""
    page_path: str = ""

    @property
    def key(self) -> tuple[str, str, str, str]:
        """Row identity survives edits; the separate fingerprint does not."""
        return self.page_id, self.region_id, self.kind, self.stage


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _text(value):
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n")).strip() if isinstance(value, str) else ""


def _canonical(value):
    """Read stable values without importing/instantiating Qt or numpy.

    The caller may supply passive QColor/enum/array values in canvas properties.
    Never hash object reprs, which frequently contain process memory addresses.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Enum):
        return _canonical(value.value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if callable(getattr(value, "tolist", None)):
        return _canonical(value.tolist())
    if callable(getattr(value, "rgba", None)):
        return {"rgba": int(value.rgba())}
    if callable(getattr(value, "toString", None)):
        return value.toString()
    return {"unsupported_type": type(value).__name__}


def _hash(payload):
    encoded = json.dumps(_canonical(payload), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _language(languages, key, block):
    value = _get(languages, key)
    if value is None:
        value = _get(languages, key + "_lang")
    if value is None:
        value = _get(block, key + "_lang", "")
    return _text(value)


def _geometry(block):
    return {key: _get(block, key) for key in ("xyxy", "bubble_xyxy", "angle", "direction")}


_LAYOUT_FIELDS = (
    "text", "font_family", "font_size", "document_font", "position", "rotation",
    "scale", "transform_origin", "width", "height", "direction", "vertical",
    "alignment", "line_spacing", "bold", "italic", "underline", "text_color",
    "outline", "outline_color", "outline_width", "selection_outlines", "layout_preferences",
)


def fingerprint_for_stage(stage, block, rendered_props=None, *, diagnostics=None, workflow=None, languages=None) -> str:
    """Hash stage-relevant content, excluding IDs, paths and approval/protection state.

    Languages accepts source/target or source_lang/target_lang keys, with block
    fields as fallback. Properties accepts a mapping or a properties object.
    All region bounds and item positions must be page-local. Diagnostics are
    accepted for a common calling convention; their evidence belongs to the
    specific measured issue fingerprint, not the stage-content fingerprint.
    """
    if stage not in REVIEW_STAGES:
        raise ValueError("Unknown review stage")
    source = _text(_get(block, "text", ""))
    target = _text(_get(block, "translation", ""))
    source_language = _language(languages, "source", block)
    target_language = _language(languages, "target", block)
    if stage == "source":
        content = {"text": source, "bounds": _get(block, "xyxy"), "lines": _get(block, "lines"),
                   "angle": _get(block, "angle", 0), "direction": _get(block, "direction", ""),
                   "language": source_language}
    elif stage == "translation":
        content = {"source": source, "target": target,
                   "source_language": source_language, "target_language": target_language}
    elif stage == "layout":
        content = {"translation": target, "source_geometry": _geometry(block),
                   "rendered": {key: _get(rendered_props, key) for key in _LAYOUT_FIELDS}}
        if _get(rendered_props, "lettering_transform"):
            # Preserve existing fingerprints for ordinary/legacy lettering.
            content["rendered"]["lettering_transform"] = _get(rendered_props, "lettering_transform")
    else:
        content = {"evidence": _get(workflow, "cleanup_fingerprint", "")}
    return _hash({"schema": 1, "stage": stage, "content": content})


class _VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"style", "script", "head"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag.lower() in {"style", "script", "head"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _has_rendered_text(props):
    text = _get(props, "text", "")
    if not isinstance(text, str):
        return False
    if re.search(r"<(?:html|body|p|span|div|br|style)(?:\s|>|/)", text, re.IGNORECASE):
        parser = _VisibleText()
        parser.feed(text)
        text = "".join(parser.parts)
    return bool(text.strip())


def _diagnostic_reasons(diagnostics):
    reasons = _get(diagnostics, "overflow_reasons", ())
    if not isinstance(reasons, (tuple, list)):
        return ()
    return tuple(sorted(set(reason for reason in reasons if isinstance(reason, str))))


def evaluate_region(page_id, block, rendered_props=None, diagnostics=None, workflow=None, languages=None) -> list[ReviewIssue]:
    """Return current issues, omitting only matching reviews/dismissals.

    Reviewing an intentionally blank source/translation suppresses that missing
    stage warning. Layout approval never suppresses overflow or small text;
    those require an explicit dismissal. A populated stage can await review
    even when measurements are unknown; this makes no fit-success claim.
    No region means no missing-OCR issue for an unscreened page.
    """
    if block is None:
        return []
    region_id = _get(block, "region_id", "")
    metadata = normalize_region_metadata(_get(block, "region_metadata"))
    review = metadata["review"]
    approved = review["approved_fingerprints"]
    dismissed = review["dismissed_issue_fingerprints"]
    fingerprints = {stage: fingerprint_for_stage(stage, block, rendered_props,
                    diagnostics=diagnostics, workflow=workflow, languages=languages) for stage in REVIEW_STAGES}
    source = _text(_get(block, "text", ""))
    target = _text(_get(block, "translation", ""))
    path = _get(workflow, "page_path", "")
    path = path if isinstance(path, str) else ""
    issues = []

    def append(kind, stage, detail="", evidence=None, reviewable=True):
        stage_fingerprint = fingerprints[stage]
        if reviewable and approved.get(stage) == stage_fingerprint:
            return
        fingerprint = stage_fingerprint if evidence is None else _hash({"stage": stage_fingerprint, "evidence": evidence})
        if fingerprint in dismissed.get(kind, []):
            return
        issues.append(ReviewIssue(page_id, region_id, kind, stage, fingerprint, detail, path))

    if not source:
        append("missing_ocr", "source")
    else:
        evidence = uncertain_ocr_evidence(block)
        if evidence:
            append("uncertain_ocr", "source", evidence=evidence)
        append("awaiting_review", "source")
    if source and not target:
        append("missing_translation", "translation")
    elif target:
        append("awaiting_review", "translation")

    if _has_rendered_text(rendered_props):
        reasons = _diagnostic_reasons(diagnostics)
        overflow = tuple(reason for reason in reasons if reason in {"width", "height", "unbreakable_word", "boundary"})
        if overflow or (_get(diagnostics, "fits") is False and not reasons):
            append("overflow", "layout", ",".join(overflow) or "does_not_fit",
                   evidence={"fits": _get(diagnostics, "fits"), "reasons": reasons}, reviewable=False)
        if _get(diagnostics, "below_minimum") is True:
            append("small_text", "layout", "below_minimum",
                   evidence={"below_minimum": True,
                             "minimum_font_size": _get(diagnostics, "minimum_font_size", _get(diagnostics, "min_font_size")),
                             "font_size": _get(diagnostics, "font_size", _get(rendered_props, "font_size"))}, reviewable=False)
        measured = bool(overflow) or _get(diagnostics, "fits") is True or (
            _get(diagnostics, "fits") is False and not reasons)
        if "unsupported_content" in reasons or "empty" in reasons:
            measured = False
        append("awaiting_review", "layout", "" if measured else "diagnostics_unknown")

    cleanup = _get(workflow, "cleanup_fingerprint")
    if isinstance(cleanup, str) and cleanup.strip():
        append("awaiting_review", "cleanup")
    return issues


def mark_reviewed(metadata, stage, fingerprint) -> dict:
    """Return independent region metadata approving exactly this stage fingerprint."""
    value = normalized_review_fingerprint(fingerprint)
    if stage not in REVIEW_STAGES or not value:
        raise ValueError("Review requires a known stage and SHA256 fingerprint")
    result = normalize_region_metadata(metadata)
    result["review"]["approved_fingerprints"][stage] = value
    return result


def dismiss_issue(metadata, issue: ReviewIssue) -> dict:
    """Ignore this result only; content/diagnostic edits make it stale."""
    value = normalized_review_fingerprint(issue.fingerprint)
    if issue.kind not in REVIEW_ISSUE_KINDS or not value:
        raise ValueError("Dismissal requires a known issue kind and SHA256 fingerprint")
    result = normalize_region_metadata(metadata)
    dismissed = result["review"]["dismissed_issue_fingerprints"]
    previous = [fingerprint for fingerprint in dismissed.get(issue.kind, []) if fingerprint != value]
    dismissed[issue.kind] = [*previous, value][-MAX_DISMISSED_FINGERPRINTS:]
    return result


def clear_review(metadata, stage) -> dict:
    if stage not in REVIEW_STAGES:
        raise ValueError("Unknown review stage")
    result = normalize_region_metadata(metadata)
    result["review"]["approved_fingerprints"].pop(stage, None)
    return result


ignore_issue = dismiss_issue
