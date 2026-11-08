"""Text-bound recognizer evidence for an advisory source review."""
from collections.abc import Mapping
import hashlib
import math
from numbers import Real


# Applied only to the PP-OCR decoder's mean emitted-character score per line.
# This is a review threshold, not a calibrated probability of a correct reading.
PPOCR_REVIEW_THRESHOLD = 0.75
_SCORE_SCOPES = {
    "PP-OCR": "lowest_line",
    "Pororo OCR": "lowest_sequence_product",
    "Microsoft OCR": "lowest_word",
}


def _score(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    value = float(value)
    return value if math.isfinite(value) and 0 <= value <= 1 else None


def _signature(text):
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def lowest_reported_score(values):
    """Return the lowest score only when all of the contributing values are usable."""
    scores = [_score(value) for value in values]
    return min(scores) if scores and all(score is not None for score in scores) else None


def make_ocr_evidence(text, engine, model, *, line_scores=None):
    """Bind the recognizer and its scoped score to this reading."""
    if (not isinstance(text, str) or not text.strip()
            or not all(isinstance(value, str) and value.strip() for value in (engine, model))):
        return None
    record = {"engine": engine, "model": model, "scope": "unscored",
              "score": None, "line_count": 0, "text_sha256": _signature(text)}
    if engine in _SCORE_SCOPES and line_scores is not None:
        scores = list(line_scores)
        lowest = lowest_reported_score(scores)
        if lowest is not None:
            record.update(scope=_SCORE_SCOPES[engine], score=lowest, line_count=len(scores))
    return record


def current_ocr_evidence(block):
    """Return detached evidence only for its original, exact source wording."""
    if isinstance(block, Mapping):
        text, record = block.get("text"), block.get("ocr_evidence")
    else:
        text, record = getattr(block, "text", None), getattr(block, "ocr_evidence", None)
    if not isinstance(record, Mapping) or not isinstance(text, str) or not text.strip():
        return None
    if record.get("text_sha256") != _signature(text):
        return None
    if not all(isinstance(record.get(key), str) and record[key] for key in ("engine", "model")):
        return None
    scope = record.get("scope")
    count = record.get("line_count")
    if scope != "unscored":
        if (not scope or scope != _SCORE_SCOPES.get(record["engine"]) or type(count) is not int or count < 1
                or _score(record.get("score")) is None):
            return None
    elif "score" not in record or record["score"] is not None or type(count) is not int or count != 0:
        return None
    return {key: record[key] for key in ("engine", "model", "scope", "score", "line_count", "text_sha256")}


def assign_unscored_reading(block, text, engine, model):
    """Bind a new, unscored reading to the recognizer that produced it."""
    block.text = text
    block.ocr_evidence = make_ocr_evidence(text, engine, model)


def uncertain_ocr_evidence(block):
    """Return the scored evidence that warrants a PP-OCR review hint, if any."""
    evidence = current_ocr_evidence(block)
    if evidence and evidence["scope"] == "lowest_line" and evidence["score"] < PPOCR_REVIEW_THRESHOLD:
        return evidence
    return None
