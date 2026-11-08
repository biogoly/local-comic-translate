"""Conservative, advisory comparison of requested lettering and OCR output.

This module does not perform OCR or determine whether an image is correct.
``match`` means only that the OCR string matches after NFC/whitespace
normalization. Case is meaningful by default; a caller may explicitly allow
capitalization changes. Punctuation and accents remain meaningful. Callers must
display the strings as plain text (or escape them in their own rich-text UI).
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal
import unicodedata


@dataclass(frozen=True)
class LetteringDifference:
    """One changed span, with offsets into the normalized result strings."""

    kind: Literal["replace", "missing", "extra"]
    expected: str
    recognized: str
    expected_start: int
    expected_end: int
    recognized_start: int
    recognized_end: int


@dataclass(frozen=True)
class LetteringCheckResult:
    """OCR agreement, never automatic approval of a generated image."""

    status: Literal["match", "mismatch", "uncertain"]
    expected: str
    recognized: str
    differences: tuple[LetteringDifference, ...] = ()


def normalize_lettering_text(text: str) -> str:
    """Normalize canonical Unicode spelling and runs of whitespace only.

    Compatibility characters, case, punctuation, accents and zero-width
    characters are deliberately retained. In particular, this must not make
    an OCR-dropped accent or a differently spelled word appear correct.
    """
    if not isinstance(text, str):
        raise TypeError("Lettering must be a string.")
    return " ".join(unicodedata.normalize("NFC", text).split())


def _unspaced_character(character: str) -> bool:
    """Scripts whose ordinary words cannot be split reliably on spaces."""
    value = ord(character)
    return any(start <= value <= end for start, end in (
        (0x0E00, 0x0EFF),   # Thai and Lao
        (0x1000, 0x109F),   # Myanmar
        (0x1780, 0x17FF),   # Khmer
        (0x3040, 0x30FF),   # Hiragana and Katakana
        (0x3100, 0x312F),   # Bopomofo
        (0x3130, 0x318F),   # Hangul compatibility jamo
        (0x31A0, 0x31BF),   # Bopomofo extended
        (0x31F0, 0x31FF),   # Katakana phonetic extensions
        (0x3400, 0x4DBF),   # CJK extension A
        (0x4E00, 0x9FFF),   # CJK unified ideographs
        (0xA960, 0xA97F),   # Hangul extended A
        (0xAC00, 0xD7FF),   # Hangul syllables and extended B
        (0xF900, 0xFAFF),   # CJK compatibility ideographs
        (0xFF66, 0xFF9D),   # Half-width Katakana (not compatibility-normalized)
        (0x1100, 0x11FF),   # Hangul jamo
        (0x1B000, 0x1B16F), # Kana supplements
        (0x20000, 0x323AF), # Supplementary CJK ideographs
    ))


def _tokens(text: str) -> tuple[list[str], list[int]]:
    # Keep whitespace as tokens so removing a word boundary is a difference.
    # Combining marks stay attached to the preceding character/token. This is
    # comparison tokenization, not a typography or linguistic word segmenter.
    tokens: list[str] = []
    starts: list[int] = []
    previous_word = False
    for index, character in enumerate(text):
        category = unicodedata.category(character)
        if category.startswith("M") and tokens:
            tokens[-1] += character
            continue
        word = character.isalnum() and not _unspaced_character(character)
        if word and previous_word:
            tokens[-1] += character
        else:
            tokens.append(character)
            starts.append(index)
        previous_word = word
    starts.append(len(text))
    return tokens, starts


def compare_lettering(
    expected: str, recognized: str | None, *, case_sensitive: bool = True
) -> LetteringCheckResult:
    """Compare exact requested wording with OCR, without changing either input.

    Blank, symbol-only or replacement-character OCR is uncertain. No heuristic
    claims to distinguish plausible-looking OCR errors from actual spelling
    mistakes; nonmatching readable strings are advisory mismatches. An absent
    expected wording is invalid and must not be inferred from a freeform prompt.
    Case-insensitive checks compare lowercase tokens, retaining original strings
    and offsets for display. This deliberately avoids compatibility folding,
    such as treating the ligature ``\ufb01`` as the two letters ``fi``.
    """
    expected = normalize_lettering_text(expected)
    if not expected:
        raise ValueError("An exact requested wording is required to check lettering.")
    recognized = "" if recognized is None else normalize_lettering_text(recognized)
    if (not recognized or "\ufffd" in recognized
            or not any(character.isalnum() for character in recognized)):
        return LetteringCheckResult("uncertain", expected, recognized)
    if recognized == expected:
        return LetteringCheckResult("match", expected, recognized)

    expected_tokens, expected_offsets = _tokens(expected)
    recognized_tokens, recognized_offsets = _tokens(recognized)
    if not case_sensitive:
        expected_tokens = [token.lower() for token in expected_tokens]
        recognized_tokens = [token.lower() for token in recognized_tokens]
        if expected_tokens == recognized_tokens:
            return LetteringCheckResult("match", expected, recognized)
    differences = []
    for operation, a, b, c, d in SequenceMatcher(
            None, expected_tokens, recognized_tokens, autojunk=False).get_opcodes():
        if operation == "equal":
            continue
        expected_start, expected_end = expected_offsets[a], expected_offsets[b]
        recognized_start, recognized_end = recognized_offsets[c], recognized_offsets[d]
        differences.append(LetteringDifference(
            {"replace": "replace", "delete": "missing", "insert": "extra"}[operation],
            expected[expected_start:expected_end], recognized[recognized_start:recognized_end],
            expected_start, expected_end, recognized_start, recognized_end,
        ))
    return LetteringCheckResult("mismatch", expected, recognized, tuple(differences))
