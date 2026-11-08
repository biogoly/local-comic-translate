"""Isolated, explicitly requested OCR of generated lettering previews.

Settings are captured on the GUI thread. Recognition uses only that immutable
snapshot and copied preview pixels, never the live page or its OCR caches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from modules.artistic_edit.image_ops import crop_to_bbox, mask_bbox
from modules.translation.providers import credential_service


OCR_KEYS = ('Default', 'Microsoft OCR', 'Google Cloud Vision',
            'GPT-4.1-mini', 'Gemini-2.5-Flash-Lite')


@dataclass(frozen=True)
class _SnapshotUI:
    @staticmethod
    def tr(text):
        return text


@dataclass(frozen=True)
class OCRSettingsSnapshot:
    ocr_key: str
    use_gpu: bool
    _credentials: tuple[tuple[str, str | bool | int | float | None], ...] = field(repr=False)
    ui: _SnapshotUI = field(default_factory=_SnapshotUI, repr=False)

    @property
    def hosted(self):
        return self.ocr_key != 'Default'

    def get_tool_selection(self, tool):
        if tool != 'ocr':
            raise ValueError('The lettering check can only use OCR settings.')
        return self.ocr_key

    def is_gpu_enabled(self):
        return self.use_gpu

    def get_credentials(self, service=''):
        if service and credential_service(service) == credential_service(self.ocr_key):
            return dict(self._credentials)
        return {}


def capture_ocr_settings(settings) -> OCRSettingsSnapshot:
    """Read live widgets once, before dispatching any background work."""
    selected = settings.get_tool_selection('ocr')
    aliases = {settings.ui.tr(key): key for key in OCR_KEYS}
    key = aliases.get(selected, selected)
    if key not in OCR_KEYS:
        raise ValueError('The selected OCR engine is not supported for lettering checks.')
    credentials = settings.get_credentials(credential_service(key)) if key != 'Default' else {}
    if not isinstance(credentials, dict):
        raise ValueError('The selected OCR credentials could not be captured.')
    for name, value in credentials.items():
        if not isinstance(name, str) or not isinstance(value, (str, bool, int, float, type(None))):
            raise ValueError('The selected OCR credentials have an unsupported format.')
    return OCRSettingsSnapshot(key, bool(settings.is_gpu_enabled()), tuple(sorted(credentials.items())))


@dataclass(frozen=True)
class LetteringCrop:
    image: np.ndarray = field(repr=False)
    partial_mask: bool = False


def prepare_lettering_crop(prepared, patch, protection=None) -> LetteringCrop:
    """Copy final writable pixels, excluding protected/unselected surroundings.

``protection`` is an optional decoded binary mask in full-page coordinates.
The modest white padding aids recognition without leaking neighboring text.
``partial_mask`` warns that the writable bounding rectangle contains holes.
    """
    pixels = np.asarray(patch.image)
    alpha = np.asarray(patch.alpha)
    if (pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3
            or not pixels.size or alpha.shape != pixels.shape[:2]
            or alpha.dtype.kind not in 'bui'):
        raise ValueError('The generated lettering crop has invalid pixels or mask dimensions.')
    # Verify the final patch belongs to the captured source context. The crop
    # uses its final composed pixels, including feathering and protection.
    x, y, width, height = patch.bbox
    cx, cy, cw, ch = prepared.context_bbox
    if (width != pixels.shape[1] or height != pixels.shape[0] or x < cx or y < cy
            or x + width > cx + cw or y + height > cy + ch):
        raise ValueError('The generated lettering patch is outside its preview context.')
    writable = alpha > 0
    if protection is not None:
        protected = np.asarray(protection)
        if protected.ndim != 2 or protected.dtype.kind not in 'bui':
            raise ValueError('Artwork protection must be a decoded page mask.')
        protected_crop = crop_to_bbox(protected, patch.bbox)
        writable &= protected_crop == 0
    bounds = mask_bbox(writable.astype(np.uint8))
    if bounds is None:
        raise ValueError('There are no writable pixels to check in this preview.')
    selection = crop_to_bbox(writable, bounds)
    selected_pixels = crop_to_bbox(pixels, bounds).copy()
    selected_pixels[~selection] = 255
    padded = np.pad(selected_pixels, ((8, 8), (8, 8), (0, 0)), constant_values=255)
    padded.setflags(write=False)
    return LetteringCrop(padded, partial_mask=not bool(selection.all()))


def run_lettering_ocr(image, target_language, settings_snapshot: OCRSettingsSnapshot) -> str:
    """Recognize generated target text with the selected OCR route, once.

No expected wording is supplied to the recognizer. A private factory cache
prevents stateful recognizers from sharing buffers with normal page OCR.
Errors propagate; an empty reading stays empty rather than invoking a fallback.
    """
    from modules.ocr.factory import OCRFactory
    from modules.ocr.processor import OCRProcessor
    from modules.utils.language_utils import language_codes
    from modules.utils.textblock import TextBlock

    if not isinstance(settings_snapshot, OCRSettingsSnapshot) or settings_snapshot.ocr_key not in OCR_KEYS:
        raise ValueError('A captured OCR configuration is required for the lettering check.')
    if target_language not in language_codes:
        raise ValueError('Choose a supported target language before checking lettering.')
    pixels = np.asarray(image)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3 or not pixels.size:
        raise ValueError('Lettering OCR requires a nonempty RGB preview image.')
    pixels = pixels.copy(order='C')

    class _JobOCRFactory(OCRFactory):
        _engines = {}

        @classmethod
        def create_engine(cls, settings, source_lang_english, ocr_model, backend=None):
            engine = super().create_engine(settings, source_lang_english, ocr_model, backend)
            if engine is None:
                raise ValueError('The selected OCR engine does not support this target language.')
            return engine

    processor = OCRProcessor(engine_factory=_JobOCRFactory)
    processor.initialize(SimpleNamespace(settings_page=settings_snapshot, lang_mapping={}), target_language)
    height, width = pixels.shape[:2]
    # This is a fresh OCR crop, not a detected "text_free" page region. That
    # page-relative label would falsely trigger the large-title PP-OCR override
    # for every full-crop Japanese check, after loading Manga OCR as well.
    region = TextBlock(text_bbox=np.array([0, 0, width, height], dtype=int), text='',
                       lines=None, direction='', text_class='')
    try:
        processor.process(pixels, [region])
        return region.text if isinstance(region.text, str) else ''
    finally:
        _JobOCRFactory._engines.clear()
