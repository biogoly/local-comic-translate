"""Explicit, local alternative recognizers for one original-page region."""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image

from modules.detection.heuristic_lines import annotate_blocks_with_heuristic_lines
from modules.ocr.evidence import current_ocr_evidence, make_ocr_evidence
from modules.ocr.factory import OCRFactory
from modules.ocr.processor import _should_override_to_ppocr
from modules.translation.spatial import clip_polygon, geometry_signature, region_polygon
from modules.utils.download import ModelID
from modules.utils.language_utils import get_dominant_page_script, is_supported_script, normalize_script, language_codes
from modules.utils.textblock import TextBlock


@dataclass(frozen=True)
class LocalRecognizer:
    label: str
    model: ModelID
    source_code: str
    pp_language: str = ''

    def load(self, settings):
        # Each attempt owns its engine. Do not touch the normal OCR cache or
        # invoke OCRProcessor's Japanese page-size override for this crop.
        if self.pp_language:
            return OCRFactory._create_ppocr(settings, self.pp_language)
        if self.model == ModelID.MANGA_OCR_MOBILE_ONNX:
            return OCRFactory._create_manga_ocr(settings)
        if self.model == ModelID.PORORO_ONNX:
            return OCRFactory._create_pororo_ocr(settings)
        if self.model == ModelID.PPOCR_V6_REC_VI:
            return OCRFactory._create_vietnamese_ocr(settings)
        raise ValueError('Unknown local OCR model.')


def retry_recognizers(block, language, *, default_selected, page_size=None, page_blocks=()):
    """Offer only supported alphabets and exclude the model responsible for the current reading."""
    if language == 'Auto':
        script = normalize_script(getattr(block, 'script', ''))
        if not is_supported_script(script):
            script = get_dominant_page_script(page_blocks)
        language = {'HanS': 'Chinese', 'HanT': 'Chinese', 'Latin': 'Latin',
                    'Cyrillic': 'Russian', 'Japanese': 'Japanese', 'Korean': 'Korean'}.get(script)
    code = language_codes.get(language, '')
    multilingual = LocalRecognizer('PP-OCRv6 (Multilingual)', ModelID.PPOCR_V6_REC_SMALL, code or 'en', 'multi')
    if language == 'English':
        routes = [LocalRecognizer('PP-OCRv5 (English)', ModelID.PPOCR_V5_REC_EN_MOBILE, code, 'en'),
                  multilingual, LocalRecognizer('PP-OCRv5 (Latin)', ModelID.PPOCR_V5_REC_LATIN_MOBILE, code, 'latin')]
    elif language == 'Japanese':
        routes = [LocalRecognizer('Manga OCR', ModelID.MANGA_OCR_MOBILE_ONNX, code),
                  LocalRecognizer('PP-OCRv6', ModelID.PPOCR_V6_REC_SMALL, code, 'ja')]
        if page_size and _should_override_to_ppocr(block, *page_size):
            routes.reverse()
    elif language == 'Korean':
        routes = [LocalRecognizer('PP-OCRv5 (Korean)', ModelID.PPOCR_V5_REC_KOREAN_MOBILE, code, 'ko'),
                  LocalRecognizer('Pororo OCR', ModelID.PORORO_ONNX, code)]
    elif language in ('Latin', 'French', 'Spanish', 'Italian', 'German', 'Dutch',
                      'Portuguese', 'Brazilian Portuguese'):
        routes = [LocalRecognizer('PP-OCRv5 (Latin)', ModelID.PPOCR_V5_REC_LATIN_MOBILE, code or 'en', 'latin'),
                  multilingual]
    elif language in ('Chinese', 'Traditional Chinese', 'Simplified Chinese'):
        routes = [LocalRecognizer('PP-OCRv6', ModelID.PPOCR_V6_REC_SMALL, code, 'ch')]
    elif language == 'Vietnamese':
        routes = [LocalRecognizer('PP-OCRv6 (Vietnamese)', ModelID.PPOCR_V6_REC_VI, code)]
    elif language == 'Russian':
        routes = [LocalRecognizer('PP-OCRv5 (Cyrillic)', ModelID.PPOCR_V5_REC_ESLAV_MOBILE, code, 'ru')]
    else:
        return []

    evidence = current_ocr_evidence(block)
    if evidence:
        return [route for route in routes if route.model.value != evidence['model']]
    if default_selected and language == 'Japanese' and page_size is None and block.text_class == 'text_free':
        # Older projects have no provenance; either Japanese recognizer might
        # have run for this box. Avoid offering the current one as an alternative.
        return []
    return routes[1:] if default_selected else routes


@dataclass(frozen=True)
class _DeviceSettings:
    use_gpu: bool

    def is_gpu_enabled(self):
        return self.use_gpu


def _original_crop(image, block):
    h, w = image.shape[:2]
    x1, y1, x2, y2, angle, _, _ = geometry_signature(block)
    if not angle:
        return image[max(0, math.floor(y1)):max(0, math.ceil(y2)),
                     max(0, math.floor(x1)):max(0, math.ceil(x2))].copy()
    points = region_polygon(block)
    size = math.ceil(x2 - x1), math.ceil(y2 - y1)
    if not clip_polygon(points, w, h) or size[0] * size[1] > 2 * w * h:
        raise ValueError('The selected region is outside the original page.')
    # PIL expects upper-left, lower-left, lower-right, upper-right.
    quad = tuple(coord for i in (0, 3, 2, 1) for coord in points[i])
    return np.array(Image.fromarray(image).transform(size, Image.Transform.QUAD, quad,
                    Image.Resampling.BILINEAR, fillcolor='white'))


def recognize_alternative(image, block, route: LocalRecognizer, use_gpu: bool):
    """Read the selected original pixels in isolation; publish only text and its evidence."""
    crop = _original_crop(image, block)
    if crop.size == 0:
        return '', None
    height, width = crop.shape[:2]
    region = TextBlock(text_bbox=np.array([0, 0, width, height]), source_lang=route.source_code,
                       script=block.script, text_class=block.text_class)
    annotate_blocks_with_heuristic_lines(crop, [region])
    # Always use our known line(s), even for a blank crop. This keeps the
    # recognizer from detecting other regions or loading an extra detector.
    if not region.lines:
        region.lines = [region.xyxy.tolist()]
    engine = route.load(_DeviceSettings(use_gpu))
    engine.process_image(crop, [region])
    text = region.text or ''
    evidence = current_ocr_evidence(region)
    if text and evidence is None:
        family = {ModelID.MANGA_OCR_MOBILE_ONNX: 'Manga OCR',
                  ModelID.PORORO_ONNX: 'Pororo OCR'}.get(route.model, 'PP-OCR')
        evidence = make_ocr_evidence(text, family, route.model.value)
    return text, evidence
