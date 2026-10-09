"""Vietnamese PP-OCRv6 recognition with a matched, full tone-mark alphabet.

Reuses the existing detection/cropping/block workflow. Only explicitly selected
Vietnamese uses this community fine-tune; stock Latin recognition is untouched.
"""
from __future__ import annotations

import numpy as np

from modules.utils.device import get_providers
from modules.utils.download import ModelDownloader, ModelID
from modules.utils.onnx import make_session
from .ppocr.engine import PPOCRv5Engine, _make_ppocr_session_options
from .ppocr.postprocessing import CTCLabelDecoder
from .ppocr.preprocessing import rec_resize_norm


def vietnamese_rec_width(crop: np.ndarray) -> int:
    """Use the fine-tune's qualified 320/640 widths, not arbitrary widths."""
    height, width = crop.shape[:2]
    return 320 if width * 48 <= max(1, height) * 320 else 640


class VietnameseOCREngine(PPOCRv5Engine):
    def initialize(self, device='cpu', det_model='mobile', use_text_lines=True):
        self.device = device
        self.det_model = det_model
        self.use_text_lines = use_text_lines
        # Bounded batches keep long comic pages from creating large allocations.
        self.rec_batch_size = 4
        self.rec_threads = 3
        paths = ModelDownloader.file_path_map(ModelID.PPOCR_V6_REC_VI)
        self.decoder = CTCLabelDecoder(dict_path=paths['charset.txt'])
        self.rec_sess = make_session(
            paths['recognition.onnx'], providers=get_providers(device),
            sess_options=_make_ppocr_session_options(self.rec_threads),
        )
        # This pair is exactly 228 dictionary entries + CTC blank + space.
        # Fail clearly if assets drift instead of silently decoding wrong glyphs.
        if len(self.decoder.dict_chars) != 228 or self.rec_sess.get_outputs()[0].shape[-1] != 230:
            self.rec_sess = None
            self.decoder = None
            raise RuntimeError('Vietnamese OCR model and character dictionary do not match.')

    def _rec_infer(self, crops):
        assert self.rec_sess is not None and self.decoder is not None
        texts, confidences = [''] * len(crops), [0.0] * len(crops)
        buckets: dict[int, list[int]] = {}
        for index, crop in enumerate(crops):
            buckets.setdefault(vietnamese_rec_width(crop), []).append(index)
        input_name = self.rec_sess.get_inputs()[0].name
        output_name = self.rec_sess.get_outputs()[0].name
        for width, indices in buckets.items():
            for start in range(0, len(indices), self.rec_batch_size):
                batch_indices = indices[start:start + self.rec_batch_size]
                batch = np.stack([
                    rec_resize_norm(crops[index], self.rec_img_shape, width / 48)
                    for index in batch_indices
                ]).astype(np.float32)
                logits = self.rec_sess.run([output_name], {input_name: batch})[0]
                if (logits.ndim != 3 or logits.shape[-1] != 230
                        or logits.shape[0] != len(batch_indices)):
                    raise RuntimeError('Unexpected Vietnamese OCR output shape.')
                decoded, scores = self.decoder(logits, prob_threshold=0.0)
                for index, text, score in zip(batch_indices, decoded, scores):
                    texts[index], confidences[index] = text, float(score)
        return texts, confidences
