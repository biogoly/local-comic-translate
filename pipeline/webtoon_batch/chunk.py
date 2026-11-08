from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import imkit as imk
import numpy as np
import requests
from PySide6.QtCore import QCoreApplication

from app.ui.messages import Messages
from modules.detection.processor import TextBlockDetector
from modules.translation.processor import Translator
from modules.utils.device import resolve_device
from modules.utils.exceptions import InsufficientCreditsException
from modules.utils.image_utils import generate_mask
from modules.utils.pipeline_config import get_config, get_inpainter_backend, inpaint_map
from modules.utils.textblock import TextBlock, sort_blk_list
from modules.utils.translator_utils import is_renderable_translation
from modules.utils.region_policy import (
    permits_region_operation, subtract_cleanup_protections, shifted_region_copy,
    should_preserve_region,
)
from pipeline.inpainting import call_inpaint_image
from modules.repair.protection import PixelProtectionError, get_protection

if TYPE_CHECKING:
    from .processor import WebtoonBatchProcessor

logger = logging.getLogger(__name__)


def _record_pixel_protection(main, record, pixels):
    """Map a physical page's protection into this virtual page's coordinates."""
    vpage = record.get("vpage")
    if vpage is None:
        return get_protection(main, record["path"], pixels.shape)
    protected = get_protection(main, record["path"], (vpage.physical_height, vpage.physical_width))
    if protected is None:
        return None
    top, bottom = int(vpage.crop_top), int(vpage.crop_bottom)
    if top < 0 or bottom > protected.shape[0] or bottom <= top:
        raise PixelProtectionError("Protected virtual page bounds are invalid.")
    cropped = protected[top:bottom, :pixels.shape[1]]
    if cropped.shape != pixels.shape[:2]:
        raise PixelProtectionError("Protection dimensions do not match this virtual page.")
    return cropped.copy()


class ChunkMixin:
    @staticmethod
    def _union_xyxy(a: List[float], b: List[float]) -> List[float]:
        return [
            min(float(a[0]), float(b[0])),
            min(float(a[1]), float(b[1])),
            max(float(a[2]), float(b[2])),
            max(float(a[3]), float(b[3])),
        ]

    @staticmethod
    def _shift_xyxy(xyxy: List[float], dx: float, dy: float) -> List[float]:
        return [
            float(xyxy[0]) + dx,
            float(xyxy[1]) + dy,
            float(xyxy[2]) + dx,
            float(xyxy[3]) + dy,
        ]

    def _is_cancelled(self: WebtoonBatchProcessor) -> bool:
        worker = getattr(self.main_page, "current_worker", None)
        return bool(worker and worker.is_cancelled)

    def _get_page_scene_offset(
        self: WebtoonBatchProcessor, page_index: int
    ) -> int:
        webtoon_manager = self.main_page.image_viewer.webtoon_manager
        if (
            webtoon_manager
            and page_index < len(getattr(webtoon_manager, "image_positions", []))
        ):
            return int(webtoon_manager.image_positions[page_index])
        return 0

    def _ensure_detector(self: WebtoonBatchProcessor):
        if self.block_detection.block_detector_cache is None:
            self.block_detection.block_detector_cache = TextBlockDetector(
                self.main_page.settings_page
            )

    def _ensure_inpainter(self: WebtoonBatchProcessor):
        settings_page = self.main_page.settings_page
        inpainter_key = settings_page.get_tool_selection("inpainter")
        if (
            self.inpainting.inpainter_cache is None
            or self.inpainting.cached_inpainter_key != inpainter_key
        ):
            InpainterClass = inpaint_map[inpainter_key]
            backend = get_inpainter_backend(inpainter_key)
            device = resolve_device(settings_page.is_gpu_enabled(), backend=backend)
            self.inpainting.inpainter_cache = InpainterClass(device, backend=backend)
            self.inpainting.cached_inpainter_key = inpainter_key

    def _detect_blocks_for_page(
        self: WebtoonBatchProcessor, image: np.ndarray
    ) -> List[TextBlock]:
        self._ensure_detector()
        blocks = self.block_detection.block_detector_cache.detect(image)
        return blocks or []

    def _extract_error_message(
        self: WebtoonBatchProcessor, error: Exception, context: str
    ) -> str:
        if isinstance(error, requests.exceptions.ConnectionError):
            return QCoreApplication.translate(
                "Messages",
                "Unable to connect to the server.\nPlease check your internet connection.",
            )
        if isinstance(error, requests.exceptions.HTTPError):
            status_code = error.response.status_code if error.response is not None else 500
            if status_code >= 500:
                return Messages.get_server_error_text(status_code, context=context)
            try:
                err_json = error.response.json()
                if "detail" in err_json and isinstance(err_json["detail"], dict):
                    return err_json["detail"].get("error_description", str(error))
                return err_json.get("error_description", str(error))
            except Exception:
                return str(error)
        return str(error)

    def _run_ocr_on_blocks(
        self: WebtoonBatchProcessor,
        image: np.ndarray,
        blocks: List[TextBlock],
        source_lang: str,
        affected_paths: List[str],
        reason: str,
        sort_after: bool = True,
    ) -> List[TextBlock]:
        if not blocks:
            return blocks

        source_lang_en = self.main_page.lang_mapping.get(source_lang, source_lang)
        eligible = [block for block in blocks if permits_region_operation(block, 'source')
                    and block.region_id not in getattr(self, '_review_manual_ids', ())]
        bounded = []
        for block in eligible:
            if should_preserve_region(block) and (block.xyxy[1] < 0 or block.xyxy[3] > image.shape[0]):
                logger.info('Retaining source for region %s: full seam OCR context unavailable', block.region_id)
            else:
                bounded.append(block)
        self.block_detection.annotate_language_if_auto(image, bounded, source_lang_en)

        if bounded:
            self.ocr_handler.ocr.initialize(self.main_page, source_lang)
        try:
            if bounded:
                self.ocr_handler.ocr.process(image, bounded)
            if sort_after:
                rtl = source_lang_en == "Japanese"
                return sort_blk_list(blocks, rtl)
            return blocks
        except InsufficientCreditsException:
            raise
        except Exception as error:
            err_msg = self._extract_error_message(error, context="ocr")
            logger.exception("OCR failed (%s): %s", reason, err_msg)
            for path in affected_paths:
                self.main_page.image_skipped.emit(path, reason, err_msg)
            return blocks

    def _run_translation_on_blocks(
        self: WebtoonBatchProcessor,
        image: np.ndarray,
        blocks: List[TextBlock],
        source_lang: str,
        target_lang: str,
        image_path: str,
    ) -> None:
        if not blocks:
            return
        eligible = [block for block in blocks if permits_region_operation(block, 'translation')
                    and block.region_id not in getattr(self, '_review_manual_ids', ())]
        try:
            if eligible:
                from pipeline.translation_job import BatchTranslationSession
                session = getattr(self, '_translation_session', None)
                if session is None:
                    session = BatchTranslationSession(self.main_page, [image_path], Translator)
                session.translate(eligible, image, source_lang, target_lang, image_path, self.cache_manager)
        except InsufficientCreditsException:
            raise
        except Exception as error:
            err_msg = self._extract_error_message(error, context="translation")
            logger.exception("Translation failed for %s: %s", image_path, err_msg)
            self.main_page.image_skipped.emit(image_path, "Translation", err_msg)
            # Retain prior targets; a failed response never erases completed work.
            raise

    def _inpaint_image_with_blocks(
        self: WebtoonBatchProcessor,
        image: np.ndarray,
        blocks: List[TextBlock],
        protected_blocks=None,
        pixel_protection=None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not blocks:
            return None, None
        config = get_config(self.main_page.settings_page)
        mask_blocks: List[TextBlock] = []
        img_h, img_w = image.shape[:2]
        for block in blocks:
            if (not permits_region_operation(block, 'cleanup')
                    or block.region_id in getattr(self, '_review_manual_ids', ())):
                continue
            mask_block = block.deep_copy()
            if (
                not getattr(mask_block, "text", "")
                or not mask_block.text.strip()
                or not getattr(mask_block, "translation", "")
                or not mask_block.translation.strip()
                or not is_renderable_translation(mask_block.translation)
            ):
                continue
            x1, y1, x2, y2 = [float(v) for v in mask_block.xyxy]
            x1_i = int(np.floor(x1))
            y1_i = int(np.floor(y1))
            x2_i = int(np.ceil(x2))
            y2_i = int(np.ceil(y2))
            x1_i = max(0, min(x1_i, img_w))
            y1_i = max(0, min(y1_i, img_h))
            x2_i = max(0, min(x2_i, img_w))
            y2_i = max(0, min(y2_i, img_h))
            if x2_i <= x1_i or y2_i <= y1_i:
                continue
            mask_block.xyxy = [x1_i, y1_i, x2_i, y2_i]
            if mask_block.bubble_xyxy is not None:
                bx1, by1, bx2, by2 = [float(v) for v in mask_block.bubble_xyxy]
                mask_block.bubble_xyxy = [
                    max(0, min(int(np.floor(bx1)), img_w)),
                    max(0, min(int(np.floor(by1)), img_h)),
                    max(0, min(int(np.ceil(bx2)), img_w)),
                    max(0, min(int(np.ceil(by2)), img_h)),
                ]
            mask_blocks.append(mask_block)
        if not mask_blocks:
            return None, None
        protection_regions = list(protected_blocks or ()) + list(blocks)
        mask = subtract_cleanup_protections(generate_mask(image, mask_blocks), protection_regions)
        if mask is None or not np.any(mask):
            return None, None
        inpainted = call_inpaint_image(self.inpainting, image, mask, config, blk_list=mask_blocks,
                                      automatic=True, protected_blocks=protection_regions,
                                      pixel_protection=pixel_protection)
        inpainted = imk.convert_scale_abs(inpainted)
        return mask, inpainted

    def _extract_page_patches_from_mask(
        self: WebtoonBatchProcessor,
        mask: np.ndarray,
        inpainted: np.ndarray,
        page_index: int,
        file_path: str,
        y_offset: int = 0,
    ) -> List[Dict]:
        contours, _ = imk.find_contours(mask)
        if not contours:
            return []

        scene_offset_y = self._get_page_scene_offset(page_index)
        patches: List[Dict] = []
        for contour in contours:
            x, y, w, h = [int(v) for v in imk.bounding_rect(contour)]
            if w <= 0 or h <= 0:
                continue
            patch = inpainted[y : y + h, x : x + w]
            physical_y = y + int(y_offset)
            patches.append(
                {
                    "bbox": [x, physical_y, w, h],
                    "image": patch.copy(),
                    "page_index": page_index,
                    "file_path": file_path,
                    "scene_pos": [x, physical_y + scene_offset_y],
                }
            )
        return patches

    def _match_split_blocks(
        self: WebtoonBatchProcessor,
        top_blocks: List[TextBlock],
        top_height: int,
        bottom_blocks: List[TextBlock],
    ) -> List[Tuple[int, int]]:
        if not top_blocks or not bottom_blocks:
            return []

        edge = float(max(10, self.edge_threshold))
        top_candidates = []
        bottom_candidates = []

        for idx, blk in enumerate(top_blocks):
            x1, y1, x2, y2 = [float(v) for v in blk.xyxy]
            if y2 >= top_height - edge and y1 < top_height:
                top_candidates.append((idx, x1, y1, x2, y2))

        for idx, blk in enumerate(bottom_blocks):
            x1, y1, x2, y2 = [float(v) for v in blk.xyxy]
            if y1 <= edge and y2 > 0:
                bottom_candidates.append((idx, x1, y1, x2, y2))

        if not top_candidates or not bottom_candidates:
            return []

        candidate_pairs = []
        for top_idx, tx1, ty1, tx2, ty2 in top_candidates:
            top_w = max(1.0, tx2 - tx1)
            top_cx = 0.5 * (tx1 + tx2)
            top_edge_dist = abs(top_height - ty2)

            for bot_idx, bx1, by1, bx2, by2 in bottom_candidates:
                bot_w = max(1.0, bx2 - bx1)
                bot_cx = 0.5 * (bx1 + bx2)
                bot_edge_dist = abs(by1)

                width_ratio = max(top_w, bot_w) / max(1.0, min(top_w, bot_w))
                if width_ratio > 2.2:
                    continue
                if top_edge_dist > 1.6 * edge or bot_edge_dist > 1.6 * edge:
                    continue

                inter_x = max(0.0, min(tx2, bx2) - max(tx1, bx1))
                overlap_ratio = inter_x / max(1.0, min(top_w, bot_w))
                center_ratio = abs(top_cx - bot_cx) / max(1.0, max(top_w, bot_w))
                if overlap_ratio < 0.35 and center_ratio > 0.35:
                    continue

                score = (
                    2.6 * overlap_ratio
                    - 0.75 * center_ratio
                    - 0.35 * abs(width_ratio - 1.0)
                    - (top_edge_dist + bot_edge_dist) / max(1.0, 3.0 * edge)
                )
                if score <= 0.0:
                    continue

                candidate_pairs.append((score, top_idx, bot_idx))

        if not candidate_pairs:
            return []

        candidate_pairs.sort(key=lambda item: item[0], reverse=True)
        matched_top = set()
        matched_bottom = set()
        matches: List[Tuple[int, int]] = []

        for _, top_idx, bot_idx in candidate_pairs:
            if top_idx in matched_top or bot_idx in matched_bottom:
                continue
            matched_top.add(top_idx)
            matched_bottom.add(bot_idx)
            matches.append((top_idx, bot_idx))

        return matches

    def _build_stitched_pair(
        self: WebtoonBatchProcessor, top_image: np.ndarray, bottom_image: np.ndarray
    ) -> np.ndarray:
        top_h, top_w = top_image.shape[:2]
        bottom_h, bottom_w = bottom_image.shape[:2]
        stitched_width = max(top_w, bottom_w)
        stitched = np.full((top_h + bottom_h, stitched_width, 3), 255, dtype=np.uint8)
        stitched[:top_h, :top_w] = top_image
        stitched[top_h : top_h + bottom_h, :bottom_w] = bottom_image
        return stitched

    def _compute_union_crop(
        self: WebtoonBatchProcessor,
        boxes: List[List[float]],
        image_shape: Tuple[int, int, int],
        pad_x: int,
        pad_y: int,
    ) -> Optional[List[int]]:
        if not boxes:
            return None
        h, w = image_shape[:2]
        x1 = int(max(0, min(box[0] for box in boxes) - pad_x))
        y1 = int(max(0, min(box[1] for box in boxes) - pad_y))
        x2 = int(min(w, max(box[2] for box in boxes) + pad_x))
        y2 = int(min(h, max(box[3] for box in boxes) + pad_y))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _localize_blocks_to_crop(
        self: WebtoonBatchProcessor, blocks: List[TextBlock], crop_xyxy: List[int]
    ) -> List[TextBlock]:
        crop_x1, crop_y1, _, _ = crop_xyxy
        localized = []
        for block in blocks:
            local_block = block.deep_copy()
            local_block.xyxy = self._shift_xyxy(block.xyxy, -float(crop_x1), -float(crop_y1))
            if local_block.bubble_xyxy is not None:
                local_block.bubble_xyxy = self._shift_xyxy(
                    local_block.bubble_xyxy, -float(crop_x1), -float(crop_y1)
                )
            localized.append(local_block)
        return localized

    def _extract_seam_patches_from_mask(
        self: WebtoonBatchProcessor,
        mask: np.ndarray,
        inpainted_crop: np.ndarray,
        crop_xyxy: List[int],
        top_page_index: int,
        bottom_page_index: int,
        top_global_page_index: int,
        bottom_global_page_index: int,
        top_path: str,
        bottom_path: str,
        top_y_offset: int,
        bottom_y_offset: int,
        top_image: np.ndarray,
        bottom_image: np.ndarray,
    ) -> Dict[int, List[Dict]]:
        contours, _ = imk.find_contours(mask)
        if not contours:
            return {top_page_index: [], bottom_page_index: []}

        crop_x1, crop_y1, _, _ = crop_xyxy
        top_h, top_w = top_image.shape[:2]
        bottom_h, bottom_w = bottom_image.shape[:2]
        seam_y = top_h

        top_scene_offset = self._get_page_scene_offset(top_global_page_index)
        bottom_scene_offset = self._get_page_scene_offset(bottom_global_page_index)

        top_patches: List[Dict] = []
        bottom_patches: List[Dict] = []

        for contour in contours:
            x, y, w, h = [int(v) for v in imk.bounding_rect(contour)]
            if w <= 0 or h <= 0:
                continue

            gx1 = crop_x1 + x
            gy1 = crop_y1 + y
            gx2 = gx1 + w
            gy2 = gy1 + h

            tx1 = max(0, gx1)
            tx2 = min(top_w, gx2)
            ty1 = max(0, gy1)
            ty2 = min(top_h, gy2)
            if tx2 > tx1 and ty2 > ty1:
                sx1 = tx1 - crop_x1
                sx2 = tx2 - crop_x1
                sy1 = ty1 - crop_y1
                sy2 = ty2 - crop_y1
                patch = inpainted_crop[sy1:sy2, sx1:sx2]
                top_physical_y = ty1 + int(top_y_offset)
                top_patches.append(
                    {
                        "bbox": [tx1, top_physical_y, tx2 - tx1, ty2 - ty1],
                        "image": patch.copy(),
                        "page_index": top_global_page_index,
                        "file_path": top_path,
                        "scene_pos": [tx1, top_physical_y + top_scene_offset],
                    }
                )

            bx1 = max(0, gx1)
            bx2 = min(bottom_w, gx2)
            by1 = max(seam_y, gy1)
            by2 = min(seam_y + bottom_h, gy2)
            if bx2 > bx1 and by2 > by1:
                sx1 = bx1 - crop_x1
                sx2 = bx2 - crop_x1
                sy1 = by1 - crop_y1
                sy2 = by2 - crop_y1
                patch = inpainted_crop[sy1:sy2, sx1:sx2]
                page_local_y = (by1 - seam_y) + int(bottom_y_offset)
                bottom_patches.append(
                    {
                        "bbox": [bx1, page_local_y, bx2 - bx1, by2 - by1],
                        "image": patch.copy(),
                        "page_index": bottom_global_page_index,
                        "file_path": bottom_path,
                        "scene_pos": [bx1, page_local_y + bottom_scene_offset],
                    }
                )

        return {top_page_index: top_patches, bottom_page_index: bottom_patches}

    def _process_seam_job_ocr_and_inpaint(
        self: WebtoonBatchProcessor,
        seam_job,
        page_records: List[Dict],
    ) -> Dict[int, List[Dict]]:
        top_record = page_records[seam_job.top_page_index]
        bottom_record = page_records[seam_job.bottom_page_index]
        top_image = top_record.get("composed_image", top_record.get("image"))
        bottom_image = bottom_record.get("composed_image", bottom_record.get("image"))
        if top_image is None or bottom_image is None:
            return {seam_job.top_page_index: [], seam_job.bottom_page_index: []}

        owner_blocks = [match.owner_block for match in seam_job.matches]
        if not owner_blocks:
            return {seam_job.top_page_index: [], seam_job.bottom_page_index: []}

        stitched = self._build_stitched_pair(top_image, bottom_image)
        crop_xyxy = self._compute_union_crop(
            [list(block.xyxy) for block in owner_blocks],
            stitched.shape,
            pad_x=self.seam_crop_pad_x,
            pad_y=self.seam_crop_pad_y,
        )
        if crop_xyxy is None:
            return {seam_job.top_page_index: [], seam_job.bottom_page_index: []}

        x1, y1, x2, y2 = crop_xyxy
        seam_crop = stitched[y1:y2, x1:x2].copy()
        seam_blocks_local = self._localize_blocks_to_crop(owner_blocks, crop_xyxy)

        # OCR has already been performed by the unified per-current-record pass.
        processed_local_blocks = seam_blocks_local

        protection_regions = list(top_record.get('protection_blocks', ()))
        protection_regions += [shifted_region_copy(block, dy=top_image.shape[0])
                               for block in bottom_record.get('protection_blocks', ())]
        protection_regions = [shifted_region_copy(block, -x1, -y1) for block in protection_regions]

        protected = []
        for record, pixels in ((top_record, top_image), (bottom_record, bottom_image)):
            protected.append(_record_pixel_protection(getattr(self, "main_page", None), record, pixels))
        seam_protection = None
        if any(value is not None for value in protected):
            stitched_protection = np.zeros(stitched.shape[:2], dtype=np.uint8)
            if protected[0] is not None:
                stitched_protection[:top_image.shape[0], :top_image.shape[1]] = protected[0]
            if protected[1] is not None:
                stitched_protection[top_image.shape[0]:top_image.shape[0]+bottom_image.shape[0],
                                     :bottom_image.shape[1]] = protected[1]
            seam_protection = stitched_protection[y1:y2, x1:x2].copy()
        mask, inpainted_crop = self._inpaint_image_with_blocks(
            seam_crop, processed_local_blocks, protected_blocks=protection_regions,
            pixel_protection=seam_protection,
        )
        if mask is None or inpainted_crop is None:
            return {seam_job.top_page_index: [], seam_job.bottom_page_index: []}

        # Keep the in-flight composed records current before processing the
        # next virtual page; queued GUI patch publication may not have run yet.
        stitched[y1:y2, x1:x2] = inpainted_crop
        top_record['composed_image'] = stitched[:top_image.shape[0], :top_image.shape[1]].copy()
        bottom_record['composed_image'] = stitched[top_image.shape[0]:top_image.shape[0]+bottom_image.shape[0],
                                                   :bottom_image.shape[1]].copy()

        return self._extract_seam_patches_from_mask(
            mask=mask,
            inpainted_crop=inpainted_crop,
            crop_xyxy=crop_xyxy,
            top_page_index=seam_job.top_page_index,
            bottom_page_index=seam_job.bottom_page_index,
            top_global_page_index=int(top_record["global_index"]),
            bottom_global_page_index=int(bottom_record["global_index"]),
            top_path=top_record["path"],
            bottom_path=bottom_record["path"],
            top_y_offset=int(top_record.get("y_offset", 0)),
            bottom_y_offset=int(bottom_record.get("y_offset", 0)),
            top_image=top_image,
            bottom_image=bottom_image,
        )
