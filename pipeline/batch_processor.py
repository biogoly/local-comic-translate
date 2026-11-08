from __future__ import annotations

import os
import json
import requests
import logging
import traceback
import imkit as imk
import time
from copy import deepcopy
from typing import TYPE_CHECKING
from datetime import datetime
from typing import List
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QColor

from modules.detection.processor import TextBlockDetector
from modules.translation.processor import Translator
from modules.utils.textblock import sort_blk_list
from modules.utils.pipeline_config import get_config
from modules.utils.image_utils import generate_mask, get_smart_text_color
from modules.utils.language_utils import get_language_code, is_no_space_lang
from modules.utils.translator_utils import get_raw_translation, get_raw_text, format_translations, is_renderable_translation
from modules.rendering.render import (
    block_for_render_area, get_best_render_area, is_vertical_block, orient_new_lettering, pyside_word_wrap,
)
from modules.rendering.auto_bubble_fit import fit_new_text_item
from modules.utils.device import resolve_device
from modules.utils.manual_regions import merge_manual_regions, has_manual_translation
from modules.utils.region_policy import permits_region_operation, retained_layout_states, subtract_cleanup_protections
from modules.utils.exceptions import InsufficientCreditsException
from app.path_materialization import ensure_path_materialized
from app.ui.canvas.text_item import OutlineInfo, OutlineType
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.messages import Messages
from .cache_manager import CacheManager
from .block_detection import BlockDetectionHandler
from .inpainting import InpaintingHandler, call_inpaint_image
from modules.repair.protection import get_protection
from .ocr_handler import OCRHandler

if TYPE_CHECKING:
    from controller import ComicTranslate

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Handles batch processing of comic translation."""
    
    def __init__(
            self, 
            main_page: ComicTranslate, 
            cache_manager: CacheManager, 
            block_detection_handler: BlockDetectionHandler, 
            inpainting_handler: InpaintingHandler, 
            ocr_handler: OCRHandler 
        ):
        
        self.main_page = main_page
        self.cache_manager = cache_manager
        # Use shared handlers from the main pipeline
        self.block_detection = block_detection_handler
        self.inpainting = inpainting_handler
        self.ocr_handler = ocr_handler

    def skip_save(self, directory, timestamp, base_name, extension, archive_bname, image):
        logger.info("Skipping fallback translated image save for '%s'.", base_name)

    def emit_progress(self, index, total, step, steps, change_name):
        """Wrapper around main_page.progress_update.emit that logs a human-readable stage."""
        stage_map = {
            0: 'start-image',
            1: 'text-block-detection',
            2: 'ocr-processing',
            3: 'translation',
            5: 'pre-inpaint-setup',
            7: 'inpainting',
            9: 'text-rendering-prepare',
            10: 'save-and-finish',
        }
        stage_name = stage_map.get(step, f'stage-{step}')
        logger.info(f"Progress: image_index={index}/{total} step={step}/{steps} ({stage_name}) change_name={change_name}")
        self.main_page.progress_update.emit(index, total, step, steps, change_name)

    def log_skipped_image(self, directory, timestamp, image_path, reason="", full_traceback=""):
        # Deprecated: skip details are captured by batch reporting/UI signals.
        return

    def _is_cancelled(self) -> bool:
        worker = getattr(self.main_page, "current_worker", None)
        return bool(worker and worker.is_cancelled)

    def batch_process(self, selected_paths: List[str] = None):
        from pipeline.translation_job import BatchTranslationSession, speaker_region_ids
        timestamp = datetime.now().strftime("%b-%d-%Y_%I-%M-%S%p")
        image_list = selected_paths if selected_paths is not None else self.main_page.image_files
        session = getattr(self, '_prepared_translation_session', None)
        self._prepared_translation_session = None
        total_images = len(image_list)
        auto_fit_at_start = bool(getattr(self.main_page.render_settings(), "auto_bubble_fit", False))
        preserved_pages = {
            path for path in image_list
            if self.main_page.image_states.get(path, {}).get("viewer_state", {}).get("text_items_state")
            and not self.main_page.image_states.get(path, {}).get('blk_list')
        } if auto_fit_at_start else set()
        if session is None:
            session = BatchTranslationSession(self.main_page, [p for p in image_list if p not in preserved_pages], Translator)
        try:
            if self.main_page.file_handler.should_pre_materialize(image_list):
                count = self.main_page.file_handler.pre_materialize(image_list)
                logger.info("Batch pre-materialized %d paths before full-run processing.", count)
        except Exception:
            logger.debug("Batch pre-materialization failed; continuing lazily.", exc_info=True)

        for index, image_path in enumerate(image_list):
            if self._is_cancelled():
                return

            file_on_display = self.main_page.image_files[self.main_page.curr_img_idx]

            # index, step, total_steps, change_name
            self.emit_progress(index, total_images, 0, 10, True)

            if image_path in preserved_pages:
                # Redetection creates new region IDs. Preserve the entire
                # edited page rather than guessing matches and losing edits.
                reason = QCoreApplication.translate(
                    "Messages",
                    "Page already contains rendered text; use manual Render to add missing regions.",
                )
                logger.info("Preserving already-rendered page '%s': %s", image_path, reason)
                self.main_page.image_skipped.emit(image_path, "Existing Rendered Text", reason)
                self.emit_progress(index, total_images, 10, 10, False)
                continue

            settings_page = self.main_page.settings_page
            source_lang = self.main_page.image_states[image_path]['source_lang']
            target_lang = self.main_page.image_states[image_path]['target_lang']

            trg_lng_cd = get_language_code(target_lang)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0].strip()
            extension = os.path.splitext(image_path)[1]
            directory = os.path.dirname(image_path)

            archive_bname = ""
            for archive in self.main_page.file_handler.archive_info:
                images = archive['extracted_images']
                archive_path = archive['archive_path']

                for img_pth in images:
                    if img_pth == image_path:
                        directory = os.path.dirname(archive_path)
                        archive_bname = os.path.splitext(os.path.basename(archive_path))[0].strip()

            ensure_path_materialized(image_path)
            image = imk.read_image(image_path)

            # skip UI-skipped images
            state = self.main_page.image_states.get(image_path, {})
            if state.get('skip', False):
                self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                self.log_skipped_image(directory, timestamp, image_path, "User-skipped")
                continue

            previous_blocks = deepcopy(state.get('blk_list', []))
            previous_items = deepcopy(state.get('viewer_state', {}).get('text_items_state', []))
            legacy_manual_ids = {block.region_id for block in previous_blocks if has_manual_translation(block)}
            existing_layouts = retained_layout_states(previous_blocks, previous_items, retain_all=auto_fit_at_start)
            composed_image = image.copy()
            compose = getattr(getattr(self.main_page, 'image_ctrl', None), 'get_composited_page_image', None)
            if callable(compose):
                composed_image = compose(image_path)
                if composed_image is None or composed_image.shape != image.shape:
                    self.main_page.image_skipped.emit(image_path, 'Cleanup', 'Current page pixels are unavailable')
                    continue
                composed_image = composed_image.copy()

            # Text Block Detection
            self.emit_progress(index, total_images, 1, 10, False)
            if self._is_cancelled():
                return

            # Use the shared block detector from the handler
            if self.block_detection.block_detector_cache is None:
                self.block_detection.block_detector_cache = TextBlockDetector(settings_page)
            
            blk_list = self.block_detection.block_detector_cache.detect(image) or []
            retained_ids = set(existing_layouts) | speaker_region_ids(self.main_page, state.get('page_id'))
            blk_list = merge_manual_regions(previous_blocks, blk_list, retain_ids=retained_ids)

            self.emit_progress(index, total_images, 2, 10, False)
            if self._is_cancelled():
                return

            ocr_blocks = [block for block in blk_list if permits_region_operation(block, 'source')
                          and block.region_id not in legacy_manual_ids]
            self.block_detection.annotate_language_if_auto(image, ocr_blocks, source_lang)

            logger.info("Automatic workflow: processing all %d detected text blocks", len(blk_list))

            if blk_list:
                # Get ocr cache key for batch processing
                ocr_model = settings_page.get_tool_selection('ocr')
                device = resolve_device(settings_page.is_gpu_enabled())
                cache_key = self.cache_manager._get_ocr_cache_key(image, source_lang, ocr_model, device)
                # Use the shared OCR processor from the handler
                if ocr_blocks:
                    self.ocr_handler.ocr.initialize(self.main_page, source_lang)
                try:
                    if ocr_blocks:
                        self.ocr_handler.ocr.process(image, ocr_blocks)
                    # Cache the OCR results for potential future use
                    self.cache_manager._cache_ocr_results(cache_key, blk_list)
                    rtl = True if source_lang == 'Japanese' else False
                    blk_list = sort_blk_list(blk_list, rtl)
                    
                except InsufficientCreditsException:
                    raise
                except Exception as e:
                    # if it's a connection/network error, give a short message
                    if isinstance(e, requests.exceptions.ConnectionError):
                        err_msg = QCoreApplication.translate("Messages", "Unable to connect to the server.\nPlease check your internet connection.")
                    # if it's an HTTPError, try to pull the "error_description" field
                    elif isinstance(e, requests.exceptions.HTTPError):
                        status_code = e.response.status_code if e.response is not None else 500
                        if status_code >= 500:
                            err_msg = Messages.get_server_error_text(status_code, context='ocr')
                        else:
                            try:
                                err_json = e.response.json()
                                if "detail" in err_json and isinstance(err_json["detail"], dict):
                                    err_msg = err_json["detail"].get("error_description", str(e))
                                else:
                                    err_msg = err_json.get("error_description", str(e))
                            except Exception:
                                err_msg = str(e)
                    else:
                        err_msg = str(e)

                    logger.exception(f"OCR processing failed: {err_msg}")
                    reason = f"OCR: {err_msg}"
                    full_traceback = traceback.format_exc()
                    self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                    self.main_page.image_skipped.emit(image_path, "OCR", err_msg)
                    self.log_skipped_image(directory, timestamp, image_path, reason, full_traceback)
                    continue
            else:
                self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                self.main_page.image_skipped.emit(image_path, "Text Blocks", "")
                self.log_skipped_image(directory, timestamp, image_path, "No text blocks detected")
                continue

            self.emit_progress(index, total_images, 3, 10, False)
            if self._is_cancelled():
                return

            try:
                # Use the complete block list regardless of detector class.
                translation_blocks = [block for block in blk_list if permits_region_operation(block, 'translation')
                                      and block.region_id not in legacy_manual_ids]
                if translation_blocks:
                    session.translate(translation_blocks, image, source_lang, target_lang, image_path, self.cache_manager)
            except InsufficientCreditsException:
                raise
            except Exception as e:
                # if it's a connection/network error, give a short message
                if isinstance(e, requests.exceptions.ConnectionError):
                    err_msg = QCoreApplication.translate("Messages", "Unable to connect to the server.\nPlease check your internet connection.")
                # if it's an HTTPError, try to pull the "error_description" field
                elif isinstance(e, requests.exceptions.HTTPError):
                    status_code = e.response.status_code if e.response is not None else 500
                    if status_code >= 500:
                        err_msg = Messages.get_server_error_text(status_code, context='translation')
                    else:
                        try:
                            err_json = e.response.json()
                            if "detail" in err_json and isinstance(err_json["detail"], dict):
                                err_msg = err_json["detail"].get("error_description", str(e))
                            else:
                                err_msg = err_json.get("error_description", str(e))
                        except Exception:
                            err_msg = str(e)
                else:
                    err_msg = str(e)

                logger.exception(f"Translation failed: {err_msg}")
                reason = f"Translator: {err_msg}"
                full_traceback = traceback.format_exc()
                self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                self.main_page.image_skipped.emit(image_path, "Translator", err_msg)
                self.log_skipped_image(directory, timestamp, image_path, reason, full_traceback)
                continue

            if self._is_cancelled():
                return

            entire_raw_text = get_raw_text(blk_list)
            entire_translated_text = get_raw_translation(blk_list)

            # Parse JSON strings and check if they're empty objects or invalid
            try:
                raw_text_obj = json.loads(entire_raw_text)
                translated_text_obj = json.loads(entire_translated_text)
                
                if (not raw_text_obj) or (not translated_text_obj):
                    self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                    self.main_page.image_skipped.emit(image_path, "Translator", "")
                    self.log_skipped_image(directory, timestamp, image_path, "Translator: empty JSON")
                    continue
            except json.JSONDecodeError as e:
                # Handle invalid JSON
                error_message = str(e)
                reason = f"Translator: JSONDecodeError: {error_message}"
                logger.exception(reason)
                full_traceback = traceback.format_exc()
                self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                self.main_page.image_skipped.emit(image_path, "Translator", error_message)
                self.log_skipped_image(directory, timestamp, image_path, reason, full_traceback)
                continue

            export_settings = settings_page.get_export_settings()

            if export_settings['export_raw_text']:
                path = os.path.join(directory, f"comic_translate_{timestamp}", "raw_texts", archive_bname)
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                with open(
                    os.path.join(path, os.path.splitext(os.path.basename(image_path))[0] + "_raw.json"),
                    'w',
                    encoding='UTF-8',
                ) as file:
                    file.write(entire_raw_text)

            if export_settings['export_translated_text']:
                path = os.path.join(directory, f"comic_translate_{timestamp}", "translated_texts", archive_bname)
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                with open(
                    os.path.join(path, os.path.splitext(os.path.basename(image_path))[0] + "_translated.json"),
                    'w',
                    encoding='UTF-8',
                ) as file:
                    file.write(entire_translated_text)

            self.emit_progress(index, total_images, 5, 10, False)
            if self._is_cancelled():
                return

            # Clean Image of text
            config = get_config(settings_page)
            render_settings = self.main_page.render_settings()
            auto_bubble_fit = bool(getattr(render_settings, "auto_bubble_fit", False))
            # Keep an untouched source for typography. The inpainter may modify
            # its input in place; cleanup masks/cleaned pixels are not boundaries.
            original_for_fitting = image.copy() if auto_bubble_fit else None
            if original_for_fitting is not None:
                original_for_fitting.setflags(write=False)
            
            # Filter blocks to only inpaint if both OCR text and Translation are non-empty
            # and the translation will actually be rendered (single-character translations
            # like an echoed "?" are skipped at render time).
            inpaint_blk_list = [
                blk for blk in blk_list
                if blk.text and blk.text.strip() and blk.translation and blk.translation.strip()
                and is_renderable_translation(blk.translation)
                and permits_region_operation(blk, 'cleanup') and blk.region_id not in legacy_manual_ids
            ]
            
            logger.info("pre-inpaint: generating mask (inpaint_blk_list=%d blocks out of %d)", len(inpaint_blk_list), len(blk_list))
            t0 = time.time()
            mask = subtract_cleanup_protections(generate_mask(image, inpaint_blk_list), blk_list)
            t1 = time.time()
            logger.info("pre-inpaint: mask generated in %.2fs (mask shape=%s)", t1 - t0, getattr(mask, 'shape', None))

            self.emit_progress(index, total_images, 7, 10, False)
            if self._is_cancelled():
                return

            if inpaint_blk_list:
                inpaint_input_img = call_inpaint_image(
                    self.inpainting,
                    composed_image,
                    mask,
                    config,
                    blk_list=inpaint_blk_list,
                    automatic=True,
                    protected_blocks=blk_list,
                    pixel_protection=get_protection(self.main_page, image_path, composed_image.shape),
                )
                inpaint_input_img = imk.convert_scale_abs(inpaint_input_img)
                patches = self.inpainting.get_inpainted_patches(mask, inpaint_input_img)
            else:
                inpaint_input_img = composed_image.copy()
                patches = []

            if export_settings['export_inpainted_image']:
                path = os.path.join(directory, f"comic_translate_{timestamp}", "cleaned_images", archive_bname)
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                imk.write_image(os.path.join(path, f"{base_name}_cleaned{extension}"), inpaint_input_img)

            self.emit_progress(index, total_images, 9, 10, False)
            if self._is_cancelled():
                return

            # Text Rendering
            upper_case = render_settings.upper_case
            outline = render_settings.outline
            format_translations([block for block in blk_list if permits_region_operation(block, 'translation')
                                 and block.region_id not in legacy_manual_ids], trg_lng_cd, upper_case=upper_case)
            layout_blocks = [block_for_render_area(blk) for blk in blk_list]
            fit_blocks = {
                id(layout): original.deep_copy() for original, layout in zip(blk_list, layout_blocks)
            } if auto_bubble_fit else {}
            get_best_render_area([block for block in layout_blocks if block.region_id not in existing_layouts
                                  and permits_region_operation(block, 'layout')],
                                 image, inpaint_input_img)

            font = render_settings.font_family
            setting_font_color = QColor(render_settings.color)

            max_font_size = render_settings.max_font_size
            min_font_size = render_settings.min_font_size
            line_spacing = float(render_settings.line_spacing) 
            outline_width = float(render_settings.outline_width)
            outline_color = QColor(render_settings.outline_color) if outline else None
            bold = render_settings.bold
            italic = render_settings.italic
            underline = render_settings.underline
            alignment_id = render_settings.alignment_id
            alignment = self.main_page.button_to_alignment[alignment_id]
            direction = render_settings.direction
                
            text_items_state = []
            for blk in layout_blocks:
                if blk.region_id in existing_layouts:
                    text_items_state.append(deepcopy(existing_layouts[blk.region_id]))
                    continue
                if not permits_region_operation(blk, 'layout'):
                    # Absence of lettering is also a layout decision.
                    text_items_state.extend(deepcopy(item) for item in previous_items
                                            if item.get('region_id') == blk.region_id)
                    continue
                x1, y1, block_width, block_height = blk.xywh

                translation = blk.translation
                if not is_renderable_translation(translation):
                    continue
                logical_translation = translation
                
                # Determine if this block should use vertical rendering
                vertical = is_vertical_block(blk, trg_lng_cd)

                translation, font_size, rendered_width, rendered_height = pyside_word_wrap(
                    translation, 
                    font, 
                    block_width, 
                    block_height,
                    line_spacing, 
                    outline_width, 
                    bold, 
                    italic, 
                    underline,
                    alignment, 
                    direction, 
                    max_font_size, 
                    min_font_size,
                    vertical,
                    is_no_space_lang(trg_lng_cd),
                    return_metrics=True
                )
                
                # Smart Color Override
                font_color = get_smart_text_color(blk.font_color, setting_font_color)

                # Use TextItemProperties for consistent text item creation
                text_props = TextItemProperties(
                    text=translation,
                    region_id=blk.region_id,
                    font_family=font,
                    font_size=font_size,
                    text_color=font_color,
                    alignment=alignment,
                    line_spacing=line_spacing,
                    outline_color=outline_color,
                    outline_width=outline_width,
                    bold=bold,
                    italic=italic,
                    underline=underline,
                    position=(x1, y1),
                    scale=1.0,
                    transform_origin=blk.tr_origin_point,
                    width=rendered_width,
                    height=rendered_height,
                    direction=direction,
                    vertical=vertical,
                    selection_outlines=[
                        OutlineInfo(0, len(translation), 
                        outline_color, 
                        outline_width, 
                        OutlineType.Full_Document)
                    ] if outline else [],
                )
                orient_new_lettering(text_props, blk)
                if auto_bubble_fit:
                    text_props = fit_new_text_item(
                        text_props, fit_blocks[id(blk)], original_for_fitting,
                        render_settings, source_text=logical_translation,
                    )
                text_items_state.append(text_props.to_dict())

            session.validate(image_path)
            self.main_page.image_states[image_path]['viewer_state'].update({
                'text_items_state': text_items_state
                })
            
            self.main_page.image_states[image_path]['viewer_state'].update({
                'push_to_stack': True
                })
            
            self.emit_progress(index, total_images, 9, 10, False)
            if self._is_cancelled():
                return

            # Saving blocks with texts to history
            self.main_page.image_states[image_path].update({
                'blk_list': blk_list                   
            })

            # Publish the page only after every patch and text item is ready. The
            # surrounding macro makes automatic cleanup and translated text a
            # single Undo/Redo operation on the displayed page. A page containing
            # no rendered text must not create an empty undo entry.
            if patches or text_items_state:
                self.main_page.batch_page_edit_started.emit(image_path)
                if patches:
                    self.main_page.patches_processed.emit(patches, image_path)
                self.main_page.render_state_ready.emit(image_path)
                self.main_page.batch_page_edit_finished.emit(image_path)
            else:
                self.main_page.render_state_ready.emit(image_path)

            if image_path == file_on_display:
                self.main_page.blk_list = blk_list

            self.emit_progress(index, total_images, 10, 10, False)
