from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from PySide6 import QtCore

from modules.detection.processor import TextBlockDetector
from modules.ocr.processor import OCRProcessor
from modules.rendering.render import pyside_word_wrap, is_vertical_block, get_best_render_area
from modules.translation.processor import Translator
from modules.utils.common_utils import is_close
from modules.utils.device import resolve_device
from modules.utils.language_utils import get_language_code, is_no_space_lang
from modules.utils.language_utils import to_canonical_language_name
from modules.utils.manual_regions import has_manual_translation, merge_manual_regions
from modules.utils.region_policy import is_protected, permits_region_operation
from modules.utils.pipeline_config import validate_ocr, validate_translator
from modules.utils.textblock import sort_blk_list
from modules.utils.translator_utils import (
    format_translations,
    is_renderable_translation,
    is_there_text,
    set_upper_case,
)
from pipeline.webtoon_utils import (
    filter_and_convert_visible_blocks,
    find_block_page_index,
    get_first_visible_block,
    get_visible_text_items,
    restore_original_block_coordinates,
)

if TYPE_CHECKING:
    from app.ui.canvas.text_item import TextBlockItem
    from controller import ComicTranslate
    from modules.utils.textblock import TextBlock


class ManualWorkflowController:
    def __init__(self, main: ComicTranslate) -> None:
        self.main = main

    def _current_file_path(self) -> str | None:
        if 0 <= self.main.curr_img_idx < len(self.main.image_files):
            return self.main.image_files[self.main.curr_img_idx]
        return None

    def _selected_page_paths(self) -> list[str]:
        return self.main.get_selected_page_paths()

    def _load_page_image(self, file_path: str):
        img = self.main.image_data.get(file_path)
        if img is None:
            img = self.main.image_ctrl.load_image(file_path)
            if img is not None:
                self.main.image_data[file_path] = img
        return img

    def _prepare_multi_page_context(self, selected_paths: list[str]) -> dict[str, Any]:
        current_file = self._current_file_path()
        current_page_unloaded = False
        if self.main.webtoon_mode:
            manager = getattr(self.main.image_viewer, "webtoon_manager", None)
            scene_mgr = getattr(manager, "scene_item_manager", None) if manager is not None else None
            if scene_mgr is not None:
                scene_mgr.save_all_scene_items_to_states()
                if (
                    current_file in selected_paths
                    and 0 <= self.main.curr_img_idx < len(self.main.image_files)
                    and self.main.curr_img_idx in manager.loaded_pages
                ):
                    scene_mgr.unload_page_scene_items(self.main.curr_img_idx)
                    current_page_unloaded = True
        else:
            self.main.image_ctrl.save_current_image_state()

        return {
            "current_file": current_file,
            "current_page_unloaded": current_page_unloaded,
        }

    def _reload_current_webtoon_page(self) -> None:
        if not self.main.webtoon_mode:
            return
        manager = getattr(self.main.image_viewer, "webtoon_manager", None)
        if manager is None:
            return
        scene_mgr = getattr(manager, "scene_item_manager", None)
        if scene_mgr is None:
            return
        page_idx = self.main.curr_img_idx
        if not (0 <= page_idx < len(self.main.image_files)):
            return
        if page_idx not in manager.loaded_pages:
            return
        scene_mgr.load_page_scene_items(page_idx)
        self.main.text_ctrl.clear_text_edits()

    def _copy_blocks_for_current_webtoon_page(self, blk_list: list[TextBlock]) -> list[TextBlock]:
        blocks = [
            blk.deep_copy() if hasattr(blk, "deep_copy") else blk
            for blk in (blk_list or [])
        ]
        if not blocks or not self.main.webtoon_mode:
            return blocks

        manager = getattr(self.main.image_viewer, "webtoon_manager", None)
        scene_mgr = getattr(manager, "scene_item_manager", None) if manager is not None else None
        text_block_mgr = getattr(scene_mgr, "text_block_manager", None) if scene_mgr is not None else None
        page_idx = self.main.curr_img_idx
        if text_block_mgr is None or page_idx < 0 or page_idx >= len(self.main.image_files):
            return blocks

        for blk in blocks:
            text_block_mgr._convert_textblock_coordinates(blk, page_idx, to_scene=True)
        return blocks

    def _set_current_blocks_from_page_state(
        self,
        blk_list: list[TextBlock],
        *,
        current_page_unloaded: bool = False,
    ) -> None:
        if not self.main.webtoon_mode:
            self.main.blk_list = [blk.deep_copy() if hasattr(blk, "deep_copy") else blk for blk in blk_list]
            return

        if current_page_unloaded:
            self._reload_current_webtoon_page()
            return

        self.main.blk_list = self._copy_blocks_for_current_webtoon_page(blk_list)

    def _serialize_rectangles_from_blocks(self, blk_list: list[TextBlock]) -> list[dict]:
        rects: list[dict] = []
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            rects.append(
                {
                    "rect": (float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                    "region_id": blk.region_id,
                    "rotation": float(getattr(blk, "angle", 0)),
                    "transform_origin": tuple(blk.tr_origin_point) if getattr(blk, "tr_origin_point", None) else (0.0, 0.0),
                }
            )
        return rects

    @staticmethod
    def _has_rendered_manual_translation(blk: TextBlock, text_items: Sequence) -> bool:
        coords = getattr(blk, "xyxy", None)
        if coords is None or len(coords) < 4 or not (has_manual_translation(blk) or is_protected(blk, 'keep_layout')):
            return False
        for item in text_items:
            if isinstance(item, dict):
                identity = item.get('region_id', '')
                x, y = item.get("position", (0, 0))
                angle = item.get("rotation", 0)
            else:
                identity = getattr(item, 'region_id', '')
                x, y, angle = item.pos().x(), item.pos().y(), item.rotation()
            if identity:
                if identity == getattr(blk, 'region_id', ''):
                    return True
                continue
            if (
                is_close(x, blk.xyxy[0], 5)
                and is_close(y, blk.xyxy[1], 5)
                and is_close(angle, blk.angle, 1)
            ):
                return True
        return False

    def _clear_text_items_for_segmentation(self) -> None:
        """Keep completed manual layers; ordinary layers are rebuilt by Render."""
        viewer = self.main.image_viewer
        preserved = [
            item for item in viewer.text_items
            if any(
                self._has_rendered_manual_translation(blk, [item])
                for blk in self.main.blk_list
            )
        ]
        for item in list(viewer.text_items):
            if item not in preserved:
                viewer._scene.removeItem(item)
        viewer.text_items[:] = preserved

    def _serialize_segmentation_strokes(self, blk_list: list[TextBlock], image=None) -> list[dict]:
        strokes: list[dict] = []
        build_stroke = self.main.image_viewer.drawing_manager.make_segmentation_stroke_data
        for blk in blk_list:
            if blk.xyxy is None or len(blk.xyxy) < 4 or not permits_region_operation(blk, 'cleanup'):
                continue
            stroke = build_stroke(blk, image)
            if stroke is not None:
                strokes.append(stroke)
        return strokes

    def block_detect(self, load_rects: bool = True) -> None:
        selected_paths = self._selected_page_paths()
        if len(selected_paths) > 1:
            self.main.loading.setVisible(True)
            self.main.disable_hbutton_group()
            context = self._prepare_multi_page_context(selected_paths)
            source_lang_fallback = to_canonical_language_name(
                self.main.s_combo.currentText(),
                self.main.lang_mapping,
            )

            def detect_selected_pages() -> dict[str, list[TextBlock]]:
                if self.main.pipeline.block_detection.block_detector_cache is None:
                    self.main.pipeline.block_detection.block_detector_cache = TextBlockDetector(self.main.settings_page)
                detector = self.main.pipeline.block_detection.block_detector_cache
                results: dict[str, list[TextBlock]] = {}
                for file_path in selected_paths:
                    image = self._load_page_image(file_path)
                    if image is None:
                        continue
                    blk_list = detector.detect(image)
                    state = self.main.image_states.get(file_path, {})
                    source_lang = state.get("source_lang", source_lang_fallback)
                    if blk_list:
                        get_best_render_area(blk_list, image)
                    self.main.pipeline.block_detection.annotate_language_if_auto(image, blk_list, source_lang)
                    from pipeline.translation_job import speaker_region_ids
                    blk_list = merge_manual_regions(state.get("blk_list", []), blk_list,
                        retain_ids=speaker_region_ids(self.main, state.get('page_id')))
                    rtl = source_lang == "Japanese"
                    results[file_path] = sort_blk_list(blk_list, rtl)
                return results

            def on_detect_ready(results: dict[str, list[TextBlock]]) -> None:
                current_file = context["current_file"]
                current_blocks: list[TextBlock] | None = None
                for file_path, blk_list in (results or {}).items():
                    state = self.main.image_states.get(file_path)
                    if state is None:
                        continue
                    state["blk_list"] = blk_list
                    viewer_state = state.setdefault("viewer_state", {})
                    viewer_state["rectangles"] = self._serialize_rectangles_from_blocks(blk_list)
                    if file_path == current_file:
                        current_blocks = blk_list

                if current_blocks is not None:
                    if self.main.webtoon_mode:
                        self._set_current_blocks_from_page_state(
                            current_blocks,
                            current_page_unloaded=context["current_page_unloaded"],
                        )
                    elif load_rects:
                        self.main.blk_list = [blk.deep_copy() if hasattr(blk, "deep_copy") else blk for blk in current_blocks]
                        self.main.pipeline.load_box_coords(self.main.blk_list)

                if results:
                    self.main.mark_project_dirty()

            self.main.run_threaded(
                detect_selected_pages,
                on_detect_ready,
                self.main.default_error_handler,
                self.main.on_manual_finished,
            )
            return

        self.main.loading.setVisible(True)
        self.main.disable_hbutton_group()
        self.main.run_threaded(
            self.main.pipeline.detect_blocks,
            self.main.pipeline.on_blk_detect_complete,
            self.main.default_error_handler,
            self.main.on_manual_finished,
            load_rects,
        )

    def finish_ocr_translate(self, single_block: bool = False) -> None:
        if self.main.blk_list:
            if single_block:
                rect = self.main.image_viewer.selected_rect
            else:
                if self.main.webtoon_mode:
                    first_block = get_first_visible_block(
                        self.main.blk_list, self.main.image_viewer
                    )
                    if first_block is None:
                        first_block = self.main.blk_list[0]
                else:
                    first_block = self.main.blk_list[0]
                rect = self.main.rect_item_ctrl.find_corresponding_rect(first_block, 0.5)
            self.main.image_viewer.select_rectangle(rect)
        self.main.set_tool("box")
        self.main.on_manual_finished()

    def ocr(self, single_block: bool = False) -> None:
        if not validate_ocr(self.main):
            return
        review = getattr(self.main, 'review_ctrl', None)
        if single_block and review is not None and review.preview_replacement('source'):
            return
        selected_paths = self._selected_page_paths()
        if len(selected_paths) > 1 and not single_block:
            self.main.loading.setVisible(True)
            self.main.disable_hbutton_group()
            context = self._prepare_multi_page_context(selected_paths)
            source_lang_fallback = to_canonical_language_name(
                self.main.s_combo.currentText(),
                self.main.lang_mapping,
            )

            def ocr_selected_pages() -> dict[str, list[TextBlock]]:
                cache_manager = self.main.pipeline.cache_manager
                ocr = OCRProcessor()
                ocr_model = self.main.settings_page.get_tool_selection("ocr")
                device = resolve_device(self.main.settings_page.is_gpu_enabled())
                results: dict[str, list[TextBlock]] = {}
                for file_path in selected_paths:
                    state = self.main.image_states.get(file_path, {})
                    blk_list = state.get("blk_list", [])
                    if not blk_list:
                        continue
                    image = self._load_page_image(file_path)
                    if image is None:
                        continue
                    source_lang = state.get("source_lang", source_lang_fallback)
                    cache_key = cache_manager._get_ocr_cache_key(image, source_lang, ocr_model, device)
                    pending = [blk for blk in blk_list if not has_manual_translation(blk) and permits_region_operation(blk, 'source')]
                    if pending and cache_manager._can_serve_all_blocks_from_ocr_cache(cache_key, pending):
                        cache_manager._apply_cached_ocr_to_blocks(cache_key, pending)
                    elif pending:
                        ocr.initialize(self.main, source_lang)
                        ocr.process(image, pending)
                        cache_manager._cache_ocr_results(cache_key, pending)
                    results[file_path] = blk_list
                return results

            def on_ocr_ready(results: dict[str, list[TextBlock]]) -> None:
                current_file = context["current_file"]
                for file_path, blk_list in (results or {}).items():
                    state = self.main.image_states.get(file_path)
                    if state is None:
                        continue
                    state["blk_list"] = blk_list
                    if file_path == current_file:
                        self._set_current_blocks_from_page_state(
                            blk_list,
                            current_page_unloaded=context["current_page_unloaded"],
                        )

                if results:
                    self.main.mark_project_dirty()

            self.main.run_threaded(
                ocr_selected_pages,
                on_ocr_ready,
                self.main.default_error_handler,
                lambda: self.finish_ocr_translate(single_block),
            )
            return

        self.main.loading.setVisible(True)
        self.main.disable_hbutton_group()

        if self.main.webtoon_mode:
            self.main.run_threaded(
                lambda: self.main.pipeline.OCR_webtoon_visible_area(single_block),
                None,
                self.main.default_error_handler,
                lambda: self.finish_ocr_translate(single_block),
            )
        else:
            self.main.run_threaded(
                lambda: self.main.pipeline.OCR_image(single_block),
                None,
                self.main.default_error_handler,
                lambda: self.finish_ocr_translate(single_block),
            )

    def translate_image(self, single_block: bool = False) -> None:
        review = getattr(self.main, 'review_ctrl', None)
        if single_block and review is not None and review.preview_replacement('translation', update_lettering=False):
            return
        selected_paths = self._selected_page_paths()
        if len(selected_paths) > 1 and not single_block:
            has_any_text = False
            for file_path in selected_paths:
                blk_list = self.main.image_states.get(file_path, {}).get("blk_list", [])
                if is_there_text(blk_list):
                    has_any_text = True
                    break
            if not has_any_text:
                return
            for file_path in selected_paths:
                target_lang = self.main.image_states.get(file_path, {}).get(
                    "target_lang",
                    to_canonical_language_name(
                        self.main.t_combo.currentText(),
                        self.main.lang_mapping,
                    ),
                )
                if not validate_translator(self.main, target_lang):
                    return

            self.main.loading.setVisible(True)
            self.main.disable_hbutton_group()
            context = self._prepare_multi_page_context(selected_paths)
            source_lang_fallback = to_canonical_language_name(
                self.main.s_combo.currentText(),
                self.main.lang_mapping,
            )
            target_lang_fallback = to_canonical_language_name(
                self.main.t_combo.currentText(),
                self.main.lang_mapping,
            )
            settings_page = self.main.settings_page
            extra_context = settings_page.get_llm_settings()["extra_context"]
            translator_key = settings_page.get_tool_selection("translator")
            upper_case = settings_page.ui.uppercase_checkbox.isChecked()

            from pipeline.translation_job import TranslationJob, eligible_translation_blocks
            jobs = []
            try:
                for file_path in selected_paths:
                    state = self.main.image_states.get(file_path, {})
                    pending = eligible_translation_blocks(state.get('blk_list', []))
                    if not pending:
                        continue
                    source_lang = state.get('source_lang', source_lang_fallback)
                    target_lang = state.get('target_lang', target_lang_fallback)
                    translator = Translator(self.main, source_lang, target_lang)
                    jobs.append(TranslationJob(self.main, translator, pending, None, extra_context,
                        self.main.pipeline.cache_manager, path=file_path, upper_case=upper_case,
                        resolve_blocks=lambda path=file_path: self._translation_blocks_for_page(path),
                        image_loader=lambda path=file_path: self._load_page_image(path)))
            except Exception:
                self._translation_setup_failed()
                return

            def refresh_current():
                current_file = self._current_file_path()
                if current_file == context['current_file'] and current_file in selected_paths:
                    if self.main.webtoon_mode:
                        self._set_current_blocks_from_page_state(self.main.image_states[current_file].get('blk_list', []),
                            current_page_unloaded=context['current_page_unloaded'])
                    self._finish_translation(False)
                else:
                    self.main.on_manual_finished()

            self._run_translation_jobs(jobs, refresh_current)
            return

        target_lang = self.main.t_combo.currentText()
        if not is_there_text(self.main.blk_list) or not validate_translator(
            self.main, target_lang
        ):
            return
        self.main.loading.setVisible(True)
        self.main.disable_hbutton_group()

        selected_id = getattr(self.main.pipeline.get_selected_block(), 'region_id', None) if single_block else None
        start_path = self._current_file_path()
        try:
            job = self.main.pipeline.translation_handler.prepare_job(single_block, visible=self.main.webtoon_mode)
        except Exception:
            self._translation_setup_failed()
            return

        def finish_current():
            current_id = getattr(self.main.pipeline.get_selected_block(), 'region_id', None) if single_block else None
            if self._current_file_path() == start_path and (not single_block or selected_id == current_id):
                self._finish_translation(single_block)
            else:
                self.main.on_manual_finished()

        self._run_translation_jobs([job] if job is not None else [], finish_current)

    def _translation_setup_failed(self):
        import sys
        import traceback
        kind, error = sys.exc_info()[:2]
        self.main.on_manual_finished()
        self.main.default_error_handler((kind, error, traceback.format_exc()))

    def _translation_blocks_for_page(self, path):
        if not self.main.webtoon_mode and path == self._current_file_path():
            return self.main.blk_list
        return self.main.image_states.get(path, {}).get('blk_list', [])

    def _run_translation_jobs(self, jobs, finish):
        """Only the GUI callback changes page text; failures retain prior targets."""
        import traceback
        from pipeline.translation_job import StaleTranslationError
        if not jobs:
            self.main.on_manual_finished()
            return
        applied = []

        def run():
            results = []
            for job in jobs:
                try:
                    results.append((job.run(), None))
                except Exception as error:
                    results.append((job, (type(error), error, traceback.format_exc())))
                    if isinstance(error, StaleTranslationError):
                        break
            return results

        def ready(results):
            errors = []
            for job, error in results:
                if error is not None:
                    errors.append(error)
                    continue
                try:
                    if job.commit():
                        applied.append(job.path)
                except Exception as error:
                    errors.append((type(error), error, traceback.format_exc()))
            if applied:
                self.main.mark_project_dirty()
                self.main.batch_report_ctrl.resolve_translated_pages([p for p in applied if p])
            for error in errors:
                self.main.default_error_handler(error)

        def finished():
            if applied:
                finish()
            else:
                self.main.on_manual_finished()

        self.main.run_threaded(run, ready, self.main.default_error_handler, finished)

    def _get_visible_text_items(self) -> list[TextBlockItem]:
        if not self.main.webtoon_mode:
            return self.main.image_viewer.text_items
        return get_visible_text_items(
            self.main.image_viewer.text_items, self.main.image_viewer.webtoon_manager
        )

    @staticmethod
    def _find_text_item_for_block(
        blk: TextBlock,
        text_items: Sequence[TextBlockItem],
    ) -> TextBlockItem | None:
        # Lettering can be moved or rotated independently of the OCR box.
        identity = getattr(blk, 'region_id', '')
        if identity:
            for item in text_items:
                if getattr(item, 'region_id', '') == identity:
                    return item
        return next(
            (
                item
                for item in text_items
                if not getattr(item, 'region_id', '') and is_close(item.pos().x(), blk.xyxy[0], 5)
                and is_close(item.pos().y(), blk.xyxy[1], 5)
                and is_close(item.rotation(), blk.angle, 1)
            ),
            None,
        )

    def _file_path_for_block(self, blk: TextBlock) -> str:
        if self.main.webtoon_mode:
            page_idx = find_block_page_index(
                blk,
                self.main.image_viewer.webtoon_manager,
            )
            if page_idx is not None and 0 <= page_idx < len(self.main.image_files):
                return self.main.image_files[page_idx]
        return self._current_file_path() or ""

    def _visible_blocks_for_translation_refresh(self) -> list[TextBlock]:
        if not self.main.webtoon_mode:
            return list(self.main.blk_list)
        _image, mappings = self.main.image_viewer.get_visible_area_image()
        if not mappings:
            return []
        blocks = filter_and_convert_visible_blocks(
            self.main,
            self.main.pipeline,
            mappings,
            single_block=False,
        )
        restore_original_block_coordinates(blocks)
        return blocks

    def _finish_translation(self, single_block: bool) -> None:
        """Selected Translate updates target text; Clean and Render remain explicit."""
        if not single_block:
            self.update_translated_text_items(False)
            return

        self._resolve_current_page_if_translated()
        self.finish_ocr_translate(True)

    def update_translated_text_items(self, single_blk: bool) -> None:

        def set_new_text(
            text_item: TextBlockItem, 
            wrapped: str, 
            font_size: int
        ) -> None:
            text_item.set_plain_text(wrapped)
            text_item.set_font_size(font_size)

        text_items_to_process = self._get_visible_text_items()
        if single_blk:
            selected_blk = self.main.pipeline.get_selected_block()
            blocks_to_process = [selected_blk] if selected_blk is not None else []
        else:
            # Manual mode processes the user's block list, including free text
            # and newly drawn boxes that have no detector classification.
            blocks_to_process = [
                blk for blk in self._visible_blocks_for_translation_refresh()
                if not self._has_rendered_manual_translation(blk, text_items_to_process)
                and permits_region_operation(blk, 'layout')
            ]

        rs = self.main.render_settings()
        upper = rs.upper_case
        target_lang_en = self.main.lang_mapping.get(self.main.t_combo.currentText(), None)
        trg_lng_cd = get_language_code(target_lang_en)

        def on_format_finished() -> None:
            self._resolve_current_page_if_translated()
            wrap_count = 0
            default_alignment = self.main.button_to_alignment[rs.alignment_id]

            for blk in blocks_to_process:
                if not (blk and is_renderable_translation(blk.translation)):
                    continue

                text_item = self._find_text_item_for_block(blk, text_items_to_process)
                image_path = self._file_path_for_block(blk)
                if text_item is not None:
                    text_item.handleDeselection()

                vertical = is_vertical_block(blk, trg_lng_cd)
                wrap_args = (
                    blk.translation,
                    text_item.font_family if text_item is not None else rs.font_family,
                    blk.xyxy[2] - blk.xyxy[0],
                    blk.xyxy[3] - blk.xyxy[1],
                    float(text_item.line_spacing) if text_item is not None else float(rs.line_spacing),
                    float(text_item.outline_width) if text_item is not None else float(rs.outline_width),
                    text_item.bold if text_item is not None else rs.bold,
                    text_item.italic if text_item is not None else rs.italic,
                    text_item.underline if text_item is not None else rs.underline,
                    text_item.alignment if text_item is not None else default_alignment,
                    text_item.direction if text_item is not None else rs.direction,
                    rs.max_font_size,
                    rs.min_font_size,
                    vertical,
                    is_no_space_lang(trg_lng_cd),
                )

                def apply_wrapped_text(
                    wrap_res,
                    ti=text_item,
                    block=blk,
                    path=image_path,
                ) -> None:
                    if ti is None:
                        self.main.text_ctrl.on_blk_rendered(
                            wrap_res[0], wrap_res[1], block, path
                        )
                    else:
                        controller = self.main.text_ctrl
                        suspended = controller._suspend_text_command
                        controller._suspend_text_command = suspended or (not single_blk and not permits_region_operation(block, 'translation'))
                        try:
                            set_new_text(ti, wrap_res[0], wrap_res[1])
                        finally:
                            controller._suspend_text_command = suspended

                self.main.run_threaded(
                    pyside_word_wrap,
                    apply_wrapped_text,
                    self.main.default_error_handler,
                    None,
                    *wrap_args,
                )
                wrap_count += 1

            if wrap_count:
                self.main.run_finish_only(finished_callback=self.main.on_manual_finished)
            else:
                self.finish_ocr_translate(single_blk)

        self.main.run_threaded(
            lambda: format_translations([blk for blk in blocks_to_process
                                         if single_blk or permits_region_operation(blk, 'translation')],
                                        trg_lng_cd, upper_case=upper),
            None,
            self.main.default_error_handler,
            on_format_finished,
        )

    def _resolve_current_page_if_translated(self) -> None:
        """Remove a fully translated manual page from the batch report."""
        file_path = self._current_file_path()
        if not file_path:
            return
        blocks = self.main.image_states.get(file_path, {}).get("blk_list", self.main.blk_list)
        text_blocks = [blk for blk in blocks if getattr(blk, "text", "")]
        if text_blocks and all(getattr(blk, "translation", "") for blk in text_blocks):
            self.main.batch_report_ctrl.resolve_translated_pages([file_path])

    def inpaint_and_set(self, automatic: bool = True) -> None:
        if not self.main.image_viewer.hasPhoto():
            return

        selected_paths = self._selected_page_paths()
        if len(selected_paths) > 1:
            self.main.text_ctrl.clear_text_edits()
            self.main.loading.setVisible(True)
            self.main.disable_hbutton_group()
            context = self._prepare_multi_page_context(selected_paths)

            def inpaint_selected_pages() -> dict[str, list[dict]]:
                results: dict[str, list[dict]] = {}
                path_to_index = {p: i for i, p in enumerate(self.main.image_files)}

                for file_path in selected_paths:
                    state = self.main.image_states.get(file_path, {})
                    strokes = state.get("brush_strokes", [])
                    if not strokes:
                        continue
                    blk_list = state.get("blk_list", [])
                    # Repeated cleanup must start from the current composited
                    # page, including every earlier inpainting patch.
                    image = self.main.image_ctrl.get_composited_page_image(file_path)
                    if image is None:
                        continue

                    patches = self.main.pipeline.inpainting.inpaint_page_from_saved_strokes(
                        image,
                        strokes,
                        blk_list=blk_list,
                        automatic=automatic,
                    )

                    if self.main.webtoon_mode and patches:
                        page_idx = path_to_index.get(file_path)
                        if page_idx is not None:
                            for patch in patches:
                                x, y, _w, _h = patch['bbox']
                                scene_pos = self.main.image_viewer.page_to_scene_coordinates(
                                    page_idx,
                                    QtCore.QPointF(x, y),
                                )
                                if scene_pos is not None:
                                    patch['scene_pos'] = [scene_pos.x(), scene_pos.y()]
                                    patch['page_index'] = page_idx

                    results[file_path] = patches

                return results

            def on_selected_inpaint_ready(results: dict[str, list[dict]]) -> None:
                current_file = context["current_file"]
                processed_any = False

                for file_path, patches in (results or {}).items():
                    stack = self.main.undo_stacks.get(file_path)
                    if stack is not None:
                        stack.beginMacro("inpaint")
                    try:
                        if patches:
                            self.main.image_ctrl.on_inpaint_patches_processed(patches, file_path)
                    finally:
                        if stack is not None:
                            stack.endMacro()

                    state = self.main.image_states.get(file_path)
                    if state is not None:
                        state['brush_strokes'] = []
                    processed_any = True

                if not self.main.webtoon_mode and current_file in (results or {}):
                    self.main.image_viewer.clear_brush_strokes(page_switch=True)

                if self.main.webtoon_mode and context["current_page_unloaded"]:
                    self._reload_current_webtoon_page()

                if processed_any:
                    self.main.mark_project_dirty()

            self.main.run_threaded(
                inpaint_selected_pages,
                on_selected_inpaint_ready,
                self.main.default_error_handler,
                self.main.on_manual_finished,
            )
            return

        if self.main.image_viewer.has_drawn_elements():
            self.main.text_ctrl.clear_text_edits()
            self.main.loading.setVisible(True)
            self.main.disable_hbutton_group()
            self.main.undo_group.activeStack().beginMacro("inpaint")
            self.main.run_threaded(
                lambda: self.main.pipeline.inpaint(automatic=automatic),
            self.main.pipeline.inpaint_complete,
                self.main.default_error_handler,
                self.main.on_manual_finished,
            )

    def blk_detect_segment(
        self, 
        result: tuple[list[TextBlock], bool] | tuple[list[TextBlock], bool, Any]
    ) -> None:
        
        if len(result) == 3:
            blk_list, load_rects, _ = result
        else:
            blk_list, load_rects = result
        self.main.blk_list = blk_list
        self.main.undo_group.activeStack().beginMacro("draw_segmentation_boxes")
        image = self.main.image_viewer.get_image_array()
        for blk in self.main.blk_list:
            if blk.xyxy is not None and permits_region_operation(blk, 'cleanup'):
                stroke = self.main.image_viewer.drawing_manager.make_segmentation_stroke_data(blk, image)
                self.main.image_viewer.draw_segmentation_lines(blk.xyxy, stroke=stroke)
        self.main.undo_group.activeStack().endMacro()

    def load_segmentation_points(self) -> None:
        if self.main.image_viewer.hasPhoto():
            self.main.text_ctrl.clear_text_edits()
            self.main.set_tool("brush")
            self.main.disable_hbutton_group()
            self.main.image_viewer.clear_rectangles()
            self._clear_text_items_for_segmentation()

            self.main.loading.setVisible(True)
            self.main.disable_hbutton_group()

            selected_paths = self._selected_page_paths()
            if len(selected_paths) > 1:
                self.main.undo_group.activeStack().beginMacro("draw_segmentation_boxes")
                context = self._prepare_multi_page_context(selected_paths)

                def compute_selected_bboxes() -> dict[str, tuple[list[TextBlock], list[dict]]]:
                    results = {}
                    for file_path in selected_paths:
                        state = self.main.image_states.get(file_path, {})
                        blk_list = state.get("blk_list", [])
                        if not blk_list:
                            continue
                        image = self._load_page_image(file_path)
                        if image is None:
                            continue
                        strokes = self._serialize_segmentation_strokes(blk_list, image)
                        results[file_path] = (blk_list, strokes)
                    return results

                def on_selected_bboxes_ready(results: dict[str, tuple[list[TextBlock], list[dict]]]) -> None:
                    current_file = context["current_file"]
                    for file_path, (blk_list, strokes) in (results or {}).items():
                        state = self.main.image_states.get(file_path)
                        if state is None:
                            continue
                        state["blk_list"] = blk_list
                        viewer_state = state.setdefault("viewer_state", {})
                        viewer_state["rectangles"] = []
                        state["brush_strokes"] = strokes
                        if file_path == current_file:
                            self._set_current_blocks_from_page_state(
                                blk_list,
                                current_page_unloaded=context["current_page_unloaded"],
                            )

                    if (
                        not self.main.webtoon_mode
                        and current_file is not None
                        and current_file in (results or {})
                    ):
                        for stroke in results[current_file][1]:
                            self.main.image_viewer.draw_segmentation_lines(None, stroke=stroke)

                    if results:
                        self.main.mark_project_dirty()
                    self.main.undo_group.activeStack().endMacro()

                def on_selected_bboxes_error(error_tuple: tuple) -> None:
                    try:
                        self.main.undo_group.activeStack().endMacro()
                    except Exception:
                        pass
                    self.main.default_error_handler(error_tuple)

                self.main.run_threaded(
                    compute_selected_bboxes,
                    on_selected_bboxes_ready,
                    on_selected_bboxes_error,
                    self.main.on_manual_finished,
                )
                return

            if self.main.blk_list:
                self.main.undo_group.activeStack().beginMacro("draw_segmentation_boxes")

                if self.main.webtoon_mode:
                    self.main.run_threaded(
                        lambda: self.main.pipeline.segment_webtoon_visible_area(),
                        self._on_segmentation_bboxes_ready,
                        self.main.default_error_handler,
                        self.main.on_manual_finished,
                    )
                else:

                    def compute_all_strokes() -> list[tuple[TextBlock, Any]]:
                        image = self.main.image_viewer.get_image_array()
                        results = []
                        for blk in self.main.blk_list:
                            if not permits_region_operation(blk, 'cleanup'):
                                continue
                            stroke = self.main.image_viewer.drawing_manager.make_segmentation_stroke_data(blk, image)
                            results.append((blk, stroke))
                        return results

                    self.main.run_threaded(
                        compute_all_strokes,
                        self._on_segmentation_bboxes_ready,
                        self.main.default_error_handler,
                        self.main.on_manual_finished,
                    )

            else:
                self.main.run_threaded(
                    self.main.pipeline.detect_blocks,
                    self.blk_detect_segment,
                    self.main.default_error_handler,
                    self.main.on_manual_finished,
                )

    def _on_segmentation_bboxes_ready(
        self, 
        results: Sequence[tuple[TextBlock, Any] | TextBlock]
    ) -> None:
        if self.main.webtoon_mode:
            for blk in results:
                if blk.xyxy is not None and permits_region_operation(blk, 'cleanup'):
                    stroke = self.main.image_viewer.drawing_manager.make_segmentation_stroke_data(blk)
                    self.main.image_viewer.draw_segmentation_lines(blk.xyxy, stroke=stroke)
        else:
            for blk, stroke in results:
                if stroke is not None and permits_region_operation(blk, 'cleanup'):
                    self.main.image_viewer.draw_segmentation_lines(blk.xyxy, stroke=stroke)
        self.main.undo_group.activeStack().endMacro()
