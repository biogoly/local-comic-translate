from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import imkit as imk
from PySide6 import QtCore

from modules.detection.processor import TextBlockDetector
from modules.ocr.processor import OCRProcessor
from modules.rendering.render import pyside_word_wrap, is_vertical_block, get_best_render_area
from modules.translation.processor import Translator
from modules.utils.common_utils import is_close
from modules.utils.device import resolve_device
from modules.utils.image_utils import generate_mask
from modules.utils.language_utils import get_language_code, is_no_space_lang
from modules.utils.language_utils import to_canonical_language_name
from modules.utils.pipeline_config import get_config, validate_ocr, validate_translator
from modules.utils.textblock import sort_blk_list
from modules.utils.translator_utils import (
    format_translations,
    is_renderable_translation,
    is_there_text,
    set_upper_case,
)
from pipeline.inpainting import call_inpaint_image
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
                    "rotation": float(getattr(blk, "angle", 0)),
                    "transform_origin": tuple(blk.tr_origin_point) if getattr(blk, "tr_origin_point", None) else (0.0, 0.0),
                }
            )
        return rects

    def _serialize_segmentation_strokes(self, blk_list: list[TextBlock], image=None) -> list[dict]:
        strokes: list[dict] = []
        build_stroke = self.main.image_viewer.drawing_manager.make_segmentation_stroke_data
        for blk in blk_list:
            if blk.xyxy is None or len(blk.xyxy) < 4:
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
                    if cache_manager._can_serve_all_blocks_from_ocr_cache(cache_key, blk_list):
                        cache_manager._apply_cached_ocr_to_blocks(cache_key, blk_list)
                    else:
                        ocr.initialize(self.main, source_lang)
                        ocr.process(image, blk_list)
                        cache_manager._cache_ocr_results(cache_key, blk_list)
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

            def translate_selected_pages() -> dict[str, list[TextBlock]]:
                cache_manager = self.main.pipeline.cache_manager
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
                    target_lang = state.get("target_lang", target_lang_fallback)
                    translator = Translator(self.main, source_lang, target_lang)
                    cache_key = cache_manager._get_translation_cache_key(
                        image,
                        source_lang,
                        target_lang,
                        translator_key,
                        extra_context,
                    )
                    if cache_manager._can_serve_all_blocks_from_translation_cache(cache_key, blk_list):
                        cache_manager._apply_cached_translations_to_blocks(cache_key, blk_list)
                    else:
                        translator.translate(blk_list, image, extra_context)
                        cache_manager._cache_translation_results(cache_key, blk_list)
                    set_upper_case(blk_list, upper_case)
                    results[file_path] = blk_list
                return results

            def on_translation_ready(results: dict[str, list[TextBlock]]) -> None:
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
                    self.main.batch_report_ctrl.resolve_translated_pages(list(results))
                    self.main.mark_project_dirty()

            self.main.run_threaded(
                translate_selected_pages,
                on_translation_ready,
                self.main.default_error_handler,
                lambda: self._finish_translation(single_block),
            )
            return

        target_lang = self.main.t_combo.currentText()
        if not is_there_text(self.main.blk_list) or not validate_translator(
            self.main, target_lang
        ):
            return
        self.main.loading.setVisible(True)
        self.main.disable_hbutton_group()

        if self.main.webtoon_mode:
            self.main.run_threaded(
                lambda: self.main.pipeline.translate_webtoon_visible_area(single_block),
                None,
                self.main.default_error_handler,
                lambda: self._finish_translation(single_block),
            )
        else:
            self.main.run_threaded(
                lambda: self.main.pipeline.translate_image(single_block),
                None,
                self.main.default_error_handler,
                lambda: self._finish_translation(single_block),
            )

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
        return next(
            (
                item
                for item in text_items
                if is_close(item.pos().x(), blk.xyxy[0], 5)
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
        """Finish manual translation, cleaning a newly translated selected box first."""
        if not single_block:
            self.update_translated_text_items(False)
            return

        blk = self.main.pipeline.get_selected_block()
        if not (
            blk
            and getattr(blk, "text", "").strip()
            and is_renderable_translation(getattr(blk, "translation", ""))
        ):
            self.update_translated_text_items(True)
            return

        # Existing text layers already sit on a cleaned background. Only run a
        # focused cleanup when materializing a translation that has no layer yet.
        if self._find_text_item_for_block(blk, self._get_visible_text_items()) is not None:
            self.update_translated_text_items(True)
            return

        self.main.run_threaded(
            lambda: self._inpaint_translated_block(blk),
            self._apply_translated_block_patches,
            self.main.default_error_handler,
            lambda: self.update_translated_text_items(True),
        )

    def _inpaint_translated_block(self, blk: TextBlock) -> dict[str, list[dict]]:
        """Inpaint one explicitly selected block and return patches grouped by page."""
        inpainting = self.main.pipeline.inpainting
        blocks: list[TextBlock]

        if self.main.webtoon_mode:
            image, mappings = self.main.image_viewer.get_visible_area_image()
            if image is None or not mappings:
                return {}
            blocks = filter_and_convert_visible_blocks(
                self.main,
                self.main.pipeline,
                mappings,
                single_block=True,
            )
        else:
            file_path = self._current_file_path()
            if not file_path:
                return {}
            image = self.main.image_ctrl.get_composited_page_image(file_path)
            if image is None:
                return {}
            blocks = [blk.deep_copy() if hasattr(blk, "deep_copy") else blk]

        if not blocks:
            return {}

        try:
            mask = generate_mask(image, blocks)
            if mask is None or not mask.any():
                return {}
            config = get_config(self.main.settings_page)
            inpainted = call_inpaint_image(
                inpainting,
                image,
                mask,
                config,
                blk_list=blocks,
            )
            inpainted = imk.convert_scale_abs(inpainted)
            patches = inpainting.get_inpainted_patches(mask, inpainted)
        finally:
            if self.main.webtoon_mode:
                restore_original_block_coordinates(blocks)

        if not patches:
            return {}
        if not self.main.webtoon_mode:
            return {file_path: patches}

        grouped: dict[str, list[dict]] = {}
        for patch in patches:
            patch_file = patch.get("file_path")
            if not patch_file:
                continue
            clean_patch = {"bbox": patch["bbox"], "image": patch["image"]}
            for key in ("scene_pos", "page_index"):
                if key in patch:
                    clean_patch[key] = patch[key]
            grouped.setdefault(patch_file, []).append(clean_patch)
        return grouped

    def _apply_translated_block_patches(self, grouped: dict[str, list[dict]]) -> None:
        applied = False
        for file_path, patches in (grouped or {}).items():
            if not patches:
                continue
            self.main.image_ctrl.on_inpaint_patches_processed(patches, file_path)
            applied = True
        if applied:
            self.main.mark_project_dirty()

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
            blocks_to_process = self._visible_blocks_for_translation_refresh()

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
                        set_new_text(ti, wrap_res[0], wrap_res[1])

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
            lambda: format_translations(self.main.blk_list, trg_lng_cd, upper_case=upper),
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

    def inpaint_and_set(self) -> None:
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
                self.main.pipeline.inpaint,
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
            if blk.xyxy is not None:
                stroke = self.main.image_viewer.drawing_manager.make_segmentation_stroke_data(blk, image)
                self.main.image_viewer.draw_segmentation_lines(blk.xyxy, stroke=stroke)
        self.main.undo_group.activeStack().endMacro()

    def load_segmentation_points(self) -> None:
        if self.main.image_viewer.hasPhoto():
            self.main.text_ctrl.clear_text_edits()
            self.main.set_tool("brush")
            self.main.disable_hbutton_group()
            self.main.image_viewer.clear_rectangles()
            self.main.image_viewer.clear_text_items()

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
                if blk.xyxy is not None:
                    stroke = self.main.image_viewer.drawing_manager.make_segmentation_stroke_data(blk)
                    self.main.image_viewer.draw_segmentation_lines(blk.xyxy, stroke=stroke)
        else:
            for blk, stroke in results:
                if stroke is not None:
                    self.main.image_viewer.draw_segmentation_lines(blk.xyxy, stroke=stroke)
        self.main.undo_group.activeStack().endMacro()
