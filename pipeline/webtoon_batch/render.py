from __future__ import annotations

import logging
import os
from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING, List

import imkit as imk
from PySide6.QtGui import QColor

from app.path_materialization import ensure_path_materialized
from app.ui.canvas.save_renderer import ImageSaveRenderer
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.text_item import OutlineInfo, OutlineType
from modules.rendering.render import (
    block_for_render_area, get_best_render_area, is_vertical_block, orient_new_lettering, pyside_word_wrap,
)
from modules.rendering.auto_bubble_fit import fit_new_text_item, load_original_for_fitting
from modules.utils.image_utils import get_smart_text_color
from modules.utils.language_utils import get_language_code, is_no_space_lang
from modules.utils.textblock import TextBlock
from modules.utils.translator_utils import format_translations, get_raw_text, get_raw_translation, is_renderable_translation
from modules.utils.region_policy import permits_region_operation, retained_layout_states

if TYPE_CHECKING:
    from .processor import WebtoonBatchProcessor

logger = logging.getLogger(__name__)


class RenderMixin:
    def _prepare_page_blocks_for_render(
        self: WebtoonBatchProcessor,
        image_path: str,
        blocks: List[TextBlock],
        has_patches: bool,
    ) -> List[TextBlock]:
        page_state = self.main_page.image_states[image_path]
        session = getattr(self, '_translation_session', None)
        if session is not None:
            session.validate(image_path)
        if not blocks:
            page_state["blk_list"] = []
            page_state["skip_render"] = not has_patches
            if session is not None:
                session.mark_published(image_path)
            return []

        render_settings = self.main_page.render_settings()
        target_lang = page_state["target_lang"]
        target_lang_code = get_language_code(target_lang)

        format_translations(
            [block for block in blocks if permits_region_operation(block, 'translation')
             and block.region_id not in getattr(self, '_review_manual_ids', ())], target_lang_code, upper_case=render_settings.upper_case
        )

        page_state["blk_list"] = blocks
        page_state["skip_render"] = False
        if session is not None:
            # This checkpoint includes our deliberate case formatting. The
            # renderer below keeps wrapping in text properties/copies; advancing
            # again after rendering could accept an intervening user text edit.
            session.mark_published(image_path)
        return blocks

    def _store_page_text_items(
        self: WebtoonBatchProcessor,
        page_index: int,
        image_path: str,
        blocks: List[TextBlock],
        image_shape: tuple,
        live_blocks: List[TextBlock] | None = None,
    ) -> None:
        page_state = self.main_page.image_states[image_path]
        viewer_state = page_state.setdefault("viewer_state", {})
        session = getattr(self, '_translation_session', None)
        spatial = image_path in getattr(session, 'spatial_paths', ())
        # Sizing a rotated layer or a layer referenced by the vision request
        # must preserve the canonical OCR/cleanup box, including at page seams.
        canonical_blocks = blocks
        blocks = [block.deep_copy() if spatial else block_for_render_area(block)
                  for block in canonical_blocks]
        render_settings = self.main_page.render_settings()
        auto_bubble_fit = bool(getattr(render_settings, "auto_bubble_fit", False))
        previous_items = viewer_state.get("text_items_state", [])
        existing_by_id = retained_layout_states(blocks, previous_items, retain_all=auto_bubble_fit)
        viewer_state["text_items_state"] = []
        viewer_state["push_to_stack"] = True

        if not blocks:
            return

        font = render_settings.font_family
        base_font_color = QColor(render_settings.color)
        max_font_size = render_settings.max_font_size
        min_font_size = render_settings.min_font_size
        line_spacing = float(render_settings.line_spacing)
        outline_width = float(render_settings.outline_width)
        outline = render_settings.outline
        outline_color = QColor(render_settings.outline_color) if outline else None
        bold = render_settings.bold
        italic = render_settings.italic
        underline = render_settings.underline
        alignment = self.main_page.button_to_alignment[render_settings.alignment_id]
        direction = render_settings.direction

        target_lang = page_state["target_lang"]
        target_lang_code = get_language_code(target_lang)

        virtual_img = SimpleNamespace(shape=image_shape)
        fit_blocks = {
            id(block): block.deep_copy() for block in blocks
        } if auto_bubble_fit else {}
        original_for_fitting = (
            load_original_for_fitting(image_path, expected_shape=image_shape)
            if auto_bubble_fit and any(block.region_id not in existing_by_id
                                       and permits_region_operation(block, 'layout') for block in blocks)
            else None
        )
        # Avoid clipping seam-owned overflow blocks (needed for page-spanning text).
        in_bounds_blocks = [
            block
            for block in blocks
            if float(block.xyxy[1]) >= 0 and float(block.xyxy[3]) <= float(image_shape[0])
            and block.region_id not in existing_by_id
            and permits_region_operation(block, 'layout')
        ]
        if in_bounds_blocks:
            get_best_render_area(in_bounds_blocks, virtual_img)

        should_emit_live = False
        webtoon_manager = getattr(self.main_page.image_viewer, "webtoon_manager", None)
        if self.main_page.webtoon_mode and webtoon_manager:
            should_emit_live = page_index in webtoon_manager.loaded_pages
        live_block_ids = (
            None if live_blocks is None else {id(block) for block in live_blocks}
        )
        live_region_ids = None if live_blocks is None else {block.region_id for block in live_blocks}
        page_scene_offset = self._get_page_scene_offset(page_index)

        for block, canonical in zip(blocks, canonical_blocks):
            # Incremental webtoon publishing revisits accumulated blocks. Keep
            # their exact fitted/manual properties rather than fitting twice.
            # Never guess identity from overlap after a fresh detection pass.
            if block.region_id in existing_by_id:
                viewer_state["text_items_state"].append(deepcopy(existing_by_id[block.region_id]))
                continue
            if not permits_region_operation(block, 'layout'):
                viewer_state['text_items_state'].extend(deepcopy(item) for item in previous_items
                                                        if item.get('region_id') == block.region_id)
                continue
            x1, y1, x2, y2 = [float(v) for v in block.xyxy]
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)

            translation = block.translation
            if not is_renderable_translation(translation):
                continue

            vertical = is_vertical_block(block, target_lang_code)
            (
                wrapped_translation,
                font_size,
                rendered_width,
                rendered_height,
            ) = pyside_word_wrap(
                translation,
                font,
                width,
                height,
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
                is_no_space_lang(target_lang_code),
                return_metrics=True,
            )

            font_color = get_smart_text_color(block.font_color, base_font_color)
            text_props = TextItemProperties(
                text=wrapped_translation,
                region_id=block.region_id,
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
                transform_origin=block.tr_origin_point if block.tr_origin_point else (0, 0),
                width=rendered_width,
                height=rendered_height,
                direction=direction,
                vertical=vertical,
                selection_outlines=[
                    OutlineInfo(
                        0,
                        len(wrapped_translation),
                        outline_color,
                        outline_width,
                        OutlineType.Full_Document,
                    )
                ]
                if outline
                else [],
            )
            orient_new_lettering(text_props, block)
            if auto_bubble_fit:
                text_props = fit_new_text_item(
                    text_props, fit_blocks[id(block)], original_for_fitting,
                    render_settings, source_text=translation,
                )
            properties = text_props.to_dict()
            viewer_state["text_items_state"].append(properties)

            # Construct saved properties first so the live item can use the
            # same HTML, font, width and page-local position as reload/export.
            emit_block = ((live_region_ids is None or block.region_id in live_region_ids) if spatial
                          else (live_block_ids is None or id(canonical) in live_block_ids))
            if should_emit_live and emit_block:
                render_block = canonical.deep_copy()
                render_block.translation = wrapped_translation
                render_block.xyxy = list(render_block.xyxy)
                render_block.xyxy[1] += page_scene_offset
                render_block.xyxy[3] += page_scene_offset
                if render_block.bubble_xyxy is not None:
                    render_block.bubble_xyxy = list(render_block.bubble_xyxy)
                    render_block.bubble_xyxy[1] += page_scene_offset
                    render_block.bubble_xyxy[3] += page_scene_offset
                if auto_bubble_fit or block is not canonical:
                    render_block._render_text_properties = deepcopy(properties)
                    render_block._render_page_origin = (0, page_scene_offset)
                self.main_page.blk_rendered.emit(
                    wrapped_translation, text_props.font_size, render_block, image_path
                )

    def _save_final_rendered_page(
        self: WebtoonBatchProcessor, page_idx: int, image_path: str, timestamp: str
    ):
        """
        Handle per-page exports once page results are finalized.
        """
        logger.info(
            "Starting final render process for page %s at path: %s", page_idx, image_path
        )

        ensure_path_materialized(image_path)
        image = imk.read_image(image_path)
        compose = getattr(getattr(self.main_page, 'image_ctrl', None), 'get_composited_page_image', None)
        if callable(compose):
            image = compose(image_path)
        if image is None:
            logger.error("Failed to load physical image for rendering: %s", image_path)
            return

        base_name = os.path.splitext(os.path.basename(image_path))[0].strip()
        extension = os.path.splitext(image_path)[1]
        directory = os.path.dirname(image_path)

        archive_bname = ""
        for archive in self.main_page.file_handler.archive_info:
            if image_path in archive["extracted_images"]:
                archive_path = archive["archive_path"]
                directory = os.path.dirname(archive_path)
                archive_bname = os.path.splitext(os.path.basename(archive_path))[0].strip()
                break

        if self.main_page.image_states[image_path].get("skip_render"):
            logger.info("Skipping final render for page %s, copying original.", page_idx)
            reason = "No text blocks detected or processed successfully."
            self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
            self.log_skipped_image(directory, timestamp, image_path, reason)
            return

        settings_page = self.main_page.settings_page
        export_settings = settings_page.get_export_settings()

        if export_settings["export_inpainted_image"]:
            renderer = ImageSaveRenderer(image)
            patches = self.final_patches_for_save.get(image_path, [])
            renderer.apply_patches(patches)
            path = os.path.join(
                directory,
                f"comic_translate_{timestamp}",
                "cleaned_images",
                archive_bname,
            )
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            cleaned_image_rgb = renderer.render_to_image()
            imk.write_image(
                os.path.join(path, f"{base_name}_cleaned{extension}"), cleaned_image_rgb
            )

        blk_list = self.main_page.image_states[image_path].get("blk_list", [])

        if export_settings["export_raw_text"] and blk_list:
            path = os.path.join(
                directory, f"comic_translate_{timestamp}", "raw_texts", archive_bname
            )
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            raw_text = get_raw_text(blk_list)
            with open(os.path.join(path, f"{base_name}_raw.json"), "w", encoding="UTF-8") as f:
                f.write(raw_text)

        if export_settings["export_translated_text"] and blk_list:
            path = os.path.join(
                directory,
                f"comic_translate_{timestamp}",
                "translated_texts",
                archive_bname,
            )
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            translated_text = get_raw_translation(blk_list)
            with open(
                os.path.join(path, f"{base_name}_translated.json"),
                "w",
                encoding="UTF-8",
            ) as f:
                f.write(translated_text)
