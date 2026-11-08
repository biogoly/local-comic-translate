"""Automatic fitting integration without detector, OCR or model execution."""
import os
import unittest
from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.controllers.batch_report import BatchReportController
from app.controllers.image import ImageStateController
from app.ui.dayu_widgets.message import MMessage
from modules.rendering.auto_bubble_fit import fit_new_text_item
from modules.utils.textblock import TextBlock
from pipeline.batch_processor import BatchProcessor
from pipeline.cache_manager import CacheManager
from pipeline.webtoon_batch.processor import WebtoonBatchProcessor
from pipeline.webtoon_batch.render import RenderMixin


PRESERVED_REASON = "Page already contains rendered text; use manual Render to add missing regions."


def settings(enabled=True):
    return SimpleNamespace(
        upper_case=False, outline=False, font_family="Arial", color="#000000",
        max_font_size=30, min_font_size=8, line_spacing=1.0, outline_width=0.0,
        bold=False, italic=False, underline=False, alignment_id=0,
        direction=Qt.LayoutDirection.LeftToRight,
        auto_bubble_fit=enabled, bubble_padding=8,
    )


def block(bounds=(10, 10, 60, 60)):
    return TextBlock(text_bbox=np.array(bounds), text_class="text_bubble",
                     text="original words", translation="logical translated words")


def fitted_properties(props, *_args, **_kwargs):
    fitted = deepcopy(props)
    fitted.text = '<p><b>logical</b> translated words</p>'
    fitted.position = (24.5, 23.5)
    fitted.width = 77.0
    fitted.height = 42.0
    fitted.font_size = 16.5
    fitted.layout_preferences = {"mode": "bubble", "bound_width": 90.0, "bound_height": 70.0}
    fitted.document_font = "Arial,16.5,-1,5,400,0,0,0,0,0,0,0,0,0,0,1"
    return fitted


def main_page(enabled=True, previous=None):
    render = settings(enabled)
    main = SimpleNamespace(
        image_files=["page.png"], curr_img_idx=0, blk_list=[], current_worker=None,
        settings_page=SimpleNamespace(
            get_tool_selection=lambda name: "Local LLM" if name == "translator" else "Default",
            is_gpu_enabled=lambda: False,
            get_llm_settings=lambda: {"extra_context": ""},
            get_export_settings=lambda: {
                "export_raw_text": False, "export_translated_text": False,
                "export_inpainted_image": False,
            },
        ),
        image_states={"page.png": {
            "source_lang": "Spanish", "target_lang": "English", "viewer_state": {
                "text_items_state": deepcopy(previous or []),
            },
        }},
        file_handler=SimpleNamespace(should_pre_materialize=lambda _paths: False, archive_info=[]),
        render_settings=lambda: render,
        button_to_alignment={0: Qt.AlignmentFlag.AlignCenter},
        image_viewer=SimpleNamespace(webtoon_manager=SimpleNamespace(loaded_pages={0}, image_positions=[1000])),
        webtoon_mode=False, lang_mapping={},
        s_combo=SimpleNamespace(currentText=lambda: "Spanish"),
        t_combo=SimpleNamespace(currentText=lambda: "English"),
    )
    for name in ("progress_update", "image_skipped", "batch_page_edit_started",
                 "patches_processed", "render_state_ready", "batch_page_edit_finished", "blk_rendered"):
        setattr(main, name, Mock())
    return main


class AutomaticBubbleBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def run_regular(self, main, blocks, *, mutate_inpaint=False):
        image = np.full((100, 200, 3), 255, dtype=np.uint8)
        detector = SimpleNamespace(block_detector_cache=Mock(), annotate_language_if_auto=Mock())
        detector.block_detector_cache.detect.return_value = blocks
        ocr = SimpleNamespace(ocr=Mock())
        batch = BatchProcessor(main, CacheManager(), detector,
                               SimpleNamespace(get_inpainted_patches=lambda *_a: []), ocr)

        def inpaint(_handler, pixels, *_args, **_kwargs):
            if mutate_inpaint:
                pixels[:] = 0
            return pixels

        def adjust(block_list, *_args):
            for blk in block_list:
                blk.xyxy += np.array([2, 2, -2, -2])

        with ExitStack() as stack:
            mocks = {}
            for name, kwargs in {
                "ensure_path_materialized": {},
                "imk.read_image": {"return_value": image},
                "Translator": {},
                "get_config": {"return_value": SimpleNamespace()},
                "generate_mask": {"return_value": np.ones((100, 200), dtype=np.uint8)},
                "call_inpaint_image": {"side_effect": inpaint},
                "get_best_render_area": {"side_effect": adjust},
                "pyside_word_wrap": {"return_value": ("wrapper\nline breaks", 12, 50, 50)},
                "fit_new_text_item": {"side_effect": fitted_properties},
            }.items():
                mocks[name] = stack.enter_context(patch("pipeline.batch_processor." + name, **kwargs))
            batch.batch_process()
        return mocks, image, detector, ocr

    def test_regular_batch_saves_fitted_properties_from_original_pixels_and_seed(self):
        main, region = main_page(), block()
        mocks, image, _, _ = self.run_regular(main, [region], mutate_inpaint=True)
        args = mocks["fit_new_text_item"].call_args
        self.assertEqual(args.kwargs["source_text"], "logical translated words")
        np.testing.assert_array_equal(args.args[1].xyxy, [10, 10, 60, 60])
        self.assertTrue(np.all(args.args[2] == 255))
        self.assertFalse(args.args[2].flags.writeable)
        self.assertTrue(np.all(image == 255))  # Cleanup now starts from an independent composed snapshot.
        saved = main.image_states["page.png"]["viewer_state"]["text_items_state"][0]
        self.assertEqual(saved, fitted_properties(args.args[0]).to_dict())
        # The fitted item position must not replace the underlying region bounds.
        np.testing.assert_array_equal(region.xyxy, [12, 12, 58, 58])
        self.assertEqual(saved["region_id"], region.region_id)
        main.render_state_ready.emit.assert_called_once_with("page.png")

    def test_regular_existing_page_is_preserved_before_any_model_or_cleanup(self):
        existing = {"text": "<p>My <i>manual</i> edit</p>", "position": (42, 7), "width": 99}
        main = main_page(previous=[existing])
        before = deepcopy(main.image_states)
        mocks, _, detector, ocr = self.run_regular(main, [block()])
        self.assertEqual(main.image_states, before)
        for name in ("imk.read_image", "Translator", "generate_mask", "call_inpaint_image", "fit_new_text_item"):
            mocks[name].assert_not_called()
        detector.block_detector_cache.detect.assert_not_called()
        ocr.ocr.process.assert_not_called()
        main.render_state_ready.emit.assert_not_called()
        main.image_skipped.emit.assert_called_once_with("page.png", "Existing Rendered Text", PRESERVED_REASON)

    def test_disabled_regular_option_retains_previous_rebuild_behavior(self):
        main = main_page(False, [{"text": "old rendered text"}])
        mocks, _, detector, _ = self.run_regular(main, [block()])
        mocks["fit_new_text_item"].assert_not_called()
        detector.block_detector_cache.detect.assert_called_once()
        self.assertEqual(main.image_states["page.png"]["viewer_state"]["text_items_state"][0]["text"], "wrapper\nline breaks")
        main.image_skipped.emit.assert_not_called()

    def webtoon_handler(self, enabled=True, previous=None):
        main = main_page(enabled, previous)
        main.webtoon_mode = True
        return SimpleNamespace(main_page=main, _get_page_scene_offset=lambda _index: 1000)

    def store_webtoon(self, handler, blocks, *, live_blocks=None, fit_effect=fitted_properties):
        source = np.full((100, 200, 3), 247, dtype=np.uint8)
        source.setflags(write=False)
        with (
            patch("pipeline.webtoon_batch.render.load_original_for_fitting", return_value=source) as load,
            patch("pipeline.webtoon_batch.render.get_best_render_area") as area,
            patch("pipeline.webtoon_batch.render.pyside_word_wrap", return_value=("wrapper\nline breaks", 12, 50, 50)),
            patch("pipeline.webtoon_batch.render.fit_new_text_item", side_effect=fit_effect) as fit,
        ):
            RenderMixin._store_page_text_items(handler, 0, "page.png", blocks, source.shape, live_blocks=live_blocks)
        return load, fit, area

    def test_webtoon_live_payload_matches_saved_page_local_properties(self):
        handler, region = self.webtoon_handler(), block()
        load, fit, _ = self.store_webtoon(handler, [region])
        load.assert_called_once_with("page.png", expected_shape=(100, 200, 3))
        self.assertEqual(fit.call_args.kwargs["source_text"], region.translation)
        main = handler.main_page
        saved = main.image_states["page.png"]["viewer_state"]["text_items_state"][0]
        event = main.blk_rendered.emit.call_args.args
        emitted_block = event[2]
        self.assertEqual(emitted_block._render_text_properties, saved)
        self.assertIsNot(emitted_block._render_text_properties, saved)
        self.assertEqual(emitted_block._render_page_origin, (0, 1000))
        self.assertEqual(event[1], saved["font_size"])
        np.testing.assert_array_equal(region.xyxy, [10, 10, 60, 60])
        np.testing.assert_array_equal(emitted_block.xyxy, [10, 1010, 60, 1060])
        self.assertEqual(saved["position"], (24.5, 23.5))

    def test_incremental_webtoon_keeps_exact_existing_properties_and_fits_only_new(self):
        handler, first, second = self.webtoon_handler(), block(), block((70, 10, 120, 60))
        self.store_webtoon(handler, [first])
        items = handler.main_page.image_states["page.png"]["viewer_state"]["text_items_state"]
        items[0]["text"] = "<p>Manually revised <u>text</u></p>"
        items[0]["position"] = (27, 28)
        previous = deepcopy(items[0])
        handler.main_page.blk_rendered.emit.reset_mock()
        _, fit, area = self.store_webtoon(handler, [first, second], live_blocks=[second])
        fit.assert_called_once()
        self.assertEqual(fit.call_args.args[1].region_id, second.region_id)
        self.assertEqual(area.call_args.args[0], [second])
        saved = handler.main_page.image_states["page.png"]["viewer_state"]["text_items_state"]
        self.assertEqual(saved[0], previous)
        handler.main_page.blk_rendered.emit.assert_called_once()

    def test_webtoon_does_not_associate_new_identity_by_geometry(self):
        old, fresh = block(), block()
        handler = self.webtoon_handler(previous=[{"region_id": old.region_id, "text": "old", "position": (10, 10)}])
        _, fit, _ = self.store_webtoon(handler, [fresh])
        fit.assert_called_once()
        saved = handler.main_page.image_states["page.png"]["viewer_state"]["text_items_state"]
        self.assertEqual([item["region_id"] for item in saved], [fresh.region_id])

    def test_webtoon_seam_region_is_passed_unclipped_for_box_fallback(self):
        handler, region = self.webtoon_handler(), block((10, 70, 60, 130))
        _, fit, area = self.store_webtoon(handler, [region], fit_effect=fit_new_text_item)
        area.assert_not_called()
        np.testing.assert_array_equal(fit.call_args.args[1].xyxy, [10, 70, 60, 130])
        self.assertEqual(fit.call_args.args[2].shape, (100, 200, 3))
        saved = handler.main_page.image_states["page.png"]["viewer_state"]["text_items_state"][0]
        self.assertEqual(saved["text"], "wrapper\nline breaks")
        self.assertEqual(saved["position"], (10, 70))
        self.assertEqual(saved["width"], 50)
        self.assertEqual(saved["layout_preferences"], {})

    def test_disabled_webtoon_does_not_load_source_or_fit_or_add_payload(self):
        handler = self.webtoon_handler(False)
        load, fit, _ = self.store_webtoon(handler, [block()])
        load.assert_not_called()
        fit.assert_not_called()
        emitted = handler.main_page.blk_rendered.emit.call_args.args[2]
        self.assertFalse(hasattr(emitted, "_render_text_properties"))

    def test_webtoon_protected_page_is_a_seam_barrier_and_remains_unchanged(self):
        main = main_page()
        main.webtoon_mode = True
        main.image_files = ["before.png", "protected.png", "after.png"]
        main.image_states = {
            path: {"source_lang": "Spanish", "target_lang": "English", "viewer_state": {"text_items_state": []}}
            for path in main.image_files
        }
        protected = main.image_states["protected.png"]
        protected["viewer_state"]["text_items_state"] = [{"text": "My edited text", "position": (9, 10)}]
        protected["sentinel_patches"] = ["keep existing background"]
        before = deepcopy(protected)
        processor = WebtoonBatchProcessor(main, Mock(), Mock(), Mock(), Mock())
        source = np.full((100, 200, 3), 255, dtype=np.uint8)
        with ExitStack() as stack:
            stack.enter_context(patch("pipeline.webtoon_batch.flow.ensure_path_materialized"))
            stack.enter_context(patch('modules.translation.processor.Translator'))
            opened = stack.enter_context(patch("pipeline.webtoon_batch.flow.Image.open"))
            opened.return_value.__enter__.return_value.size = (200, 100)
            read = stack.enter_context(patch.object(processor, "_read_virtual_image", return_value=source))
            detect = stack.enter_context(patch.object(processor, "_detect_blocks_for_page", return_value=[]))
            seam = stack.enter_context(patch.object(processor, "_build_pair_split_matches", return_value=([], set())))
            for name in ("_run_ocr_on_blocks", "_run_translation_on_blocks", "_save_final_rendered_page"):
                stack.enter_context(patch.object(processor, name))
            clean = stack.enter_context(patch.object(processor, "_inpaint_image_with_blocks", return_value=(None, None)))
            processor.webtoon_batch_process()
        self.assertEqual(protected, before)
        self.assertEqual([call.args[0].physical_page_path for call in read.call_args_list], ["before.png", "after.png"])
        self.assertEqual(detect.call_count, 2)
        self.assertEqual(clean.call_count, 2)
        seam.assert_not_called()
        main.image_skipped.emit.assert_called_once_with("protected.png", "Existing Rendered Text", PRESERVED_REASON)
        self.assertNotIn("protected.png", [call.args[0] for call in main.render_state_ready.emit.call_args_list])

    def test_preservation_notice_is_informational_and_report_is_actionable(self):
        image_controller = ImageStateController.__new__(ImageStateController)
        self.assertEqual(image_controller._resolve_skip_message_type("Existing Rendered Text", PRESERVED_REASON), MMessage.InfoType)
        message = image_controller._build_skip_message("page.png", "Existing Rendered Text", PRESERVED_REASON)
        self.assertEqual(message, "page.png\n" + PRESERVED_REASON)
        report = BatchReportController(SimpleNamespace(tr=lambda text: text))
        self.assertEqual(report._format_batch_skip_reason("Existing Rendered Text", PRESERVED_REASON), PRESERVED_REASON)


if __name__ == "__main__":
    unittest.main()
