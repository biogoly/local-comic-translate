"""Regression tests: detector classes must not gate the user's text regions."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from modules.ocr.processor import OCRProcessor
from modules.utils.textblock import TextBlock
from pipeline.batch_processor import BatchProcessor
from pipeline.cache_manager import CacheManager
from pipeline.webtoon_batch.chunk import ChunkMixin
from pipeline.webtoon_batch.render import RenderMixin


def mixed_blocks():
    return [
        TextBlock(
            text_bbox=np.array([10 + 60 * i, 10, 60 + 60 * i, 60]),
            text_class=kind,
            text="original " + str(i),
            translation="translated " + str(i),
        )
        for i, kind in enumerate(["text_bubble", "text_free", ""])
    ]


def render_settings():
    return SimpleNamespace(
        upper_case=False, outline=False, font_family="Arial", color="#000000",
        max_font_size=30, min_font_size=8, line_spacing=1.0, outline_width=0.0,
        bold=False, italic=False, underline=False, alignment_id=0, direction="ltr",
    )


class AllRegionWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_ocr_receives_bubbles_free_text_and_hand_drawn_regions(self):
        blocks = mixed_blocks()
        main = SimpleNamespace(
            lang_mapping={},
            settings_page=SimpleNamespace(
                get_tool_selection=lambda _name: "Default",
                ui=SimpleNamespace(tr=lambda text: text),
            ),
        )
        processor = OCRProcessor()
        processor.initialize(main, "Spanish")
        engine = Mock()
        with patch("modules.ocr.processor.OCRFactory.create_engine", return_value=engine):
            processor.process(np.zeros((100, 200, 3), dtype=np.uint8), blocks)
        self.assertIs(engine.process_image.call_args.args[1], blocks)
        self.assertEqual(len(engine.process_image.call_args.args[1]), 3)

    def test_regular_batch_translates_inpaints_and_renders_all_classes(self):
        blocks = mixed_blocks()
        image = np.full((100, 200, 3), 255, dtype=np.uint8)
        settings = SimpleNamespace(
            get_tool_selection=lambda name: "Local LLM" if name == "translator" else "Default",
            is_gpu_enabled=lambda: False,
            get_llm_settings=lambda: {"extra_context": ""},
            get_export_settings=lambda: {
                "export_raw_text": False, "export_translated_text": False,
                "export_inpainted_image": False,
            },
        )
        main = SimpleNamespace(
            image_files=["page.png"], curr_img_idx=0, blk_list=blocks,
            current_worker=None, settings_page=settings,
            image_states={"page.png": {
                "source_lang": "Spanish", "target_lang": "English", "viewer_state": {},
            }},
            file_handler=SimpleNamespace(
                should_pre_materialize=lambda _paths: False, archive_info=[],
            ),
            render_settings=render_settings,
            button_to_alignment={0: Qt.AlignmentFlag.AlignCenter},
        )
        for name in (
            "progress_update", "image_skipped", "batch_page_edit_started",
            "patches_processed", "render_state_ready", "batch_page_edit_finished",
        ):
            setattr(main, name, Mock())
        detector = SimpleNamespace(
            block_detector_cache=SimpleNamespace(detect=lambda _image: blocks),
            annotate_language_if_auto=Mock(),
        )
        inpainter = SimpleNamespace(get_inpainted_patches=lambda *_args: [])
        cache = CacheManager()
        batch = BatchProcessor(
            main, cache, detector, inpainter, SimpleNamespace(ocr=Mock())
        )
        with (
            patch("pipeline.batch_processor.ensure_path_materialized"),
            patch("pipeline.batch_processor.imk.read_image", return_value=image),
            patch("pipeline.batch_processor.Translator") as translator,
            patch("pipeline.batch_processor.get_config", return_value=SimpleNamespace()),
            patch("pipeline.batch_processor.generate_mask", return_value=np.ones((100, 200), dtype=np.uint8)) as mask,
            patch("pipeline.batch_processor.call_inpaint_image", return_value=image) as inpaint,
            patch("pipeline.batch_processor.get_best_render_area"),
            patch("pipeline.batch_processor.pyside_word_wrap", side_effect=lambda text, *_a, **_kw: (text, 12, 50, 50)),
        ):
            batch.batch_process()
        self.assertCountEqual(translator.return_value.translate.call_args.args[0], blocks)
        self.assertCountEqual(mask.call_args.args[1], blocks)
        self.assertCountEqual(inpaint.call_args.kwargs["blk_list"], blocks)
        self.assertEqual(len(main.image_states["page.png"]["viewer_state"]["text_items_state"]), 3)
        self.assertEqual(len(next(iter(cache.translation_cache.values()))), 3)
        main.image_skipped.emit.assert_not_called()

    def test_webtoon_inpainting_does_not_filter_by_text_class(self):
        blocks = mixed_blocks()
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        handler = SimpleNamespace(
            main_page=SimpleNamespace(settings_page=None),
            inpainting=Mock(),
        )
        with (
            patch("pipeline.webtoon_batch.chunk.get_config", return_value=SimpleNamespace()),
            patch("pipeline.webtoon_batch.chunk.generate_mask", return_value=np.ones((100, 200), dtype=np.uint8)) as mask,
            patch("pipeline.webtoon_batch.chunk.call_inpaint_image", return_value=image) as inpaint,
        ):
            ChunkMixin._inpaint_image_with_blocks(handler, image, blocks)
        self.assertEqual([block.text_class for block in mask.call_args.args[1]], ["text_bubble", "text_free", ""])
        self.assertEqual(len(inpaint.call_args.kwargs["blk_list"]), 3)

    def test_webtoon_rendering_does_not_filter_by_text_class(self):
        blocks = mixed_blocks()
        state = {"target_lang": "English", "viewer_state": {}}
        handler = SimpleNamespace(
            main_page=SimpleNamespace(
                image_states={"page.png": state}, render_settings=render_settings,
                button_to_alignment={0: Qt.AlignmentFlag.AlignCenter},
                image_viewer=SimpleNamespace(), webtoon_mode=False,
            ),
            _get_page_scene_offset=lambda _index: 0,
        )
        with (
            patch("pipeline.webtoon_batch.render.get_best_render_area") as area,
            patch("pipeline.webtoon_batch.render.pyside_word_wrap", side_effect=lambda text, *_a, **_kw: (text, 12, 50, 50)),
        ):
            RenderMixin._store_page_text_items(handler, 0, "page.png", blocks, (100, 200, 3))
        self.assertEqual(area.call_args.args[0], blocks)
        self.assertEqual(len(state["viewer_state"]["text_items_state"]), 3)


if __name__ == "__main__":
    unittest.main()
