import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from modules.utils.textblock import TextBlock
from pipeline.cache_manager import CacheManager
from pipeline.translation_handler import TranslationHandler


class _Combo:
    def __init__(self, value):
        self.value = value

    def currentText(self):
        return self.value


class _CheckBox:
    def isChecked(self):
        return False


class _Viewer:
    def __init__(self, source, composited):
        self.source = source
        self.composited = composited
        self.include_patches_calls = []

    def hasPhoto(self):
        return True

    def get_image_array(self, include_patches=True):
        self.include_patches_calls.append(include_patches)
        return self.composited if include_patches else self.source


class TranslationCacheTests(unittest.TestCase):
    def test_single_block_translation_uses_source_image_cache_after_inpainting(self):
        source = np.zeros((40, 40, 3), dtype=np.uint8)
        composited = source.copy()
        composited[10:20, 10:20] = 255
        viewer = _Viewer(source, composited)
        first = TextBlock(
            text_bbox=np.array([1, 1, 9, 9]),
            text="uno",
            translation="one",
        )
        selected = TextBlock(
            text_bbox=np.array([20, 20, 35, 35]),
            text="dos",
        )
        cache = CacheManager()
        cache_key = cache._get_translation_cache_key(
            source, "Spanish", "English", "Local LLM", ""
        )
        cache._cache_translation_results(cache_key, [first])
        first.translation = ""

        translated_block_counts = []

        class _Translator:
            def translate(self, blocks, image, extra_context):
                del image, extra_context
                translated_block_counts.append(len(blocks))
                for block in blocks:
                    block.translation = "two"

        settings = SimpleNamespace(
            ui=SimpleNamespace(uppercase_checkbox=_CheckBox()),
            get_llm_settings=lambda: {"extra_context": ""},
            get_tool_selection=lambda name: "Local LLM" if name == "translator" else "",
        )
        main = SimpleNamespace(
            s_combo=_Combo("Spanish"),
            t_combo=_Combo("English"),
            lang_mapping={"Spanish": "Spanish", "English": "English"},
            image_viewer=viewer,
            settings_page=settings,
            blk_list=[first, selected],
        )
        pipeline = SimpleNamespace(get_selected_block=lambda: selected)
        handler = TranslationHandler(main, cache, pipeline)

        with patch("pipeline.translation_handler.Translator", return_value=_Translator()):
            handler.translate_image(single_block=True)

        self.assertEqual(viewer.include_patches_calls, [False])
        self.assertEqual(translated_block_counts, [1])
        self.assertEqual(selected.translation, "two")
        self.assertEqual(len(cache.translation_cache[cache_key]), 2)


if __name__ == "__main__":
    unittest.main()
