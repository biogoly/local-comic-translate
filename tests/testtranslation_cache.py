import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from modules.utils.textblock import TextBlock
from pipeline.cache_manager import CacheManager
from pipeline.translation_handler import TranslationHandler
from modules.translation.llm.local import LocalLLMTranslation
from modules.translation.request import build_request


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
        translated_block_counts = []
        engine = LocalLLMTranslation()
        engine.source_lang, engine.target_lang = 'Spanish', 'English'
        engine.img_as_llm_input = True
        observed_images = []

        class _Translator:
            source_lang, target_lang, translator_key = 'Spanish', 'English', 'Local LLM'

            def prepare_request(self, blocks, image, extra_context):
                return build_request(engine, blocks, image, extra_context)

            def translate(self, blocks, image, extra_context, *, request):
                observed_images.append(image.copy())
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

        translator = _Translator()
        request = translator.prepare_request([selected], source, '')
        cache_key = cache._get_translation_cache_key(source, 'Spanish', 'English', 'Local LLM', '', request=request)
        with patch("pipeline.translation_handler.Translator", return_value=translator):
            handler.translate_image(single_block=True)
            self.assertEqual(selected.translation, 'two')
            # A further pixel repair must not invalidate a translation whose
            # visual context still comes from the original page.
            viewer.composited[:] = 100
            selected.translation = ''
            handler.translate_image(single_block=True)

        self.assertEqual(viewer.include_patches_calls, [False, False])
        self.assertEqual(translated_block_counts, [1])
        np.testing.assert_array_equal(observed_images[0], source)
        self.assertEqual(selected.translation, 'two')
        self.assertEqual(first.translation, 'one')
        self.assertEqual(len(cache.translation_cache[cache_key]), 1)
        self.assertIsNone(cache._get_cached_translation_for_block(cache_key, first))


if __name__ == "__main__":
    unittest.main()
