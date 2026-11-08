"""Actual provider payloads with synthetic images; all HTTP calls are mocked."""
import base64
import io
import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import requests

from modules.translation.llm.claude import ClaudeTranslation
from modules.translation.llm.deepseek import DeepseekTranslation
from modules.translation.llm.gemini import GeminiTranslation
from modules.translation.llm.gpt import GPTTranslation
from modules.translation.llm.local import LocalLLMTranslation
from modules.translation.request import build_request
from modules.utils.textblock import TextBlock
from testtranslation_request import translator_fixture


PROVIDERS = (GPTTranslation, ClaudeTranslation, GeminiTranslation,
             LocalLLMTranslation, DeepseekTranslation)


def configured_engine(cls, *, enabled=True, images=True, deepseek_model='deepseek-flash'):
    settings = MagicMock()
    settings.ui.tr.side_effect = lambda value: value
    settings.get_llm_settings.return_value = {
        'spatial_context_enabled': enabled, 'image_input_enabled': images,
        'local_runtime': 'external', 'local_endpoint': 'http://127.0.0.1:9/v1',
        'local_model': 'fixture-model', 'local_context_size': 8192,
        'local_max_tokens': 512,
    }
    settings.get_credentials.return_value = {
        'api_key': 'fixture-key',
        'model': deepseek_model if cls is DeepseekTranslation else 'fixture-model',
    }
    engine = cls()
    engine.initialize(settings, 'English', 'French', 'fixture-provider')
    return engine


def response_for(content):
    response = MagicMock(status_code=200)
    # Each real adapter reads its own wire-format response envelope.
    response.json.return_value = {
        'choices': [{'message': {'content': content}}],
        'content': [{'text': content}],
        'candidates': [{'content': {'parts': [{'text': content}]}}],
    }
    return response


def payload_from(call):
    kwargs = call.kwargs
    return kwargs['json'] if 'json' in kwargs else json.loads(kwargs['data'])


def user_parts(payload):
    if 'contents' in payload:
        return payload['contents'][0]['parts']
    content = next(message['content'] for message in payload['messages'] if message['role'] == 'user')
    return [{'text': content}] if isinstance(content, str) else content


def user_text(payload):
    return '\n'.join(part['text'] for part in user_parts(payload) if 'text' in part)


def image_bytes(payload):
    result = []
    for part in user_parts(payload):
        if 'image_url' in part:
            result.append(base64.b64decode(part['image_url']['url'].split(',', 1)[1]))
        elif part.get('type') == 'image':
            result.append(base64.b64decode(part['source']['data']))
        elif 'inline_data' in part:
            result.append(base64.b64decode(part['inline_data']['data']))
    return result


class SpatialContextProviderTests(unittest.TestCase):
    def blocks(self):
        # Deliberately repeated OCR: ordering must follow stable request IDs,
        # rather than matching identical source strings to the first region.
        return [TextBlock(text_bbox=np.array([6., 4., 30., 20.]), text='Hello', translation='old one'),
                TextBlock(text_bbox=np.array([36., 24., 54., 36.]), text='Hello', translation='old two')]

    def image(self):
        y, x = np.indices((40, 60))
        return np.stack((x * 3, y * 5, x + y), axis=-1).astype(np.uint8)

    def test_every_vision_adapter_delivers_matching_image_and_spatial_metadata(self):
        for cls in PROVIDERS:
            with self.subTest(provider=cls.__name__):
                engine = configured_engine(cls)
                translator, _ = translator_fixture(engine)
                blocks, pixels = self.blocks(), self.image()
                original = pixels.copy()
                request = translator.prepare_request(blocks, pixels)
                spatial = json.loads(request.spatial_reference_json)
                self.assertEqual((spatial['image_width'], spatial['image_height']), (60, 40))
                self.assertEqual([row['id'] for row in spatial['regions']], ['block_0', 'block_1'])
                np.testing.assert_allclose(spatial['regions'][0]['bbox_normalized'], [.1, .1, .5, .5])
                np.testing.assert_allclose(spatial['regions'][1]['bbox_normalized'], [.6, .6, .9, .9])
                pixels[:] = 0  # A prepared request owns its original pixels.
                blocks.reverse()  # IDs, not list position, route the response.
                with patch('requests.post', return_value=response_for(
                        '{"block_0":"Bonjour premier","block_1":"Bonjour second"}')) as post:
                    translator.translate(blocks, request=request)
                self.assertEqual(post.call_count, 1)
                payload = payload_from(post.call_args)
                self.assertEqual(user_text(payload), request.user_prompt)
                self.assertIn(request.spatial_reference_json, user_text(payload))
                sent_images = image_bytes(payload)
                self.assertEqual(len(sent_images), 1)
                decoded = Image.open(io.BytesIO(sent_images[0]))
                self.assertEqual(decoded.size, (60, 40))
                if cls is LocalLLMTranslation:  # Local uses lossless PNG.
                    np.testing.assert_array_equal(np.asarray(decoded), original)
                self.assertEqual([block.translation for block in blocks], ['Bonjour second', 'Bonjour premier'])

    def test_disabled_spatial_mode_keeps_images_and_ordinary_prompt(self):
        for cls in PROVIDERS:
            with self.subTest(provider=cls.__name__):
                engine = configured_engine(cls, enabled=False)
                request = build_request(engine, self.blocks(), self.image())
                self.assertEqual(request.spatial_reference_json, '')
                with patch('requests.post', return_value=response_for(
                        '{"block_0":"Un","block_1":"Deux"}')) as post:
                    engine.translate_request(request)
                payload = payload_from(post.call_args)
                self.assertNotIn('Visual reference data:', user_text(payload))
                self.assertEqual(len(image_bytes(payload)), 1)

    def test_image_disabled_and_text_only_models_never_send_spatial_metadata(self):
        engines = [configured_engine(cls, images=False) for cls in PROVIDERS]
        engines.append(configured_engine(DeepseekTranslation, deepseek_model='deepseek-v4-flash'))
        for engine in engines:
            with self.subTest(provider=type(engine).__name__, model=engine.model):
                request = build_request(engine, self.blocks(), self.image())
                self.assertEqual(request.spatial_reference_json, '')
                self.assertIsNone(request.image)
                with patch('requests.post', return_value=response_for(
                        '{"block_0":"Un","block_1":"Deux"}')) as post:
                    engine.translate_request(request)
                payload = payload_from(post.call_args)
                self.assertNotIn('Visual reference data:', user_text(payload))
                self.assertEqual(image_bytes(payload), [])

    def test_local_schema_retry_preserves_image_spatial_context_and_requested_keys(self):
        engine = configured_engine(LocalLLMTranslation)
        blocks = self.blocks()
        request = build_request(engine, blocks, self.image())
        with patch('requests.post', side_effect=[response_for('{"block_0":"Incomplete"}'),
                   response_for('{"block_0":"Un","block_1":"Deux"}')]) as post:
            translated = engine.translate_request(request)
        self.assertEqual(translated, {'block_0': 'Un', 'block_1': 'Deux'})
        self.assertEqual(post.call_count, 2)
        first, second = [payload_from(call) for call in post.call_args_list]
        self.assertEqual(user_text(first), request.user_prompt)
        self.assertTrue(user_text(second).startswith(request.user_prompt))
        self.assertIn(request.spatial_reference_json, user_text(second))
        self.assertEqual(image_bytes(first), image_bytes(second))
        self.assertEqual([block.translation for block in blocks], ['old one', 'old two'])

    def test_provider_errors_and_wrong_ids_leave_existing_translations_intact(self):
        for cls in PROVIDERS:
            for failure in ('timeout', 'wrong_ids'):
                with self.subTest(provider=cls.__name__, failure=failure):
                    engine = configured_engine(cls)
                    translator, _ = translator_fixture(engine)
                    blocks = self.blocks()
                    with patch('requests.post') as post:
                        if failure == 'timeout':
                            post.side_effect = requests.Timeout('fixture timeout')
                        else:
                            post.return_value = response_for('{"block_0":"Un","block_99":"Deux"}')
                        with self.assertRaises(Exception):
                            translator.translate(blocks, self.image())
                    self.assertEqual([block.translation for block in blocks], ['old one', 'old two'])
                    self.assertEqual(post.call_count, 3 if failure == 'wrong_ids' and cls is LocalLLMTranslation else 1)


if __name__ == '__main__':
    unittest.main()
