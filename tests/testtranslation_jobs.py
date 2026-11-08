"""Real request/cache staging with a deterministic LLM transport, no network."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from modules.translation.context import normalize_translation_profile
from modules.translation.llm.base import BaseLLMTranslation
from modules.translation.llm.local_response import TranslationResponseError
from modules.translation.processor import Translator
from modules.utils.textblock import TextBlock
from app.controllers.manual_workflow import ManualWorkflowController
from pipeline.cache_manager import CacheManager
from pipeline.translation_job import (BatchTranslationSession, StaleTranslationError, TranslationJob,
                                      eligible_translation_blocks)


class FixtureLLM(BaseLLMTranslation):
    def __init__(self):
        super().__init__()
        self.model = 'test-model'
        self.api_key = 'VERYSECRET'
        self.api_url = 'https://username:password@example.test/v1?key=VERYSECRET'
        self.source_lang, self.target_lang = 'English', 'French'
        self.temperature, self.top_p, self.top_k = .2, .9, 40
        self.max_tokens = 512
        self.context_size = 16384
        self.img_as_llm_input = True
        self.transport = {'calls': [], 'callback': None, 'error': None, 'response': None}

    def _perform_translation(self, user_prompt, system_prompt, image):
        raw = json.loads(user_prompt.split('Translate this JSON:\n')[-1])
        self.transport['calls'].append((raw, image, user_prompt))
        callback = self.transport['callback']
        if callback is not None:
            callback()
        if self.transport['error'] is not None:
            raise self.transport['error']
        if self.transport['response'] is not None:
            return self.transport['response']
        return json.dumps({key: 'Translated ' + text for key, text in raw.items()})


class TranslationJobTests(unittest.TestCase):
    def setUp(self):
        self.blocks = [TextBlock(text='Alpha', translation='Old A', text_class='text_free'),
                       TextBlock(text='Beta', translation='Old B', text_class='text_bubble')]
        self.image = np.full((8, 12, 3), 128, np.uint8)
        settings = SimpleNamespace(ui=SimpleNamespace(tr=lambda text: text),
            get_tool_selection=lambda tool: 'OpenAI',
            get_llm_settings=lambda: {'extra_context': 'Natural dialogue'})
        self.main = SimpleNamespace(settings_page=settings, lang_mapping={},
            translation_profile=normalize_translation_profile({'glossary': [
                {'id': 'term', 'source': 'Alpha', 'target': 'Alfa', 'source_lang': 'English', 'target_lang': 'French'}]}),
            image_files=['page.png', 'other.png'], curr_img_idx=0, blk_list=self.blocks, webtoon_mode=False,
            image_states={'page.png': {'page_id': 'page1', 'source_lang': 'English', 'target_lang': 'French', 'blk_list': self.blocks},
                          'other.png': {'page_id': 'page2', 'source_lang': 'English', 'target_lang': 'French', 'blk_list': []}},
            _translation_project_epoch=0, _translation_cancel_epoch=0)
        self.engine = FixtureLLM()
        self.cache = CacheManager()
        self.factory_patch = patch('modules.translation.processor.TranslationFactory.create_engine', return_value=self.engine)
        self.factory_patch.start()
        self.addCleanup(self.factory_patch.stop)

    def make_job(self, *, upper_case=False, image=True, loader=None):
        translator = Translator(self.main, 'English', 'French')
        return TranslationJob(self.main, translator, self.blocks, self.image if image else None,
            'Natural dialogue', self.cache, resolve_blocks=lambda: self.main.blk_list,
            path='page.png', require_current_page=True, upper_case=upper_case, image_loader=loader)

    def make_session(self):
        return BatchTranslationSession(self.main, ['page.png', 'other.png'], Translator)

    def test_run_stages_results_and_image_until_commit(self):
        job = self.make_job()
        self.image[:] = 0
        job.run()
        self.assertEqual([b.translation for b in self.blocks], ['Old A', 'Old B'])
        self.assertTrue(np.all(self.engine.transport['calls'][0][1] == 128))
        self.assertTrue(job.commit())
        self.assertEqual([b.translation for b in self.blocks], ['Translated Alpha', 'Translated Beta'])
        self.assertIsNot(job.blocks[0], self.blocks[0])

    def test_reordering_during_inference_applies_by_stable_identity(self):
        by_id = {b.region_id: b.text for b in self.blocks}
        job = self.make_job()
        self.engine.transport['callback'] = self.blocks.reverse
        job.run()
        job.commit()
        for block in self.blocks:
            self.assertEqual(block.translation, 'Translated ' + by_id[block.region_id])

    def test_changed_regions_reject_whole_commit_without_partial_application(self):
        changes = ('source', 'target', 'protection', 'delete', 'duplicate', 'identity')
        for change in changes:
            with self.subTest(change=change):
                self.setUp_scenario()
                job = self.make_job()
                job.run()
                if change == 'source':
                    self.blocks[-1].text = 'User source edit'
                elif change == 'target':
                    self.blocks[-1].translation = 'User translation'
                elif change == 'protection':
                    self.blocks[-1].region_metadata['protections']['keep_translation'] = True
                elif change == 'delete':
                    self.blocks.pop()
                elif change == 'duplicate':
                    self.blocks.append(self.blocks[-1].deep_copy())
                else:
                    self.blocks[-1].region_id = 'new-identity'
                before = [b.translation for b in self.blocks]
                with self.assertRaises(StaleTranslationError):
                    job.commit()
                self.assertEqual([b.translation for b in self.blocks], before)
                self.assertEqual(self.blocks[0].translation, 'Old A')

    def setUp_scenario(self):
        self.blocks[:] = [TextBlock(text='Alpha', translation='Old A', text_class='text_free'),
                          TextBlock(text='Beta', translation='Old B', text_class='text_bubble')]

    def test_cancel_close_page_switch_and_context_edits_reject_complete_apply(self):
        actions = (
            lambda: setattr(self.main, '_translation_cancel_epoch', 1),
            lambda: setattr(self.main, '_translation_project_epoch', 1),
            lambda: setattr(self.main, 'curr_img_idx', 1),
            lambda: self.main.image_files.remove('page.png'),
            lambda: self.main.image_states['page.png'].update(page_id='replacement-page'),
            lambda: self.main.translation_profile.update(notes='New context'),
        )
        for action in actions:
            with self.subTest(action=action):
                self.main._translation_cancel_epoch = self.main._translation_project_epoch = 0
                self.main.curr_img_idx = 0
                self.main.image_files[:] = ['page.png', 'other.png']
                self.main.image_states['page.png']['page_id'] = 'page1'
                self.main.translation_profile['notes'] = ''
                job = self.make_job().run()
                action()
                with self.assertRaises(StaleTranslationError):
                    job.commit()
                self.assertEqual([b.translation for b in self.blocks], ['Old A', 'Old B'])

    def test_cancel_before_run_never_loads_image_or_calls_transport(self):
        loader = Mock(return_value=self.image)
        job = self.make_job(image=False, loader=loader)
        self.main._translation_cancel_epoch += 1
        with self.assertRaises(StaleTranslationError):
            job.run()
        loader.assert_not_called()
        self.assertEqual(self.engine.transport['calls'], [])

    def test_cache_stores_raw_wording_and_uppercase_is_postprocess(self):
        first = self.make_job(upper_case=True).run()
        first.commit()
        self.assertEqual(self.blocks[0].translation, 'TRANSLATED ALPHA')
        self.assertEqual(self.cache._get_cached_translation_for_block(first.key, self.blocks[0]), 'Translated Alpha')
        second = self.make_job(upper_case=False).run()
        second.commit()
        self.assertEqual(second.key, first.key)
        self.assertEqual(len(self.engine.transport['calls']), 1)
        self.assertEqual(self.blocks[0].translation, 'Translated Alpha')
        for secret in ('VERYSECRET', 'username', 'password'):
            self.assertNotIn(secret, repr(second.key))
            self.assertNotIn(secret, second.request.preview)

    def test_relevant_context_and_model_changes_invalidate_real_cache(self):
        first = self.make_job().run()
        self.main.translation_profile['glossary'].append({'id': 'unrelated', 'source': 'Unrelated', 'target': 'Inutile'})
        irrelevant = self.make_job().run()
        self.assertEqual(first.key, irrelevant.key)
        self.assertEqual(len(self.engine.transport['calls']), 1)
        self.main.translation_profile['glossary'][0]['target'] = 'Alpha nouvelle'
        relevant = self.make_job().run()
        self.assertNotEqual(first.key, relevant.key)
        self.assertEqual(len(self.engine.transport['calls']), 2)
        self.engine.model = 'different-model'
        different_model = self.make_job().run()
        self.assertNotEqual(relevant.key, different_model.key)
        self.assertEqual(len(self.engine.transport['calls']), 3)

    def test_failed_or_incomplete_response_preserves_all_existing_targets(self):
        for response in ('{"block_0":"Partial"}', '{"block_0":"First","block_1":""}',
                         '{"block_0":"First","block_0":"Duplicate","block_1":"Second"}'):
            with self.subTest(response=response):
                self.engine.transport['response'] = response
                job = self.make_job()
                with self.assertRaises(TranslationResponseError):
                    job.run()
                self.assertIsNone(job.result)
                self.assertFalse(job.commit())
                self.assertEqual([b.translation for b in self.blocks], ['Old A', 'Old B'])
                self.assertFalse(self.cache.translation_cache)

    def test_text_free_is_eligible_and_keep_translation_is_independent(self):
        self.blocks[0].region_metadata['protections']['skip_auto_cleanup'] = True
        self.blocks[1].region_metadata['protections']['keep_translation'] = True
        self.assertEqual(eligible_translation_blocks(self.blocks), [self.blocks[0]])
        self.assertEqual(eligible_translation_blocks(self.blocks, single_block=self.blocks[1]), [self.blocks[1]])

    def test_selected_pages_current_live_edit_invalidates_saved_snapshot(self):
        for field in ('text', 'translation'):
            with self.subTest(field=field):
                self.setUp_scenario()
                saved = [block.deep_copy() for block in self.blocks]
                self.main.image_states['page.png']['blk_list'] = saved
                workflow = ManualWorkflowController(self.main)
                self.assertIs(workflow._translation_blocks_for_page('page.png'), self.blocks)
                job = TranslationJob(self.main, Translator(self.main, 'English', 'French'), saved,
                    self.image, '', self.cache,
                    resolve_blocks=lambda: workflow._translation_blocks_for_page('page.png'),
                    path='page.png', require_current_page=False)
                job.run()
                setattr(self.blocks[-1], field, 'Live user correction')
                with self.assertRaises(StaleTranslationError):
                    job.commit()
                self.assertEqual(getattr(self.blocks[-1], field), 'Live user correction')
                self.assertEqual(self.blocks[0].translation, 'Old A')
                self.assertEqual([block.translation for block in saved], ['Old A', 'Old B'])
                self.assertEqual(saved[-1].text, 'Beta')

    def test_commit_mirrors_target_only_and_preserves_distinct_live_saved_layouts(self):
        saved = [block.deep_copy() for block in self.blocks]
        self.main.image_states['page.png']['blk_list'] = list(reversed(saved))
        self.blocks[0].xyxy = np.array([40, 50, 140, 120])
        self.blocks[0].font_color = (230, 20, 70)
        self.blocks[0].region_metadata['layout_preferences']['bound_width'] = 100
        saved[0].xyxy = np.array([10, 20, 70, 60])
        saved[0].font_color = (10, 50, 200)
        saved[0].region_metadata['layout_preferences']['bound_width'] = 60
        other_owner = saved[0].deep_copy()
        other_owner.translation = 'Other page target'
        self.main.image_states['other.png']['blk_list'] = [other_owner]
        live_before = copy.deepcopy(self.blocks[0].__dict__)
        saved_before = copy.deepcopy(saved[0].__dict__)
        workflow = ManualWorkflowController(self.main)
        job = TranslationJob(self.main, Translator(self.main, 'English', 'French'), saved,
            self.image, '', self.cache,
            resolve_blocks=lambda: workflow._translation_blocks_for_page('page.png'),
            path='page.png', require_current_page=False)
        job.run()
        self.assertTrue(job.commit())
        for block in self.blocks + saved:
            self.assertEqual(block.translation, 'Translated ' + block.text)
        for block, prior in ((self.blocks[0], live_before), (saved[0], saved_before)):
            np.testing.assert_array_equal(block.xyxy, prior['xyxy'])
            self.assertEqual(block.font_color, prior['font_color'])
            self.assertEqual(block.region_metadata, prior['region_metadata'])
            self.assertEqual(block.text, prior['text'])
        self.assertEqual(other_owner.translation, 'Other page target')
        self.assertIsNot(self.blocks[0], saved[0])

    def test_batch_validate_detects_pending_source_changes_before_transport(self):
        session = self.make_session()
        self.blocks[-1].text = 'Revised before processing'
        with self.assertRaises(StaleTranslationError):
            session.translate(self.blocks, self.image, 'English', 'French', 'page.png', self.cache)
        self.assertEqual(self.engine.transport['calls'], [])
        self.assertEqual([b.translation for b in self.blocks], ['Old A', 'Old B'])

    def test_job_and_batch_keep_owner_scoped_speakers_through_copies(self):
        identity = self.blocks[0].region_id
        self.main.translation_profile = normalize_translation_profile({
            'characters': [{'id': 'quiet', 'name': 'Quiet', 'notes': 'Quiet voice'},
                           {'id': 'loud', 'name': 'Loud', 'notes': 'Loud voice'}],
            'speakers': [{'page_id': 'page1', 'region_id': identity, 'character_id': 'quiet'},
                         {'page_id': 'page2', 'region_id': identity, 'character_id': 'loud'}],
        })
        for explicit_owner, expected in ((None, 'Quiet voice'), ('page2', 'Loud voice')):
            with self.subTest(owner=explicit_owner):
                if explicit_owner:
                    self.blocks[0]._page_id = explicit_owner
                job = self.make_job().run()
                self.assertEqual(json.loads(job.request.compiled_context)['speakers']['block_0']['voice_notes'], expected)
                session = self.make_session()
                request = session.translate(self.blocks, self.image, 'English', 'French', 'page.png', self.cache)
                self.assertEqual(json.loads(request.compiled_context)['speakers']['block_0']['voice_notes'], expected)

    def test_batch_reordering_preserves_response_identity(self):
        session = self.make_session()
        self.engine.transport['callback'] = self.blocks.reverse
        session.translate(self.blocks, self.image, 'English', 'French', 'page.png', self.cache)
        self.assertEqual({b.text: b.translation for b in self.blocks},
                         {'Alpha': 'Translated Alpha', 'Beta': 'Translated Beta'})

    def test_batch_cancel_and_page_edits_after_transport_never_apply_partial_targets(self):
        for mutation in ('cancel', 'project', 'source', 'target', 'deleted', 'profile'):
            with self.subTest(mutation=mutation):
                self.setUp_scenario()
                self.cache = CacheManager()
                self.main._translation_cancel_epoch = self.main._translation_project_epoch = 0
                self.main.translation_profile['notes'] = ''
                session = self.make_session()
                def change():
                    if mutation == 'cancel':
                        self.main._translation_cancel_epoch += 1
                    elif mutation == 'project':
                        self.main._translation_project_epoch += 1
                    elif mutation == 'source':
                        self.blocks[-1].text = 'User source'
                    elif mutation == 'target':
                        self.blocks[-1].translation = 'User target'
                    elif mutation == 'deleted':
                        self.blocks.pop()
                    else:
                        self.main.translation_profile['notes'] = 'Context changed'
                self.engine.transport['callback'] = change
                with self.assertRaises(StaleTranslationError):
                    session.translate(self.blocks, self.image, 'English', 'French', 'page.png', self.cache)
                self.assertEqual(self.blocks[0].translation, 'Old A')
                if mutation == 'target':
                    self.assertEqual(self.blocks[-1].translation, 'User target')

    def test_batch_response_failure_leaves_page_and_cache_untouched(self):
        self.engine.transport['error'] = RuntimeError('Transport failed')
        session = self.make_session()
        with self.assertRaisesRegex(RuntimeError, 'Transport failed'):
            session.translate(self.blocks, self.image, 'English', 'French', 'page.png', self.cache)
        self.assertEqual([b.translation for b in self.blocks], ['Old A', 'Old B'])
        self.assertFalse(self.cache.translation_cache)
