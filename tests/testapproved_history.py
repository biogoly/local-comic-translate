"""Project-bound approved history from real review metadata to the LLM payload."""
import json
from types import SimpleNamespace
from unittest import TestCase, mock
from pathlib import Path

import numpy as np

from modules.review.issues import fingerprint_for_stage, mark_reviewed
from modules.translation.context import empty_translation_profile, normalize_translation_profile
from modules.translation.history import capture_approved_history
from modules.translation.processor import Translator
from modules.translation.request import build_request, TranslationContextError, StaleTranslationRequest
from modules.utils.textblock import TextBlock
from pipeline.cache_manager import CacheManager
from pipeline.translation_job import BatchTranslationSession, TranslationJob, StaleTranslationError
from testtranslation_request import engine_fixture
from testtranslation_profile_persistence import main_fixture, prepared_main, project_directory
from app.projects.project_state import load_state_from_proj_file
from app.projects.project_state_v2 import save_state_to_proj_file_v2


def block(source, target=''):
    return TextBlock(text_bbox=[5, 5, 40, 40], text=source, translation=target)


def approve(region, source_lang='English', target_lang='French'):
    fp = fingerprint_for_stage('translation', region, languages={'source': source_lang, 'target': target_lang})
    region.region_metadata = mark_reviewed(region.region_metadata, 'translation', fp)
    return region


class ApprovedHistoryTests(TestCase):
    def setUp(self):
        self.previous = approve(block('My dear', 'Ma chérie'))
        self.current = block('Meet me at noon.')
        self.future = approve(block('The ending', 'La fin'))
        self.paths = ['prior.png', 'now.png', 'future.png']
        groups = [[self.previous], [self.current], [self.future]]
        self.main = SimpleNamespace(image_files=self.paths, curr_img_idx=1, webtoon_mode=False,
            blk_list=[self.current], lang_mapping={}, settings_page=mock.MagicMock(),
            translation_profile=normalize_translation_profile({'include_approved_history': True}),
            image_states={path: {'page_id': str(i), 'blk_list': group,
                'source_lang': 'English', 'target_lang': 'French'}
                for i, (path, group) in enumerate(zip(self.paths, groups))})
        self.engine = engine_fixture()
        self.settings = self.main.settings_page
        self.settings.ui.tr.side_effect = lambda value: value
        self.settings.get_tool_selection.return_value = 'OpenAI'
        self.settings.get_llm_settings.return_value = {'extra_context': ''}

    def translator(self):
        with mock.patch('modules.translation.processor.TranslationFactory.create_engine', return_value=self.engine):
            return Translator(self.main, 'English', 'French')

    def request(self, **kwargs):
        args = dict(profile=self.main.translation_profile, history=capture_approved_history(self.main), page_id='1')
        return build_request(self.engine, [self.current], **{**args, **kwargs})

    def test_approved_fingerprint_not_protection_controls_collection(self):
        prior = self.main.image_states['prior.png']['blk_list']
        prior.append(approve(block('before the spelling edit', 'ancienne version')))
        prior[-1].text = 'after the spelling edit'
        kept = block('kept only', 'gardée')
        kept.region_metadata['protections']['keep_translation'] = True
        prior += [kept, approve(block('blank approval', '  '))]
        data = json.loads(self.request().compiled_context)
        self.assertEqual(data['approved_history'], [{'source': 'My dear', 'translation': 'Ma chérie'}])
        self.assertEqual(len(capture_approved_history(self.main).for_pair('en', 'fr')), 2)

    def test_boundaries_project_order_missing_ids_and_language_pairs(self):
        for page in ('', 'unknown', '0'):
            self.assertEqual(json.loads(self.request(page_id=page).compiled_context), {})
        self.current._page_id = '0'
        self.assertNotIn('approved_history', self.request().compiled_context)
        del self.current._page_id
        self.main.image_files = ['now.png', 'prior.png', 'future.png']
        self.assertEqual(self.request().compiled_context, '{}')
        self.main.image_files = self.paths
        for source, target in [('English', 'Brazilian Portuguese'), ('Auto', 'French')]:
            self.main.image_states['prior.png'].update(source_lang=source, target_lang=target)
            approve(self.previous, source, target)
            self.assertEqual(self.request().compiled_context, '{}')

    def test_current_canvas_overrides_saved_edits_and_deletions(self):
        self.main.curr_img_idx = 0
        self.main.blk_list = [self.previous.deep_copy()]
        self.main.blk_list[0].translation = 'edited but unapproved'
        self.assertEqual(self.request().compiled_context, '{}')
        self.main.blk_list.clear()
        self.assertEqual(self.request().compiled_context, '{}')

    def test_webtoon_loaded_mirrors_must_still_be_approved(self):
        self.main.webtoon_mode = True
        self.main.image_viewer = SimpleNamespace(webtoon_manager=SimpleNamespace(loaded_pages={0, 1}))
        self.main.blk_list = [self.previous.deep_copy(), self.current]
        self.assertIn('Ma chérie', self.request().compiled_context)
        self.main.blk_list[0].region_metadata['review']['approved_fingerprints'].clear()
        self.assertEqual(self.request().compiled_context, '{}')

    def test_default_off_unsupported_or_ambiguous_ownership_omits_history(self):
        for profile in (empty_translation_profile(), normalize_translation_profile({'include_approved_history': 'true'})):
            self.assertEqual(self.request(profile=profile).compiled_context, '{}')
        self.assertEqual(self.request(structured_context_supported=False).compiled_context, '{}')
        self.main.image_states['prior.png']['blk_list'].append(self.previous.deep_copy())
        self.assertEqual(self.request().compiled_context, '{}')

    def test_budget_prioritizes_glossary_and_accepts_only_complete_examples(self):
        profile = self.main.translation_profile
        profile.update(context_token_limit=128, glossary=[{'source': 'noon', 'target': 'midi'}])
        self.main.image_states['prior.png']['blk_list'].append(approve(block('x' * 4096, 'y' * 2000)))
        req = self.request()
        ref = json.loads(req.compiled_context)
        self.assertEqual(ref['glossary'][0]['preferred_translation'], 'midi')
        self.assertEqual(len(ref['approved_history']), 1)
        self.assertLessEqual(req.context_estimated_tokens, 128)
        base = self.request(profile={**profile, 'include_approved_history': False})
        self.engine.context_size = base.prompt_estimated_tokens + base.output_tokens_reserved + 256
        self.assertNotIn('approved_history', self.request().compiled_context)
        profile['notes'] = 'long ' * 600
        with self.assertRaises(TranslationContextError):
            self.request()

    def test_recent_order_and_dedup_and_portable_prompt(self):
        prior = self.main.image_states['prior.png']['blk_list']
        prior.extend(approve(block('hello '+str(i), 'bonjour '+str(i))) for i in range(30))
        prior.append(approve(block('hello 15', 'bonjour 15')))
        req = self.request()
        rows = json.loads(req.compiled_context)['approved_history']
        self.assertEqual(len(rows), 24)
        self.assertEqual(rows[0]['source'], 'hello 6')
        self.assertEqual(rows[-1]['source'], 'hello 15')
        self.assertEqual(set(rows[-1]), {'source', 'translation'})
        self.assertNotIn('prior.png', req.user_prompt)

    def test_effective_cache_identity_and_immutable_guard(self):
        before = self.request()
        self.future.translation = 'future edits have no effect'
        self.assertEqual(before.fingerprint, self.request().fingerprint)
        self.assertTrue(before.history_guard.current(self.main))
        self.previous.translation = 'no longer approved'
        after = self.request()
        self.assertNotEqual(before.fingerprint, after.fingerprint)
        self.assertFalse(before.history_guard.current(self.main))
        self.assertIn('Ma chérie', before.compiled_context)

    def test_job_commit_and_batch_guard_reject_revoked_reference(self):
        cache = CacheManager()
        img = np.ones((3, 3, 3), dtype='uint8')
        def transport(_request):
            return {'block_0': 'Viens à midi.'}
        translator = self.translator()
        translator.engine.translate_request = transport
        job = TranslationJob(self.main, translator, [self.current], img, '', cache,
                             resolve_blocks=lambda: self.main.blk_list, path='now.png')
        job.run()
        self.previous.translation = 'revoked'
        with self.assertRaises(StaleTranslationError):
            job.commit()
        self.assertEqual(self.current.translation, '')
        approve(self.previous)
        session = BatchTranslationSession(self.main, ['now.png'], lambda *_: self.translator())
        batch_translator = session.translators['English', 'French']
        batch_translator.engine.translate_request = transport
        copied = self.current.deep_copy()
        req = session.translate([copied], img, 'English', 'French', 'now.png', cache)
        self.assertIn('revoked', req.compiled_context)
        self.main.image_files = ['now.png', 'prior.png', 'future.png']
        with self.assertRaises(StaleTranslationError):
            session.validate('now.png')

    def test_provider_edit_during_generation_rejects_output(self):
        translator = self.translator()
        self.current._page_id = '1'
        def transport(request):
            self.assertIn('Ma chérie', request.user_prompt)
            self.previous.region_metadata['review']['approved_fingerprints'].clear()
            return {'block_0': 'Viens à midi.'}
        translator.engine.translate_request = transport
        with self.assertRaises(StaleTranslationRequest):
            translator.translate([self.current])
        self.assertEqual(self.current.translation, '')

    def test_each_virtual_request_is_guarded_until_its_page_is_published(self):
        session = BatchTranslationSession(self.main, ['now.png'], lambda *_: self.translator())
        translator = session.translators['English', 'French']
        translator.engine.translate_request = lambda req: {'block_0': 'Texte'}
        long_block = block('long dialogue ' * 200)
        image = np.ones((3, 3, 3), dtype='uint8')
        base = build_request(translator.engine, [long_block], image=image)
        translator.engine.context_size = base.prompt_estimated_tokens + base.output_tokens_reserved + 256
        cache = CacheManager()
        first = session.translate([self.current.deep_copy()], image, 'English', 'French', 'now.png', cache)
        second = session.translate([long_block], image, 'English', 'French', 'now.png', cache)
        self.assertTrue(first.history_guard.included)
        self.assertFalse(second.history_guard.included)
        self.previous.translation = 'revoked'
        with self.assertRaises(StaleTranslationError):
            session.validate('now.png')

    def test_history_is_derived_after_reopen_and_never_stored_as_duplicate_profile_text(self):
        with project_directory() as directory:
            main = prepared_main(directory)
            state = main.image_states[main.image_files[0]]
            state.update(source_lang='Japanese', target_lang='Brazilian Portuguese')
            row = approve(state['blk_list'][0], 'Japanese', 'Brazilian Portuguese')
            main.translation_profile['include_approved_history'] = True
            file = str(Path(directory) / 'reviewed.ctpr')
            save_state_to_proj_file_v2(main, file)
            reopened = main_fixture(Path(directory) / 'loaded')
            load_state_from_proj_file(reopened, file)
            self.assertTrue(reopened.translation_profile['include_approved_history'])
            self.assertNotIn('approved_history', reopened.translation_profile)
            records = capture_approved_history(reopened).for_pair('ja', 'pt-br')
            self.assertEqual([(r.region_id, r.target) for r in records], [(row.region_id, row.translation)])
            self.assertFalse(capture_approved_history(reopened).for_pair('ja', 'Portuguese'))
