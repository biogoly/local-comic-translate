"""Project reference selection: Unicode, dialect scope, and manual approval."""
import copy
import unittest

from modules.translation.context import (empty_translation_profile, normalize_translation_profile,
                                         compile_context, term_matches)
from modules.utils.textblock import TextBlock


def term(source='Rasputin', target='Raspoutine', **extra):
    return dict(id='term-1', source=source, target=target, aliases=[], source_lang='', target_lang='',
                match='auto', enabled=True, approved=True, note='', **extra)


class TranslationContextTests(unittest.TestCase):
    def test_legacy_and_malformed_profiles_have_independent_empty_defaults(self):
        for value in (None, [], '', {'glossary': 'bad', 'speakers': 3, 'characters': None}):
            profile = normalize_translation_profile(value)
            self.assertEqual(profile, empty_translation_profile())
        first, second = empty_translation_profile(), empty_translation_profile()
        first['glossary'].append({})
        self.assertEqual(second['glossary'], [])

    def test_allowlist_does_not_store_credentials_paths_or_mutable_input(self):
        original = {'schema_version': 1, 'api_key': 'not-for-projects', 'runtime_path': 'C:/models',
                    'glossary': [dict(term(), api_key='hidden')], 'notes': 'Context'}
        normalized = normalize_translation_profile(original)
        self.assertNotIn('api_key', normalized)
        self.assertNotIn('runtime_path', normalized)
        self.assertNotIn('api_key', normalized['glossary'][0])
        normalized['glossary'][0]['aliases'].append('alias')
        self.assertEqual(original['glossary'][0]['aliases'], [])

    def test_future_profile_version_cannot_be_silently_downgraded(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported translation profile version'):
            normalize_translation_profile({'schema_version': 2, 'notes': 'future format'})
        self.assertEqual(normalize_translation_profile({})['schema_version'], 1)

    def test_auto_matches_japanese_without_space_boundaries(self):
        self.assertTrue(term_matches('山田', '昨日山田さんと話しました', 'auto', 'Japanese'))
        self.assertFalse(term_matches('山田', '昨日山田さんと話しました', 'whole_word', 'Japanese'))
        self.assertTrue(term_matches('ＡＢＣ', 'abc', 'whole_word', 'English'))

    def test_whole_word_does_not_match_inside_other_words(self):
        self.assertTrue(term_matches('Ann', 'ANN says hello.', source_lang='English'))
        self.assertFalse(term_matches('Ann', 'annual announcement', source_lang='English'))
        self.assertTrue(term_matches('Ann', 'announcement', 'substring', 'English'))

    def test_vietnamese_accents_normalize_without_removing_diacritics(self):
        self.assertTrue(term_matches('Cảm ơn', 'Cảm ơn bạn!', source_lang='Vietnamese'))
        self.assertFalse(term_matches('ma', 'má', source_lang='Vietnamese'))

    def test_aliases_select_only_matching_approved_entries(self):
        profile = empty_translation_profile()
        profile['glossary'] = [dict(term(), aliases=['Grigori']),
                               dict(term('Duma', 'Douma'), id='2', approved=False),
                               dict(term('tower', 'tour'), id='3', enabled=False),
                               dict(term('elsewhere', 'ailleurs'), id='4')]
        result = compile_context(profile, [TextBlock(text='Grigori visits the Duma tower.')], 'English', 'French')
        self.assertEqual(len(result['glossary']), 1)
        self.assertEqual(result['glossary'][0]['matched_terms'], ['Grigori'])
        self.assertEqual(result['glossary'][0]['preferred_translation'], 'Raspoutine')

    def test_portuguese_dialects_and_language_pairs_do_not_leak(self):
        profile = empty_translation_profile()
        profile['glossary'] = [dict(term('bus', 'autocarro'), source_lang='English', target_lang='Portuguese'),
                               dict(term('bus', 'ônibus'), id='2', source_lang='en', target_lang='pt-BR')]
        blocks = [TextBlock(text='Take the bus.')]
        portugal = compile_context(profile, blocks, 'English', 'Portuguese')
        brazil = compile_context(profile, blocks, 'English', 'Brazilian Portuguese')
        self.assertEqual(portugal['glossary'][0]['preferred_translation'], 'autocarro')
        self.assertEqual(brazil['glossary'][0]['preferred_translation'], 'ônibus')
        self.assertEqual(compile_context(profile, blocks, 'English', 'French'), {})
        self.assertEqual(compile_context(profile, blocks, 'Auto', 'Portuguese'), {})

    def test_speaker_notes_require_explicit_unambiguous_assignment_and_approval(self):
        profile = empty_translation_profile()
        block, other = TextBlock(text='Hello'), TextBlock(text='Anna is here')
        profile['characters'] = [{'id': 'anna', 'name': 'Anna', 'target': 'Anne', 'notes': 'Formal, restrained.'},
                                 {'id': 'draft', 'name': 'Draft', 'notes': 'Do not use', 'approved': False}]
        profile['speakers'] = [{'page_id': 'p1', 'region_id': block.region_id, 'character_id': 'anna'},
                               {'page_id': 'p1', 'region_id': other.region_id, 'character_id': 'draft'}]
        result = compile_context(profile, [block, other], 'English', 'French', page_id='p1')
        self.assertEqual(result['speakers'], {'block_0': {'name': 'Anna', 'preferred_translation': 'Anne',
                                                         'voice_notes': 'Formal, restrained.'}})
        self.assertEqual(compile_context(profile, [block], 'English', 'French', page_id='p2'), {})
        profile['speakers'].append({'page_id': 'p2', 'region_id': block.region_id, 'character_id': 'anna'})
        self.assertEqual(compile_context(profile, [block], 'English', 'French'), {})

    def test_user_notes_and_summary_are_explicit_reference_data(self):
        profile = empty_translation_profile()
        profile.update(notes='The scene is in 1916.', summary='The characters have not met before.')
        expected = {'project_notes': profile['notes'], 'user_written_summary': profile['summary']}
        before = copy.deepcopy(profile)
        self.assertEqual(compile_context(profile, [], 'English', 'French'), expected)
        self.assertEqual(profile, before)


if __name__ == '__main__':
    unittest.main()
