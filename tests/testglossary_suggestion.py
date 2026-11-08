"""Grounded glossary candidates, request budgeting and the configured model transport."""
import json
import unittest
from unittest.mock import patch

import requests

from modules.translation.glossary_suggestion import glossary_source_conflicts, prepare_glossary_suggestion
from modules.translation.history import ApprovedExample
from modules.translation.llm.claude import ClaudeTranslation
from modules.translation.llm.deepseek import DeepseekTranslation
from modules.translation.llm.gemini import GeminiTranslation
from modules.translation.llm.gpt import GPTTranslation
from modules.translation.llm.local import LocalLLMTranslation
from modules.translation.request import estimate_tokens
from modules.translation.suggestion_reference import SuggestionError
from testsummary_suggestion import translator
from testspatial_context_providers import image_bytes, payload_from, response_for, user_text


def row(source, target, index=0, *, source_lang='en', target_lang='fr'):
    return ApprovedExample('internal-page-id', f'internal-region-{index}', source_lang, target_lang, source, target)


def candidate(source, target, example=1):
    return {'source': source, 'target': target, 'example': example}


class GlossarySuggestionTests(unittest.TestCase):
    def setUp(self):
        self.translator = translator()
        self.pairs = [row('Captain Star visits Gray Harbor.', 'Capitaine Étoile visite Port Gris.'),
                      row('Mara carries a key.', 'Mara porte une clé.', 1)]

    def prepare(self, entries=(), pairs=None, **kwargs):
        return prepare_glossary_suggestion(self.translator, entries, self.pairs if pairs is None else pairs, **kwargs)

    def reply(self, req, terms):
        with patch('requests.post', return_value=response_for(json.dumps({'terms': terms}, ensure_ascii=False))):
            return [(c.source, c.target, c.example_index) for c in req.run()]

    def test_result_requires_both_literal_terms_in_the_same_cited_pair(self):
        request = self.prepare()
        values = [
            candidate('Captain Star', 'Capitaine Étoile'),
            candidate('Gray Harbor', 'Mara'),  # Only present in a different pair.
            candidate('Captain Star', 'Capitaine Etoile'),  # Accent was removed.
            candidate('Captain Star', 'Capitaine Étoile', -1),
            candidate('Mara', 'Mara', 2), candidate('Mara', 'Mara', 2),
            candidate('Mara', 'Mara', 3), candidate('Gray Harbor', 'Port Gris', 0)]
        self.assertEqual(self.reply(request, values), [('Captain Star', 'Capitaine Étoile', 0), ('Mara', 'Mara', 1)])

    def test_unicode_literal_boundaries_and_wrapped_canonically_equivalent_terms(self):
        req = self.prepare(pairs=[row('Joanna sees Gray\nHarbor.', 'La Canne et l’Étoile.'),
                                  row('先輩と空港', 'Café de la Lune', 1)])
        self.assertEqual(self.reply(req, [
            candidate('anna', 'Canne'), candidate('Gray Harbor', 'anne'),
            candidate('Gray Harbor', 'E\u0301toile'), candidate('Joanna', 'Ca'),
            candidate('先輩', 'Cafe\u0301', 2)]),
            [('Gray Harbor', 'Étoile', 0), ('先輩', 'Café', 1)])

    def test_conflicts_and_duplicates_preserve_every_existing_glossary_row(self):
        entries = [{'source': 'Captain', 'aliases': ['CAPTAIN STAR'], 'approved': False, 'enabled': False,
                    'target': 'Protected draft', 'note': 'private'},
                   {'source': 'Gray Harbor', 'source_lang': 'ja', 'target_lang': 'fr'}]
        before = json.dumps(entries)
        request = self.prepare(entries)
        self.assertNotIn('Protected draft', request.user_prompt)
        self.assertNotIn('private', request.user_prompt)
        self.assertEqual(self.reply(request, [candidate('Captain Star', 'Capitaine Étoile'),
                                             candidate('Gray Harbor', 'Port Gris')]),
                         [('Gray Harbor', 'Port Gris', 0)])
        self.assertTrue(glossary_source_conflicts(' Captain   Star ', entries, 'English', 'French'))
        self.assertFalse(glossary_source_conflicts('Gray Harbor', entries, 'English', 'French'))
        self.assertEqual(json.dumps(entries), before)
        request = self.prepare(pairs=[*self.pairs, row('Captain Star leaves.', 'Étoile part.', 2)])
        self.assertEqual(self.reply(request, [candidate('Captain Star', 'Capitaine Étoile'),
                                             candidate('Captain Star', 'Étoile', 3)]), [])

    def test_only_same_dialect_complete_pairs_are_disclosed_and_fingerprint_tracks_collisions(self):
        self.translator.target_lang_en = 'Brazilian Portuguese'
        rows = [row('A lighthouse', 'Farol', target_lang='pt-BR'),
                row('Secret French', 'Un phare', 1),
                row('Secret European Portuguese', 'Outro farol', 2, target_lang='pt-PT'),
                row('A long caption' * 1000, 'Farol', 3, target_lang='pt-BR')]
        request = self.prepare(pairs=rows, context_token_limit=128)
        self.assertEqual(len(request.examples), 1)
        self.assertEqual(json.loads(request.user_prompt)['approved_translations'],
                         [dict(example=1, source='A lighthouse', target='Farol')])
        for forbidden in ('Secret', 'internal-', 'A long caption'):
            self.assertNotIn(forbidden, request.user_prompt)
        self.assertLessEqual(estimate_tokens(request.user_prompt), 128)
        updated = self.prepare([{'source': 'Lighthouse', 'target_lang':'pt-BR'}], rows, context_token_limit=128)
        self.assertNotEqual(request.fingerprint, updated.fingerprint)
        self.assertEqual(request.user_prompt, updated.user_prompt)
        with self.assertRaisesRegex(SuggestionError, '^no_examples$'):
            self.prepare(pairs=rows[1:3])
        with self.assertRaisesRegex(SuggestionError, '^budget$'):
            self.prepare(pairs=rows[-1:], context_token_limit=128)

    def test_structural_errors_and_oversized_outputs_are_redacted(self):
        req = self.prepare()
        for reply in ('junk', '["private"]', '{"terms":[],"terms":[]}', '{"terms":[{}]}',
                      '{"terms":[{"source":"a","source":"b","target":"a","example":1}]}',
                      json.dumps({'terms': [candidate('Mara', 'Mara', True)]}),
                      json.dumps({'terms': [candidate('Mara', 'Mara', 1.5)]}),
                      json.dumps({'terms': [candidate('Mara', 'Mara', 2)] * 9}), 'A' * 16001):
            with self.subTest(reply=reply[:60]), patch('requests.post', return_value=response_for(reply)):
                with self.assertRaisesRegex(SuggestionError, '^response$'):
                    req.run()
        self.assertEqual(self.reply(req, []), [])
        with patch('requests.post', side_effect=requests.ConnectionError('private API key')) as post:
            with self.assertRaisesRegex(SuggestionError, '^request$'):
                req.run()
        post.assert_called_once()

    def test_summary_and_notes_stay_local_and_all_five_adapters_send_one_text_request(self):
        routes = ((LocalLLMTranslation, 'Local LLM'), (GPTTranslation, 'OpenAI'), (ClaudeTranslation, 'Anthropic Claude'),
                  (GeminiTranslation, 'Google Gemini'), (DeepseekTranslation, 'Deepseek'))
        reply = json.dumps({'terms': [candidate('Mara', 'Mara', 2)]})
        for cls, name in routes:
            with self.subTest(name=name):
                t = translator(cls, name)
                t.engine.img_as_llm_input = True
                req = prepare_glossary_suggestion(t, [], self.pairs)
                with patch('requests.post', return_value=response_for(reply)) as post:
                    self.assertEqual(req.run()[0].source, 'Mara')
                post.assert_called_once()
                payload = payload_from(post.call_args)
                self.assertEqual(image_bytes(payload), [])
                self.assertEqual(user_text(payload), req.user_prompt)
                self.assertTrue(t.engine.img_as_llm_input)
