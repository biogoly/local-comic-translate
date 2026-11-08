"""Exact-wording OCR comparison without models, Qt or network access."""

from dataclasses import FrozenInstanceError
import unittest

from modules.artistic_edit.lettering_check import compare_lettering, normalize_lettering_text


class LetteringCheckTests(unittest.TestCase):
    def assert_spans(self, result):
        for difference in result.differences:
            self.assertEqual(difference.expected,
                             result.expected[difference.expected_start:difference.expected_end])
            self.assertEqual(difference.recognized,
                             result.recognized[difference.recognized_start:difference.recognized_end])

    def test_whitespace_and_canonically_equivalent_accents_match(self):
        result = compare_lettering(" MA CHÉRIE!\n\nCome here. ", "MA CHE\u0301RIE!\tCome\u00a0here.")
        self.assertEqual(result.status, "match")
        self.assertEqual(result.expected, "MA CHÉRIE! Come here.")
        self.assertEqual(result.recognized, result.expected)
        self.assertEqual(result.differences, ())

    def test_deliberate_misspelling_reports_the_changed_word(self):
        result = compare_lettering("A revolution called Rasputin", "A revoluton called Rasputin")
        self.assertEqual(result.status, "mismatch")
        self.assertEqual(len(result.differences), 1)
        difference, = result.differences
        self.assertEqual((difference.kind, difference.expected, difference.recognized),
                         ("replace", "revolution", "revoluton"))
        self.assert_spans(result)

    def test_accents_case_and_punctuation_are_not_silently_ignored(self):
        for expected, recognized in (("CHÉRIE", "CHERIE"), ("Étoile", "étoile"),
                                      ("Don't go!", "Don’t go!"), ("Go!", "Go"),
                                      ("ＡＢＣ", "ABC"), ("maçã", "maca"),
                                      ("hòa", "hoa"), ("cœur", "coeur")):
            with self.subTest(expected=expected, recognized=recognized):
                result = compare_lettering(expected, recognized)
                self.assertEqual(result.status, "mismatch")
                self.assertTrue(result.differences)
                self.assert_spans(result)

    def test_optional_case_insensitive_wording_preserves_original_strings(self):
        expected = 'COVER ILLUSTRATION COLORING: BRUSCO'
        recognized = 'Cover illustration coloring: BRUSCO'
        self.assertEqual(compare_lettering(expected, recognized).status, 'mismatch')
        result = compare_lettering(expected, recognized, case_sensitive=False)
        self.assertEqual(result.status, 'match')
        self.assertEqual(result.expected, expected)
        self.assertEqual(result.recognized, recognized)
        self.assertEqual(result.differences, ())

    def test_case_insensitive_checks_retain_spelling_accents_and_compatibility_differences(self):
        for expected, recognized in (
            ('COLORING', 'colorng'), ('CHÉRIE', 'cherie'), ('GO!', 'go'),
            ('ＡＢＣ', 'abc'), ('ﬁlm', 'film'), ('Straße', 'strasse'),
            ('ICE CREAM', 'icecream'), ('HÒA', 'hoa'), ('HELLO', 'he\u200bllo'),
        ):
            with self.subTest(expected=expected, recognized=recognized):
                result = compare_lettering(expected, recognized, case_sensitive=False)
                self.assertEqual(result.status, 'mismatch')
                self.assertTrue(result.differences)
                self.assert_spans(result)
        self.assertEqual(compare_lettering('CHÉRIE', 'chéRIE', case_sensitive=False).status, 'match')

    def test_case_insensitive_differences_keep_original_spans(self):
        result = compare_lettering('COVER ILLUSTRATION COLORING: BRUSCO',
                                   'Cover illustration colring: BRUSCO', case_sensitive=False)
        self.assertEqual(result.status, 'mismatch')
        difference, = result.differences
        self.assertEqual((difference.expected, difference.recognized), ('COLORING', 'colring'))
        self.assert_spans(result)
        for expected, recognized, kind in (
            ('COME HOME NOW!', 'Come now!', 'missing'),
            ('COME NOW!', 'Come home now!', 'extra'),
        ):
            with self.subTest(kind=kind):
                result = compare_lettering(expected, recognized, case_sensitive=False)
                self.assertEqual(result.status, 'mismatch')
                self.assertEqual(result.differences[0].kind, kind)
                self.assert_spans(result)

    def test_missing_and_extra_words_have_directional_differences(self):
        missing = compare_lettering("Come home now!", "Come now!")
        extra = compare_lettering("Come now!", "Come home now!")
        self.assertEqual(missing.status, "mismatch")
        self.assertEqual(extra.status, "mismatch")
        self.assertEqual(missing.differences[0].kind, "missing")
        self.assertEqual(extra.differences[0].kind, "extra")
        self.assertIn("home", missing.differences[0].expected)
        self.assertIn("home", extra.differences[0].recognized)
        self.assert_spans(missing)
        self.assert_spans(extra)

    def test_repeated_words_and_word_order_are_not_dropped(self):
        for expected, recognized in (("NO NO NO!", "NO NO!"), ("Never say never", "Never never say"),
                                      ("yes " * 250 + "no", "yes " * 250 + "go")):
            result = compare_lettering(expected, recognized)
            self.assertEqual(result.status, "mismatch")
            self.assertTrue(result.differences)
            self.assert_spans(result)

    def test_cjk_kana_hangul_differences_are_local_character_spans(self):
        for expected, recognized, old, new in (("你好世界", "你好世畀", "界", "畀"),
                                               ("こんにちは", "こんちは", "に", ""),
                                               ("안녕하세요", "안녕하세오", "요", "오"),
                                               ("新しい本", "新しい木", "本", "木")):
            with self.subTest(expected=expected):
                result = compare_lettering(expected, recognized)
                self.assertEqual(result.status, "mismatch")
                difference, = result.differences
                self.assertEqual((difference.expected, difference.recognized), (old, new))
                self.assert_spans(result)

    def test_word_boundaries_are_preserved_even_in_unspaced_scripts(self):
        for expected, recognized in (("ice cream", "icecream"), ("你好", "你 好")):
            result = compare_lettering(expected, recognized)
            self.assertEqual(result.status, "mismatch")
            self.assertTrue(result.differences)
            self.assert_spans(result)

    def test_unreadable_or_missing_ocr_is_uncertain_never_a_mismatch_or_match(self):
        for recognized in (None, "", " \n\t ", "???", "...", "\ufffd", "H\ufffdLLO", "😀"):
            with self.subTest(recognized=recognized):
                result = compare_lettering("HELLO", recognized)
                self.assertEqual(result.status, "uncertain")
                self.assertEqual(result.differences, ())
        self.assertEqual(compare_lettering("...", "...").status, "uncertain")

    def test_numbers_and_non_latin_readable_text_can_match(self):
        for text in ("123!", "Привет!", "こんにちは", "안녕하세요", "مرحبا", "ภาษาไทย", "Tiếng Việt"):
            with self.subTest(text=text):
                self.assertEqual(compare_lettering(text, text).status, "match")

    def test_blank_expected_cannot_be_checked_and_nonstring_inputs_are_rejected(self):
        for expected in ("", " \n\t "):
            with self.assertRaises(ValueError):
                compare_lettering(expected, "some OCR")
        for expected, recognized in ((None, "text"), (123, "text"), ("text", ["text"])):
            with self.assertRaises(TypeError):
                compare_lettering(expected, recognized)

    def test_output_contains_plain_text_data_without_invented_markup(self):
        result = compare_lettering('Read <title> & "quote"', 'Read <script> & "quote"')
        self.assertEqual(result.status, "mismatch")
        self.assertEqual(result.expected, 'Read <title> & "quote"')
        self.assertEqual(result.recognized, 'Read <script> & "quote"')
        self.assertEqual(result.differences[0].expected, "title")
        self.assertEqual(result.differences[0].recognized, "script")
        self.assert_spans(result)

    def test_comparison_snapshot_and_differences_are_immutable(self):
        result = compare_lettering("Hello", "Hallo")
        with self.assertRaises(FrozenInstanceError):
            result.status = "match"
        with self.assertRaises(FrozenInstanceError):
            result.differences[0].recognized = "Hello"

    def test_normalization_preserves_compatibility_and_zero_width_characters(self):
        self.assertEqual(normalize_lettering_text(" \t ﬁＡB\u200b \r\n"), "ﬁＡB\u200b")
        result = compare_lettering("HELLO", "HE\u200bLLO")
        self.assertEqual(result.status, "mismatch")
        self.assert_spans(result)


if __name__ == "__main__":
    unittest.main()
