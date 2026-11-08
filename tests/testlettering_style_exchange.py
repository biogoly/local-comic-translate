"""Portable presets cross a file boundary without carrying project state."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from modules.utils import lettering_style_exchange as exchange
from modules.utils.lettering_styles import MAX_STYLE_NAME, MAX_STYLE_PRESETS, empty_style_library


def entry(name, **properties):
    return {'name': name, 'properties': properties}


def library(*entries):
    result = empty_style_library()
    result['presets'] = [{'id': str(uuid4()), **item} for item in entries]
    return result


def document(presets):
    return {'format': exchange.STYLE_FILE_FORMAT, 'version': 1, 'presets': presets}


class StyleExchangeTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / '本文.json'

    def assert_error(self, code, function, *args):
        with self.assertRaises(exchange.StyleExchangeError) as failure:
            function(*args)
        self.assertEqual(failure.exception.code, code)

    def test_round_trip_strips_private_fields_ids_and_keeps_only_selected_properties(self):
        name = 'Tiêu đề · 本文'
        source = library(entry(name, font_family='Noto Sans 日本語', text_color='#aBc',
                               bold=False, font_file='private.otf'))
        source['presets'][0]['text'] = 'private dialogue'
        source['endpoint'] = 'http://private'
        before = copy.deepcopy(source)
        exchange.write_style_file(self.path, source)
        expected = [entry(name, font_family='Noto Sans 日本語', text_color='#aabbcc', bold=False)]
        raw = self.path.read_bytes()
        self.assertIn(name.encode('utf-8'), raw)
        self.assertNotIn(b'private', raw)
        self.assertEqual(json.loads(raw), document(expected))
        entries = exchange.read_style_file(self.path)
        self.assertEqual(entries, expected)
        merged = exchange.merge_imported_styles(empty_style_library(), entries)
        self.assertEqual((merged.added, merged.renamed, merged.skipped), (1, 0, 0))
        preset = merged.library['presets'][0]
        self.assertEqual(UUID(preset['id']).version, 4)
        self.assertNotEqual(preset['id'], source['presets'][0]['id'])
        self.assertEqual(source, before)
        preset['properties']['bold'] = True
        self.assertFalse(entries[0]['properties']['bold'])

    def test_numbered_conflicts_including_literal_numbered_names_are_repeatable(self):
        current = library(entry('Title', font_size=14), entry('Caption', text_color='#DDDDDD'))
        before = copy.deepcopy(current)
        entries = [entry('title', font_size=19), entry('title (2)', font_size=23),
                   entry('TITLE', font_size=27), entry('caption', text_color='#ddd')]
        merged = exchange.merge_imported_styles(current, entries)
        self.assertEqual(current, before)
        self.assertEqual((merged.added, merged.skipped, merged.renamed), (3, 1, 2))
        self.assertEqual([p['name'] for p in merged.library['presets']],
                         ['Title', 'Caption', 'title (3)', 'title (2)', 'TITLE (4)'])
        self.assertEqual(merged.library['presets'][:2], before['presets'])
        again = exchange.merge_imported_styles(merged.library, entries)
        self.assertEqual((again.added, again.skipped, again.renamed), (0, 4, 0))
        self.assertEqual(again.library, merged.library)

    def test_numbered_copy_is_found_beyond_gap_and_long_names_fit(self):
        name = 'É' * MAX_STYLE_NAME
        prefix = name[:-4]
        current = library(entry(name, bold=False), entry(name[:-5] + ' (78)', bold=True))
        merged = exchange.merge_imported_styles(current, [entry(name, bold=True),
                                                          entry(name, underline=True)])
        self.assertEqual((merged.skipped, merged.added, merged.renamed), (1, 1, 1))
        self.assertEqual(merged.library['presets'][-1]['name'], prefix + ' (2)')

    def test_merge_is_atomic_at_project_cap_and_accepts_duplicate_only_when_full(self):
        current = library(*(entry(f'Style {i}', bold=False) for i in range(MAX_STYLE_PRESETS - 1)))
        before = copy.deepcopy(current)
        self.assert_error('capacity', exchange.merge_imported_styles, current,
                          [entry('new', bold=True), entry('also new', italic=True)])
        self.assertEqual(current, before)
        full = exchange.merge_imported_styles(current, [entry('new', bold=True)])
        again = exchange.merge_imported_styles(full.library, [entry('new', bold=True)])
        self.assertEqual((again.added, again.skipped), (0, 1))

    def test_strict_entries_reject_unknown_fields_invalid_ranges_and_font_paths(self):
        current = library(entry('Keep', italic=True))
        before = copy.deepcopy(current)
        bad = [
            {'id': str(uuid4()), **entry('A', bold=True)}, entry('A', html='private'),
            entry('A'), entry('A', font_size=0), entry('A', line_spacing=11),
            entry('A', bold=1), entry('A', alignment=True), entry('A', text_color='#ffff'),
            entry('A', outline_width=float('nan')), entry('A', font_size=10**500),
            entry('A', font_family=r'C:\private\a.ttf'), entry('A', font_family='file:///local/font'),
            entry('A', font_family='fonts/private.otf'), entry('A', font_family='data:base64,AAA'),
            entry('A', font_family='A' * 129), entry('A' * (MAX_STYLE_NAME + 1), bold=True),
            entry('\ud800', bold=True), entry('A', font_family='\udff0'), entry(' ', bold=True),
            {'name': 'A', 'properties': []},
        ]
        for item in bad:
            with self.subTest(item=item):
                self.assert_error('invalid', exchange.merge_imported_styles, current,
                                  [entry('Good', bold=False), item])
                self.assertEqual(current, before)
                self.path.write_text(json.dumps(document([item])), encoding='utf-8')
                self.assert_error('invalid', exchange.read_style_file, self.path)

    def test_strict_file_envelope_encoding_limits_and_errors(self):
        self.assert_error('read', exchange.read_style_file, self.path)
        data = document([entry('A', italic=False)])
        for bad, code in [
            ([], 'format'), (dict(data, format='another-app'), 'format'),
            (dict(data, version=True), 'version'), (dict(data, version=2), 'version'),
            (dict(data, unknown=True), 'format'), (document([]), 'empty'),
            (document('unexpected'), 'invalid'),
            (document([entry('A', bold=True)] * (MAX_STYLE_PRESETS + 1)), 'invalid'),
        ]:
            with self.subTest(code=code, data=str(bad)[:60]):
                self.path.write_text(json.dumps(bad), encoding='utf-8')
                self.assert_error(code, exchange.read_style_file, self.path)
        for raw in [b'\xff', b'{}{}', b'[[[' * 1200,
                    json.dumps(data).replace('"italic": false', '"italic": false, "italic": true').encode()]:
            self.path.write_bytes(raw)
            self.assert_error('format', exchange.read_style_file, self.path)
        self.path.write_bytes(b'x' * (exchange.MAX_STYLE_FILE_BYTES + 1))
        self.assert_error('oversized', exchange.read_style_file, self.path)
        self.path.write_bytes(json.dumps(data).encode('utf-8-sig'))
        self.assertEqual(exchange.read_style_file(self.path), data['presets'])

    def test_export_failure_leaves_existing_target_and_no_partial_files(self):
        self.path.write_bytes(b'old')
        current = library(entry('Title', bold=False))
        with patch.object(exchange.os, 'replace', side_effect=PermissionError):
            self.assert_error('write', exchange.write_style_file, self.path, current)
        self.assert_error('empty', exchange.write_style_file, self.path, empty_style_library())
        invalid = library(entry('Title', font_family='C:/private/font.ttf'))
        self.assert_error('invalid', exchange.write_style_file, self.path, invalid)
        self.assertEqual(self.path.read_bytes(), b'old')
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_full_library_at_max_unicode_text_lengths_fits_byte_limit(self):
        items = []
        for index in range(MAX_STYLE_PRESETS):
            name = ('/SFX/ ' + str(index)).ljust(MAX_STYLE_NAME, '\U00020000')
            items.append(entry(name, font_family='\U00020001' * 128, font_size=499.99999999999994,
                               text_color='#fffffffe', outline=False, outline_color='#00000000',
                               outline_width=49.99999999999999, line_spacing=9.999999999999998,
                               alignment=4, bold=False, italic=True, underline=False))
        exchange.write_style_file(self.path, library(*items))
        self.assertLess(self.path.stat().st_size, exchange.MAX_STYLE_FILE_BYTES)
        self.assertEqual(exchange.read_style_file(self.path), items)


if __name__ == '__main__':
    unittest.main()
