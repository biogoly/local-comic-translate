"""Portable project lettering data, including old formats and project isolation."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock
from types import SimpleNamespace
from uuid import uuid4
import zipfile

import msgpack

from app.projects.lettering_styles import reset_lettering_project, styles_for_project
from app.projects.project_state import load_state_from_proj_file
from app.projects.project_state_v2 import (
    _init_schema, close_cached_connection, load_state_from_proj_file_v2, save_state_to_proj_file_v2,
)
from modules.utils.lettering_styles import (
    empty_style_library, normalize_style_library, normalize_style_properties, validate_style_library,
)
from testtranslation_profile_persistence import database, main_fixture, prepared_main, project_directory


def library():
    return {'schema_version': 1, 'presets': [
        {'id': str(uuid4()), 'name': 'Dialogue · 日本語 · Đối thoại',
         'properties': {'font_family': 'Segoe UI', 'font_size': 24, 'text_color': '#fed',
                        'outline': True, 'outline_color': '#FF001122', 'outline_width': 1.5,
                        'line_spacing': 1.2, 'alignment': 4, 'bold': False, 'italic': True,
                        'underline': False, 'font_file': 'private-font.ttf'},
         'text': 'private comic text'},
    ], 'endpoint': 'https://private.invalid'}


class LetteringStyleSchemaTests(unittest.TestCase):
    def test_plain_portable_data_is_canonical_and_independent(self):
        raw = library()
        before = copy.deepcopy(raw)
        result = normalize_style_library(raw)
        self.assertEqual(raw, before)
        self.assertEqual(result['presets'][0]['properties']['text_color'], '#ffeedd')
        self.assertEqual(result['presets'][0]['properties']['outline_color'], '#ff001122')
        serialized = json.dumps(result, ensure_ascii=False)
        self.assertNotIn('private', serialized)
        self.assertNotIn('font_file', serialized)
        result['presets'][0]['properties']['font_family'] = 'Another font'
        self.assertEqual(raw, before)

    def test_partial_properties_never_add_unspecified_formatting(self):
        self.assertEqual(normalize_style_properties({'text_color': '#AbC', 'unknown': True}),
                         {'text_color': '#aabbcc'})
        self.assertEqual(normalize_style_properties({'italic': True}), {'italic': True})
        self.assertEqual(normalize_style_properties(None), {})

    def test_invalid_values_are_omitted_or_rejected_in_strict_mode(self):
        for key, value in [('font_size', float('nan')), ('font_size', 0), ('font_size', True),
                           ('line_spacing', float('inf')), ('line_spacing', -1),
                           ('outline_width', 10000), ('text_color', 'red'),
                           ('outline_color', '#abcd'), ('font_family', 'abc\nxyz'),
                           ('font_family', 'a' * 129), ('bold', 1), ('alignment', 132),
                           ('alignment', True)]:
            with self.subTest(key=key, value=value):
                self.assertEqual(normalize_style_properties({key: value}), {})
                with self.assertRaises(ValueError):
                    normalize_style_properties({key: value}, strict=True)

    def test_safe_old_defaults_and_unknown_version_are_distinct(self):
        for value in (None, {}, [], 'invalid'):
            self.assertEqual(normalize_style_library(value), empty_style_library())
        for version in (999, True, '1'):
            with self.assertRaises(ValueError):
                normalize_style_library({'schema_version': version, 'presets': []})
        with self.assertRaises(ValueError):
            validate_style_library({})

    def test_duplicates_and_invalid_entries_do_not_create_ambiguous_ids(self):
        raw = library()
        original = copy.deepcopy(raw['presets'][0])
        raw['presets'] += [original, {'id': 'invalid', 'name': 'Bad', 'properties': {'bold': True}},
                           {'id': str(uuid4()), 'name': '', 'properties': {'bold': True}},
                           {'id': str(uuid4()), 'name': 'No properties', 'properties': {}}]
        self.assertEqual(len(normalize_style_library(raw)['presets']), 1)
        with self.assertRaises(ValueError):
            validate_style_library(raw)

    def test_limits_and_names_are_bounded(self):
        raw = {'schema_version': 1, 'presets': [
            {'id': str(uuid4()), 'name': 'Dialogue', 'properties': {'bold': False}} for _ in range(101)]}
        self.assertEqual(len(normalize_style_library(raw)['presets']), 100)
        with self.assertRaises(ValueError):
            validate_style_library(raw)
        raw['presets'] = raw['presets'][:1]
        raw['presets'][0]['name'] = 'n' * 81
        with self.assertRaises(ValueError):
            validate_style_library(raw)

    def test_reset_and_project_projection_do_not_share_mutable_defaults(self):
        first, second = SimpleNamespace(), SimpleNamespace()
        reset_lettering_project(first)
        reset_lettering_project(second)
        first.lettering_styles['presets'].append(library()['presets'][0])
        self.assertEqual(second.lettering_styles, empty_style_library())
        projected = styles_for_project(first)
        projected['presets'].clear()
        self.assertEqual(len(first.lettering_styles['presets']), 1)


class LetteringStylePersistenceTests(unittest.TestCase):
    def tearDown(self):
        close_cached_connection()

    def test_current_sqlite_save_reopen_and_save_as_keep_unicode_presets(self):
        with project_directory() as directory:
            source = prepared_main(directory)
            source.lettering_styles = library()
            expected = normalize_style_library(source.lettering_styles)
            project = str(Path(directory) / 'styles.ctpr')
            save_state_to_proj_file_v2(source, project)
            with database(project) as connection:
                blob = connection.execute('SELECT manifest_blob FROM project_manifest').fetchone()[0]
                manifest = msgpack.unpackb(blob, raw=False)
            self.assertEqual(manifest['lettering_styles'], expected)
            self.assertNotIn(b'private', msgpack.packb(manifest['lettering_styles']))
            loaded = main_fixture(Path(directory) / 'loaded')
            load_state_from_proj_file(loaded, project)
            self.assertEqual(loaded.lettering_styles, expected)
            save_as = str(Path(directory) / 'copy.ctpr')
            save_state_to_proj_file_v2(loaded, save_as)
            reopened = main_fixture(Path(directory) / 'reopened')
            load_state_from_proj_file_v2(reopened, save_as)
            self.assertEqual(reopened.lettering_styles, expected)
            reopened.lettering_styles['presets'].clear()
            self.assertEqual(loaded.lettering_styles, expected)

    def test_zip_legacy_without_library_clears_previous_project(self):
        with project_directory() as directory:
            main = prepared_main(directory)
            main.lettering_styles = library()
            project = str(Path(directory) / 'old.ctpr')
            with zipfile.ZipFile(project, 'w') as archive:
                archive.writestr('state.msgpack', msgpack.packb({'original_image_files': [], 'image_states': {}}))
            load_state_from_proj_file(main, project)
            self.assertEqual(main.lettering_styles, empty_style_library())

    def test_zip_and_interim_sqlite_preserve_style_library(self):
        with project_directory() as directory:
            expected = normalize_style_library(library())
            state = {'original_image_files': [], 'image_states': {}, 'lettering_styles': expected}
            for kind in ('zip', 'sqlite'):
                with self.subTest(kind=kind):
                    project = str(Path(directory) / f'{kind}.ctpr')
                    if kind == 'zip':
                        with zipfile.ZipFile(project, 'w') as archive:
                            archive.writestr('state.msgpack', msgpack.packb(state, use_bin_type=True))
                    else:
                        with database(project) as connection:
                            _init_schema(connection)
                            connection.execute('INSERT INTO project_state VALUES(1, ?)',
                                               (msgpack.packb(state, use_bin_type=True),))
                    loaded = main_fixture(Path(directory) / kind)
                    load_state_from_proj_file(loaded, project)
                    self.assertEqual(loaded.lettering_styles, expected)

    def test_current_sqlite_without_library_clears_previous_project(self):
        with project_directory() as directory:
            main = prepared_main(directory)
            main.lettering_styles = library()
            project = str(Path(directory) / 'old.ctpr')
            save_state_to_proj_file_v2(main, project)
            with database(project) as connection:
                manifest = msgpack.unpackb(connection.execute('SELECT manifest_blob FROM project_manifest').fetchone()[0], raw=False)
                del manifest['lettering_styles']
                connection.execute('UPDATE project_manifest SET manifest_blob = ?', (msgpack.packb(manifest),))
            load_state_from_proj_file(main, project)
            self.assertEqual(main.lettering_styles, empty_style_library())

    def test_failed_read_never_keeps_previous_library(self):
        with project_directory() as directory:
            main = prepared_main(directory)
            main.lettering_styles = library()
            with self.assertRaises(FileNotFoundError):
                load_state_from_proj_file(main, str(Path(directory) / 'missing.ctpr'))
            self.assertEqual(main.lettering_styles, empty_style_library())

    def test_future_schema_rejects_load_without_installing_partial_metadata(self):
        with project_directory() as directory:
            source = prepared_main(directory)
            source.lettering_styles = library()
            project = str(Path(directory) / 'future.ctpr')
            save_state_to_proj_file_v2(source, project)
            with database(project) as connection:
                manifest = msgpack.unpackb(connection.execute('SELECT manifest_blob FROM project_manifest').fetchone()[0], raw=False)
                manifest['lettering_styles']['schema_version'] = 99
                connection.execute('UPDATE project_manifest SET manifest_blob = ?', (msgpack.packb(manifest),))
            loaded = main_fixture(Path(directory) / 'loaded')
            loaded.lettering_styles = library()
            with self.assertRaises(ValueError):
                load_state_from_proj_file(loaded, project)
            self.assertEqual(loaded.lettering_styles, empty_style_library())
            self.assertEqual(loaded.translation_profile['glossary'], [])

    def test_later_load_failure_does_not_install_new_presets(self):
        with project_directory() as directory:
            source = prepared_main(directory)
            source.lettering_styles = library()
            project = str(Path(directory) / 'styles.ctpr')
            save_state_to_proj_file_v2(source, project)
            loaded = main_fixture(Path(directory) / 'loaded')
            loaded.lettering_styles = library()
            loaded.batch_report_ctrl = SimpleNamespace(restore_latest_batch_report=Mock(side_effect=ValueError('broken report')))
            with self.assertRaisesRegex(ValueError, 'broken report'):
                load_state_from_proj_file(loaded, project)
            self.assertEqual(loaded.lettering_styles, empty_style_library())


if __name__ == '__main__':
    unittest.main()
