"""Project context follows portable project identity and never the last project."""
import copy
from contextlib import closing, contextmanager
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from uuid import uuid4
import zipfile

import imkit as imk
import msgpack
import numpy as np

from app.projects.parsers import ProjectEncoder
from app.projects.project_metadata import normalize_project_pages
from app.projects.project_state import load_state_from_proj_file
from app.projects.project_state_v2 import (
    _init_schema, close_cached_connection, load_state_from_proj_file_v2, save_state_to_proj_file_v2,
)
from app.projects.translation_profile import profile_for_project, reset_translation_project
from modules.translation.context import empty_translation_profile, normalize_translation_profile
from modules.utils.textblock import TextBlock


@contextmanager
def project_directory():
    with tempfile.TemporaryDirectory() as directory:
        try:
            yield directory
        finally:
            close_cached_connection()


@contextmanager
def database(path):
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            yield connection


def main_fixture(root, pages=()):
    Path(root).mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        temp_dir=str(root), image_files=list(pages), image_data={}, in_memory_history={},
        image_history={}, image_states={}, image_patches={}, current_history_index={},
        displayed_images=set(), loaded_images=[], curr_img_idx=0, webtoon_mode=False,
        image_viewer=SimpleNamespace(webtoon_view_state={}),
        settings_page=SimpleNamespace(get_llm_settings=lambda: {'extra_context': 'Legacy context'}),
        translation_profile=empty_translation_profile(), _translation_project_epoch=3,
    )


def prepared_main(root):
    root = Path(root)
    page = str(root / 'page.png')
    imk.write_image(page, np.full((18, 24, 3), 239, dtype=np.uint8))
    main = main_fixture(root, [page])
    block = TextBlock(text_bbox=np.array([1, 2, 14, 12]), text='お嬢様', translation='Senhorita')
    main.image_states[page] = {'blk_list': [block]}
    normalize_project_pages(main)
    character_id = str(uuid4())
    profile = {
        'schema_version': 1,
        'glossary': [{'id': str(uuid4()), 'source': 'お嬢様', 'aliases': ['おじょうさま'],
                      'target': 'Senhorita', 'source_lang': 'Japanese', 'target_lang': 'Portuguese (Brazil)',
                      'note': 'Título formal; giữ nguyên dấu.', 'match': 'substring',
                      'enabled': True, 'approved': True, 'api_key': 'must-not-save'}],
        'characters': [{'id': character_id, 'name': 'Đặng', 'aliases': ['ダン'], 'target': 'Đặng',
                        'source_lang': '', 'target_lang': '', 'notes': 'Formal, porém afetuoso.',
                        'enabled': True, 'approved': True}],
        'notes': '日本語 → português; giữ dấu tiếng Việt.', 'summary': 'Uma história sobre família.',
        'speakers': [{'page_id': main.image_states[page]['page_id'], 'region_id': block.region_id,
                      'character_id': character_id}],
        'context_token_limit': 2048, 'api_key': 'must-not-save', 'endpoint': 'https://private.invalid/key',
    }
    main.translation_profile = profile
    return main


class TranslationProfilePersistenceTests(unittest.TestCase):
    def tearDown(self):
        close_cached_connection()

    def test_history_setting_and_accepted_summary_survive_reopen_and_stay_in_their_project(self):
        from modules.review.issues import fingerprint_for_stage, mark_reviewed
        with project_directory() as directory:
            project_a = prepared_main(directory)
            state = project_a.image_states[project_a.image_files[0]]
            state.update(source_lang='Japanese', target_lang='Portuguese (Brazil)')
            block, = state['blk_list']
            fingerprint = fingerprint_for_stage('translation', block, languages={
                'source': state['source_lang'], 'target': state['target_lang']})
            block.region_metadata = mark_reviewed(block.region_metadata, 'translation', fingerprint)
            project_a.translation_profile.update(include_approved_history=True, summary='Mara perdeu a chave.')
            path = str(Path(directory) / 'with-history.ctpr')
            save_state_to_proj_file_v2(project_a, path)
            reopened = main_fixture(Path(directory) / 'reopened')
            load_state_from_proj_file(reopened, path)
            self.assertTrue(reopened.translation_profile['include_approved_history'])
            self.assertEqual(reopened.translation_profile['summary'], 'Mara perdeu a chave.')
            saved_block, = reopened.image_states[reopened.image_files[0]]['blk_list']
            self.assertEqual(saved_block.region_metadata['review']['approved_fingerprints']['translation'], fingerprint)
            other_path = str(Path(directory) / 'legacy.ctpr')
            with zipfile.ZipFile(other_path, 'w') as archive:
                archive.writestr('state.msgpack', msgpack.packb({'original_image_files': [], 'image_states': {}},
                                                                use_bin_type=True))
            load_state_from_proj_file(reopened, other_path)
            self.assertFalse(reopened.translation_profile['include_approved_history'])
            self.assertEqual(reopened.translation_profile['summary'], '')

    def test_sqlite_profile_unicode_speaker_and_save_as_are_portable(self):
        with project_directory() as directory:
            main = prepared_main(directory)
            project = str(Path(directory) / 'first.ctpr')
            expected = profile_for_project(main)
            save_state_to_proj_file_v2(main, project)
            with database(project) as connection:
                blob = connection.execute('SELECT manifest_blob FROM project_manifest').fetchone()[0]
                manifest = msgpack.unpackb(blob, raw=False)
            self.assertEqual(manifest['translation_profile'], expected)
            self.assertNotIn(b'must-not-save', blob)
            self.assertNotIn(b'private.invalid', blob)

            loaded = main_fixture(Path(directory) / 'loaded')
            self.assertEqual(load_state_from_proj_file(loaded, project), 'Legacy context')
            self.assertEqual(loaded.translation_profile, expected)
            self.assertGreater(loaded._translation_project_epoch, 3)
            self.assertNotEqual(loaded.image_files, main.image_files)
            self.assertEqual(loaded.translation_profile['speakers'][0]['page_id'],
                             loaded.image_states[loaded.image_files[0]]['page_id'])
            save_as = str(Path(directory) / 'copy.ctpr')
            save_state_to_proj_file_v2(loaded, save_as)
            reopened = main_fixture(Path(directory) / 'reopened')
            load_state_from_proj_file_v2(reopened, save_as)
            self.assertEqual(reopened.translation_profile, expected)
            reopened.translation_profile['glossary'][0]['aliases'].append('new alias')
            self.assertEqual(loaded.translation_profile, expected)

    def test_opening_legacy_project_after_project_a_does_not_retain_context(self):
        with project_directory() as directory:
            main = prepared_main(directory)
            project = str(Path(directory) / 'legacy.ctpr')
            state = {'original_image_files': [], 'image_states': {}, 'llm_extra_context': ''}
            with zipfile.ZipFile(project, 'w') as archive:
                archive.writestr('state.msgpack', msgpack.packb(state, use_bin_type=True))
            epoch = main._translation_project_epoch
            load_state_from_proj_file(main, project)
            self.assertEqual(main.translation_profile, empty_translation_profile())
            self.assertGreater(main._translation_project_epoch, epoch)

    def test_zip_profile_loads_unicode_and_valid_ids(self):
        with project_directory() as directory:
            source = prepared_main(directory)
            expected = profile_for_project(source)
            state = {'original_image_files': source.image_files, 'image_states': source.image_states,
                     'translation_profile': source.translation_profile}
            project = str(Path(directory) / 'zip.ctpr')
            with zipfile.ZipFile(project, 'w') as archive:
                archive.writestr('state.msgpack', msgpack.packb(state, default=ProjectEncoder().encode, use_bin_type=True))
            loaded = main_fixture(Path(directory) / 'loaded')
            load_state_from_proj_file(loaded, project)
            self.assertEqual(loaded.translation_profile, expected)

    def test_interim_sqlite_state_blob_preserves_profile(self):
        with project_directory() as directory:
            source = prepared_main(directory)
            expected = profile_for_project(source)
            state = {'original_image_files': source.image_files, 'image_states': source.image_states,
                     'translation_profile': source.translation_profile}
            project = str(Path(directory) / 'interim.ctpr')
            with database(project) as connection:
                _init_schema(connection)
                connection.execute('INSERT INTO project_state VALUES(1, ?)',
                                   (msgpack.packb(state, default=ProjectEncoder().encode, use_bin_type=True),))
            loaded = main_fixture(Path(directory) / 'loaded')
            load_state_from_proj_file_v2(loaded, project)
            self.assertEqual(loaded.translation_profile, expected)

    def test_sqlite_project_b_without_profile_clears_project_a(self):
        with project_directory() as directory:
            main = prepared_main(directory)
            second = main_fixture(Path(directory) / 'b')
            project = str(Path(directory) / 'b.ctpr')
            save_state_to_proj_file_v2(second, project)
            with database(project) as connection:
                blob = connection.execute('SELECT manifest_blob FROM project_manifest').fetchone()[0]
                manifest = msgpack.unpackb(blob, raw=False)
                del manifest['translation_profile']
                connection.execute('UPDATE project_manifest SET manifest_blob = ?',
                                   (msgpack.packb(manifest, use_bin_type=True),))
            load_state_from_proj_file(main, project)
            self.assertEqual(main.translation_profile, empty_translation_profile())

    def test_failed_load_discards_prior_profile_before_reading(self):
        with project_directory() as directory:
            main = prepared_main(directory)
            missing = str(Path(directory) / 'missing.ctpr')
            with self.assertRaises(FileNotFoundError):
                load_state_from_proj_file(main, missing)
            self.assertEqual(main.translation_profile, empty_translation_profile())
            main.translation_profile['notes'] = 'Project A again'
            broken = str(Path(directory) / 'broken.ctpr')
            with database(broken) as connection:
                _init_schema(connection)
                connection.execute('INSERT INTO project_manifest VALUES(1, ?)', (b'\xc1',))
            with self.assertRaises(msgpack.FormatError):
                load_state_from_proj_file_v2(main, broken)
            self.assertEqual(main.translation_profile, empty_translation_profile())

    def test_profile_projection_drops_dangling_speakers_without_mutating_live_data(self):
        with project_directory() as directory:
            main = prepared_main(directory)
            valid = copy.deepcopy(main.translation_profile['speakers'][0])
            for field in ('page_id', 'region_id', 'character_id'):
                invalid = {**valid, field: str(uuid4())}
                main.translation_profile['speakers'].append(invalid)
            before = copy.deepcopy(main.translation_profile)
            result = profile_for_project(main)
            self.assertEqual(result['speakers'], [valid])
            self.assertEqual(main.translation_profile, before)

    def test_profile_is_not_installed_when_later_load_state_fails(self):
        with project_directory() as directory:
            source = prepared_main(directory)
            project = str(Path(directory) / 'source.ctpr')
            save_state_to_proj_file_v2(source, project)
            loaded = main_fixture(Path(directory) / 'loaded')
            loaded.translation_profile['notes'] = 'Unrelated earlier project'
            loaded.batch_report_ctrl = SimpleNamespace(restore_latest_batch_report=Mock(side_effect=ValueError('broken report')))
            with self.assertRaisesRegex(ValueError, 'broken report'):
                load_state_from_proj_file_v2(loaded, project)
            self.assertEqual(loaded.translation_profile, empty_translation_profile())

    def test_future_profile_version_fails_without_retaining_previous_context(self):
        with project_directory() as directory:
            main = prepared_main(directory)
            project = str(Path(directory) / 'future.ctpr')
            save_state_to_proj_file_v2(main, project)
            with database(project) as connection:
                manifest = msgpack.unpackb(connection.execute('SELECT manifest_blob FROM project_manifest').fetchone()[0], raw=False)
                manifest['translation_profile']['schema_version'] = 999
                connection.execute('UPDATE project_manifest SET manifest_blob = ?',
                                   (msgpack.packb(manifest, use_bin_type=True),))
            with self.assertRaises(ValueError):
                load_state_from_proj_file_v2(main, project)
            self.assertEqual(main.translation_profile, empty_translation_profile())

    def test_real_recovery_autosave_restores_project_profile(self):
        from app.controllers.projects import ProjectController
        with project_directory() as directory:
            main = prepared_main(directory)
            expected = profile_for_project(main)
            recovery = str(Path(directory) / 'recovery.ctpr')
            main.project_file = None
            main._dirty_revision = 4
            main.text_ctrl = SimpleNamespace(_commit_pending_text_command=lambda: None)
            main.has_unsaved_changes = lambda: True
            main.settings_page.get_export_settings = lambda: {'project_autosave_enabled': False}
            controller = SimpleNamespace(
                main=main, _autosave_save_pending=False, _autosave_retrigger_requested=False,
                save_current_state=lambda: None, _ensure_autosave_project_file_if_needed=lambda: None,
                _recovery_project_path=lambda: recovery,
            )
            controller.save_project = lambda path: ProjectController.save_project(controller, path)
            def run_threaded(fn, result, error, finished, path):
                fn(path)
                finished()
            main.run_threaded = run_threaded
            ProjectController.autosave_project(controller, prefer_project_file=False)
            self.assertFalse(controller._autosave_save_pending)
            loaded = main_fixture(Path(directory) / 'restored')
            load_state_from_proj_file(loaded, recovery)
            self.assertEqual(loaded.translation_profile, expected)

    def test_clear_resets_data_and_editor_before_clean_state(self):
        from app.controllers.image import ImageStateController
        with project_directory() as directory:
            main = prepared_main(directory)
            reset_observations = []
            main.translation_context_ctrl = SimpleNamespace(
                reset_for_project=lambda: reset_observations.append(copy.deepcopy(main.translation_profile)))
            main.setWindowTitle = Mock()
            main.batch_report_ctrl = SimpleNamespace(clear_latest_batch_report=Mock())
            main.blk_list = []
            main.image_viewer = Mock()
            main.s_text_edit = Mock()
            main.t_text_edit = Mock()
            main.undo_stacks = {}
            main.in_memory_patches = {}
            main.page_list = Mock()
            main.project_file = 'A.ctpr'
            main.set_project_clean = lambda: self.assertEqual(main.translation_profile, empty_translation_profile())
            controller = SimpleNamespace(main=main, _close_transient_skip_notice=Mock(),
                _hide_active_page_skip_error=Mock(), _page_skip_errors={}, _suppress_dismiss_message_ids=set(),
                page_list_loader=Mock())
            ImageStateController.clear_state(controller)
            self.assertEqual(reset_observations, [empty_translation_profile()])
            self.assertEqual(main._translation_project_epoch, 4)

    def test_successful_load_resets_editor_to_loaded_profile(self):
        from app.controllers.projects import ProjectController
        main = SimpleNamespace(translation_profile=normalize_translation_profile({'notes': 'Project B'}),
                               settings_page=SimpleNamespace(ui=SimpleNamespace(extra_context=Mock())))
        snapshots = []
        main.translation_context_ctrl = SimpleNamespace(
            reset_for_project=lambda: snapshots.append(copy.deepcopy(main.translation_profile)))
        ProjectController.load_state_to_ui(SimpleNamespace(main=main), 'Existing context')
        self.assertEqual(snapshots[0]['notes'], 'Project B')
        main.settings_page.ui.extra_context.setPlainText.assert_called_once_with('Existing context')


if __name__ == '__main__':
    unittest.main()
