"""Gradient preview commit boundaries using real masks, patches, undo and codecs."""
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import imkit as imk
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.controllers.local_repair import LocalRepairController, repair_patch, rasterize_paths
from app.path_materialization import ensure_path_materialized
from app.projects.project_state_v2 import close_cached_connection, load_state_from_proj_file_v2, save_state_to_proj_file_v2
from app.ui.canvas.image_viewer import ImageViewer
from app.ui.commands.base import PatchCommandBase
from app.ui.local_repair_panel import LocalRepairPanel


class GradientWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / 'page.png')
        y, x = np.indices((80, 100))
        self.original = np.stack((60 + x, 80 + y, 180 - y), axis=-1).astype(np.uint8)
        self.current = self.original.copy()
        self.current[20:40, 20:40] = (15, 10, 5)
        imk.write_image(self.path, self.current)
        self.viewer = ImageViewer(None)
        self.viewer.display_image_array(self.current, fit=False)
        self.panel = LocalRepairPanel()
        self.stack = QtGui.QUndoStack()
        self.errors, self.notices, self.dialogs = [], [], []
        self.main = SimpleNamespace(
            image_viewer=self.viewer, local_repair_panel=self.panel,
            image_files=[self.path], curr_img_idx=0, undo_stacks={self.path: self.stack},
            temp_dir=self.tmp.name, image_patches={}, in_memory_patches={},
            image_states={self.path: {'page_id': 'test-page', 'blk_list': []}},
            _translation_project_epoch=1,
            image_ctrl=SimpleNamespace(load_original_image=lambda _: self.original.copy(),
                get_composited_page_image=lambda _: self.viewer.get_image_array()),
            set_tool=self.viewer.set_tool, set_slider_size=lambda _: None,
            default_error_handler=self.errors.append,
            image_data={}, image_history={}, in_memory_history={}, current_history_index={},
            displayed_images=set(), loaded_images=[], webtoon_mode=False,
            settings_page=SimpleNamespace(get_llm_settings=lambda: {}),
        )
        self.repair = LocalRepairController(self.main)
        self.repair._notice = self.notices.append
        self.panel.fill_mode.setCurrentIndex(self.panel.fill_mode.findData('gradient'))
        self.add_mask()

    def tearDown(self):
        close_cached_connection()
        self.repair.cancel()
        self.viewer.close()
        self.panel.close()
        self.tmp.cleanup()
        self.assertEqual(self.errors, [])

    def add_mask(self, rect=(20, 20, 20, 20)):
        path = QtGui.QPainterPath()
        path.addRect(*rect)
        return self.viewer._scene.addPath(path, QtGui.QPen(QtCore.Qt.NoPen),
            QtGui.QBrush(QtGui.QColor(255, 0, 0, 128)))

    def run_preview(self, *, accept=True, during=None, source='current'):
        def construct(current, original, mask, parent=None, protection=None):
            result = SimpleNamespace(
                current=current.copy(), original=original.copy() if original is not None else None,
                mask=mask.copy(), patch=repair_patch(current, self.original, mask),
                metadata={'repair_method': 'gradient', 'source': source,
                          'sample_points': [5, 5, 90, 5, 5, 70, 90, 70]},
            )
            def execute():
                if during:
                    during(result)
                return QtWidgets.QDialog.Accepted if accept else QtWidgets.QDialog.Rejected
            result.exec = execute
            result.deleteLater = lambda: None
            self.dialogs.append(result)
            return result
        with patch('app.controllers.local_repair.GradientRepairDialog', side_effect=construct):
            self.repair.fill_mask()

    def test_apply_is_one_undo_step_with_exact_outside_pixels_and_mask_restoration(self):
        before = self.viewer.get_image_array()
        self.assertIsNone(self.repair.color)  # gradient does not require a solid swatch
        self.run_preview()
        self.assertEqual(len(self.dialogs), 1)
        mask = self.dialogs[0].mask
        after = self.viewer.get_image_array()
        np.testing.assert_array_equal(after[mask == 0], before[mask == 0])
        np.testing.assert_array_equal(after[25, 25], self.original[25, 25])
        self.assertEqual(self.stack.count(), 1)
        self.assertFalse(self.viewer.has_drawn_elements())
        record = self.main.image_patches[self.path][0]
        self.assertEqual(record['kind'], 'inpaint')
        self.assertEqual(record['metadata']['source'], 'current')
        self.stack.undo()
        np.testing.assert_array_equal(self.viewer.get_image_array(), before)
        self.assertTrue(self.viewer.has_drawn_elements())
        self.stack.redo()
        np.testing.assert_array_equal(self.viewer.get_image_array(), after)
        self.assertFalse(self.viewer.has_drawn_elements())

    def test_protection_changed_during_preview_rejects_apply(self):
        from modules.repair.protection import encode_protection
        protected = np.ones(self.current.shape[:2], np.uint8)
        self.run_preview(during=lambda _: self.main.image_states[self.path].update(
            pixel_protection=encode_protection(protected)))
        self.assertEqual(self.stack.count(), 0)
        self.assertTrue(self.notices)

    def prepare_heal(self):
        from modules.utils.clone_brush import capture_sample
        self.repair.clone.source = dict(file=self.path, stack=self.stack,
            sample=capture_sample(self.original, 65, 45, 50), shape=self.current.shape)
        self.repair.color = (100, 130, 160)
        self.panel.set_color(QtGui.QColor(*self.repair.color))
        self.panel.fill_mode.setCurrentIndex(self.panel.fill_mode.findData('heal'))

    def test_heal_apply_is_one_undo_step_and_restores_mask(self):
        self.prepare_heal()
        from modules.repair.protection import encode_protection
        protection = np.zeros(self.current.shape[:2], np.uint8)
        protection[15:45, 28:32] = 255
        self.main.image_states[self.path]['pixel_protection'] = encode_protection(protection)
        before = self.viewer.get_image_array()
        def accept(dialog):
            self.assertTrue(dialog.apply_button.isEnabled(), dialog.status_label.text())
            dialog.apply_button.click()
            return dialog.result()
        with patch('app.ui.dialogs.heal_repair.HealRepairDialog.exec', accept):
            self.repair.fill_mask()
        self.assertEqual(self.stack.count(), 1)
        after = self.viewer.get_image_array()
        self.assertFalse(np.array_equal(after, before))
        np.testing.assert_array_equal(after[protection != 0], before[protection != 0])
        self.assertFalse(self.viewer.has_drawn_elements())
        self.assertEqual(self.main.image_patches[self.path][0]['metadata']['repair_method'], 'heal')
        self.stack.undo()
        np.testing.assert_array_equal(self.viewer.get_image_array(), before)
        self.assertTrue(self.viewer.has_drawn_elements())
        self.stack.redo()
        np.testing.assert_array_equal(self.viewer.get_image_array(), after)

    def test_heal_stale_mask_rejects_actual_dialog_result(self):
        self.prepare_heal()
        def accept(dialog):
            dialog.apply_button.click()
            self.add_mask((70, 60, 5, 5))
            return dialog.result()
        with patch('app.ui.dialogs.heal_repair.HealRepairDialog.exec', accept):
            self.repair.fill_mask()
        self.assertEqual(self.stack.count(), 0)
        self.assertTrue(self.notices)

    def test_heal_requires_explicit_sample(self):
        self.prepare_heal()
        self.repair.clone.source = None
        self.repair.fill_mask()
        self.assertEqual(self.stack.count(), 0)
        self.assertIn('capture a sample', self.notices[0])

    def test_cancel_keeps_pixels_mask_and_undo_stack(self):
        before = self.viewer.get_image_array()
        self.run_preview(accept=False)
        np.testing.assert_array_equal(self.viewer.get_image_array(), before)
        self.assertTrue(self.viewer.has_drawn_elements())
        self.assertEqual(self.stack.count(), 0)
        self.assertFalse(self.main.image_patches)

    def test_preview_snapshots_current_and_original_separately(self):
        self.run_preview(accept=False, source='original')
        np.testing.assert_array_equal(self.dialogs[0].current, self.current)
        np.testing.assert_array_equal(self.dialogs[0].original, self.original)
        self.assertFalse(np.array_equal(self.dialogs[0].current, self.dialogs[0].original))

    def test_stale_mask_change_without_undo_command_is_rejected(self):
        self.run_preview(during=lambda _: self.add_mask((60, 50, 8, 8)))
        self.assertEqual(self.stack.count(), 0)
        self.assertFalse(self.main.image_patches)
        self.assertTrue(self.viewer.has_drawn_elements())
        self.assertTrue(self.notices)

    def test_stale_composed_pixels_without_undo_command_are_rejected(self):
        def change(_):
            replacement = self.current.copy()
            replacement[3, 3] = 0
            self.main.image_ctrl.get_composited_page_image = lambda _: replacement
        self.run_preview(during=change)
        self.assertEqual(self.stack.count(), 0)
        self.assertFalse(self.main.image_patches)
        self.assertTrue(self.notices)

    def test_stale_project_page_and_stack_identity_are_rejected(self):
        mutations = (
            lambda: setattr(self.main, '_translation_project_epoch', 2),
            lambda: self.main.image_states[self.path].update(page_id='another-page'),
            lambda: self.main.undo_stacks.update({self.path: QtGui.QUndoStack()}),
            lambda: setattr(self.main, 'curr_img_idx', -1),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.main._translation_project_epoch = 1
                self.main.image_states[self.path]['page_id'] = 'test-page'
                self.main.undo_stacks[self.path] = self.stack
                self.main.curr_img_idx = 0
                self.notices.clear()
                self.run_preview(during=lambda _, mutate=mutate: mutate())
                self.assertFalse(self.main.image_patches)
                self.assertEqual(self.stack.count(), 0)
                self.assertTrue(self.notices)

    def test_original_for_gradient_restore_and_clone_ignores_edited_history_and_cache(self):
        from app.controllers.image import ImageStateController

        imk.write_image(self.path, self.original)
        history_path = str(Path(self.tmp.name) / 'edited.png')
        imk.write_image(history_path, self.current)
        self.main.image_history[self.path] = [history_path]
        self.main.current_history_index[self.path] = 0
        reader = ImageStateController.__new__(ImageStateController)
        reader.main = self.main
        self.main.image_ctrl = reader

        self.run_preview(accept=False, source='original')
        np.testing.assert_array_equal(self.dialogs[-1].original, self.original)
        np.testing.assert_array_equal(self.dialogs[-1].current, self.current)
        self.assertIn(self.path, self.main.image_data)

        anchor = QtCore.QPointF(30, 30)
        self.viewer.set_tool('clone')
        self.panel.clone_original.setChecked(True)
        self.repair.press(anchor, set_source=True)
        sample = self.repair.clone.source['sample']
        half = sample.pixels.shape[0] // 2
        np.testing.assert_array_equal(sample.pixels[half, half], self.original[30, 30])
        self.panel.clone_original.setChecked(False)
        self.repair.press(anchor, set_source=True)
        np.testing.assert_array_equal(self.repair.clone.source['sample'].pixels[half, half], self.current[30, 30])

        self.viewer.set_tool('restore')
        self.repair.press(anchor)
        self.repair.release()
        np.testing.assert_array_equal(self.viewer.get_image_array()[30, 30], self.original[30, 30])
        self.stack.undo()
        np.testing.assert_array_equal(self.viewer.get_image_array(), self.current)

        # A failed resample must not leave an old stamp armed or use the edited cache.
        self.viewer.set_tool('clone')
        self.panel.clone_original.setChecked(True)
        self.repair.press(anchor, set_source=True)
        self.assertIsNotNone(self.repair.clone.source)
        imk.write_image(self.path, self.original[:10, :10])
        self.repair.press(anchor, set_source=True)
        self.assertIsNone(self.repair.clone.source)

    def test_stale_original_source_rejected_only_when_sampled(self):
        def change(_):
            self.main.image_ctrl.load_original_image = lambda _: np.zeros_like(self.original)
        self.run_preview(source='original', during=change)
        self.assertFalse(self.main.image_patches)
        self.assertTrue(self.notices)
        self.main.image_ctrl.load_original_image = lambda _: self.original.copy()
        self.run_preview(source='current', during=change)
        self.assertEqual(self.stack.count(), 1)

    def test_repeated_cancelled_previews_do_not_accumulate_pixels(self):
        for _ in range(3):
            self.run_preview(accept=False)
        for dialog in self.dialogs:
            np.testing.assert_array_equal(dialog.current, self.current)
            np.testing.assert_array_equal(dialog.original, self.original)
        self.assertEqual(self.stack.count(), 0)
        self.run_preview()
        self.assertEqual(self.stack.count(), 1)

    def test_edited_page_repair_can_apply_when_original_cannot_be_loaded(self):
        def unavailable(_):
            raise OSError('Original fixture is unavailable')
        self.main.image_ctrl.load_original_image = unavailable
        self.run_preview(source='current')
        self.assertIsNone(self.dialogs[0].original)
        self.assertEqual(self.stack.count(), 1)
        self.assertFalse(self.viewer.has_drawn_elements())

    def test_original_becoming_unavailable_rejects_original_sampled_preview(self):
        def unavailable(_):
            raise OSError('Original fixture disappeared')
        self.run_preview(source='original',
                         during=lambda _: setattr(self.main.image_ctrl, 'load_original_image', unavailable))
        self.assertFalse(self.main.image_patches)
        self.assertEqual(self.stack.count(), 0)
        self.assertTrue(self.viewer.has_drawn_elements())
        self.assertTrue(self.notices)

    def test_gradient_controls_use_all_compiled_interface_languages(self):
        from app.ui.dialogs.gradient_repair import GradientRepairDialog
        folder = Path(__file__).resolve().parents[1] / 'resources/translations/compiled'
        mask = self.repair._current_mask(self.current.shape, QtCore.QPointF())
        for locale in ('de', 'es', 'fr', 'it', 'ja', 'ko', 'ru', 'tr', 'zh-CN'):
            translator = QtCore.QTranslator()
            self.assertTrue(translator.load(str(folder / f'ct_{locale}.qm')))
            self.app.installTranslator(translator)
            panel, dialog = None, None
            try:
                panel = LocalRepairPanel()
                panel.fill_mode.setCurrentIndex(panel.fill_mode.findData('gradient'))
                self.assertEqual(panel.fill_mode.currentText(), translator.translate('LocalRepairPanel', 'Gradient'))
                self.assertEqual(panel.fill.text(), translator.translate('LocalRepairPanel', 'Preview gradient fill…'))
                dialog = GradientRepairDialog(self.current, self.original, mask)
                self.assertEqual(dialog.windowTitle(), translator.translate('GradientRepairDialog', 'Gradient repair'))
                self.assertNotEqual(dialog.windowTitle(), 'Gradient repair')
                self.assertEqual(dialog.source_combo.currentText(), translator.translate('GradientRepairDialog', 'Edited page'))
                self.assertEqual(dialog.generate_button.text(), translator.translate('GradientRepairDialog', 'Generate preview'))
                self.assertEqual(dialog.sample_count.text(), translator.translate('GradientRepairDialog', 'Samples: {count}').format(count=0))
            finally:
                if panel is not None:
                    panel.close()
                if dialog is not None:
                    dialog.close()
                self.app.removeTranslator(translator)

    def test_accepted_patch_and_metadata_survive_project_roundtrip(self):
        self.run_preview()
        expected = self.viewer.get_image_array()
        original_record = copy.deepcopy(self.main.image_patches[self.path][0])
        project = str(Path(self.tmp.name) / 'repair.ctpr')
        save_state_to_proj_file_v2(self.main, project)
        loaded_root = Path(self.tmp.name) / 'loaded'
        loaded_root.mkdir()
        loaded = SimpleNamespace(temp_dir=str(loaded_root), image_files=[], image_data={},
            in_memory_history={}, image_history={}, image_states={}, image_patches={},
            current_history_index={}, displayed_images=set(), loaded_images=[], curr_img_idx=0,
            image_viewer=SimpleNamespace(webtoon_view_state={}), webtoon_mode=False)
        load_state_from_proj_file_v2(loaded, project)
        path = loaded.image_files[0]
        record = loaded.image_patches[path][0]
        self.assertEqual(record['metadata'], original_record['metadata'])
        self.assertEqual(record['patch_id'], original_record['patch_id'])
        self.assertEqual(record['bbox'], original_record['bbox'])
        ensure_path_materialized(path)
        self.viewer.display_image_array(imk.read_image(path), fit=False)
        PatchCommandBase.create_patch_item(record, self.viewer)
        np.testing.assert_array_equal(self.viewer.get_image_array(), expected)

    def test_webtoon_offset_and_cross_page_mask_are_page_local_and_undoable(self):
        for item in list(self.viewer._scene.items()):
            if isinstance(item, QtWidgets.QGraphicsPathItem):
                self.viewer._scene.removeItem(item)
        origin = QtCore.QPointF(110, 200)
        self.current[65:, 20:40] = 0
        self.main.image_ctrl.get_composited_page_image = lambda _: self.current.copy()
        self.viewer.webtoon_manager.layout_manager.image_positions = [200, 280]
        self.main.webtoon_mode = True
        self.main.image_files.append(str(Path(self.tmp.name) / 'next.png'))
        self.viewer.webtoon_manager.image_loader.loaded_pages.add(0)
        self.viewer.page_to_scene_coordinates = lambda index, point: point + origin + QtCore.QPointF(0, index * 80)
        original_path = self.add_mask((130, 265, 20, 40)).path()
        self.run_preview()
        self.assertEqual(self.stack.count(), 1)
        record = self.main.image_patches[self.path][0]
        x, y, w, h = record['bbox']
        self.assertGreaterEqual(y, 65)
        self.assertLessEqual(y + h, 80)
        pixmap = next(item for item in self.viewer._scene.items()
                      if item.data(PatchCommandBase.PATCH_ID_KEY) == record['patch_id'])
        self.assertEqual(pixmap.pos(), origin + QtCore.QPointF(x, y))
        paths = [item.path() for item in self.viewer._scene.items() if isinstance(item, QtWidgets.QGraphicsPathItem)]
        self.assertEqual(len(paths), 1)
        self.assertFalse(paths[0].contains(QtCore.QPointF(140, 270)))
        self.assertTrue(paths[0].contains(QtCore.QPointF(140, 290)))
        self.stack.undo()
        paths = [item.path() for item in self.viewer._scene.items() if isinstance(item, QtWidgets.QGraphicsPathItem)]
        self.assertEqual(paths, [original_path])
        self.stack.redo()
        self.assertEqual(len(self.main.image_patches[self.path]), 1)


if __name__ == '__main__':
    unittest.main()
