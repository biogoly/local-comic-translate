"""Editable transform state survives project, rendering and webtoon boundaries."""
import copy
from collections import defaultdict
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import imkit as imk
import msgpack
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.controllers.lettering_style_format import apply_style_properties, styled_properties
from app.controllers.text_fit import apply_layout_properties
from app.projects.parsers import ProjectEncoder
from app.projects.project_state import load_state_from_proj_file
from app.projects.project_state_v2 import close_cached_connection, save_state_to_proj_file_v2
from app.ui.canvas.image_viewer import ImageViewer
from app.ui.canvas.save_renderer import ImageSaveRenderer
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.text_item import TextBlockItem
from app.ui.canvas.webtoons.scene_items.text_item_manager import TextItemManager
from modules.utils.lettering_transform import normalize_lettering_transform
from modules.utils.region_metadata import new_id
from testtranslation_profile_persistence import main_fixture, prepared_main, project_directory


class LetteringTransformPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.item = TextBlockItem(font_family='Arial', font_size=24, outline_color=None)
        self.item.set_plain_text('Bonjour mon ami!\nAnother line')
        self.item.setTextWidth(210)
        self.item.setPos(70, 90)
        self.item.setRotation(13)
        self.item.setScale(1.1)
        self.item.setTransformOriginPoint(30, 25)
        self.item.region_id = new_id()
        cursor = QtGui.QTextCursor(self.item.document())
        cursor.setPosition(8)
        cursor.setPosition(11, QtGui.QTextCursor.KeepAnchor)
        fmt = QtGui.QTextCharFormat()
        fmt.setFontWeight(QtGui.QFont.Bold)
        fmt.setFontItalic(True)
        cursor.mergeCharFormat(fmt)
        self.warp = normalize_lettering_transform({'scale_x': 1.3, 'scale_y': .8,
                    'skew_x': 18, 'perspective_x': .3, 'perspective_y': -.2})
        self.item.set_lettering_transform(self.warp)
        self.state = TextItemProperties.from_text_item(self.item).to_dict()

    def tearDown(self):
        close_cached_connection()

    def assert_same_geometry(self, original, restored):
        self.assertEqual(original.toPlainText(), restored.toPlainText())
        self.assertEqual(original.lettering_transform, restored.lettering_transform)
        for point in (original.boundingRect().topLeft(), original.boundingRect().topRight(),
                      original.boundingRect().bottomRight(), original.boundingRect().center()):
            first, second = original.mapToScene(point), restored.mapToScene(point)
            self.assertAlmostEqual(first.x(), second.x(), places=5)
            self.assertAlmostEqual(first.y(), second.y(), places=5)
        cursor = QtGui.QTextCursor(restored.document())
        cursor.setPosition(9)
        self.assertTrue(cursor.charFormat().fontItalic())
        self.assertEqual(cursor.charFormat().fontWeight(), QtGui.QFont.Bold)

    def test_legacy_default_and_independent_property_snapshots(self):
        legacy = dict(self.state)
        del legacy['lettering_transform']
        self.assertEqual(TextItemProperties.from_dict(legacy).lettering_transform, {})
        props = TextItemProperties.from_dict(self.state)
        props.lettering_transform['scale_x'] = 2
        self.assertEqual(self.state['lettering_transform']['scale_x'], 1.3)
        emitted = props.to_dict()
        emitted['lettering_transform']['scale_x'] = 3
        self.assertEqual(props.lettering_transform['scale_x'], 2)

    def test_viewer_save_reload_and_export_factory_keep_mapped_geometry(self):
        viewer = ImageViewer(None)
        viewer.display_image_array(np.full((400, 400, 3), 255, np.uint8), fit=False)
        restored = viewer.add_text_item(self.state)
        self.assert_same_geometry(self.item, restored)
        exported = ImageSaveRenderer.create_text_item(self.state)
        self.assert_same_geometry(self.item, exported)
        state = viewer.save_state()
        self.assertEqual(state['text_items_state'][0]['lettering_transform'], self.warp)
        viewer.load_state(state)
        self.assert_same_geometry(self.item, viewer.text_items[0])
        viewer.deleteLater()
        self.app.processEvents()

    def test_style_and_fit_restoration_retain_transform_and_word_emphasis(self):
        styled = styled_properties(self.state, {'font_size': 27}, {'font_size'})
        apply_style_properties(self.item, styled)
        self.assertEqual(self.item.lettering_transform, self.warp)
        self.assertFalse(self.item.transform().isIdentity())
        apply_layout_properties(self.item, self.state)
        expected = ImageSaveRenderer.create_text_item(self.state)
        self.assert_same_geometry(expected, self.item)

    def test_fit_controls_do_not_claim_unwarped_bounds_fit_transformed_text(self):
        from testmanual_text_box import _ManualTextMain
        main = _ManualTextMain()
        main.fit_text_button = QtWidgets.QPushButton()
        main.text_fit_status = QtWidgets.QLabel()
        item = main.image_viewer.add_text_item(self.state)
        item._page_path = 'page.png'
        item.selected = True
        fit = main.text_ctrl.fit_controller
        fit.refresh()
        self.assertFalse(main.fit_text_button.isEnabled())
        self.assertEqual(main.text_fit_status.text(), 'Reset text shape to fit.')
        self.assertIn('unavailable for transformed lettering', main.text_fit_status.toolTip())
        with patch('app.controllers.text_fit.QtWidgets.QMessageBox.information') as notice, \
                patch('app.controllers.text_fit.TextFitDialog') as dialog:
            fit.open_preview()
            notice.assert_called_once()
            dialog.assert_not_called()
        item.set_lettering_transform({})
        fit.refresh()
        self.assertTrue(main.fit_text_button.isEnabled())
        self.assertEqual(main.text_fit_status.toolTip(), '')
        main.deleteLater()
        self.app.processEvents()

    def test_review_measurement_and_fingerprint_respect_transform(self):
        from app.controllers.review import ReviewController
        from modules.review.issues import fingerprint_for_stage
        from testreview_issues import region
        self.assertIsNone(ReviewController._diagnostics(None, {'props': self.state}))
        ordinary = dict(self.state)
        ordinary.pop('lettering_transform')
        identity = dict(ordinary, lettering_transform={})
        self.assertEqual(fingerprint_for_stage('layout', region(), ordinary),
                         fingerprint_for_stage('layout', region(), identity))
        self.assertNotEqual(fingerprint_for_stage('layout', region(), ordinary),
                            fingerprint_for_stage('layout', region(), self.state))

    def test_project_save_reopen_save_as_retains_editable_geometry(self):
        with project_directory() as directory:
            source = prepared_main(directory)
            path = source.image_files[0]
            source.image_states[path]['viewer_state'] = {'text_items_state': [self.state]}
            project = str(Path(directory) / 'transform.ctpr')
            save_state_to_proj_file_v2(source, project)
            loaded = main_fixture(Path(directory) / 'loaded')
            load_state_from_proj_file(loaded, project)
            restored_state = loaded.image_states[loaded.image_files[0]]['viewer_state']['text_items_state'][0]
            self.assert_same_geometry(self.item, ImageSaveRenderer.create_text_item(restored_state))
            second = str(Path(directory) / 'copy.ctpr')
            save_state_to_proj_file_v2(loaded, second)
            reopened = main_fixture(Path(directory) / 'reopened')
            load_state_from_proj_file(reopened, second)
            saved = reopened.image_states[reopened.image_files[0]]['viewer_state']['text_items_state'][0]
            self.assertEqual(saved['lettering_transform'], self.warp)
            self.assert_same_geometry(self.item, ImageSaveRenderer.create_text_item(saved))

    def test_legacy_zip_project_retains_optional_transform(self):
        with project_directory() as directory:
            source = prepared_main(directory)
            source.image_states[source.image_files[0]]['viewer_state'] = {'text_items_state': [self.state]}
            payload = {'original_image_files': source.image_files, 'image_states': source.image_states}
            project = str(Path(directory) / 'legacy.ctpr')
            with zipfile.ZipFile(project, 'w') as archive:
                archive.writestr('state.msgpack', msgpack.packb(payload, default=ProjectEncoder().encode, use_bin_type=True))
            loaded = main_fixture(Path(directory) / 'loaded')
            load_state_from_proj_file(loaded, project)
            saved = loaded.image_states[loaded.image_files[0]]['viewer_state']['text_items_state'][0]
            self.assert_same_geometry(self.item, ImageSaveRenderer.create_text_item(saved))

    def test_actual_export_pixels_match_live_viewer_geometry(self):
        image = np.full((400, 400, 3), 255, np.uint8)
        viewer = ImageViewer(None)
        viewer.display_image_array(image, fit=False)
        viewer.add_text_item(self.state)
        viewer._scene.setSceneRect(0, 0, 400, 400)
        live = ImageSaveRenderer(image)
        live.scene = viewer._scene
        saved = ImageSaveRenderer(image)
        saved.add_state_to_image({'text_items_state': [self.state]})
        np.testing.assert_array_equal(live.render_to_image(), saved.render_to_image())
        self.assertTrue(np.any(saved.render_to_image() != 255))
        viewer.deleteLater()
        self.app.processEvents()

    def _overlapping_items(self, viewer, transformed_on_top):
        items = []
        for index, color in enumerate(('#1634df', '#df2616')):
            item = TextBlockItem('OVERLAP', font_family='Arial', font_size=36,
                                 render_color=QtGui.QColor(color), outline_color=None)
            item.setTextWidth(280)
            item.setPos(15, 8)
            item.region_id = new_id()
            if bool(index) == transformed_on_top:
                item.set_lettering_transform({'scale_x': 1.1, 'perspective_x': .18})
            props = TextItemProperties.from_text_item(item).to_dict()
            if not item.lettering_transform:
                props.pop('lettering_transform')  # Legacy native item state.
            items.append(viewer.add_text_item(props))
        return items

    def _scene_pixels(self, viewer, image):
        viewer._scene.setSceneRect(0, 0, image.shape[1], image.shape[0])
        renderer = ImageSaveRenderer(image)
        renderer.scene = viewer._scene
        return renderer.render_to_image()

    def test_overlapping_native_and_transformed_items_keep_stack_on_export_and_reload(self):
        image = np.full((100, 320, 3), 255, np.uint8)
        for transformed_on_top in (False, True):
            with self.subTest(transformed_on_top=transformed_on_top):
                viewer = ImageViewer(None)
                viewer.display_image_array(image, fit=False)
                bottom, top = self._overlapping_items(viewer, transformed_on_top)
                expected = self._scene_pixels(viewer, image)
                state = viewer.save_state()
                expected_order = [top.region_id, bottom.region_id]
                self.assertEqual([data['region_id'] for data in state['text_items_state']], expected_order)
                exported = ImageSaveRenderer(image)
                exported.add_state_to_image(state)
                np.testing.assert_array_equal(exported.render_to_image(), expected)
                # This is a visible overlap, not a vacuous order-only assertion.
                flipped = ImageSaveRenderer(image)
                flipped.add_state_to_image({'text_items_state': list(reversed(state['text_items_state']))})
                self.assertTrue(np.any(flipped.render_to_image() != expected))
                for _ in range(2):
                    viewer.clear_text_items()
                    viewer.load_state(state)
                    np.testing.assert_array_equal(self._scene_pixels(viewer, image), expected)
                    state = viewer.save_state()
                    self.assertEqual([data['region_id'] for data in state['text_items_state']], expected_order)
                viewer.deleteLater()
                self.app.processEvents()

    def test_webtoon_stack_survives_save_unload_reload_and_export(self):
        image = np.full((100, 320, 3), 255, np.uint8)
        viewer, manager = self._webtoon_manager()
        viewer.display_image_array(image, fit=False)
        manager.layout_manager.get_pages_for_scene_bounds = lambda rect: {0}
        bottom, top = self._overlapping_items(viewer, True)
        for item in (bottom, top):
            item._page_path = 'a.png'
        expected = self._scene_pixels(viewer, image)
        expected_order = [top.region_id, bottom.region_id]
        pages = defaultdict(lambda: {'text_items': []})
        manager.save_text_items_to_states(pages)
        self.assertEqual([data['region_id'] for data in pages[0]['text_items']], expected_order)
        for _ in range(2):
            manager.unload_text_items(0, 0, 100, 'a.png')
            saved = manager.main_controller.image_states['a.png']
            self.assertEqual([data['region_id'] for data in saved['viewer_state']['text_items_state']], expected_order)
            exported = ImageSaveRenderer(image)
            exported.add_state_to_image(saved['viewer_state'])
            np.testing.assert_array_equal(exported.render_to_image(), expected)
            manager.load_text_items(saved, 0)
            np.testing.assert_array_equal(self._scene_pixels(viewer, image), expected)
        viewer.deleteLater()
        self.app.processEvents()

    def test_render_state_refresh_keeps_saved_text_stack(self):
        from app.controllers.image import ImageStateController
        image = np.full((100, 320, 3), 255, np.uint8)
        viewer = ImageViewer(None)
        viewer.display_image_array(image, fit=False)
        bottom, top = self._overlapping_items(viewer, True)
        expected = self._scene_pixels(viewer, image)
        state = viewer.save_state()
        main = SimpleNamespace(webtoon_mode=False, curr_img_idx=0, image_files=['a.png'],
            image_viewer=viewer, image_states={'a.png': {'viewer_state': state}},
            text_ctrl=SimpleNamespace(clear_text_edits=Mock()))
        controller = SimpleNamespace(main=main, _nav_request_id=0)
        ImageStateController.on_render_state_ready(controller, 'a.png')
        np.testing.assert_array_equal(self._scene_pixels(viewer, image), expected)
        self.assertEqual([data['region_id'] for data in viewer.save_state()['text_items_state']],
                         [top.region_id, bottom.region_id])
        viewer.deleteLater()
        self.app.processEvents()

    def _webtoon_manager(self):
        viewer = ImageViewer(None)
        loader = SimpleNamespace(image_file_paths=['a.png', 'b.png'])
        converter = SimpleNamespace(
            page_local_to_scene_position=lambda point, page: point + QtCore.QPointF(0, page * 100),
            scene_to_page_local_position=lambda point, page: point - QtCore.QPointF(0, page * 100),
            clip_text_item_to_page=Mock(side_effect=AssertionError('Warped text must not be split')),
        )
        layout = SimpleNamespace(get_pages_for_scene_bounds=lambda rect: {0, 1},
                                 get_page_at_position=lambda y: int(y // 100))
        manager = TextItemManager(viewer, layout, converter, loader)
        manager.main_controller = SimpleNamespace(image_files=loader.image_file_paths,
            image_states={path: {'viewer_state': {'text_items_state': []}} for path in loader.image_file_paths})
        return viewer, manager

    def test_webtoon_save_and_lazy_unload_keep_one_full_source_on_owner_page(self):
        viewer, manager = self._webtoon_manager()
        item = viewer.add_text_item(self.state)
        item.setPos(20, 110)  # The origin has crossed into page B, but A owns it.
        item._page_path = 'a.png'
        # SceneItemManager starts with an empty defaultdict, not prefilled pages.
        pages = defaultdict(lambda: {'text_items': []})
        manager.save_text_items_to_states(pages)
        self.assertEqual(len(pages[0]['text_items']), 1)
        self.assertEqual(pages[1]['text_items'], [])
        saved = pages[0]['text_items'][0]
        self.assertEqual(saved['text'], item.toHtml())
        self.assertEqual(saved['width'], item.boundingRect().width())
        self.assertEqual(saved['lettering_transform'], self.warp)
        manager.unload_text_items(1, 100, 200, 'b.png')
        self.assertIn(item, viewer.text_items)
        manager.unload_text_items(0, 0, 100, 'a.png')
        self.assertNotIn(item, viewer.text_items)
        saved = manager.main_controller.image_states['a.png']['viewer_state']['text_items_state'][0]
        manager.load_text_items({'viewer_state': {'text_items_state': [saved]}}, 0)
        self.assertEqual(viewer.text_items[0].lettering_transform, self.warp)
        self.assertEqual(viewer.text_items[0].toHtml(), item.toHtml())
        viewer.deleteLater()
        self.app.processEvents()

    def test_webtoon_deduplication_keeps_distinct_transformed_regions(self):
        viewer, manager = self._webtoon_manager()
        other = copy.deepcopy(self.state)
        other['region_id'] = new_id()
        self.assertFalse(manager.is_duplicate_text_item(self.state, [other]))
        self.assertTrue(manager.is_duplicate_text_item(self.state, [copy.deepcopy(self.state)]))
        viewer.deleteLater()
        self.app.processEvents()

    def test_unloaded_webtoon_state_is_not_split_or_heuristically_merged(self):
        viewer, manager = self._webtoon_manager()
        pages = {0: {'text_items': []}, 1: {'text_items': []}}
        manager.redistribute_existing_text_items({0: [self.state]}, pages)
        self.assertEqual(pages[0]['text_items'], [self.state])
        self.assertEqual(pages[1]['text_items'], [])
        other = copy.deepcopy(self.state)
        other['region_id'] = new_id()
        manager.main_controller.image_states['a.png']['viewer_state']['text_items_state'] = [self.state]
        manager.main_controller.image_states['b.png']['viewer_state']['text_items_state'] = [other]
        manager.merge_clipped_text_items()
        self.assertEqual(len(manager.main_controller.image_states['a.png']['viewer_state']['text_items_state']), 1)
        self.assertEqual(len(manager.main_controller.image_states['b.png']['viewer_state']['text_items_state']), 1)
        viewer.deleteLater()
        self.app.processEvents()

    def test_spanning_export_uses_full_transform_and_centers_different_width_pages(self):
        with project_directory() as directory:
            paths, images = [], []
            for index, width in enumerate((100, 80, 100)):
                path = str(Path(directory) / f'{index}.png')
                image = np.full((100, width, 3), 255, np.uint8)
                imk.write_image(path, image)
                paths.append(path)
                images.append(image)
            item = TextBlockItem(font_family='Arial', font_size=22, outline_color=None)
            item.set_plain_text('ONE\nTWO\nTHREE')
            item.setTextWidth(90)
            item.setPos(5, 100)
            item.region_id = new_id()
            item.set_lettering_transform({'scale_y': 3, 'perspective_x': .3})
            source = TextItemProperties.from_text_item(item).to_dict()
            main = SimpleNamespace(image_files=paths, image_data=dict(zip(paths, images)),
                image_states={path: {'viewer_state': {'text_items_state': [source] if index == 0 else []}}
                              for index, path in enumerate(paths)})
            before = copy.deepcopy(main.image_states)
            whole = ImageSaveRenderer(np.full((300, 100, 3), 255, np.uint8))
            whole.add_state_to_image({'text_items_state': [source]})
            reference = whole.render_to_image()
            for page_idx, x in ((1, 10), (2, 0)):
                renderer = ImageSaveRenderer(images[page_idx])
                renderer.add_state_to_image(main.image_states[paths[page_idx]]['viewer_state'], page_idx, main)
                result = renderer.render_to_image()
                expected = reference[page_idx * 100:(page_idx + 1) * 100, x:x + images[page_idx].shape[1]]
                # Qt's projective rasterizer can round edge antialiasing slightly
                # differently after clipping/translating onto another canvas.
                delta = np.abs(result.astype(np.int16) - expected.astype(np.int16))
                self.assertLessEqual(int(delta.max()), 8)
                self.assertLess(float(np.count_nonzero(delta)) / delta.size, .002)
                self.assertTrue(np.any(result != 255))
            self.assertEqual(main.image_states, before)


if __name__ == '__main__':
    unittest.main()
