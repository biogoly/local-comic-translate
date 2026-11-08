"""Actual curve geometry survives editable projects and explicitly raster PSDs."""
import copy
from collections import defaultdict
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import imkit as imk
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.controllers.lettering_style_format import apply_style_properties, styled_properties
from app.controllers.psd_support import require_photoshopapi
from app.controllers.psd_exporter import PsdPageData, export_psd_pages, _render_raster_lettering
from app.controllers.psd_importer import _read_single_psd, _image_layer_to_rgba
from app.projects.project_state import load_state_from_proj_file
from app.projects.project_state_v2 import close_cached_connection, save_state_to_proj_file_v2
from app.ui.canvas.image_viewer import ImageViewer
from app.ui.canvas.save_renderer import ImageSaveRenderer
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.text_item import TextBlockItem, OutlineInfo, OutlineType
from app.ui.canvas.webtoons.scene_items.text_item_manager import TextItemManager
from modules.utils.region_metadata import new_id
from testtranslation_profile_persistence import main_fixture, prepared_main, project_directory


class CurvedLetteringPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def tearDown(self):
        close_cached_connection()
        self.app.processEvents()

    def item(self, angle=80, projective=False):
        item = TextBlockItem(font_family='Arial', font_size=25,
                             render_color=QtGui.QColor('#264953'), outline_color=None)
        item.set_plain_text('Bonjour mon ami!\nA curved line')
        item.setTextWidth(240)
        item.setPos(100, 100)
        item.region_id = new_id()
        cursor = QtGui.QTextCursor(item.document())
        cursor.setPosition(8)
        cursor.setPosition(11, QtGui.QTextCursor.KeepAnchor)
        fmt = QtGui.QTextCharFormat()
        fmt.setFontWeight(QtGui.QFont.Bold)
        fmt.setFontItalic(True)
        fmt.setFontUnderline(True)
        fmt.setForeground(QtGui.QColor('#b52061'))
        cursor.mergeCharFormat(fmt)
        settings = {'arc_angle': angle}
        if projective:
            settings.update(scale_x=1.2, scale_y=.9, skew_x=12,
                            perspective_x=.2, perspective_y=-.15)
        item.set_lettering_transform(settings)
        item.setTransformOriginPoint(80, 30)
        item.setRotation(8)
        return item

    def state(self, item):
        return TextItemProperties.from_text_item(item).to_dict()

    def assert_item_matches(self, before, after):
        self.assertEqual(before.toPlainText(), after.toPlainText())
        self.assertEqual(before.lettering_transform, after.lettering_transform)
        self.assertAlmostEqual(before.logical_text_width(), after.logical_text_width(), places=6)
        a, b = before.sceneBoundingRect(), after.sceneBoundingRect()
        np.testing.assert_allclose(a.getRect(), b.getRect(), atol=1e-5)
        for position in (1, 9):
            cursor = QtGui.QTextCursor(after.document())
            cursor.setPosition(position)
            fmt = cursor.charFormat()
            self.assertEqual(fmt.fontWeight() == QtGui.QFont.Bold, position == 9)
            self.assertEqual(fmt.fontItalic(), position == 9)
            self.assertEqual(fmt.fontUnderline(), position == 9)
            if position == 9:
                self.assertEqual(fmt.foreground().color().name(), '#b52061')

    def test_signed_curve_wrap_width_and_rich_text_survive_repeated_viewer_reload(self):
        for angle in (100, -100):
            with self.subTest(angle=angle):
                original = self.item(angle, projective=True)
                viewer = ImageViewer(None)
                viewer.display_image_array(np.full((500, 500, 3), 255, np.uint8), fit=False)
                restored = viewer.add_text_item(self.state(original))
                self.assert_item_matches(original, restored)
                for _ in range(4):
                    saved = viewer.save_state()
                    self.assertEqual(saved['text_items_state'][0]['width'], 240)
                    viewer.clear_text_items()
                    viewer.load_state(saved)
                    self.assert_item_matches(original, viewer.text_items[0])
                viewer.deleteLater()
                self.app.processEvents()

    def test_real_project_save_reopen_and_save_as_keep_curve_source(self):
        for angle in (90, -90):
            with self.subTest(angle=angle), project_directory() as directory:
                original = self.item(angle)
                source = prepared_main(directory)
                source.image_states[source.image_files[0]]['viewer_state'] = {
                    'text_items_state': [self.state(original)]}
                project = str(Path(directory) / 'curve.ctpr')
                save_state_to_proj_file_v2(source, project)
                loaded = main_fixture(Path(directory) / 'loaded')
                load_state_from_proj_file(loaded, project)
                state = loaded.image_states[loaded.image_files[0]]['viewer_state']['text_items_state'][0]
                self.assert_item_matches(original, ImageSaveRenderer.create_text_item(state))
                second = str(Path(directory) / 'copy.ctpr')
                save_state_to_proj_file_v2(loaded, second)
                reopened = main_fixture(Path(directory) / 'reopened')
                load_state_from_proj_file(reopened, second)
                state = reopened.image_states[reopened.image_files[0]]['viewer_state']['text_items_state'][0]
                self.assertEqual(state['width'], 240)
                self.assert_item_matches(original, ImageSaveRenderer.create_text_item(state))

    def test_style_and_reset_restoration_preserve_source_and_curve(self):
        item = self.item(-85)
        before = self.state(item)
        styled = styled_properties(before, {'font_size': 29}, {'font_size'})
        apply_style_properties(item, styled)
        self.assertEqual(item.lettering_transform['arc_angle'], -85)
        self.assertEqual(item.logical_text_width(), 240)
        apply_style_properties(item, dict(before, lettering_transform={}))
        self.assertEqual(item.lettering_transform, {})
        apply_style_properties(item, before)
        self.assert_item_matches(ImageSaveRenderer.create_text_item(before), item)

    def test_page_renderer_matches_live_curved_item_pixels(self):
        image = np.full((500, 500, 3), 245, np.uint8)
        for angle in (95, -95):
            with self.subTest(angle=angle):
                original = self.item(angle, projective=True)
                live = ImageSaveRenderer(image)
                live.scene.addItem(original)
                saved = ImageSaveRenderer(image)
                saved.add_state_to_image({'text_items_state': [self.state(original)]})
                expected, actual = live.render_to_image(), saved.render_to_image()
                np.testing.assert_array_equal(actual, expected)
                self.assertTrue(np.any(actual != 245))

    def test_webtoon_lazy_save_preserves_full_curve_and_logical_width_on_owner(self):
        viewer = ImageViewer(None)
        loader = SimpleNamespace(image_file_paths=['a.png', 'b.png'])
        converter = SimpleNamespace(
            page_local_to_scene_position=lambda point, page: point + QtCore.QPointF(0, page * 100),
            scene_to_page_local_position=lambda point, page: point - QtCore.QPointF(0, page * 100),
            clip_text_item_to_page=Mock(side_effect=AssertionError('Curved text must stay whole')))
        layout = SimpleNamespace(get_pages_for_scene_bounds=lambda rect: {0, 1},
                                 get_page_at_position=lambda y: int(y // 100))
        manager = TextItemManager(viewer, layout, converter, loader)
        manager.main_controller = SimpleNamespace(image_files=loader.image_file_paths,
            image_states={path: {'viewer_state': {'text_items_state': []}} for path in loader.image_file_paths})
        original = self.item(-100)
        original.setPos(10, 110)
        item = viewer.add_text_item(self.state(original))
        item._page_path = 'a.png'
        pages = defaultdict(lambda: {'text_items': []})
        manager.save_text_items_to_states(pages)
        self.assertEqual(len(pages[0]['text_items']), 1)
        self.assertEqual(pages[1]['text_items'], [])
        self.assertEqual(pages[0]['text_items'][0]['width'], 240)
        self.assertEqual(pages[0]['text_items'][0]['text'], item.toHtml())
        manager.unload_text_items(0, 0, 100, 'a.png')
        state = manager.main_controller.image_states['a.png']
        manager.load_text_items(state, 0)
        self.assert_item_matches(original, viewer.text_items[0])
        viewer.deleteLater()

    def test_curved_seam_export_matches_full_canvas_without_mutating_source(self):
        with project_directory() as directory:
            paths, images = [], []
            for index in range(2):
                image = np.full((150, 320, 3), 255, np.uint8)
                path = str(Path(directory) / f'{index}.png')
                imk.write_image(path, image)
                paths.append(path)
                images.append(image)
            original = self.item(-100)
            original.setPos(40, 100)
            original.setRotation(0)
            source = self.state(original)
            main = SimpleNamespace(image_files=paths, image_data=dict(zip(paths, images)),
                image_states={path: {'viewer_state': {'text_items_state': [source] if i == 0 else []}}
                              for i, path in enumerate(paths)})
            before = copy.deepcopy(main.image_states)
            whole = ImageSaveRenderer(np.full((300, 320, 3), 255, np.uint8))
            whole.add_state_to_image({'text_items_state': [source]})
            expected = whole.render_to_image()[150:]
            page = ImageSaveRenderer(images[1])
            page.add_state_to_image(main.image_states[paths[1]]['viewer_state'], 1, main)
            actual = page.render_to_image()
            delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
            self.assertLessEqual(delta.max(), 2)
            self.assertTrue(np.any(actual != 255))
            self.assertEqual(main.image_states, before)


class CurvedLetteringPsdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        try:
            cls.psapi = require_photoshopapi()
        except Exception as exc:
            raise unittest.SkipTest(str(exc))

    def curve_state(self, angle):
        item = TextBlockItem(font_family='Arial', font_size=28, outline_color=None)
        item.set_plain_text('A CURVED TITLE')
        item.setTextWidth(310)
        item.setPos(75, 50)
        item.set_lettering_transform({'arc_angle': angle, 'perspective_x': .1})
        return TextItemProperties.from_text_item(item).to_dict()

    def test_native_psd_keeps_other_text_and_reimports_signed_curve_pixels(self):
        for angle in (100, -100):
            with self.subTest(angle=angle), project_directory() as directory:
                state = self.curve_state(angle)
                ordinary = dict(state, lettering_transform={}, position=(100, 190))
                background = np.full((400, 500, 3), 245, np.uint8)
                page = PsdPageData('page.png', background,
                                   {'text_items_state': [state, ordinary]}, [])
                path = export_psd_pages(directory, [page], 'curve')
                document = self.psapi.LayeredFile.read(path)
                curved, native = document.layers[0].layers
                self.assertIsInstance(curved, self.psapi.ImageLayer_8bit)
                self.assertIsInstance(native, self.psapi.TextLayer_8bit)
                self.assertIn('rasterized curved', curved.name)
                rgba, x, y = _render_raster_lettering(state)
                np.testing.assert_array_equal(_image_layer_to_rgba(curved), rgba)
                actual, imported, context = _read_single_psd(path)
                self.assertEqual(len(imported), 1)
                self.assertIn('Curved or perspective', context.warning)
                self.assertTrue(np.any(actual != background))
                expected = background.copy()
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(500, x + rgba.shape[1]), min(400, y + rgba.shape[0])
                crop = rgba[y0-y:y1-y, x0-x:x1-x]
                alpha = crop[..., 3:4].astype(np.float32) / 255
                expected[y0:y1, x0:x1] = (crop[..., :3] * alpha + 245 * (1-alpha)).astype(np.uint8)
                np.testing.assert_array_equal(actual, expected)

    def test_curve_export_inside_real_app_worker_has_no_qt_thread_warnings(self):
        from app.thread_worker import GenericWorker
        state = self.curve_state(-70)
        result, errors, messages = [], [], []
        with project_directory() as directory:
            page = PsdPageData('curve.png', np.full((400, 500, 3), 255, np.uint8),
                               {'text_items_state': [state]}, [])
            def work():
                return _read_single_psd(export_psd_pages(directory, [page], 'curve'))
            loop, pool = QtCore.QEventLoop(), QtCore.QThreadPool()
            worker = GenericWorker(work)
            worker.signals.result.connect(result.append)
            worker.signals.error.connect(errors.append)
            worker.signals.finished.connect(loop.quit)
            previous = QtCore.qInstallMessageHandler(lambda kind, context, message: messages.append(message))
            try:
                pool.start(worker)
                timeout = QtCore.QTimer()
                timeout.setSingleShot(True)
                timeout.timeout.connect(loop.quit)
                timeout.start(10000)
                loop.exec()
                self.assertTrue(pool.waitForDone(1000))
                self.app.processEvents()
            finally:
                QtCore.qInstallMessageHandler(previous)
            self.assertEqual(errors, [])
            self.assertEqual(messages, [])
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0][1], [])
            self.assertTrue(np.any(result[0][0] != 255))

    def test_curved_psd_raster_matches_page_export_with_selected_word_outline(self):
        state = self.curve_state(-100)
        state.update(position=(70, 80), rotation=0, transform_origin=(0, 0),
                     lettering_transform={'arc_angle': -100})
        state['selection_outlines'] = [
            OutlineInfo(2, 8, QtGui.QColor('#f05f26'), 3, OutlineType.Selection)]
        background = np.full((400, 500, 3), 245, np.uint8)
        rgba, x, y = _render_raster_lettering(state)
        expected = background.copy()
        alpha = rgba[..., 3:4].astype(np.float32) / 255
        expected[y:y+rgba.shape[0], x:x+rgba.shape[1]] = (
            rgba[..., :3] * alpha + 245 * (1-alpha)).round().astype(np.uint8)
        renderer = ImageSaveRenderer(background)
        renderer.add_state_to_image({'text_items_state': [state]})
        actual = renderer.render_to_image()
        difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
        self.assertLessEqual(int(difference.max()), 2)
        colored = (actual[..., 0].astype(int) - actual[..., 1].astype(int)) > 50
        self.assertTrue(np.any(colored), 'The selected word outline must remain visible.')


if __name__ == '__main__':
    unittest.main()
