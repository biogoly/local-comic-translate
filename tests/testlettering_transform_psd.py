"""Actual PhotoshopAPI round trips for editable affine and perspective pixels."""
import copy
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.controllers.psd_support import require_photoshopapi
from app.controllers.psd_exporter import (
    PsdPageData, export_psd_pages, _build_text_layer, _render_perspective_lettering,
)
from app.controllers.psd_importer import _read_single_psd, _image_layer_to_rgba
from app.ui.canvas.save_renderer import ImageSaveRenderer
from app.ui.canvas.text_item import TextBlockItem
from app.ui.canvas.text.text_item_properties import TextItemProperties


class LetteringTransformPsdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        try:
            cls.psapi = require_photoshopapi()
        except Exception as exc:
            raise unittest.SkipTest(str(exc))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        item = TextBlockItem('Hello brave world', font_family='Arial', font_size=24,
                             render_color=QtGui.QColor('#17354b'), outline_color=None)
        item.setTextWidth(220)
        item.setPos(95, 90)
        item.setTransformOriginPoint(40, 24)
        item.setRotation(17)
        item.setScale(1.1)
        cursor = QtGui.QTextCursor(item.document())
        cursor.setPosition(6)
        cursor.setPosition(11, QtGui.QTextCursor.KeepAnchor)
        fmt = QtGui.QTextCharFormat()
        fmt.setFontWeight(QtGui.QFont.Bold)
        cursor.mergeCharFormat(fmt)
        self.state = TextItemProperties.from_text_item(item).to_dict()
        self.background = np.full((450, 650, 3), 245, np.uint8)

    def tearDown(self):
        self.temp.cleanup()

    def export(self, states):
        page = PsdPageData('page.png', self.background, {'text_items_state': states}, [])
        path = export_psd_pages(str(self.root), [page], 'fixture', str(self.root/'fixture.psd'))
        return path, self.psapi.LayeredFile.read(path)

    def test_affine_writes_actual_editable_six_value_transform(self):
        self.state['lettering_transform'] = {'scale_x': 1.4, 'scale_y': .8, 'skew_x': 18}
        path, doc = self.export([self.state])
        layer = doc.layers[0].layers[0]
        self.assertIsInstance(layer, self.psapi.TextLayer_8bit)
        self.assertIn('transformed', layer.name)
        item = ImageSaveRenderer.create_text_item(self.state)
        transform = item.sceneTransform()
        margin = item.document().documentMargin()
        anchor = transform.map(QtCore.QPointF(margin, margin))
        np.testing.assert_allclose(layer.transform(), [transform.m11(), transform.m12(),
            transform.m21(), transform.m22(), anchor.x(), anchor.y()], atol=1e-10)
        self.assertIn('Hello brave world', layer.text)
        self.assertGreaterEqual(layer.style_run_count, 3)

    def test_affine_import_preserves_matrix_and_anchor_at_parameter_edges(self):
        for settings, rotation, scale in (
            ({}, -18, 1),
            ({'scale_x': 1.4, 'scale_y': .8, 'skew_x': 18}, 17, 1.1),
            ({'scale_x': .25, 'scale_y': 4, 'skew_x': -45}, -40, .5),
            ({'scale_x': 4, 'scale_y': .25, 'skew_x': 45}, 80, 2),
        ):
            with self.subTest(settings=settings):
                self.state.update(lettering_transform=settings, rotation=rotation, scale=scale)
                path, _ = self.export([self.state])
                _, imported, context = _read_single_psd(path)
                self.assertEqual(len(imported), 1)
                self.assertIsNone(context.warning)
                before = ImageSaveRenderer.create_text_item(self.state)
                after = ImageSaveRenderer.create_text_item(imported[0])
                cursor = QtGui.QTextCursor(after.document())
                cursor.setPosition(8)
                self.assertEqual(cursor.charFormat().fontWeight(), QtGui.QFont.Bold)
                cursor.setPosition(2)
                self.assertNotEqual(cursor.charFormat().fontWeight(), QtGui.QFont.Bold)
                for point in ((0, 0), (110, 20), (220, 40)):
                    a = before.mapToScene(QtCore.QPointF(*point))
                    b = after.mapToScene(QtCore.QPointF(*point))
                    self.assertAlmostEqual(a.x(), b.x(), places=5)
                    self.assertAlmostEqual(a.y(), b.y(), places=5)

    def test_perspective_raster_is_labeled_transparent_and_does_not_flatten_other_text(self):
        perspective = copy.deepcopy(self.state)
        perspective['lettering_transform'] = {'perspective_x': .35, 'perspective_y': -.2}
        rgba, x, y = _render_perspective_lettering(perspective)
        path, doc = self.export([self.state, perspective])
        ordinary, projected = doc.layers[0].layers
        self.assertIsInstance(ordinary, self.psapi.TextLayer_8bit)
        self.assertIsInstance(projected, self.psapi.ImageLayer_8bit)
        self.assertIn('rasterized perspective', projected.name)
        np.testing.assert_array_equal(_image_layer_to_rgba(projected), rgba)
        self.assertEqual(projected.center_x, x + rgba.shape[1] / 2)
        self.assertEqual(projected.center_y, y + rgba.shape[0] / 2)
        self.assertTrue(np.any(rgba[..., 3] == 0))
        self.assertTrue(np.any(rgba[..., 3] > 0))
        image, imported, context = _read_single_psd(path)
        self.assertEqual(len(imported), 1)
        self.assertIn('background pixels', context.warning)
        self.assertTrue(np.any(image != self.background))
        alpha = rgba[..., 3:4].astype(np.float32) / 255
        expected = (rgba[..., :3] * alpha + 245 * (1-alpha)).astype(np.uint8)
        np.testing.assert_array_equal(image[y:y+rgba.shape[0], x:x+rgba.shape[1]], expected)

    def test_perspective_raster_matches_page_renderer(self):
        self.state.update(rotation=0, scale=1, transform_origin=(0, 0),
                          lettering_transform={'scale_x': 1.2, 'perspective_x': .3, 'perspective_y': -.25})
        rgba, x, y = _render_perspective_lettering(self.state)
        expected = self.background.copy()
        alpha = rgba[..., 3:4].astype(np.float32) / 255
        expected[y:y+rgba.shape[0], x:x+rgba.shape[1]] = (
            rgba[..., :3] * alpha + 245 * (1-alpha)).round().astype(np.uint8)
        renderer = ImageSaveRenderer(self.background)
        renderer.add_state_to_image({'text_items_state': [self.state]})
        actual = renderer.render_to_image()
        difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
        self.assertLessEqual(difference.max(), 2)

    def test_ordinary_text_keeps_existing_editable_export_path(self):
        self.state.update(rotation=0, scale=1)
        layer = _build_text_layer(self.state, 1)
        self.assertIsInstance(layer, self.psapi.TextLayer_8bit)
        self.assertEqual(layer.name, 'Text 1')
        self.assertAlmostEqual(layer.rotation_angle, 0)

    def test_perspective_at_page_edge_reimports_visible_pixels_without_shifting(self):
        self.state.update(position=(-25, -12), rotation=-10, scale=1,
                          lettering_transform={'perspective_x': -.35, 'perspective_y': .2})
        path, _ = self.export([self.state])
        imported_pixels, imported_text, context = _read_single_psd(path)
        self.assertEqual(imported_text, [])
        self.assertIsNotNone(context.warning)
        rgba, x, y = _render_perspective_lettering(self.state)
        expected = self.background.copy()
        x0, y0 = max(0, x), max(0, y)
        x1 = min(expected.shape[1], x+rgba.shape[1])
        y1 = min(expected.shape[0], y+rgba.shape[0])
        crop = rgba[y0-y:y1-y, x0-x:x1-x]
        alpha = crop[..., 3:4].astype(np.float32) / 255
        expected[y0:y1, x0:x1] = (crop[..., :3] * alpha + 245 * (1-alpha)).astype(np.uint8)
        self.assertTrue(np.any(expected != self.background))
        np.testing.assert_array_equal(imported_pixels, expected)

    def test_real_app_worker_can_export_and_import_transformed_text(self):
        from app.thread_worker import GenericWorker
        transformed = copy.deepcopy(self.state)
        transformed['lettering_transform'] = {'scale_x': 1.2, 'skew_x': 15}
        perspective = copy.deepcopy(self.state)
        perspective['lettering_transform'] = {'perspective_x': .3}
        result, errors, qt_messages = [], [], []
        loop = QtCore.QEventLoop()
        pool = QtCore.QThreadPool()
        def work():
            path, _ = self.export([transformed, perspective])
            return _read_single_psd(path)
        worker = GenericWorker(work)
        worker.signals.result.connect(result.append)
        worker.signals.error.connect(errors.append)
        worker.signals.finished.connect(loop.quit)
        previous = QtCore.qInstallMessageHandler(lambda kind, context, message: qt_messages.append(message))
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
        self.assertEqual(qt_messages, [])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0][1]), 1)
        self.assertTrue(np.any(result[0][0] != self.background))

    def test_failed_transformed_layer_cannot_silently_replace_existing_export(self):
        path, _ = self.export([self.state])
        saved = Path(path).read_bytes()
        self.state.update(width=50000, lettering_transform={'perspective_x': .3})
        with self.assertLogs('app.controllers.psd_exporter', level='ERROR'):
            with self.assertRaisesRegex(ValueError, 'PSD raster size'):
                self.export([self.state])
        self.assertEqual(Path(path).read_bytes(), saved)


if __name__ == '__main__':
    unittest.main()
