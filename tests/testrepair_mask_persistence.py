import unittest
from types import SimpleNamespace

import msgpack
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.controllers.local_repair import LocalRepairController
from app.projects.parsers import ProjectEncoder, ProjectDecoder
from app.ui.canvas.drawing_manager import DrawingManager
from app.ui.canvas.webtoons.coordinate_converter import CoordinateConverter
from app.ui.canvas.webtoons.scene_items.brush_stroke_manager import BrushStrokeManager


def project_round_trip(strokes):
    packed = msgpack.packb(strokes, default=ProjectEncoder().encode, use_bin_type=True)
    return msgpack.unpackb(packed, object_hook=ProjectDecoder().decode)


class RepairMaskPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.scene = QtWidgets.QGraphicsScene()
        self.viewer = SimpleNamespace(_scene=self.scene, photo=None)
        self.drawing = DrawingManager(self.viewer)
        self.viewer.drawing_manager = self.drawing
        self.layout = SimpleNamespace(image_positions=[0, 90], image_heights=[90, 100], webtoon_width=220,
                                      get_page_at_position=lambda y: 0 if y < 90 else 1)
        self.loader = SimpleNamespace(image_data={1:np.empty((100, 160, 3), np.uint8)}, image_file_paths=['first','page'])
        self.converter = CoordinateConverter(self.layout, self.loader)
        self.webtoon = BrushStrokeManager(self.viewer, self.layout, self.converter, self.loader)
        self.webtoon.main_controller = SimpleNamespace(image_states={'page': {}})
        self.origin = self.converter.page_local_to_scene_position(QtCore.QPointF(), 1)

    def mask(self, origin=None):
        return LocalRepairController._current_mask(SimpleNamespace(viewer=self.viewer), (100,160,3),
                                                   QtCore.QPointF() if origin is None else origin)

    def filled(self, origin):
        path = QtGui.QPainterPath()
        path.setFillRule(QtCore.Qt.WindingFill)
        # Overlap with the same winding to distinguish the serialized fill rule.
        path.addEllipse(QtCore.QRectF(9.25,10.75,56.5,37.25))
        path.addRect(QtCore.QRectF(30.5,16.75,60.75,45.5))
        pen = QtGui.QPen(QtCore.Qt.NoPen)
        brush = QtGui.QBrush(QtGui.QColor(21,90,180,130),QtCore.Qt.SolidPattern)
        item = self.scene.addPath(path,pen,brush)
        item.setPos(origin)
        return item

    def stroke(self, origin):
        path = QtGui.QPainterPath(QtCore.QPointF(8.125,47.875))
        path.cubicTo(20.5,2.5,55.5,86.5,88.5,12.25)
        path.lineTo(95.25,68.75)
        pen = QtGui.QPen(QtGui.QColor(200,0,0,80),3.625,QtCore.Qt.DashLine,QtCore.Qt.FlatCap,QtCore.Qt.MiterJoin)
        pen.setDashPattern([2,1,3,1])
        pen.setDashOffset(0.25)
        pen.setMiterLimit(3)
        item = self.scene.addPath(path,pen)
        item.setPos(origin)
        return item

    def test_normal_saved_filled_mask_has_no_new_outline_or_fill_hole(self):
        self.filled(QtCore.QPointF(1.25,2.5))
        before = self.mask()
        self.assertGreater(np.count_nonzero(before),500)
        self.drawing.load_brush_strokes(project_round_trip(self.drawing.save_brush_strokes()))
        restored, = self.scene.items()
        self.assertEqual(restored.pen().style(),QtCore.Qt.NoPen)
        self.assertEqual(restored.path().fillRule(),QtCore.Qt.WindingFill)
        np.testing.assert_array_equal(self.mask(),before)

    def test_normal_fractional_custom_dashed_stroke_keeps_every_pixel(self):
        self.stroke(QtCore.QPointF())
        before = self.mask()
        self.drawing.load_brush_strokes(project_round_trip(self.drawing.save_brush_strokes()))
        pen = self.scene.items()[0].pen()
        self.assertEqual(pen.widthF(),3.625)
        self.assertEqual(pen.capStyle(),QtCore.Qt.FlatCap)
        self.assertEqual(pen.joinStyle(),QtCore.Qt.MiterJoin)
        self.assertEqual(pen.dashOffset(),0.25)
        np.testing.assert_array_equal(self.mask(),before)

    def test_webtoon_save_redistribute_unload_reload_preserves_geometry_and_style(self):
        self.filled(self.origin)
        self.stroke(self.origin)
        before = self.mask(self.origin)
        saved = {0:{'brush_strokes':[]},1:{'brush_strokes':[]}}
        self.webtoon.save_brush_strokes_to_states(saved)
        self.assertEqual(len(saved[1]['brush_strokes']),2)
        self.assertFalse(saved[0]['brush_strokes'])
        stored = project_round_trip(saved[1]['brush_strokes'])
        redistributed = {0:{'brush_strokes':[]},1:{'brush_strokes':[]}}
        self.webtoon.redistribute_existing_brush_strokes([(s,1) for s in stored],redistributed)
        self.scene.clear()
        self.webtoon.load_brush_strokes(redistributed[1],1)
        np.testing.assert_array_equal(self.mask(self.origin),before)
        self.webtoon.unload_brush_strokes(1,self.origin.y(),self.origin.y()+100,'page')
        self.assertEqual(self.scene.items(),[])
        state=self.webtoon.main_controller.image_states['page']
        state['brush_strokes'] = project_round_trip(state['brush_strokes'])
        self.webtoon.load_brush_strokes(state,1)
        np.testing.assert_array_equal(self.mask(self.origin),before)

    def test_legacy_strokes_keep_old_normal_and_webtoon_defaults(self):
        path = QtGui.QPainterPath()
        path.addRect(13,17,28,23)
        for brush_color,is_filled in [('#80ff0000',True),('#00000000',False),('#ff251122',False)]:
            record={'path':path,'pen':'#ff000000','brush':brush_color,'width':2}
            for webtoon in (False,True):
                with self.subTest(color=brush_color,webtoon=webtoon):
                    self.scene.clear()
                    origin = self.origin if webtoon else QtCore.QPointF()
                    historical = QtGui.QPen(QtGui.QColor(record['pen']),2,QtCore.Qt.SolidLine,QtCore.Qt.RoundCap,QtCore.Qt.RoundJoin)
                    self.scene.addPath(path.translated(origin),historical,
                                       QtGui.QBrush(QtGui.QColor(brush_color)) if is_filled else QtGui.QBrush())
                    expected = self.mask(origin)
                    self.scene.clear()
                    if webtoon:
                        self.webtoon.load_brush_strokes({'brush_strokes':[record]},1)
                    else:
                        self.drawing.load_brush_strokes([record])
                    np.testing.assert_array_equal(self.mask(origin),expected)
