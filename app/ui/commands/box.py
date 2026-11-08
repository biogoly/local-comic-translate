import numpy as np
from PySide6.QtGui import QUndoCommand
from PySide6.QtCore import QRectF, QPointF, QCoreApplication

from .base import RectCommandBase
from ..canvas.rectangle import MoveableRectItem
from ..canvas.text_item import TextBlockItem
from pipeline.webtoon_utils import find_block_page_index, get_first_visible_block
from modules.utils.region_metadata import find_region


class SetLetteringAngleCommand(QUndoCommand):
    """Restore the manual angle by page and region ID, including after navigation."""
    def __init__(self, main, path, block, new_angle, *, ocr_states=None):
        super().__init__(QCoreApplication.translate("RectItemController", "Edit lettering geometry"))
        self.main = main
        self.path = path
        self.region_id = block.region_id
        self.page_id = main.image_states[path].get("page_id")
        self.epoch = getattr(main, "_translation_project_epoch", None)
        self.angles = (getattr(block, "manual_lettering_angle", None), new_angle)
        self.ocr_states = ocr_states

    def _apply(self, index):
        main = self.main
        state = main.image_states.get(self.path)
        if (state is None or self.path not in main.image_files or self.page_id != state.get("page_id")
                or self.epoch != getattr(main, "_translation_project_epoch", None)):
            return
        saved = find_region(state.get("blk_list", []), region_id=self.region_id)
        page_index = main.image_files.index(self.path)
        candidates = main.blk_list if page_index == main.curr_img_idx else []
        if main.image_viewer.webtoon_mode:
            candidates = [block for block in main.blk_list if block.region_id == self.region_id and
                          find_block_page_index(block, main.image_viewer.webtoon_manager) == page_index]
        live = find_region(candidates, region_id=self.region_id)
        for block in (saved, live):
            if block is None:
                continue
            block.manual_lettering_angle = self.angles[index]
            if self.ocr_states:
                angle, origin = self.ocr_states[index]
                block.angle = angle
                block.tr_origin_point = origin
        if self.ocr_states:
            angle, origin = self.ocr_states[index]
            if live is not None:
                rectangles = main.image_viewer.rectangles
                rects = [item for item in rectangles if getattr(item, 'region_id', None) == self.region_id]
                if len(rects) != 1:
                    candidates = rects or [r for r in rectangles if not getattr(r, 'region_id', None)]
                    rects = [item for item in candidates if self._matches_rect(
                        item.pos().x(), item.pos().y(), item.rect().width(), item.rect().height(), live)]
                if len(rects) == 1:
                    rects[0].region_id = self.region_id
                    rects[0].setTransformOriginPoint(QPointF(*origin) if origin else rects[0].boundingRect().center())
                    rects[0].setRotation(angle)
            if saved is not None:
                saved_rectangles = state.get('viewer_state', {}).get('rectangles', [])
                rects = [props for props in saved_rectangles if props.get('region_id') == self.region_id]
                if not rects:
                    rects = [props for props in saved_rectangles if not props.get('region_id')
                             and self._matches_rect(*props['rect'], saved)]
                if len(rects) == 1:
                    rects[0]['rotation'] = angle
                    rects[0]['transform_origin'] = origin or (0., 0.)
        main.mark_project_dirty()

    @staticmethod
    def _matches_rect(x, y, w, h, block):
        # Old project files store the graphics item bounds including its pen.
        return np.allclose((x, y, x + w, y + h), block.xyxy, atol=1, rtol=0)

    def redo(self):
        self._apply(1)

    def undo(self):
        self._apply(0)


class AddRectangleCommand(QUndoCommand, RectCommandBase):
    def __init__(self, main_page, rect_item, blk, blk_list):
        super().__init__()
        self.viewer = main_page.image_viewer
        self.scene = self.viewer._scene
        self.blk_list = blk_list
        self.rect_properties = self.save_rect_properties(rect_item)
        self.blk_properties = self.save_blk_properties(blk)

    def redo(self):
        if not self.find_matching_rect(self.scene, self.rect_properties):
            self.create_rect_item(self.rect_properties, self.viewer)

        if not self.find_matching_blk(self.blk_list, self.blk_properties):
            blk = self.create_new_blk(self.blk_properties)
            self.blk_list.append(blk)

    def undo(self):
        matching_item = self.find_matching_rect(self.scene, self.rect_properties)
        matching_blk = self.find_matching_blk(self.blk_list, self.blk_properties)

        if matching_item:
            self.scene.removeItem(matching_item)
            self.viewer.rectangles.remove(matching_item)

        if matching_blk:
            self.blk_list.remove(matching_blk)

class BoxesChangeCommand(QUndoCommand, RectCommandBase):
    def __init__(self, viewer, old_state, new_state, blk_list):
        super().__init__()
        self.viewer = viewer
        self.scene = self.viewer._scene
        self.blk_list = blk_list
        
        self.old_xyxy = [int(c) for c in old_state.rect]
        self.old_angle = old_state.rotation
        self.old_tr_origin = (old_state.transform_origin.x(), old_state.transform_origin.y())

        self.new_xyxy = [int(c) for c in new_state.rect]
        self.new_angle = new_state.rotation
        self.new_tr_origin = (new_state.transform_origin.x(), new_state.transform_origin.y())

    def redo(self):
        for blk in self.blk_list:
            if (np.array_equal(blk.xyxy, self.old_xyxy) and
                blk.angle == self.old_angle):
            
                blk.xyxy[:] = self.new_xyxy
                blk.angle = self.new_angle
                blk.tr_origin_point = self.new_tr_origin

                self.find_and_update_item(self.scene, self.old_xyxy, self.old_angle, 
                                                self.new_xyxy, self.new_angle, self.new_tr_origin)
                self.scene.update()

    def undo(self):
        for blk in self.blk_list:
            if (np.array_equal(blk.xyxy, self.new_xyxy) and
                blk.angle == self.new_angle ):
                
                blk.xyxy[:] = self.old_xyxy
                blk.angle = self.old_angle
                blk.tr_origin_point = self.old_tr_origin

                self.find_and_update_item(self.scene, self.new_xyxy, self.new_angle, 
                                        self.old_xyxy, self.old_angle, self.old_tr_origin)
                self.scene.update()

    @staticmethod
    def find_and_update_item(scene, old_xyxy, old_angle, new_xyxy, new_angle, new_tr_origin):
        for item in scene.items():
            # Check if item position and properties match
            if (isinstance(item, (MoveableRectItem, TextBlockItem)) and
                int(item.pos().x()) == int(old_xyxy[0]) and 
                int(item.pos().y()) == int(old_xyxy[1]) and
                int(item.rotation()) == int(old_angle)):

                new_width = new_xyxy[2] - new_xyxy[0]
                new_height = new_xyxy[3] - new_xyxy[1]

                if isinstance(item, MoveableRectItem):
                    rect = QRectF(0, 0, new_width, new_height)
                    item.setRect(rect)

                item.setTransformOriginPoint(QPointF(*new_tr_origin))
                item.setPos(new_xyxy[0], new_xyxy[1])
                item.setRotation(new_angle)


class ResizeBlocksCommand(QUndoCommand):
    def __init__(self, main_page, blk_list, diff: int):
        super().__init__()
        self.main = main_page
        self.blk_list = blk_list
        self.blocks = list(blk_list)
        self.old_xyxy = [blk.xyxy.copy() for blk in self.blocks]
        self.new_xyxy = [
            [
                old[0] - diff,
                old[1] - diff,
                old[2] + diff,
                old[3] + diff,
            ]
            for old in self.old_xyxy
        ]

    def _refresh_rectangles(self):
        viewer = self.main.image_viewer
        if self.main.webtoon_mode:
            viewer.clear_rectangles_in_visible_area()
        else:
            viewer.clear_rectangles(page_switch=True)

        if not viewer.hasPhoto() or not self.main.blk_list:
            return

        for blk in self.main.blk_list:
            x1, y1, x2, y2 = blk.xyxy
            rect = QRectF(0, 0, x2 - x1, y2 - y1)
            transform_origin = QPointF(*blk.tr_origin_point) if blk.tr_origin_point else None
            rect_item = viewer.add_rectangle(rect, QPointF(x1, y1), blk.angle, transform_origin)
            rect_item.region_id = blk.region_id
            self.main.connect_rect_item_signals(rect_item)

        if self.main.webtoon_mode:
            first_block = get_first_visible_block(self.main.blk_list, viewer)
            if first_block is None:
                first_block = self.main.blk_list[0]
        else:
            first_block = self.main.blk_list[0]

        rect = self.main.rect_item_ctrl.find_corresponding_rect(first_block, 0.5)
        viewer.select_rectangle(rect)
        self.main.set_tool('box')

    def _apply(self, coords):
        for blk, xyxy in zip(self.blocks, coords):
            if blk in self.blk_list:
                blk.xyxy[:] = xyxy
        self._refresh_rectangles()

    def redo(self):
        self._apply(self.new_xyxy)

    def undo(self):
        self._apply(self.old_xyxy)
            
class ClearRectsCommand(QUndoCommand, RectCommandBase):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.scene = viewer._scene
        self.properties_list = []
        
    def undo(self):
        for properties in self.properties_list:
            self.create_rect_item(properties, self.viewer)
        self.scene.update()
        
    def redo(self):
        self.properties_list = []
        for item in self.scene.items():
            if isinstance(item, MoveableRectItem) and item != self.viewer.photo:
                self.properties_list.append(self.save_rect_properties(item))
                self.scene.removeItem(item)
                self.viewer.selected_rect = None
        self.scene.update()
    
class DeleteBoxesCommand(QUndoCommand, RectCommandBase):
    def __init__(self, main_page, rect_item, text_item, blk, blk_list):
        super().__init__()
        self.ct = main_page
        self.viewer = main_page.image_viewer
        self.scene = self.viewer._scene
        self.rect_properties = self.save_rect_properties(rect_item) if rect_item else None
        self.txt_item_prp = self.save_txt_item_properties(text_item) if text_item else None
        self.blk_properties = self.save_blk_properties(blk)
        self.blk_list = blk_list

    def redo(self):
        matching_rect = self.find_matching_rect(self.scene, self.rect_properties) if self.rect_properties else None
        matching_txt_item = self.find_matching_txt_item(self.scene, self.txt_item_prp) if self.txt_item_prp else None
        matching_blk = self.find_matching_blk(self.blk_list, self.blk_properties)

        if matching_rect:
            self.scene.removeItem(matching_rect)
            self.viewer.rectangles.remove(matching_rect)
            self.viewer.selected_rect = None
            self.scene.update()

        if matching_blk:
            self.blk_list.remove(matching_blk)
            self.ct.curr_tblock = None

        if matching_txt_item:
            self.scene.removeItem(matching_txt_item)
            self.viewer.text_items.remove(matching_txt_item)
            self.ct.curr_tblock_item = None
            self.scene.update()

    def undo(self):
        if self.rect_properties and not self.find_matching_rect(self.scene, self.rect_properties):
            self.create_rect_item(self.rect_properties, self.viewer)
            self.scene.update()

        if not self.find_matching_blk(self.blk_list, self.blk_properties):
            blk = self.create_new_blk(self.blk_properties)
            self.blk_list.append(blk)

        if self.txt_item_prp and not self.find_matching_txt_item(self.scene, self.txt_item_prp):
            text_item = self.create_new_txt_item(self.txt_item_prp, self.viewer)

class AddTextItemCommand(QUndoCommand, RectCommandBase):
    def __init__(self, main_page, text_item):
        super().__init__()
        self.viewer = main_page.image_viewer
        self.scene = self.viewer._scene
        self.txt_item_prp = self.save_txt_item_properties(text_item)

    def redo(self):
        if not self.find_matching_txt_item(self.scene, self.txt_item_prp):
            text_item = self.create_new_txt_item(self.txt_item_prp, self.viewer)

    def undo(self):
        matching_txt_item = self.find_matching_txt_item(self.scene, self.txt_item_prp)
        if matching_txt_item:
            self.scene.removeItem(matching_txt_item)
            self.viewer.text_items.remove(matching_txt_item)
            self.scene.update()


class ReplaceDetectedBlocksCommand(QUndoCommand):
    def __init__(self, main_page, old_blocks, new_blocks):
        super().__init__()
        self.main = main_page
        self._old_blocks = [blk.deep_copy() for blk in old_blocks]
        self._new_blocks = [blk.deep_copy() for blk in new_blocks]

    def _copy_blocks(self, blocks):
        return [blk.deep_copy() for blk in blocks]

    def _serialize_rectangles(self, blocks):
        rectangles = []
        for blk in blocks:
            x1, y1, x2, y2 = blk.xyxy
            rectangles.append(
                {
                    "rect": (float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                    "rotation": float(getattr(blk, "angle", 0)),
                    "region_id": blk.region_id,
                    "transform_origin": tuple(getattr(blk, "tr_origin_point", ()) or (0.0, 0.0)),
                }
            )
        return rectangles

    def _sync_current_page_state(self, blocks):
        if not self.main.image_files or self.main.curr_img_idx >= len(self.main.image_files):
            return

        file_path = self.main.image_files[self.main.curr_img_idx]
        state = self.main.image_states.get(file_path)
        if state is None:
            return

        state["blk_list"] = self._copy_blocks(blocks)
        viewer_state = state.setdefault("viewer_state", {})
        viewer_state["rectangles"] = self._serialize_rectangles(blocks)

    def _refresh_rectangles(self):
        viewer = self.main.image_viewer
        viewer.clear_rectangles(page_switch=True)

        if not viewer.hasPhoto() or not self.main.blk_list:
            return

        for blk in self.main.blk_list:
            x1, y1, x2, y2 = blk.xyxy
            rect = QRectF(0, 0, x2 - x1, y2 - y1)
            transform_origin = QPointF(*blk.tr_origin_point) if blk.tr_origin_point else None
            rect_item = viewer.add_rectangle(rect, QPointF(x1, y1), blk.angle, transform_origin)
            rect_item.region_id = blk.region_id
            self.main.connect_rect_item_signals(rect_item)

        first_block = self.main.blk_list[0]
        rect = self.main.rect_item_ctrl.find_corresponding_rect(first_block, 0.5)
        viewer.select_rectangle(rect)
        self.main.set_tool("box")

    def _apply(self, blocks):
        copied_blocks = self._copy_blocks(blocks)
        self.main.blk_list = copied_blocks
        self._sync_current_page_state(copied_blocks)

        self.main.curr_tblock = None
        self.main.curr_tblock_item = None
        self.main.image_viewer.selected_rect = None
        self._refresh_rectangles()

    def redo(self):
        self._apply(self._new_blocks)

    def undo(self):
        self._apply(self._old_blocks)

 

