from __future__ import annotations

import numpy as np
import math
from typing import TYPE_CHECKING
from PySide6.QtCore import QRectF, QPointF

from app.ui.canvas.rectangle import MoveableRectItem
from app.ui.commands.box import AddRectangleCommand, BoxesChangeCommand, SetLetteringAngleCommand

from modules.detection.utils.geometry import do_rectangles_overlap
from modules.utils.textblock import TextBlock
from modules.utils.region_metadata import find_region
from pipeline.webtoon_utils import find_block_page_index

if TYPE_CHECKING:
    from controller import ComicTranslate


class RectItemController:
    def __init__(self, main: ComicTranslate):
        self.main = main

    def connect_rect_item_signals(self, rect_item: MoveableRectItem, force_reconnect: bool = False):
        if getattr(rect_item, "_ct_signals_connected", False) and not force_reconnect:
            return

        if force_reconnect:
            try:
                rect_item.signals.change_undo.disconnect(self.rect_change_undo)
            except (TypeError, RuntimeError):
                pass
            if hasattr(rect_item, "_ct_ocr_slot"):
                try:
                    rect_item.signals.ocr_block.disconnect(rect_item._ct_ocr_slot)
                except (TypeError, RuntimeError):
                    pass
            if hasattr(rect_item, "_ct_translate_slot"):
                try:
                    rect_item.signals.translate_block.disconnect(rect_item._ct_translate_slot)
                except (TypeError, RuntimeError):
                    pass

        if not hasattr(rect_item, "_ct_ocr_slot"):
            rect_item._ct_ocr_slot = lambda: self.main.ocr(True)
        if not hasattr(rect_item, "_ct_translate_slot"):
            rect_item._ct_translate_slot = lambda: self.main.translate_image(True)

        rect_item.signals.change_undo.connect(self.rect_change_undo)
        rect_item.signals.ocr_block.connect(rect_item._ct_ocr_slot)
        rect_item.signals.translate_block.connect(rect_item._ct_translate_slot)
        rect_item._ct_signals_connected = True

    def handle_rectangle_selection(self, rect: QRectF):
        selected = self.main.image_viewer.selected_rect
        self.main.curr_tblock = self.find_corresponding_text_block(selected or rect.getCoords(), 0.5)
        if self.main.curr_tblock:
            self.main.s_text_edit.blockSignals(True)
            self.main.t_text_edit.blockSignals(True)
            self.main.s_text_edit.setPlainText(self.main.curr_tblock.text)
            self.main.t_text_edit.setPlainText(self.main.curr_tblock.translation)
            self.main.s_text_edit.blockSignals(False)
            self.main.t_text_edit.blockSignals(False)
        else:
            self.main.s_text_edit.clear()
            self.main.t_text_edit.clear()
            self.main.curr_tblock = None

    def handle_rectangle_creation(self, rect_item: MoveableRectItem):
        self.connect_rect_item_signals(rect_item)
        new_rect = rect_item.mapRectToScene(rect_item.rect())
        x1, y1, w, h = new_rect.getRect()
        x1, y1, w, h = int(x1), int(y1), int(w), int(h)
        new_rect_coords = (x1, y1, x1 + w, y1 + h)

        new_blk = TextBlock(text_bbox=np.array(new_rect_coords), is_manual=True)
        rect_item.region_id = new_blk.region_id
        self.main.blk_list.append(new_blk)
        command = AddRectangleCommand(self.main, rect_item, new_blk, self.main.blk_list)
        self.main.undo_group.activeStack().push(command)

        # A newly drawn box is the user's active text region. Selecting it here
        # binds the source/target editors immediately; previously the user had
        # to click the box a second time before typing had any target block.
        self.main.image_viewer.select_rectangle(rect_item)

    def handle_rectangle_deletion(self, rect: QRectF):
        rect_coords = rect.getCoords()
        current_text_block = self.find_corresponding_text_block(rect_coords, 0.5)
        self.main.blk_list.remove(current_text_block)

    def handle_rectangle_change(
            self, 
            old_rect_coords: tuple, 
            new_rect_coords: tuple, 
            new_angle: float, 
            new_tr_origin: QPointF
        ):
        # Find the corresponding TextBlock in blk_list
        for blk in self.main.blk_list:
            if do_rectangles_overlap(blk.xyxy, old_rect_coords, 0.2):
                # Update the TextBlock coordinates
                blk.xyxy[:] = [int(new_rect_coords[0]), 
                               int(new_rect_coords[1]),
                               int(new_rect_coords[2]), 
                               int(new_rect_coords[3])]
                blk.angle = new_angle if new_angle else 0
                blk.tr_origin_point = (new_tr_origin.x(), new_tr_origin.y()) if new_tr_origin else ()
                break

    def rect_change_undo(self, old_state, new_state):
        old_properties = getattr(old_state, 'properties', {})
        new_properties = getattr(new_state, 'properties', {})
        if not all(math.isfinite(state.rotation) for state in (old_state, new_state)):
            return
        rotated = old_state.rotation != new_state.rotation
        region_id = old_properties.get('region_id') or getattr(old_state, 'region_id', None)
        candidates = self.main.blk_list
        path = getattr(old_state, 'page_path', None)
        if old_properties and self.main.image_viewer.webtoon_mode and path in self.main.image_files:
            page_idx = self.main.image_files.index(path)
            candidates = [b for b in candidates if b.region_id == region_id and
                          find_block_page_index(b, self.main.image_viewer.webtoon_manager) == page_idx]
        block = find_region(candidates, region_id=region_id) if region_id else None

        if rotated and not old_properties:
            # Reloaded and legacy OCR rectangles may have only their geometry.
            if block is None:
                block = self._block_at_rectangle_bounds(old_state.rect, region_id)
            if block is not None:
                idx = self.main.curr_img_idx
                if self.main.image_viewer.webtoon_mode:
                    idx = find_block_page_index(block, self.main.image_viewer.webtoon_manager)
                if idx is not None and 0 <= idx < len(self.main.image_files):
                    path = self.main.image_files[idx]
                    stack = self.main.undo_stacks.get(path)
                    if stack is not None:
                        ocr_states = ((block.angle, tuple(block.tr_origin_point or ())),
                                      (new_state.rotation, new_state.transform_origin.toTuple()))
                        stack.push(SetLetteringAngleCommand(self.main, path, block, new_state.rotation,
                                                            ocr_states=ocr_states))
            return

        independent_angle = (bool(region_id) and rotated
                             or getattr(block, 'manual_lettering_angle', None) is not None
                             or getattr(block, 'detected_lettering_angle', None) is not None)
        if old_properties and (independent_angle or old_properties.get('lettering_transform')
                               or new_properties.get('lettering_transform')):
            # A lettering edit must not move the region used by OCR and cleanup.
            from types import SimpleNamespace
            from PySide6.QtCore import QCoreApplication
            from app.controllers.lettering_styles import ApplyLetteringStyleCommand
            from app.controllers.text_fit import page_to_scene_position
            if not path and 0 <= self.main.curr_img_idx < len(self.main.image_files):
                path = self.main.image_files[self.main.curr_img_idx]
            stack = self.main.undo_stacks.get(path)
            if stack is None or path not in self.main.image_files:
                return
            origin = QPointF()
            if self.main.image_viewer.webtoon_mode:
                origin = page_to_scene_position(self.main.image_viewer, self.main.image_files.index(path), QPointF())
            command = ApplyLetteringStyleCommand(SimpleNamespace(main=self.main), path, origin,
                [old_properties], [new_properties],
                label=QCoreApplication.translate('RectItemController', 'Edit lettering geometry'))
            if rotated and block is not None:
                stack.beginMacro(command.text())
                try:
                    stack.push(command)
                    stack.push(SetLetteringAngleCommand(self.main, path, block, new_state.rotation))
                finally:
                    stack.endMacro()
            else:
                stack.push(command)
            return
        command = BoxesChangeCommand(self.main.image_viewer, old_state,
                                         new_state, self.main.blk_list)
        self.main.undo_group.activeStack().push(command)
        self.handle_rectangle_change(
            old_state.rect, 
            new_state.rect,
            new_state.rotation,
            new_state.transform_origin
        )


    def _block_at_rectangle_bounds(self, bounds, region_id):
        matches = [blk for blk in self.main.blk_list
                   if (not region_id or blk.region_id == region_id) and np.size(blk.xyxy) == 4
                   and np.allclose(blk.xyxy, bounds, atol=1, rtol=0)]
        return matches[0] if len(matches) == 1 else None

    def find_corresponding_text_block(self, rect, iou_threshold: float = 0.5):
        if isinstance(rect, MoveableRectItem):
            identity = getattr(rect, 'region_id', None)
            if identity:
                block = find_region(self.main.blk_list, region_id=identity)
                if block is not None:
                    return block
            # Older files and clipped webtoon fragments can still be resolved
            # using the original, unrotated region bounds.
            bounds = QRectF(rect.pos(), rect.rect().size()).getCoords()
            block = self._block_at_rectangle_bounds(bounds, identity)
            if block is not None:
                rect.region_id = block.region_id
                return block
            if identity:
                return None
            rect = rect.mapRectToScene(rect.rect()).getCoords()
        for blk in self.main.blk_list:
            if do_rectangles_overlap(rect, blk.xyxy, iou_threshold):
                return blk
        return None

    def find_corresponding_rect(self, tblock: TextBlock, iou_threshold: int):
        identity = getattr(tblock, 'region_id', None)
        if identity:
            matches = [rect for rect in self.main.image_viewer.rectangles
                       if getattr(rect, 'region_id', None) == identity]
            if len(matches) == 1:
                return matches[0]
            if matches:
                return next((rect for rect in matches if self.find_corresponding_text_block(rect) is tblock), None)
        for rect in self.main.image_viewer.rectangles:
            if getattr(rect, 'region_id', None):
                continue
            mp_rect = rect.mapRectToScene(rect.rect())
            x1, y1, w, h = mp_rect.getRect()
            rect_coord = (x1, y1, x1 + w, y1 + h)
            if do_rectangles_overlap(rect_coord, tblock.xyxy, iou_threshold):
                return rect
        return None
