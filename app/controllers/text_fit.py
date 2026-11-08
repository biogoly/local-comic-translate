"""Selected text layout with an atomic, page-local undo record."""
from __future__ import annotations

import copy
import math
import os
from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import isValid

from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.dialogs.text_fit import TextFitDialog, boundary_message
from modules.utils.region_metadata import new_id


def page_to_scene_position(viewer, index, position):
    """Use the same page centering as webtoon save/reload."""
    manager = getattr(viewer, 'webtoon_manager', None)
    converter = getattr(manager, 'coordinate_converter', None)
    if converter is not None:
        return converter.page_local_to_scene_position(position, index)
    return viewer.page_to_scene_coordinates(index, position)


def apply_layout_properties(item, properties):
    """Restore actual document state, without flattening its character formats."""
    props = TextItemProperties.from_dict(properties)
    item._format_selection = None
    item.font_size = props.font_size
    item.layout_preferences = copy.deepcopy(props.layout_preferences)
    item.region_id = props.region_id
    item.setHtml(props.text)
    if props.document_font:
        font = QtGui.QFont()
        if font.fromString(props.document_font):
            item.document().setDefaultFont(font)
    item.setTextWidth(props.width)
    if item.vertical and item.layout:
        item.layout.set_max_size(props.width, props.layout_preferences.get('bound_height', props.height))
    item.selection_outlines = copy.deepcopy(props.selection_outlines)
    item.setPos(QtCore.QPointF(*props.position))
    item.setRotation(props.rotation)
    item.setScale(props.scale)
    item.set_lettering_transform(props.lettering_transform)
    if props.transform_origin is not None:
        item.setTransformOriginPoint(QtCore.QPointF(*props.transform_origin))
    item.update()


class FitTextCommand(QtGui.QUndoCommand):
    def __init__(self, controller, item, page_path, before, after, origin):
        super().__init__(QtCore.QCoreApplication.translate("TextFitController", "Fit Text"))
        self.controller = controller
        self.page_path = page_path
        self.item = item
        self.origin = origin
        self.before = copy.deepcopy(before)
        self.after = copy.deepcopy(after)

    def _apply(self, props):
        main = self.controller.main
        viewer = main.image_viewer
        region_id = props['region_id']
        # Saved view state is always page-local, including in webtoon mode.
        saved = copy.deepcopy(props)
        saved['position'] = (props['position'][0] - self.origin.x(), props['position'][1] - self.origin.y())
        page_state = main.image_states.setdefault(self.page_path, {})
        items = page_state.setdefault('viewer_state', {}).setdefault('text_items_state', [])
        for index, state in enumerate(items):
            if state.get('region_id') == region_id:
                items[index] = saved
                break
        else:
            items.append(saved)
        matches = [i for i in viewer.text_items if isValid(i) and i.scene() is viewer._scene
                   and getattr(i, 'region_id', '') == region_id
                   and getattr(i, '_page_path', None) == self.page_path]
        live = self.item if self.item in matches else matches[0] if len(matches) == 1 else None
        tc = main.text_ctrl
        was_suspended = tc._suspend_text_command
        tc._suspend_text_command = True
        try:
            if live is not None:
                live_props = copy.deepcopy(saved)
                if viewer.webtoon_mode:
                    index = main.image_files.index(self.page_path)
                    point = page_to_scene_position(viewer, index, QtCore.QPointF(*saved['position']))
                    if point is not None:
                        live_props['position'] = (point.x(), point.y())
                apply_layout_properties(live, live_props)
                plain = live.toPlainText()
                tc._last_item_text[live] = plain
                tc._last_item_html[live] = live.toHtml()
                if getattr(main, 'curr_tblock_item', None) is live:
                    with QtCore.QSignalBlocker(main.t_text_edit):
                        main.t_text_edit.setPlainText(plain)
                    tc.set_values_for_blk_item(live)
            else:
                doc = QtGui.QTextDocument()
                doc.setHtml(props['text'])
                plain = doc.toPlainText()
            blocks = list(page_state.get('blk_list', []))
            # Webtoon viewport lists may contain several clipped copies with
            # the same logical ID; only use an unambiguous selected copy.
            live_blocks = [b for b in main.blk_list if getattr(b, 'region_id', '') == region_id]
            if len(live_blocks) == 1 and live is not None:
                blocks += live_blocks
            for block in blocks:
                if getattr(block, 'region_id', '') == region_id:
                    block.translation = plain
        finally:
            tc._suspend_text_command = was_suspended
        main.mark_project_dirty()
        self.controller.refresh()

    def redo(self):
        self._apply(self.after)

    def undo(self):
        self._apply(self.before)


class TextFitController(QtCore.QObject):
    def __init__(self, text_controller):
        super().__init__(text_controller.main)
        self.main = text_controller.main
        self.text_controller = text_controller
        self._boundary_cache = {}
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self.refresh)
        button = getattr(self.main, 'fit_text_button', None)
        if button is not None:
            button.clicked.connect(self.open_preview)
        self.main.image_viewer._scene.selectionChanged.connect(self.schedule)

    def schedule(self, *_):
        self.timer.start()

    def _busy(self):
        return bool(getattr(getattr(self.main, 'task_runner_ctrl', None), 'is_processing_queue', False))

    def _original_image(self, path):
        """Read the project's original image, never its composited/history pixels."""
        import imkit as imk
        from app.path_materialization import ensure_path_materialized
        try:
            ensure_path_materialized(path)
            original = imk.read_image(path)
            if original is None:
                return None, 'missing_image'
            current = getattr(self.main, 'image_data', {}).get(path)
            if current is not None and current.shape[:2] != original.shape[:2]:
                return None, 'image_dimensions'
            original = original.copy()
            original.setflags(write=False)
            return original, ''
        except (OSError, ValueError, RuntimeError):
            return None, 'missing_image'

    def _boundary_context(self, item, page):
        path, origin, _ = page
        block = self.text_controller._find_text_block_for_item(item)
        def local_bounds(bounds):
            if bounds is None:
                return None
            return [float(bounds[0]) - origin.x(), float(bounds[1]) - origin.y(),
                    float(bounds[2]) - origin.x(), float(bounds[3]) - origin.y()]
        position = (item.pos().x() - origin.x(), item.pos().y() - origin.y())
        current = getattr(self.main, 'image_data', {}).get(path)
        page_size = (current.shape[1], current.shape[0]) if current is not None else None
        if page_size is None and not self.main.image_viewer.webtoon_mode:
            pixmap = self.main.image_viewer.photo.pixmap()
            if not pixmap.isNull():
                page_size = (pixmap.width(), pixmap.height())
        return dict(position=position, image_loader=lambda: self._original_image(path),
                    page_size=page_size,
                    text_bounds=local_bounds(block.xyxy) if block is not None else
                    [*position, position[0] + item.boundingRect().width(), position[1] + item.boundingRect().height()],
                    bubble_bounds=local_bounds(getattr(block, 'bubble_xyxy', None)))

    def _source_signature(self, item, page):
        context = self._boundary_context(item, page)
        return (page[0], self._file_stamp(page[0]), tuple(context['text_bounds']), tuple(context['bubble_bounds'] or ()),
                context['page_size'], getattr(self.main, 'current_history_index', {}).get(page[0]), self._busy())

    @staticmethod
    def _file_stamp(path):
        try:
            stat = os.stat(path)
            return stat.st_mtime_ns, stat.st_size
        except OSError:
            return None

    def _diagnostic_boundary(self, item, page):
        from modules.rendering.bubble_boundary import estimate_bubble_boundary
        prefs = item.layout_preferences
        context = self._boundary_context(item, page)
        key = (getattr(item, 'region_id', ''), self._source_signature(item, page),
               prefs.get('boundary_shape', 'auto'), tuple(prefs.get('boundary_box', ())), prefs.get('padding', 8))
        if key not in self._boundary_cache:
            original, reason = (self._original_image(page[0]) if key[2] == 'auto' else (None, ''))
            boundary = estimate_bubble_boundary(original, context['text_bounds'], context['bubble_bounds'],
                override=key[2], override_bounds=prefs.get('boundary_box'), padding=prefs.get('padding', 8),
                page_size=context['page_size'])
            if len(self._boundary_cache) >= 8:
                self._boundary_cache.clear()
            self._boundary_cache[key] = boundary, reason
        return self._boundary_cache[key]

    def _selected(self):
        selected = self.main.image_viewer.get_selected_text_items()
        selected = [item for item in selected if isValid(item) and item.scene() is self.main.image_viewer._scene]
        return selected[0] if len(selected) == 1 else None

    def _limits(self, item):
        prefs = getattr(item, 'layout_preferences', {})
        block = self.text_controller._find_text_block_for_item(item)
        width, height = item.boundingRect().width(), item.boundingRect().height()
        if block is not None and block.xyxy is not None:
            scale = max(abs(item.scale()), 1e-6)
            width, height = float(block.xywh[2]) / scale, float(block.xywh[3]) / scale
        minimum = getattr(getattr(self.main.settings_page, 'ui', None), 'min_font_spinbox', None)
        maximum = getattr(getattr(self.main.settings_page, 'ui', None), 'max_font_spinbox', None)
        values = (prefs.get('bound_width', width), prefs.get('bound_height', height),
                  prefs.get('min_font_size', minimum.value() if minimum else 10),
                  maximum.value() if maximum else max(72, item.font_size))
        return tuple(max(1.0, float(v)) if math.isfinite(float(v)) else 1.0 for v in values)

    def refresh(self):
        item = self._selected()
        transformed = item is not None and bool(getattr(item, 'lettering_transform', None))
        button = getattr(self.main, 'fit_text_button', None)
        label = getattr(self.main, 'text_fit_status', None)
        if button is not None:
            button.setEnabled(not self._busy() and not transformed and item is not None and bool(item.toPlainText().strip()) and self._page(item) is not None)
        if label is None:
            return
        label.setToolTip('')
        if item is None:
            label.clear()
            return
        if self._busy():
            label.setText(self.tr('Wait for page processing to finish before fitting text.'))
            self.timer.start(300)
            return
        if transformed:
            label.setText(self.tr('Reset text shape to fit.'))
            label.setToolTip(self.tr('Fit Text is unavailable for transformed lettering. Reset the lettering transform first.'))
            return
        if self._page(item) is None:
            label.setText(self.tr("Fit Text requires a text item contained within one page."))
            return
        width, height, minimum, _ = self._limits(item)
        # Only measure Current here; candidate generation is reserved for the dialog.
        from modules.rendering.layout import measure_layout
        kwargs = dict(font_size=item.font_size, minimum_font_size=minimum,
                      outline_width=max([o.width for o in item.selection_outlines] + [0]), line_spacing=item.line_spacing)
        if item.layout_preferences.get('mode') == 'bubble':
            if item.vertical or abs(item.rotation() % 360) > 1e-6 or abs(item.scale() - 1) > 1e-6 or not item.transform().isIdentity():
                label.setText(boundary_message('unsupported_transform'))
                return
            page = self._page(item)
            boundary, reason = self._diagnostic_boundary(item, page)
            if reason or not boundary.available:
                label.setText(boundary_message(reason or boundary.reason))
                return
            from modules.rendering.bubble_layout import measure_bubble_layout
            position = self._boundary_context(item, page)['position']
            result = measure_bubble_layout(item.document(), mask=boundary.mask,
                current_position=tuple(p - o for p, o in zip(position, boundary.origin)), **kwargs)
        else:
            result = measure_layout(item.document(), width=width, height=height, vertical=item.vertical, **kwargs)
        label.setText(self.tr("Text overflows the fitting area.") if not result.fits else
                      self.tr("Text is below the minimum readable size.") if result.below_minimum else '')

    def _page(self, item):
        viewer = self.main.image_viewer
        if viewer.webtoon_mode:
            owner = getattr(item, '_page_path', None)
            if owner in self.main.image_files:
                index = self.main.image_files.index(owner)
            else:
                try:
                    index, _ = viewer.scene_to_page_coordinates(item.pos())
                except IndexError:
                    return None
            bounds = item.sceneBoundingRect().adjusted(0.01, 0.01, -0.01, -0.01)
            corners = [bounds.topLeft(), bounds.topRight(), bounds.bottomLeft(), bounds.bottomRight()]
            try:
                contained = index is not None and all(viewer.scene_to_page_coordinates(point)[0] == index for point in corners)
            except IndexError:
                # A viewport can briefly expose positions before page extents
                # have been populated during loading. Do not fit that state.
                return None
            if not contained:
                return None
            origin = page_to_scene_position(viewer, index, QtCore.QPointF())
        else:
            index = self.main.curr_img_idx
            origin = QtCore.QPointF()
        if index is None or not 0 <= index < len(self.main.image_files):
            return None
        path = self.main.image_files[index]
        if getattr(item, '_page_path', path) != path:
            return None
        return path, origin or QtCore.QPointF(), self.main.undo_stacks.get(path)

    def open_preview(self, *_):
        item = self._selected()
        if self._busy() or item is None or not item.toPlainText().strip():
            return
        if getattr(item, 'lettering_transform', None):
            QtWidgets.QMessageBox.information(self.main, self.tr('Fit Text'),
                self.tr('Fit Text is unavailable for transformed lettering. Reset the lettering transform first.'))
            return
        self.text_controller._commit_pending_text_command()
        page = self._page(item)
        if page is None or page[2] is None:
            return
        path, origin, stack = page
        item._page_path = path
        item.region_id = getattr(item, 'region_id', '') or new_id()
        before = TextItemProperties.from_text_item(item).to_dict()
        width, height, minimum, maximum = self._limits(item)
        revision = stack.index()
        source_signature = self._source_signature(item, page)
        boundary_context = self._boundary_context(item, page)
        read_original = boundary_context['image_loader']
        def load_snapshot():
            nonlocal source_signature
            result = read_original()
            # Lazy project blobs can create their original file on first read.
            # Pin that file stamp without accepting any geometry/history change.
            if source_signature[1] is None and result[0] is not None:
                source_signature = (source_signature[0], self._file_stamp(path), *source_signature[2:])
            return result
        boundary_context['image_loader'] = load_snapshot
        dialog = TextFitDialog(item, width, height, minimum, maximum, self.main,
                               boundary_context=boundary_context)
        try:
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return
            current_page = self._page(item) if isValid(item) and item.scene() is not None else None
            if (current_page is None or current_page[0] != path or current_page[2] is not stack
                    or stack.index() != revision or item.toHtml() != before['text']
                    or TextItemProperties.from_text_item(item).to_dict() != before
                    or self._source_signature(item, current_page) != source_signature):
                QtWidgets.QMessageBox.information(self.main, self.tr("Fit Text"),
                    self.tr("The text or page changed. Open Fit Text again to make a new preview."))
                return
            candidate = dialog.selected_candidate()
            if candidate is None or not candidate.fits:
                return
            position = dialog.candidate_page_position()
            scene_position = (position[0] + origin.x(), position[1] + origin.y())
            if self.main.image_viewer.webtoon_mode:
                area = item.mapRectToScene(QtCore.QRectF(0, 0, candidate.width, candidate.height))
                area.translate(scene_position[0] - item.pos().x(), scene_position[1] - item.pos().y())
                page_index = self.main.image_files.index(path)
                if any(self.main.image_viewer.scene_to_page_coordinates(p)[0] != page_index
                       for p in (area.topLeft(), area.topRight(), area.bottomLeft(), area.bottomRight())):
                    QtWidgets.QMessageBox.information(self.main, self.tr("Fit Text"),
                        self.tr("Fit Text requires a text item contained within one page."))
                    return
            after = copy.deepcopy(before)
            after.update(text=candidate.html, font_size=candidate.font_size, width=candidate.width,
                         height=candidate.height, layout_preferences=dialog.preferences(),
                         document_font=candidate.default_font.toString(), position=scene_position)
            stack.push(FitTextCommand(self, item, path, before, after, origin))
        finally:
            dialog.deleteLater()
