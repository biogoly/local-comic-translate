"""Project lettering presets and one-step application to selected page items."""
from __future__ import annotations

import copy

from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import isValid

from app.controllers.lettering_style_format import apply_style_properties
from app.controllers.text_fit import page_to_scene_position
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.dialogs.lettering_styles import LetteringStylesDialog
from modules.utils.lettering_styles import normalize_style_library
from modules.utils.region_metadata import new_id


class ApplyLetteringStyleCommand(QtGui.QUndoCommand):
    """Freeze all selected item snapshots; page reloads resolve by stable ID."""
    def __init__(self, controller, path, origin, before, after, label=None):
        super().__init__(label or controller.tr('Apply lettering style'))
        self.controller, self.path = controller, path
        self.epoch = getattr(controller.main, '_translation_project_epoch', None)
        self.page_id = controller.main.image_states.get(path, {}).get('page_id')
        self.before, self.after = copy.deepcopy(before), copy.deepcopy(after)
        for entries in (self.before, self.after):
            for props in entries:
                props['position'] = (props['position'][0] - origin.x(), props['position'][1] - origin.y())

    def _apply(self, entries):
        main = self.controller.main
        state = main.image_states.get(self.path)
        if (state is None or self.path not in main.image_files
                or state.get('page_id') != self.page_id
                or getattr(main, '_translation_project_epoch', None) != self.epoch):
            return
        viewer, tc = main.image_viewer, main.text_ctrl
        saved = state.setdefault('viewer_state', {}).setdefault('text_items_state', [])
        was_suspended = tc._suspend_text_command
        tc._suspend_text_command = True
        try:
            for props in entries:
                identity = props['region_id']
                for index, existing in enumerate(saved):
                    if existing.get('region_id') == identity:
                        saved[index] = copy.deepcopy(props)
                        break
                else:
                    saved.append(copy.deepcopy(props))
                matches = [item for item in viewer.text_items if isValid(item)
                           and item.scene() is viewer._scene
                           and getattr(item, 'region_id', '') == identity
                           and getattr(item, '_page_path', None) == self.path]
                if len(matches) != 1:
                    continue
                item = matches[0]
                live = copy.deepcopy(props)
                if viewer.webtoon_mode:
                    origin = page_to_scene_position(viewer, main.image_files.index(self.path), QtCore.QPointF())
                    live['position'] = (props['position'][0] + origin.x(), props['position'][1] + origin.y())
                apply_style_properties(item, live)
                tc._last_item_text[item] = item.toPlainText()
                tc._last_item_html[item] = item.toHtml()
                if getattr(main, 'curr_tblock_item', None) is item:
                    tc.set_values_for_blk_item(item)
        finally:
            tc._suspend_text_command = was_suspended
        main.mark_project_dirty()
        fit = getattr(tc, 'fit_controller', None)
        if fit is not None:
            fit.schedule()

    def redo(self):
        self._apply(self.after)

    def undo(self):
        self._apply(self.before)


class LetteringStylesController(QtCore.QObject):
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.dialog = None
        main.lettering_styles = normalize_style_library(getattr(main, 'lettering_styles', None))
        main.lettering_styles_button.clicked.connect(self.open_editor)

    def reset_for_project(self):
        if self.dialog is not None:
            self.dialog.reject()

    shutdown = reset_for_project

    def _busy(self):
        main = self.main
        return bool(getattr(main, '_batch_active', False)
                    or getattr(getattr(main, 'task_runner_ctrl', None), 'is_processing_queue', False)
                    or getattr(getattr(main, 'review_ctrl', None), '_preview_running', False)
                    or (getattr(main, 'threadpool', None) is not None and main.threadpool.activeThreadCount()))

    def _notice(self, message):
        QtWidgets.QMessageBox.information(self.main, self.tr('Lettering styles'), message)

    def _current_page(self):
        main = self.main
        if not 0 <= main.curr_img_idx < len(main.image_files):
            return None
        if not main.image_viewer.webtoon_mode:
            page_list = getattr(main, 'page_list', None)
            if page_list is not None and page_list.currentRow() != main.curr_img_idx:
                return None
        path = main.image_files[main.curr_img_idx]
        return path, main.undo_stacks.get(path)

    def open_editor(self):
        if self._busy():
            self._notice(self.tr('Wait for processing to finish before applying a lettering style.'))
            return
        page = self._current_page()
        if page is None or page[1] is None:
            self._notice(self.tr('Open a page before editing project lettering styles.'))
            return
        main, viewer = self.main, self.main.image_viewer
        main.text_ctrl._commit_pending_text_command()
        selected = [item for item in viewer.get_selected_text_items()
                    if isValid(item) and item.scene() is viewer._scene]
        pages = [main.text_ctrl.fit_controller._page(item) for item in selected]
        if any(entry is None or entry[0] != page[0] for entry in pages):
            self._notice(self.tr('Select text boxes on one current page to apply a style.'))
            return
        path, stack = page
        origin = pages[0][1] if pages else QtCore.QPointF()
        for item in selected:
            item.region_id = getattr(item, 'region_id', '') or new_id()
            item._page_path = path
        identities = [item.region_id for item in selected]
        if len(identities) != len(set(identities)):
            self._notice(self.tr('Select one copy of each text region before applying a style.'))
            return
        before = [TextItemProperties.from_text_item(item).to_dict() for item in selected]
        library = copy.deepcopy(main.lettering_styles)
        epoch = getattr(main, '_translation_project_epoch', None)
        page_id = main.image_states.get(path, {}).get('page_id')
        revision, mode = stack.index(), viewer.webtoon_mode
        dialog = self.dialog = LetteringStylesDialog(library, before, parent=main)
        try:
            if dialog.exec() != QtWidgets.QDialog.Accepted:
                return
            valid = (not self._busy() and self._current_page() == page
                     and getattr(main, '_translation_project_epoch', None) == epoch
                     and main.image_states.get(path, {}).get('page_id') == page_id
                     and stack.index() == revision and viewer.webtoon_mode == mode
                     and main.lettering_styles == library)
            if valid:
                valid = all(isValid(item) and item.scene() is viewer._scene
                            and TextItemProperties.from_text_item(item).to_dict() == props
                            and main.text_ctrl.fit_controller._page(item) == owner
                            for item, props, owner in zip(selected, before, pages))
            if not valid:
                self._notice(self.tr('The page, lettering or presets changed. Open a new style preview.'))
                return
            after = dialog.candidates or []
            if after and len(after) != len(before):
                raise ValueError('The style preview does not match the selected items.')
            # Preset library edits behave like project settings. The page undo
            # entry changes lettering only, never overwriting presets saved on
            # another page later in the session.
            updated = normalize_style_library(dialog.library_result)
            if updated != library:
                main.lettering_styles = updated
                main.mark_project_dirty()
            if after and after != before:
                stack.push(ApplyLetteringStyleCommand(self, path, origin, before, after))
        finally:
            self.dialog = None
            dialog.deleteLater()
