"""Apply a reviewed text shape to one stable page region."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets
from shiboken6 import isValid

from app.controllers.lettering_styles import ApplyLetteringStyleCommand, LetteringStylesController
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.dialogs.lettering_transform import LetteringTransformDialog
from modules.utils.region_metadata import new_id


class LetteringTransformController(QtCore.QObject):
    # Both lettering editors share the same page/busy boundaries.
    _busy = LetteringStylesController._busy
    _current_page = LetteringStylesController._current_page
    reset_for_project = LetteringStylesController.reset_for_project
    shutdown = reset_for_project

    def __init__(self, main):
        super().__init__(main)
        self.main, self.dialog = main, None
        main.lettering_transform_button.clicked.connect(self.open_editor)

    def _notice(self, message):
        QtWidgets.QMessageBox.information(self.main, self.tr('Text shape'), message)

    def open_editor(self):
        if self._busy():
            self._notice(self.tr('Wait for processing to finish before changing text shape.'))
            return
        page = self._current_page()
        if page is None or page[1] is None:
            self._notice(self.tr('Open a page and select one text box to change its shape.'))
            return
        main, viewer = self.main, self.main.image_viewer
        main.text_ctrl._commit_pending_text_command()
        selected = [item for item in viewer.get_selected_text_items()
                    if isValid(item) and item.scene() is viewer._scene]
        if len(selected) != 1:
            self._notice(self.tr('Select one text box on the current page to change its shape.'))
            return
        item = selected[0]
        owner = main.text_ctrl.fit_controller._page(item)
        if owner is None or owner[0] != page[0]:
            self._notice(self.tr('Select one text box on the current page to change its shape.'))
            return
        path, stack = page
        item.region_id = getattr(item, 'region_id', '') or new_id()
        item._page_path = path
        before = TextItemProperties.from_text_item(item).to_dict()
        epoch = getattr(main, '_translation_project_epoch', None)
        page_id = main.image_states.get(path, {}).get('page_id')
        revision, mode = stack.index(), viewer.webtoon_mode
        dialog = self.dialog = LetteringTransformDialog(before, parent=main)
        try:
            if dialog.exec() != QtWidgets.QDialog.Accepted:
                return
            valid = (not self._busy() and self._current_page() == page
                     and getattr(main, '_translation_project_epoch', None) == epoch
                     and main.image_states.get(path, {}).get('page_id') == page_id
                     and stack.index() == revision and viewer.webtoon_mode == mode
                     and isValid(item) and item.scene() is viewer._scene
                     and TextItemProperties.from_text_item(item).to_dict() == before
                     and main.text_ctrl.fit_controller._page(item) == owner)
            if not valid:
                self._notice(self.tr('The page or lettering changed. Open a new text shape preview.'))
                return
            after = dialog.result_properties
            if after is not None and after != before:
                stack.push(ApplyLetteringStyleCommand(self, path, owner[1], [before], [after],
                                                     label=self.tr('Transform lettering')))
        finally:
            self.dialog = None
            dialog.deleteLater()
