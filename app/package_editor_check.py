"""Optional frozen-editor qualification on a disposable project with a saved repair mask."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np
from PySide6 import QtCore, QtWidgets
from PySide6.QtTest import QTest


def _file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _wait(check, errors):
    deadline = time.monotonic() + 60
    while not check():
        if errors:
            raise AssertionError("Editor operation failed") from errors[0]
        if time.monotonic() >= deadline:
            raise TimeoutError("Editor operation did not settle")
        QTest.qWait(20)
    if errors:
        raise AssertionError("Editor operation failed") from errors[0]


def _open(window, path, errors):
    loaded = []
    window.project_ctrl.thread_load_project(str(path), on_success=lambda: loaded.append(True))
    _wait(lambda: loaded and window.image_viewer.hasPhoto() and not window.threadpool.activeThreadCount()
          and not window.task_runner_ctrl.is_processing_queue, errors)
    window.show_main_page()
    window.show()


def _gradient_gesture(app, mask, captured, errors):
    from app.ui.dialogs.gradient_repair import GradientRepairDialog

    dialog = app.activeModalWidget()
    try:
        if not isinstance(dialog, GradientRepairDialog):
            raise AssertionError("Gradient preview did not open")
        dialog.source_combo.setCurrentIndex(dialog.source_combo.findData("original"))
        dialog.fit_area()
        ys, xs = np.nonzero(mask)
        # Four short traces outside the painted area. This qualification expects
        # clean original pixels around the mask and enough margin for each trace.
        for y in (int(ys.min()) - 5, int(ys.max()) + 5):
            for x in (int(xs.min()) - 5, int(xs.max()) + 5):
                points = [dialog.sample_view.mapFromScene(QtCore.QPointF(px + .5, y + .5))
                          for px in (x - 3, x + 3)]
                viewport = dialog.sample_view.viewport()
                if not all(viewport.rect().contains(p) for p in points):
                    raise AssertionError("Sample is outside the fitted viewport")
                QTest.mousePress(viewport, QtCore.Qt.LeftButton, QtCore.Qt.ShiftModifier, points[0])
                QTest.mouseMove(viewport, points[1])
                QTest.mouseRelease(viewport, QtCore.Qt.LeftButton, QtCore.Qt.ShiftModifier, points[1])
        QTest.mouseClick(dialog.generate_button, QtCore.Qt.LeftButton)
        if not dialog.apply_button.isEnabled() or dialog.patch is None:
            raise AssertionError("The original pixels around the mask did not produce a gradient")
        captured.update(dialog.metadata)
        if captured.get("sample_count") != 4 or "sample_lines_json" not in captured:
            raise AssertionError("Gradient preview did not retain four sample lines")
        QTest.mouseClick(dialog.apply_button, QtCore.Qt.LeftButton)
    except Exception as exc:
        errors.append(exc)
        if isinstance(dialog, QtWidgets.QDialog):
            dialog.reject()


def exercise_editor_project(app, root: Path, source: Path):
    from app.projects.project_state_v2 import close_cached_connection
    from controller import ComicTranslate
    from modules.repair.protection import get_protection

    if not source.is_file() or source.suffix.lower() != ".ctpr":
        raise ValueError("Supply an existing editable project to qualify")
    original_digest = _file_digest(source)
    copied, saved = root / "input.ctpr", root / "saved-as.ctpr"
    shutil.copyfile(source, copied)
    window = ComicTranslate()
    errors = []
    window.default_error_handler = lambda event: errors.append(RuntimeError(str(event[1])))
    window._skip_close_prompt = True
    try:
        _open(window, copied, errors)
        page = window.image_files[window.curr_img_idx]
        repair = window.local_repair_ctrl
        before = window.image_ctrl.get_composited_page_image(page).copy()
        mask = repair._current_mask(before.shape, QtCore.QPointF()).copy()
        if not np.any(mask):
            raise AssertionError("Qualification project must have a saved painted mask on its active page")
        protected = get_protection(window, page, before.shape)
        locked = mask == 0 if protected is None else (mask == 0) | (protected != 0)
        stack = window.undo_stacks[page]
        start = stack.index()
        captured = {}
        panel = window.local_repair_panel
        panel.fill_mode.setCurrentIndex(panel.fill_mode.findData("gradient"))
        QtCore.QTimer.singleShot(0, lambda: _gradient_gesture(app, mask, captured, errors))
        panel.fill.click()
        if errors:
            raise AssertionError("Gradient gesture failed") from errors[0]
        if not captured or stack.index() != start + 1:
            raise AssertionError("Apply did not create precisely one undo entry")
        after = window.image_ctrl.get_composited_page_image(page).copy()
        changed = np.any(after != before, axis=2)
        assert np.any(changed) and not np.any(changed & locked)
        assert not np.any(repair._current_mask(before.shape, QtCore.QPointF()))
        stack.undo()
        np.testing.assert_array_equal(window.image_ctrl.get_composited_page_image(page), before)
        np.testing.assert_array_equal(repair._current_mask(before.shape, QtCore.QPointF()), mask)
        stack.redo()
        np.testing.assert_array_equal(window.image_ctrl.get_composited_page_image(page), after)
        assert not np.any(repair._current_mask(before.shape, QtCore.QPointF()))

        # Supply only the destination; exercise the ordinary asynchronous Save As.
        completed = []
        with patch.object(window.project_ctrl, "launch_save_proj_dialog", return_value=str(saved)):
            assert window.project_ctrl.thread_save_as_project(lambda: completed.append(True))
        _wait(lambda: completed and saved.is_file() and not window.threadpool.activeThreadCount(), errors)
        _open(window, saved, errors)
        reopened_page = window.image_files[window.curr_img_idx]
        np.testing.assert_array_equal(window.image_ctrl.get_composited_page_image(reopened_page), after)
        assert not np.any(repair._current_mask(before.shape, QtCore.QPointF()))
        np.testing.assert_array_equal(get_protection(window, reopened_page, before.shape), protected)
        stored = window.image_patches[reopened_page][-1]["metadata"]
        assert stored["sample_lines_json"] == captured["sample_lines_json"]
        assert _file_digest(source) == original_digest
        return {"page_opened": True, "saved_mask_pixels": int(np.count_nonzero(mask)),
                "sample_lines": captured["sample_count"], "changed_pixels": int(np.count_nonzero(changed)),
                "outside_or_protected_changes": 0, "undo_restored_mask_and_pixels": True,
                "redo_verified": True, "save_as_reopened": True, "input_unchanged": True,
                "qt_platform": app.platformName()}
    finally:
        window.close()
        app.processEvents()
        window.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        close_cached_connection()
