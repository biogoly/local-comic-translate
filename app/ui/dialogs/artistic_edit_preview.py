from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


class _PreviewView(QtWidgets.QGraphicsView):
    zoom_changed = QtCore.Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = self._scene.addPixmap(QtGui.QPixmap())
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setBackgroundBrush(QtGui.QColor(28, 28, 28))

    @staticmethod
    def _pixmap(image: np.ndarray) -> QtGui.QPixmap:
        rgb = np.ascontiguousarray(image, dtype=np.uint8)
        height, width = rgb.shape[:2]
        qimage = QtGui.QImage(
            rgb.data, width, height, width * 3, QtGui.QImage.Format.Format_RGB888
        ).copy()
        return QtGui.QPixmap.fromImage(qimage)

    def set_image(self, image: np.ndarray, *, preserve_view: bool = False) -> None:
        center = self.mapToScene(self.viewport().rect().center())
        self._item.setPixmap(self._pixmap(image))
        self._scene.setSceneRect(self._item.boundingRect())
        if preserve_view:
            self.centerOn(center)
        else:
            self.fit_image()

    def fit_image(self) -> None:
        if not self._item.pixmap().isNull():
            self.fitInView(self._item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.zoom_changed.emit(self.transform().m11())

    def set_zoom(self, scale: float) -> None:
        current = self.transform().m11()
        if current > 0 and scale > 0:
            self.scale(scale / current, scale / current)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else (1.0 / 1.2)
        target = max(0.02, min(20.0, self.transform().m11() * factor))
        self.set_zoom(target)
        self.zoom_changed.emit(target)


class ArtisticEditPreviewDialog(QtWidgets.QDialog):
    lettering_check_requested = QtCore.Signal()
    candidate_selected = QtCore.Signal(int)
    REGENERATE_SAME_SEED = 2
    REGENERATE_NEW_SEED = 3
    GENERATE_ALTERNATIVE = 4

    def __init__(
        self,
        before: np.ndarray,
        after: np.ndarray,
        summary: str,
        *,
        apply_available: bool,
        stale_reason: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Artistic Edit Preview"))
        self.resize(1180, 760)
        layout = QtWidgets.QVBoxLayout(self)

        self._before_image = before
        self._comparison_image = None
        self._comparison_label = ""
        self._candidate_labels = []
        self._selected_candidate_index = 0
        self.candidate_row = QtWidgets.QWidget()
        candidate_layout = QtWidgets.QHBoxLayout(self.candidate_row)
        candidate_layout.setContentsMargins(0, 0, 0, 0)
        candidate_label = QtWidgets.QLabel(self.tr("Candidate"))
        self.candidate_combo = QtWidgets.QComboBox()
        self.candidate_combo.setAccessibleName(self.tr("Candidate"))
        candidate_label.setBuddy(self.candidate_combo)
        comparison_label = QtWidgets.QLabel(self.tr("Compare with"))
        self.comparison_combo = QtWidgets.QComboBox()
        self.comparison_combo.setAccessibleName(self.tr("Compare with"))
        self.comparison_combo.addItem(self.tr("Before"), "before")
        comparison_label.setBuddy(self.comparison_combo)
        candidate_layout.addWidget(candidate_label)
        candidate_layout.addWidget(self.candidate_combo, 1)
        candidate_layout.addWidget(comparison_label)
        candidate_layout.addWidget(self.comparison_combo, 1)
        self.candidate_row.hide()
        layout.addWidget(self.candidate_row)

        panes = QtWidgets.QHBoxLayout()
        self.before_view = _PreviewView()
        self.after_view = _PreviewView()
        self.before_title = QtWidgets.QLabel(self.tr("Before"))
        self.after_title = QtWidgets.QLabel(self.tr("After"))
        for label, view, image in (
            (self.before_title, self.before_view, before),
            (self.after_title, self.after_view, after),
        ):
            column = QtWidgets.QVBoxLayout()
            label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            column.addWidget(label)
            column.addWidget(view, 1)
            panes.addLayout(column, 1)
            view.set_image(image)
        layout.addLayout(panes, 1)

        self.before_view.zoom_changed.connect(self.after_view.set_zoom)
        self.after_view.zoom_changed.connect(self.before_view.set_zoom)
        self._sync_scrollbars(self.before_view, self.after_view)
        self._sync_scrollbars(self.after_view, self.before_view)

        self.summary_label = QtWidgets.QLabel(summary)
        self.summary_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary_label)

        self.stale_label = QtWidgets.QLabel(stale_reason)
        self.stale_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.stale_label.setWordWrap(True)
        self.stale_label.setStyleSheet("color: #d97b7b;")
        self.stale_label.setVisible(bool(stale_reason))
        layout.addWidget(self.stale_label)

        self._lettering_available = False
        self._lettering_busy = False
        self._lettering_expected = ""
        lettering_row = QtWidgets.QHBoxLayout()
        self.lettering_check_button = QtWidgets.QPushButton(self.tr("Check lettering"))
        self.lettering_check_button.setAutoDefault(False)
        self.lettering_route_label = QtWidgets.QLabel()
        self.lettering_route_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.lettering_route_label.setWordWrap(True)
        self.lettering_details_toggle = QtWidgets.QToolButton()
        self.lettering_details_toggle.setText(self.tr("Lettering details"))
        self.lettering_details_toggle.setCheckable(True)
        self.lettering_details_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.lettering_details_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        lettering_row.addWidget(self.lettering_check_button)
        lettering_row.addWidget(self.lettering_route_label, 1)
        lettering_row.addWidget(self.lettering_details_toggle)
        layout.addLayout(lettering_row)

        self.lettering_status_label = QtWidgets.QLabel()
        self.lettering_status_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.lettering_status_label.setWordWrap(True)
        self.lettering_status_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.lettering_status_label)
        self.lettering_advisory_label = QtWidgets.QLabel(self.tr(
            "Stylized lettering can confuse OCR. Review the image before applying."
        ))
        self.lettering_advisory_label.setWordWrap(True)
        layout.addWidget(self.lettering_advisory_label)
        self.lettering_details = QtWidgets.QPlainTextEdit()
        self.lettering_details.setReadOnly(True)
        self.lettering_details.setAccessibleName(self.tr("Lettering check details"))
        self.lettering_details.setMaximumHeight(160)
        self.lettering_details.setMinimumHeight(90)
        self.lettering_details.hide()
        layout.addWidget(self.lettering_details)
        self.lettering_details_toggle.toggled.connect(self._toggle_lettering_details)
        self.lettering_check_button.clicked.connect(self.lettering_check_requested.emit)
        self.configure_lettering_check(None, "")

        buttons = QtWidgets.QHBoxLayout()
        fit_button = QtWidgets.QPushButton(self.tr("Fit"))
        self.alternative_button = QtWidgets.QPushButton(self.tr("Generate alternative"))
        self.alternative_button.setAutoDefault(False)
        same_button = QtWidgets.QPushButton(self.tr("Regenerate Same Seed"))
        new_button = QtWidgets.QPushButton(self.tr("Regenerate New Seed"))
        discard_button = QtWidgets.QPushButton(self.tr("Discard"))
        self.apply_button = QtWidgets.QPushButton(self.tr("Apply"))
        self.apply_button.setDefault(True)
        self.apply_button.setEnabled(apply_available)
        buttons.addWidget(fit_button)
        buttons.addStretch()
        buttons.addWidget(self.alternative_button)
        buttons.addWidget(same_button)
        buttons.addWidget(new_button)
        buttons.addWidget(discard_button)
        buttons.addWidget(self.apply_button)
        layout.addLayout(buttons)

        fit_button.clicked.connect(self._fit_both)
        self.alternative_button.clicked.connect(lambda: self.done(self.GENERATE_ALTERNATIVE))
        same_button.clicked.connect(lambda: self.done(self.REGENERATE_SAME_SEED))
        new_button.clicked.connect(lambda: self.done(self.REGENERATE_NEW_SEED))
        discard_button.clicked.connect(self.reject)
        self.apply_button.clicked.connect(self.accept)
        self.candidate_combo.currentIndexChanged.connect(self._select_candidate)
        self.comparison_combo.currentIndexChanged.connect(self._show_comparison)
        self._update_alternative_button()

    @property
    def selected_candidate_index(self) -> int:
        return self._selected_candidate_index

    def configure_candidates(self, labels, selected_index: int) -> None:
        """Populate at most two candidates without emitting a user selection."""
        labels = list(labels)
        if not 1 <= len(labels) <= 2:
            raise ValueError("The preview supports one or two candidates.")
        if not 0 <= selected_index < len(labels):
            raise ValueError("Selected candidate index is out of range.")
        self._candidate_labels = labels
        self._selected_candidate_index = selected_index
        with QtCore.QSignalBlocker(self.candidate_combo):
            self.candidate_combo.clear()
            self.candidate_combo.addItems(labels)
            self.candidate_combo.setCurrentIndex(selected_index)
        self.candidate_row.setVisible(len(labels) == 2)
        self.after_title.setText(labels[selected_index] if len(labels) == 2 else self.tr("After"))
        if len(labels) < 2:
            self.set_comparison_candidate(None)
        self._update_alternative_button()

    def _select_candidate(self, index: int) -> None:
        if not 0 <= index < len(self._candidate_labels):
            return
        self._selected_candidate_index = index
        self.after_title.setText(self._candidate_labels[index])
        self.candidate_selected.emit(index)

    def set_candidate(
        self, before: np.ndarray, after: np.ndarray, summary: str, *,
        apply_available: bool, stale_reason: str = "",
    ) -> None:
        """Refresh the selected candidate; the controller owns its OCR state."""
        self._before_image = before
        self._show_comparison()
        self.after_view.set_image(after, preserve_view=True)
        self.summary_label.setText(summary)
        self.apply_button.setEnabled(apply_available)
        self.stale_label.setText(stale_reason)
        self.stale_label.setVisible(bool(stale_reason))
        self._update_alternative_button()

    def set_comparison_candidate(self, image: np.ndarray | None, label: str = "") -> None:
        """Supply the other candidate's preview for optional side-by-side A/B."""
        compare_candidate = self.comparison_combo.currentData() == "candidate"
        self._comparison_image = image
        self._comparison_label = label
        with QtCore.QSignalBlocker(self.comparison_combo):
            self.comparison_combo.clear()
            self.comparison_combo.addItem(self.tr("Before"), "before")
            if image is not None and len(self._candidate_labels) == 2:
                self.comparison_combo.addItem(label, "candidate")
                self.comparison_combo.setCurrentIndex(1 if compare_candidate else 0)
        self.comparison_combo.setEnabled(self.comparison_combo.count() > 1)
        self._show_comparison()

    def _show_comparison(self) -> None:
        compare_candidate = (self.comparison_combo.currentData() == "candidate"
                             and self._comparison_image is not None)
        image = self._comparison_image if compare_candidate else self._before_image
        title = self._comparison_label if compare_candidate else self.tr("Before")
        self.before_title.setText(title)
        self.before_view.set_image(image, preserve_view=True)

    def _update_alternative_button(self) -> None:
        if len(self._candidate_labels) >= 2:
            enabled = False
            tooltip = self.tr("Two candidates are already available.")
        elif not self.apply_button.isEnabled() or self.stale_label.text():
            enabled = False
            tooltip = self.tr("The preview is stale. Generate a fresh preview to compare alternatives.")
        else:
            enabled = True
            tooltip = self.tr("Generate a second candidate to compare.")
        self.alternative_button.setEnabled(enabled)
        self.alternative_button.setToolTip(tooltip)

    @staticmethod
    def _sync_scrollbars(source: _PreviewView, target: _PreviewView) -> None:
        source.horizontalScrollBar().valueChanged.connect(target.horizontalScrollBar().setValue)
        source.verticalScrollBar().valueChanged.connect(target.verticalScrollBar().setValue)

    def _fit_both(self) -> None:
        self.before_view.fit_image()
        self.after_view.fit_image()

    def block_apply(self, reason: str) -> None:
        self.apply_button.setEnabled(False)
        self.stale_label.setText(reason)
        self.stale_label.show()
        self._update_alternative_button()

    def configure_lettering_check(
        self, expected_text: str | None, route_label: str, unavailable_reason: str = ""
    ) -> None:
        """Describe the captured check without treating it as an Apply gate."""
        self._lettering_expected = expected_text or ""
        self._lettering_available = bool(self._lettering_expected.strip()) and not unavailable_reason
        self._lettering_busy = False
        self.lettering_check_button.setEnabled(self._lettering_available)
        self.lettering_route_label.setText(route_label)
        self.lettering_details_toggle.setChecked(False)
        self.lettering_details_toggle.setEnabled(bool(self._lettering_expected.strip()))
        self.lettering_advisory_label.setVisible(bool(self._lettering_expected.strip()))
        self.lettering_status_label.setText(unavailable_reason or (
            self.tr("Not checked yet.") if self._lettering_available else self.tr(
                "Use selected translation with Artistic lettering to supply exact wording. "
                "Editing that prompt clears the lettering check."
            )
        ))
        self.lettering_details.setPlainText(self._lettering_report(self._lettering_expected, ""))

    def _toggle_lettering_details(self, expanded: bool) -> None:
        self.lettering_details.setVisible(expanded)
        self.lettering_details_toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow
        )

    def _lettering_report(self, expected: str, recognized: str, differences=()) -> str:
        report = [self.tr("Requested wording:"), expected, "", self.tr("OCR reading:"),
                  recognized or self.tr("(no text)")]
        if differences:
            report.extend(("", self.tr("Differences:")))
            for difference in differences:
                if difference.kind == 'missing':
                    line = self.tr("Not read by OCR: {expected}").format(expected=difference.expected)
                elif difference.kind == 'extra':
                    line = self.tr("Extra OCR text: {recognized}").format(recognized=difference.recognized)
                else:
                    line = self.tr("Requested: {expected} | OCR: {recognized}").format(
                        expected=difference.expected, recognized=difference.recognized)
                report.append(line)
        return "\n".join(report)

    def set_lettering_check_busy(self, busy: bool) -> None:
        self._lettering_busy = bool(busy)
        self.lettering_check_button.setEnabled(self._lettering_available and not self._lettering_busy)
        if busy:
            self.lettering_status_label.setText(self.tr("Checking lettering…"))
            self.lettering_details.setPlainText(self._lettering_report(self._lettering_expected, ""))

    def show_lettering_check_result(self, result) -> None:
        self.set_lettering_check_busy(False)
        if result.status == 'match':
            status = self.tr("OCR text matches the requested wording.")
        elif result.status == 'mismatch':
            status = self.tr("Possible lettering differences")
        else:
            status = self.tr("OCR could not read enough text to compare.")
        self.lettering_status_label.setText(status)
        self.lettering_details.setPlainText(self._lettering_report(
            result.expected, result.recognized, result.differences))
        self.lettering_details_toggle.setChecked(True)

    def show_lettering_check_error(self, message: str) -> None:
        self.set_lettering_check_busy(False)
        self.lettering_status_label.setText(self.tr("Lettering check unavailable: {error}").format(error=message))
        self.lettering_details.setPlainText(self._lettering_report(self._lettering_expected, ""))
