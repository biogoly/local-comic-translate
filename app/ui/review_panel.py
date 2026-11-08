"""Compact review controls; page navigation and persistence belong to the controller."""
from __future__ import annotations

from collections.abc import Mapping

from PySide6 import QtCore, QtWidgets

from app.ui.dayu_widgets.check_box import MCheckBox
from app.ui.dayu_widgets.push_button import MPushButton


class _ProtectionLabel(QtWidgets.QLabel):
    clicked = QtCore.Signal()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._pressed = True
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton and getattr(self, '_pressed', False):
            self._pressed = False
            if self.rect().contains(event.position().toPoint()):
                self.clicked.emit()
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class ReviewPanel(QtWidgets.QWidget):
    issue_selected = QtCore.Signal(object)
    navigate_requested = QtCore.Signal(int)
    refresh_requested = QtCore.Signal()
    filter_changed = QtCore.Signal()
    reviewed_requested = QtCore.Signal(str)
    ignored_requested = QtCore.Signal()
    protection_changed = QtCore.Signal(str, bool)
    fit_requested = QtCore.Signal()
    retry_ocr_requested = QtCore.Signal()
    report_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._busy = False
        self._region_enabled = False
        self._responsive_rows = []
        self._scroll_viewport = None
        self._layout_timer = QtCore.QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.timeout.connect(self._sync_layout_width)
        self._stage_available = {key: True for key in ('source', 'translation', 'layout', 'cleanup')}
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetNoConstraint)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.header_button = QtWidgets.QToolButton()
        self.header_button.setText(self.tr('Review'))
        self.header_button.setCheckable(True)
        self.header_button.setChecked(False)
        self.header_button.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.header_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header_button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.header_button.toggled.connect(self.set_expanded)
        layout.addWidget(self.header_button)
        layout.setAlignment(self.header_button, QtCore.Qt.AlignmentFlag.AlignLeft)

        self.body = QtWidgets.QWidget()
        self.body.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        body_layout = QtWidgets.QVBoxLayout(self.body)
        body_layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetNoConstraint)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(5)
        self.counts_label = QtWidgets.QLabel()
        self.counts_label.setWordWrap(True)
        body_layout.addWidget(self.counts_label)

        self.scope_combo = self._combo()
        self.scope_combo.addItem(self.tr('All pages'), 'all')
        self.scope_combo.addItem(self.tr('Current page'), 'current')
        self.scope_combo.setAccessibleName(self.tr('Review scope'))
        self.filter_combo = self._combo()
        self.filter_combo.addItem(self.tr('All issues'), 'all')
        self.filter_combo.addItem(self.tr('Missing OCR'), 'missing_ocr')
        self.filter_combo.addItem(self.tr('Uncertain OCR'), 'uncertain_ocr')
        self.filter_combo.addItem(self.tr('Missing translation'), 'missing_translation')
        self.filter_combo.addItem(self.tr('Overflow'), 'overflow')
        self.filter_combo.addItem(self.tr('Small text'), 'small_text')
        self.filter_combo.addItem(self.tr('Failed operation'), 'failed_operation')
        self.filter_combo.addItem(self.tr('Awaiting review'), 'awaiting_review')
        self.filter_combo.setAccessibleName(self.tr('Issue filter'))
        filters = QtWidgets.QHBoxLayout()
        filters.addWidget(self.scope_combo, 1)
        filters.addWidget(self.filter_combo, 1)
        self._responsive_rows.append(filters)
        body_layout.addLayout(filters)
        self.scope_combo.currentIndexChanged.connect(lambda _index: self.filter_changed.emit())
        self.filter_combo.currentIndexChanged.connect(lambda _index: self.filter_changed.emit())
        self.scope_combo.currentIndexChanged.connect(self._queue_layout)
        self.filter_combo.currentIndexChanged.connect(self._queue_layout)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.status_label.hide()
        body_layout.addWidget(self.status_label)

        self.issue_list = QtWidgets.QListWidget()
        self.issue_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.issue_list.setMinimumHeight(96)
        self.issue_list.setMaximumHeight(160)
        self.issue_list.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self.issue_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.issue_list.setAccessibleName(self.tr('Review issues'))
        self.issue_list.currentItemChanged.connect(self._on_current_item_changed)
        body_layout.addWidget(self.issue_list)
        self.empty_label = QtWidgets.QLabel(self.tr('No issues in this view.'))
        self.empty_label.setWordWrap(True)
        body_layout.addWidget(self.empty_label)

        navigation = QtWidgets.QHBoxLayout()
        self.previous_button = self._button(self.tr('Previous'))
        self.next_button = self._button(self.tr('Next'))
        self.refresh_button = self._button(self.tr('Refresh'))
        self.previous_button.clicked.connect(lambda: self.navigate_requested.emit(-1))
        self.next_button.clicked.connect(lambda: self.navigate_requested.emit(1))
        self.refresh_button.clicked.connect(self.refresh_requested.emit)
        for button in (self.previous_button, self.next_button, self.refresh_button):
            navigation.addWidget(button)
        self._responsive_rows.append(navigation)
        body_layout.addLayout(navigation)

        self.stage_combo = self._combo()
        self.stage_combo.addItem(self.tr('Source'), 'source')
        self.stage_combo.addItem(self.tr('Translation'), 'translation')
        self.stage_combo.addItem(self.tr('Layout'), 'layout')
        self.stage_combo.addItem(self.tr('Cleanup'), 'cleanup')
        self.stage_combo.setAccessibleName(self.tr('Review stage'))
        self.stage_combo.currentIndexChanged.connect(self._update_action_state)
        body_layout.addWidget(self.stage_combo)
        decisions = QtWidgets.QHBoxLayout()
        self.review_button = self._button(self.tr('Mark reviewed'))
        self.reviewed_button = self.review_button
        self.ignore_button = self._button(self.tr('Ignore issue'))
        self.review_button.clicked.connect(lambda: self.reviewed_requested.emit(self.selected_stage()))
        self.ignore_button.clicked.connect(self.ignored_requested.emit)
        decisions.addWidget(self.review_button)
        decisions.addWidget(self.ignore_button)
        self._responsive_rows.append(decisions)
        body_layout.addLayout(decisions)

        self.protection_checks = {}
        self.protection_labels = {}
        for key, label in (
            ('keep_source', self.tr('Keep source')),
            ('keep_translation', self.tr('Keep translation')),
            ('keep_layout', self.tr('Keep layout')),
            ('skip_auto_cleanup', self.tr('Skip automatic cleanup')),
        ):
            checkbox = MCheckBox()
            checkbox.setAccessibleName(label)
            checkbox.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
            checkbox.toggled.connect(lambda checked, key=key: self.protection_changed.emit(key, checked))
            text_label = _ProtectionLabel(label)
            text_label.setWordWrap(True)
            text_label.setBuddy(checkbox)
            text_label.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            text_label.clicked.connect(checkbox.setFocus)
            text_label.clicked.connect(checkbox.click)
            protection_row = QtWidgets.QHBoxLayout()
            protection_row.setSpacing(6)
            protection_row.addWidget(checkbox)
            protection_row.addWidget(text_label, 1)
            self.protection_checks[key] = checkbox
            self.protection_labels[key] = text_label
            body_layout.addLayout(protection_row)
        self.protection_hint = QtWidgets.QLabel(self.tr(
            'Page-wide operations keep protected stages. Direct edits remain allowed.'
        ))
        self.protection_hint.setWordWrap(True)
        body_layout.addWidget(self.protection_hint)
        actions = QtWidgets.QHBoxLayout()
        self.ocr_score_label = QtWidgets.QLabel()
        self.ocr_score_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.ocr_score_label.setWordWrap(True)
        self.ocr_score_label.hide()
        body_layout.addWidget(self.ocr_score_label)
        self.retry_ocr_button = self._button(self.tr('Retry OCR…'))
        self.retry_ocr_button.clicked.connect(self.retry_ocr_requested.emit)
        actions.addWidget(self.retry_ocr_button)
        self.fit_button = self._button(self.tr('Fit Text'))
        self.report_button = self._button(self.tr('Report'))
        self.fit_button.clicked.connect(self.fit_requested.emit)
        self.report_button.clicked.connect(self.report_requested.emit)
        actions.addWidget(self.fit_button)
        actions.addWidget(self.report_button)
        self._responsive_rows.append(actions)
        body_layout.addLayout(actions)
        layout.addWidget(self.body)
        layout.setAlignment(self.body, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.set_counts(0, 0, 0)
        self.set_issues([])
        self.set_expanded(False)

    @staticmethod
    def _combo():
        combo = QtWidgets.QComboBox()
        combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(5)
        combo.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        return combo

    @staticmethod
    def _button(label):
        button = MPushButton(label).small()
        button.setAccessibleName(label)
        button.setToolTip(label)
        button._review_full_label = label
        return button

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._queue_layout()

    def showEvent(self, event):
        super().showEvent(event)
        self._queue_layout()

    def event(self, event):
        result = super().event(event)
        if event.type() in (QtCore.QEvent.Type.ParentChange, QtCore.QEvent.Type.StyleChange,
                            QtCore.QEvent.Type.FontChange) and hasattr(self, '_layout_timer'):
            self._queue_layout()
        return result

    def eventFilter(self, watched, event):
        if watched is self._scroll_viewport and event.type() in (
            QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Show,
        ):
            self._queue_layout()
        return super().eventFilter(watched, event)

    def _queue_layout(self, *_args):
        self._layout_timer.start(0)

    def _sync_layout_width(self):
        ancestor = self.parentWidget()
        while ancestor is not None and not isinstance(ancestor, QtWidgets.QScrollArea):
            ancestor = ancestor.parentWidget()
        viewport = ancestor.viewport() if ancestor is not None else None
        if viewport is not self._scroll_viewport:
            if self._scroll_viewport is not None:
                self._scroll_viewport.removeEventFilter(self)
            self._scroll_viewport = viewport
            if viewport is not None:
                viewport.installEventFilter(self)

        # Sibling tool rows can make the scroll content wider than its viewport.
        # Limit this panel's content to the visible width, not that larger parent.
        available = self.contentsRect().width()
        if viewport is not None:
            left = max(0, self.mapTo(viewport, QtCore.QPoint()).x())
            parent_layout = self.parentWidget().layout() if self.parentWidget() is not None else None
            margin = parent_layout.contentsMargins().right() if parent_layout is not None else 0
            available = min(available, viewport.contentsRect().width() - left - margin)
        available = max(1, available)
        self.header_button.setFixedWidth(available)
        self.body.setFixedWidth(available)

        # Long translated labels stay readable when the sidebar is narrowed.
        for row in self._responsive_rows:
            widgets = [row.itemAt(i).widget() for i in range(row.count())]
            for widget in widgets:
                if isinstance(widget, MPushButton):
                    widget.setText(widget._review_full_label)
            needed = sum(max(widget.minimumSizeHint().width(),
                             widget.fontMetrics().horizontalAdvance(widget.currentText()) + 40)
                         if isinstance(widget, QtWidgets.QComboBox) else widget.minimumSizeHint().width()
                         for widget in widgets)
            needed += row.spacing() * max(0, row.count() - 1)
            row.setDirection(QtWidgets.QBoxLayout.Direction.LeftToRight if available >= needed
                             else QtWidgets.QBoxLayout.Direction.TopToBottom)
            if needed > available:
                for widget in widgets:
                    if isinstance(widget, MPushButton) and widget.minimumSizeHint().width() > available:
                        label = widget._review_full_label
                        metrics = widget.fontMetrics()
                        padding = widget.minimumSizeHint().width() - metrics.horizontalAdvance(label)
                        widget.setText(metrics.elidedText(label, QtCore.Qt.TextElideMode.ElideRight,
                                                         max(1, available - padding)))

    def set_expanded(self, expanded: bool) -> None:
        with QtCore.QSignalBlocker(self.header_button):
            self.header_button.setChecked(bool(expanded))
        self.header_button.setArrowType(QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow)
        self.body.setVisible(bool(expanded))
        self._queue_layout()

    def set_issues(self, rows: list[dict]) -> None:
        """Replace rows silently, preserving a surviving key without navigating."""
        selected = self.selected_key()
        with QtCore.QSignalBlocker(self.issue_list):
            self.issue_list.clear()
            for row in rows:
                item = QtWidgets.QListWidgetItem(str(row.get('label', '')))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, row['key'])
                item.setToolTip(str(row.get('tooltip', row.get('label', ''))))
                self.issue_list.addItem(item)
                if selected is not None and row['key'] == selected:
                    self.issue_list.setCurrentItem(item)
        self.empty_label.setVisible(not rows)
        self.issue_list.setVisible(bool(rows))
        self._update_action_state()

    def set_counts(self, pages: int, regions: int, issues: int) -> None:
        self.counts_label.setText(self.tr('Pages: {pages} · Regions: {regions} · Issues: {issues}').format(
            pages=pages, regions=regions, issues=issues,
        ))

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)
        self.status_label.setVisible(bool(text))

    def set_region_state(self, protections: Mapping | None, enabled: bool) -> None:
        self._region_enabled = bool(enabled)
        protections = protections or {}
        for key, checkbox in self.protection_checks.items():
            with QtCore.QSignalBlocker(checkbox):
                checkbox.setChecked(bool(protections.get(key, False)))
        self._update_action_state()

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self._update_action_state()

    def selected_key(self):
        item = self.issue_list.currentItem()
        return item.data(QtCore.Qt.ItemDataRole.UserRole) if item is not None else None

    def select_key(self, key) -> bool:
        if key is None:
            self.issue_list.setCurrentRow(-1)
            self._update_action_state()
            return True
        for index in range(self.issue_list.count()):
            item = self.issue_list.item(index)
            if item.data(QtCore.Qt.ItemDataRole.UserRole) == key:
                self.issue_list.setCurrentItem(item)
                self.issue_list.scrollToItem(item)
                return True
        return False

    def selected_scope(self) -> str:
        return self.scope_combo.currentData() or 'all'

    def selected_filter(self) -> str:
        return self.filter_combo.currentData() or 'all'

    def selected_stage(self) -> str:
        return self.stage_combo.currentData() or ''

    def set_stage_available(self, stage: str, available: bool) -> None:
        index = self.stage_combo.findData(stage)
        if index < 0:
            raise ValueError(f'Unknown review stage: {stage}')
        self._stage_available[stage] = bool(available)
        self.stage_combo.model().item(index).setEnabled(bool(available))
        if not self._stage_available.get(self.selected_stage(), False):
            self.stage_combo.setCurrentIndex(next(
                (i for i in range(self.stage_combo.count()) if self._stage_available[self.stage_combo.itemData(i)]), -1
            ))
        self._update_action_state()

    def _on_current_item_changed(self, item, _previous) -> None:
        self._update_action_state()
        if item is not None:
            self.issue_selected.emit(item.data(QtCore.Qt.ItemDataRole.UserRole))

    def _update_action_state(self, *_args) -> None:
        ready = not self._busy
        region_ready = ready and self._region_enabled
        has_issues = self.issue_list.count() > 0
        for widget in (self.scope_combo, self.filter_combo, self.issue_list, self.refresh_button, self.report_button):
            widget.setEnabled(ready)
        self.previous_button.setEnabled(ready and has_issues)
        self.next_button.setEnabled(ready and has_issues)
        self.ignore_button.setEnabled(ready and self.selected_key() is not None)
        self.stage_combo.setEnabled(region_ready)
        self.review_button.setEnabled(region_ready and self._stage_available.get(self.selected_stage(), False))
        self.fit_button.setEnabled(region_ready)
        self.retry_ocr_button.setEnabled(region_ready)
        for key, checkbox in self.protection_checks.items():
            checkbox.setEnabled(region_ready)
            self.protection_labels[key].setEnabled(region_ready)

    def score_text(self, evidence):
        if not evidence or evidence['score'] is None:
            return ''
        labels = {
            'lowest_line': self.tr('Lowest line score {score} ({model})'),
            'lowest_word': self.tr('Lowest word score {score} ({model})'),
            'lowest_sequence_product': self.tr('Lowest sequence score {score} ({model})'),
        }
        label = labels.get(evidence['scope'])
        score = self.locale().toString(evidence['score'], 'g', 4)
        return label.format(score=score, model=evidence['model']) if label else ''

    def set_ocr_evidence(self, evidence):
        text = self.score_text(evidence)
        self.ocr_score_label.setText(text)
        self.ocr_score_label.setVisible(bool(text))
