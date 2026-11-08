"""Incremental review navigation, content-bound decisions and region protections."""
from __future__ import annotations

import copy
from collections import deque
import hashlib
import json
import os
import html

from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import isValid

from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.controllers.text_fit import apply_layout_properties, page_to_scene_position
from modules.review.issues import ReviewIssue, evaluate_region, fingerprint_for_stage, mark_reviewed, dismiss_issue
from modules.utils.region_metadata import new_id, normalize_region_metadata
from modules.utils.region_policy import is_protected
from modules.ocr.evidence import current_ocr_evidence


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode('utf-8')).hexdigest()


class RegionReviewCommand(QtGui.QUndoCommand):
    """Metadata follows stable page/region identity, including after navigation."""
    def __init__(self, controller, page_id, region_id, before, after):
        super().__init__(QtCore.QCoreApplication.translate('ReviewController', 'Change review state'))
        self.controller, self.page_id, self.region_id = controller, page_id, region_id
        self.before, self.after = copy.deepcopy(before), copy.deepcopy(after)

    def _apply(self, metadata):
        for block in self.controller.region_refs(self.page_id, self.region_id):
            block.region_metadata = copy.deepcopy(metadata)
        self.controller.main.mark_project_dirty()
        self.controller.refresh()

    def redo(self):
        self._apply(self.after)

    def undo(self):
        self._apply(self.before)


class ReviewedTextCommand(QtGui.QUndoCommand):
    """Applying a replacement preserves the protection and existing text geometry."""
    def __init__(self, controller, page_id, region_id, stage, before, after, old_props=None, new_props=None,
                 old_evidence=None, new_evidence=None):
        super().__init__(QtCore.QCoreApplication.translate('ReviewController', 'Apply reviewed text'))
        self.controller, self.page_id, self.region_id, self.stage = controller, page_id, region_id, stage
        self.before, self.after = before, after
        self.old_props, self.new_props = copy.deepcopy(old_props), copy.deepcopy(new_props)
        self.old_evidence, self.new_evidence = copy.deepcopy(old_evidence), copy.deepcopy(new_evidence)

    def _apply(self, value, props, evidence):
        c, main = self.controller, self.controller.main
        path = c.path_for_page(self.page_id)
        if path is None:
            return
        for block in c.region_refs(self.page_id, self.region_id):
            setattr(block, 'text' if self.stage == 'source' else 'translation', value)
            if self.stage == 'source':
                block.ocr_evidence = copy.deepcopy(evidence)
        if props is not None:
            states = main.image_states[path].setdefault('viewer_state', {}).setdefault('text_items_state', [])
            for i, state in enumerate(states):
                if state.get('region_id') == self.region_id:
                    states[i] = copy.deepcopy(props)
                    break
            else:
                states.append(copy.deepcopy(props))
            item = c.live_item(path, self.region_id)
            if item is not None:
                live_props = copy.deepcopy(props)
                origin = c.origin(path)
                live_props['position'] = (props['position'][0] + origin.x(), props['position'][1] + origin.y())
                tc = main.text_ctrl
                suspended = tc._suspend_text_command
                tc._suspend_text_command = True
                try:
                    apply_layout_properties(item, live_props)
                    tc._last_item_text[item] = item.toPlainText()
                    tc._last_item_html[item] = item.toHtml()
                finally:
                    tc._suspend_text_command = suspended
        current = getattr(main, 'curr_tblock', None)
        if current is not None and current in c.region_refs(self.page_id, self.region_id):
            editor = main.s_text_edit if self.stage == 'source' else main.t_text_edit
            with QtCore.QSignalBlocker(editor):
                editor.setPlainText(value)
        main.mark_project_dirty()
        c.refresh()

    def redo(self):
        self._apply(self.after, self.new_props, self.new_evidence)

    def undo(self):
        self._apply(self.before, self.old_props, self.old_evidence)


class ReviewController(QtCore.QObject):
    def __init__(self, main):
        super().__init__(main)
        self.main, self.panel = main, main.review_panel
        self._cache, self._snapshots, self._issues, self._originals = {}, {}, [], {}
        self._jobs = deque()
        self._pending_navigation = None
        self._preview_running = False
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.refresh)
        self.scan_timer = QtCore.QTimer(self)
        self.scan_timer.setSingleShot(True)
        self.scan_timer.timeout.connect(self._scan_slice)
        self.navigation_timer = QtCore.QTimer(self)
        self.navigation_timer.setSingleShot(True)
        self.navigation_timer.timeout.connect(self._finish_navigation)
        p = self.panel
        p.refresh_requested.connect(self.refresh)
        p.filter_changed.connect(self.refresh)
        p.header_button.toggled.connect(self.schedule)
        p.issue_selected.connect(self.open_issue)
        p.navigate_requested.connect(self.navigate)
        p.reviewed_requested.connect(self.mark_current_reviewed)
        p.ignored_requested.connect(self.ignore_current_issue)
        p.protection_changed.connect(self.set_protection)
        p.fit_requested.connect(self.fit_current)
        p.retry_ocr_requested.connect(self.retry_ocr)
        p.report_requested.connect(main.batch_report_ctrl.show_latest_batch_report)
        main.image_viewer._scene.changed.connect(self.schedule)
        main.image_viewer._scene.selectionChanged.connect(self.schedule)
        main.image_viewer.review_controller = self
        for name in ('s_text_edit', 't_text_edit'):
            getattr(main, name).textChanged.connect(self.schedule)
        for name in ('s_combo', 't_combo'):
            getattr(main, name).currentTextChanged.connect(self.schedule)
        main.page_list.currentRowChanged.connect(self.schedule)
        main.undo_group.indexChanged.connect(self.schedule)
        for sequence, step in (('Ctrl+Alt+Right', 1), ('Ctrl+Alt+Left', -1)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), main)
            shortcut.activated.connect(lambda delta=step: self.navigate(delta))
        self.schedule()

    def busy(self):
        return bool(self._preview_running or getattr(self.main, '_batch_active', False)
                    or getattr(getattr(self.main, 'task_runner_ctrl', None), 'is_processing_queue', False))

    def schedule(self, *_):
        if not self.timer.isActive():
            self.timer.start()

    def shutdown(self):
        self.timer.stop()
        self.scan_timer.stop()
        self.navigation_timer.stop()
        self._jobs.clear()

    def path_for_page(self, page_id):
        return next((p for p in self.main.image_files
                     if self.main.image_states.get(p, {}).get('page_id') == page_id), None)

    def origin(self, path):
        if not self.main.image_viewer.webtoon_mode:
            return QtCore.QPointF()
        if path not in self.main.image_files:
            return QtCore.QPointF()
        return page_to_scene_position(self.main.image_viewer, self.main.image_files.index(path), QtCore.QPointF())

    def _live_path(self, block):
        if not self.main.image_viewer.webtoon_mode:
            index = self.main.curr_img_idx
        else:
            index, _ = self.main.image_viewer.scene_to_page_coordinates(QtCore.QPointF(
                (block.xyxy[0] + block.xyxy[2]) / 2, (block.xyxy[1] + block.xyxy[3]) / 2))
        return self.main.image_files[index] if index is not None and 0 <= index < len(self.main.image_files) else None

    def region_refs(self, page_id, region_id):
        path = self.path_for_page(page_id)
        if path is None:
            return []
        refs = [b for b in self.main.image_states[path].get('blk_list', []) if b.region_id == region_id]
        for b in self.main.blk_list:
            if b.region_id == region_id and self._live_path(b) == path and all(b is not old for old in refs):
                refs.append(b)
        return refs

    def live_item(self, path, region_id):
        found = [i for i in self.main.image_viewer.text_items if isValid(i)
                 and getattr(i, 'region_id', '') == region_id
                 and getattr(i, '_page_path', path) == path
                 and i.scene() is self.main.image_viewer._scene]
        return found[0] if len(found) == 1 else None

    def _cleanup_evidence(self, path, block):
        x1, y1, x2, y2 = block.xyxy
        descriptors = []
        for patch in getattr(self.main, 'image_patches', {}).get(path, []):
            if patch.get('kind', 'inpaint') != 'inpaint':
                continue
            x, y, w, h = patch.get('bbox', (0, 0, 0, 0))
            if x < x2 and x + w > x1 and y < y2 and y + h > y1:
                descriptors.append((patch.get('patch_id'), patch.get('hash'), (x, y, w, h)))
        return {'cleanup_fingerprint': _hash(descriptors)} if descriptors else {}

    def snapshot(self):
        """Read state and live page-local text; never load images for inventory."""
        main, viewer = self.main, self.main.image_viewer
        pages = {}
        for path in main.image_files:
            state = main.image_states.setdefault(path, {})
            page_id = state.setdefault('page_id', new_id())
            pages[path] = {'page_id': page_id, 'blocks': {b.region_id: b for b in state.get('blk_list', [])},
                           'items': {p.get('region_id'): p for p in state.get('viewer_state', {}).get('text_items_state', [])}}
        loaded = (viewer.webtoon_manager.loaded_pages if viewer.webtoon_mode else
                  {main.curr_img_idx} if 0 <= main.curr_img_idx < len(main.image_files) else set())
        for index in loaded:
            if 0 <= index < len(main.image_files):
                pages[main.image_files[index]]['blocks'] = {}
        for block in main.blk_list:
            path = self._live_path(block)
            if path not in pages:
                continue
            local = block.deep_copy()
            origin = self.origin(path)
            for name in ('xyxy', 'bubble_xyxy'):
                b = getattr(local, name, None)
                if b is not None:
                    setattr(local, name, [b[0] - origin.x(), b[1] - origin.y(), b[2] - origin.x(), b[3] - origin.y()])
            pages[path]['blocks'][block.region_id] = local
        current_path = main.image_files[main.curr_img_idx] if 0 <= main.curr_img_idx < len(main.image_files) else None
        # Live pages replace their saved item inventory, including deletions.
        live_paths = ({main.image_files[i] for i in viewer.webtoon_manager.loaded_pages if i < len(main.image_files)}
                      if viewer.webtoon_mode else {current_path})
        for path in live_paths:
            if path in pages:
                pages[path]['items'] = {}
        for item in viewer.text_items:
            if not isValid(item) or item.scene() is not viewer._scene:
                continue
            path = getattr(item, '_page_path', current_path)
            if path not in pages:
                continue
            props = TextItemProperties.from_text_item(item).to_dict()
            origin = self.origin(path)
            props['position'] = (props['position'][0] - origin.x(), props['position'][1] - origin.y())
            pages[path]['items'][props.get('region_id')] = props
        result = {}
        for path, page in pages.items():
            state = main.image_states[path]
            languages = {'source': state.get('source_lang', main.s_combo.currentText()),
                         'target': state.get('target_lang', main.t_combo.currentText())}
            for region_id, block in page['blocks'].items():
                result[(page['page_id'], region_id)] = dict(path=path, page_id=page['page_id'], block=block,
                    props=page['items'].get(region_id), languages=languages, workflow=self._cleanup_evidence(path, block))
        return result

    def _fingerprint(self, snap, stage):
        return fingerprint_for_stage(stage, snap['block'], snap['props'], languages=snap['languages'], workflow=snap['workflow'])

    def _diagnostics(self, snap):
        props = snap['props']
        if not props or props.get('lettering_transform'):
            # The layout measurers inspect an untransformed document. Keep
            # warped lettering awaiting review instead of reporting false fits.
            return None
        from modules.rendering.auto_bubble_fit import _document, load_original_for_fitting
        from modules.rendering.layout import measure_layout
        from modules.rendering.bubble_boundary import estimate_bubble_boundary
        from modules.rendering.bubble_layout import measure_bubble_layout
        properties = TextItemProperties.from_dict(props)
        document, _ = _document(properties, None)
        minimum = self.main.settings_page.ui.min_font_spinbox.value()
        prefs = properties.layout_preferences
        minimum = prefs.get('min_font_size', minimum)
        kwargs = dict(font_size=properties.font_size, minimum_font_size=max(1, minimum),
            outline_width=max([float(o.get('width', 0) if isinstance(o, dict) else o.width)
                               for o in properties.selection_outlines] + [0]), line_spacing=float(properties.line_spacing))
        if prefs.get('mode') == 'bubble':
            if properties.vertical or properties.rotation % 360 or properties.scale != 1:
                return None
            original = None
            shape = prefs.get('boundary_shape', 'auto')
            if shape == 'auto':
                # Only the open page may read its original. Other pages remain
                # awaiting layout review; no contour success is invented.
                path = snap['path']
                if not (0 <= self.main.curr_img_idx < len(self.main.image_files)
                        and path == self.main.image_files[self.main.curr_img_idx]):
                    return None
                stamp = self.main.text_ctrl.fit_controller._file_stamp(path)
                key = path, stamp
                if key not in self._originals:
                    self._originals.clear()
                    self._originals[key] = load_original_for_fitting(path)
                original = self._originals[key]
            block = snap['block']
            current = getattr(self.main, 'image_data', {}).get(snap['path'])
            page_size = (current.shape[1], current.shape[0]) if current is not None else None
            boundary = estimate_bubble_boundary(original, block.xyxy, block.bubble_xyxy,
                override=shape, override_bounds=prefs.get('boundary_box'), padding=prefs.get('padding', 8), page_size=page_size)
            if not boundary.available:
                return None
            result = measure_bubble_layout(document, mask=boundary.mask,
                current_position=(properties.position[0] - boundary.origin[0], properties.position[1] - boundary.origin[1]), **kwargs)
        else:
            width = prefs.get('bound_width', snap['block'].xywh[2] / max(abs(properties.scale), 1e-6))
            height = prefs.get('bound_height', snap['block'].xywh[3] / max(abs(properties.scale), 1e-6))
            document.setTextWidth(properties.width if properties.width and properties.width > 0 else width)
            result = measure_layout(document, width=max(1, width), height=max(1, height), vertical=properties.vertical, **kwargs)
        return dict(fits=result.fits, below_minimum=result.below_minimum, overflow_reasons=result.overflow_reasons,
                    minimum_font_size=minimum)

    def refresh(self, *_):
        self.panel.set_busy(self.busy())
        if self.busy():
            self.timer.start(500)
            return
        self._update_controls()
        if not self.panel.header_button.isChecked():
            return
        self._snapshots = self.snapshot()
        minimum = self.main.settings_page.ui.min_font_spinbox.value()
        current_path = self.main.image_files[self.main.curr_img_idx] if 0 <= self.main.curr_img_idx < len(self.main.image_files) else None
        self._jobs.clear()
        self._issues = []
        for key, snap in self._snapshots.items():
            if self.panel.selected_scope() == 'current' and snap['path'] != current_path:
                continue
            signature = _hash(([self._fingerprint(snap, stage) for stage in ('source', 'translation', 'layout', 'cleanup')],
                               snap['block'].region_metadata, current_ocr_evidence(snap['block']),
                               minimum, snap['path'] == current_path))
            cached = self._cache.get(key)
            if cached and cached[0] == signature:
                self._issues.extend(cached[1])
            else:
                self._jobs.append((key, snap, signature))
        self._cache = {key: value for key, value in self._cache.items() if key in self._snapshots}
        if self._jobs:
            self.panel.set_status(self.tr('Checking text layouts…'))
            self.scan_timer.start(0)
        else:
            self._publish()

    def _scan_slice(self):
        if self.busy():
            self.schedule()
            return
        for _ in range(min(6, len(self._jobs))):
            key, snap, signature = self._jobs.popleft()
            try:
                diagnostics = self._diagnostics(snap)
            except (ValueError, TypeError, RuntimeError, AttributeError):
                diagnostics = None
            issues = evaluate_region(snap['page_id'], snap['block'], snap['props'], diagnostics=diagnostics,
                                     workflow=snap['workflow'], languages=snap['languages'])
            self._cache[key] = signature, issues
            self._issues.extend(issues)
        if self._jobs:
            self.scan_timer.start(0)
        else:
            self._publish()

    def _publish(self):
        failures = []
        report = getattr(self.main.batch_report_ctrl, '_latest_batch_report', None) or {}
        for entry in report.get('skipped_entries', []):
            path = entry.get('image_path')
            if path not in self.main.image_files or entry.get('informational', False):
                continue
            state = self.main.image_states[path]
            failures.append(ReviewIssue(state['page_id'], '', 'failed_operation', 'operation', _hash(entry.get('reasons', [])),
                                        '\n'.join(entry.get('reasons', [])), path))
        issues = self._issues + failures
        order = {self.main.image_states[p]['page_id']: i for i, p in enumerate(self.main.image_files)}
        issues.sort(key=lambda issue: (order.get(issue.page_id, -1),
                    tuple(self._snapshots[(issue.page_id, issue.region_id)]['block'].xyxy[1::-1])
                    if (issue.page_id, issue.region_id) in self._snapshots else (-1, -1), issue.kind, issue.stage))
        self._all_issues = issues
        current_path = self.main.image_files[self.main.curr_img_idx] if 0 <= self.main.curr_img_idx < len(self.main.image_files) else None
        region_numbers, page_counts = {}, {}
        for key in self._snapshots:
            page_counts[key[0]] = page_counts.get(key[0], 0) + 1
            region_numbers[key] = page_counts[key[0]]
        rows = []
        for issue in issues:
            path = self.path_for_page(issue.page_id)
            if self.panel.selected_scope() == 'current' and path != current_path:
                continue
            if self.panel.selected_filter() not in ('all', issue.kind):
                continue
            title = self.issue_title(issue)
            name = os.path.basename(path or '')
            snap = self._snapshots.get((issue.page_id, issue.region_id))
            number = region_numbers.get((issue.page_id, issue.region_id))
            label = f'{name} · {number} — {title}' if number is not None else f'{name} — {title}'
            detail = issue.detail if issue.kind == 'failed_operation' else title
            if snap:
                if issue.kind == 'uncertain_ocr':
                    detail += '\n' + self.panel.score_text(current_ocr_evidence(snap['block']))
                detail += '\n' + (snap['block'].text or snap['block'].translation or '')[:180]
            rows.append(dict(key=issue.key, label=label, tooltip='<qt>' + html.escape(detail).replace('\n', '<br>') + '</qt>'))
        self.panel.set_issues(rows)
        page_ids = set(order)
        if self.panel.selected_scope() == 'current':
            page_ids = {self.main.image_states[current_path]['page_id']} if current_path else set()
        self.panel.set_counts(len(page_ids), sum(key[0] in page_ids for key in self._snapshots), len(rows))
        self.panel.set_status(self.tr('Detect text regions to start reviewing.') if not self._snapshots else
                             self.tr('Bubble outlines are checked when their page is open.'))
        self._update_controls()

    def issue_title(self, issue):
        if issue.kind == 'uncertain_ocr':
            return QtCore.QCoreApplication.translate('ReviewPanel', 'Uncertain OCR')
        titles = {'missing_ocr': self.tr('Missing OCR'), 'missing_translation': self.tr('Missing translation'),
                  'overflow': self.tr('Overflow'), 'small_text': self.tr('Small text'),
                  'failed_operation': self.tr('Failed operation')}
        if issue.kind == 'awaiting_review':
            stages = {'source': self.tr('Source'), 'translation': self.tr('Translation'),
                      'layout': self.tr('Layout'), 'cleanup': self.tr('Cleanup')}
            return self.tr('Awaiting review: {stage}').format(stage=stages.get(issue.stage, issue.stage))
        return titles.get(issue.kind, issue.kind)

    def selected_region(self):
        block = getattr(self.main, 'curr_tblock', None)
        if block is None:
            return None
        path = self._live_path(block)
        if path not in self.main.image_states:
            return None
        page_id = self.main.image_states[path].setdefault('page_id', new_id())
        return page_id, block.region_id

    def _update_controls(self):
        selected = self.selected_region()
        refs = self.region_refs(*selected) if selected else []
        self.panel.set_region_state(refs[0].region_metadata.get('protections', {}) if refs else {}, bool(refs))
        self.panel.set_ocr_evidence(current_ocr_evidence(self.main.curr_tblock) if selected else None)
        self.panel.set_busy(self.busy() or self._pending_navigation is not None)
        snap = self._snapshots.get(selected) if selected else None
        cleanup = bool(snap and snap['workflow'])
        self.panel.set_stage_available('cleanup', cleanup)
        key = self.panel.selected_key()
        matching = selected is not None and key is not None and tuple(key[:2]) == selected
        self.panel.ignore_button.setEnabled(bool(matching) and not self.busy())
        self.panel.fit_button.setEnabled(bool(selected and self.live_item(self.path_for_page(selected[0]), selected[1])) and not self.busy())
        self.panel.report_button.setEnabled(bool(getattr(self.main.batch_report_ctrl, '_latest_batch_report', None)) and not self.busy())

    def navigate(self, step):
        if self.busy():
            return
        self.panel.set_expanded(True)
        if not self.panel.issue_list.count():
            self.refresh()
            return
        row = self.panel.issue_list.currentRow()
        row = (row + step) % self.panel.issue_list.count() if row >= 0 else (0 if step > 0 else self.panel.issue_list.count() - 1)
        if row == self.panel.issue_list.currentRow():
            # One remaining issue still needs to be framed on repeated Next.
            self.open_issue(self.panel.selected_key())
        else:
            self.panel.issue_list.setCurrentRow(row)

    def open_issue(self, key):
        if self.busy() or not key:
            return
        page_id, region_id = key[:2]
        if len(key) > 3:
            stage_index = self.panel.stage_combo.findData(key[3])
            if stage_index >= 0:
                self.panel.stage_combo.setCurrentIndex(stage_index)
        path = self.path_for_page(page_id)
        if path is None:
            self.refresh()
            return
        self.main.text_ctrl._commit_pending_text_command()
        self.main.show_main_page()
        self._pending_navigation = (page_id, region_id, 80)
        self.main.page_list.setCurrentRow(self.main.image_files.index(path))
        if self.main.image_viewer.webtoon_mode:
            self.main.image_viewer.webtoon_manager.scroll_to_page(self.main.image_files.index(path))
        self.navigation_timer.start(0)

    def _finish_navigation(self):
        pending = self._pending_navigation
        if pending is None:
            return
        page_id, region_id, attempts = pending
        path = self.path_for_page(page_id)
        viewer = self.main.image_viewer
        ready = path is not None and viewer.hasPhoto() and not self.busy()
        if ready and viewer.webtoon_mode:
            ready = self.main.image_files.index(path) in viewer.webtoon_manager.loaded_pages
        elif ready:
            ready = self.main.curr_img_idx == self.main.image_files.index(path) and not getattr(self.main, '_processing_page_change', False)
        if not ready:
            if attempts > 0 and path is not None:
                self._pending_navigation = (page_id, region_id, attempts - 1)
                self.navigation_timer.start(50)
                return
            self._pending_navigation = None
            self.panel.set_status(self.tr('The page is not ready. Try selecting the issue again.'))
            self._update_controls()
            return
        self._pending_navigation = None
        if not region_id:
            self._update_controls()
            return
        refs = [b for b in self.main.blk_list if b.region_id == region_id and self._live_path(b) == path]
        if len(refs) != 1:
            self.panel.set_status(self.tr('This region has changed. Refresh the review list.'))
            return
        block = refs[0]
        item = self.live_item(path, region_id)
        if item is not None:
            viewer._scene.clearSelection()
            item.setSelected(True)
            item.selected = True
            self.main.text_ctrl.on_text_item_selected(item)
            bounds = item.sceneBoundingRect()
        else:
            x, y, w, h = block.xywh
            rect = next((r for r in viewer.rectangles if abs(r.pos().x()-x) < 1 and abs(r.pos().y()-y) < 1), None)
            if rect is None:
                rect = viewer.add_rectangle(QtCore.QRectF(0, 0, w, h), QtCore.QPointF(x, y), block.angle)
            viewer.select_rectangle(rect)
            self.main.curr_tblock = block
            bounds = rect.sceneBoundingRect()
        visible = viewer.mapFromScene(bounds).boundingRect()
        if visible.width() < 70 or visible.height() < 35 or visible.width() > viewer.viewport().width() * .9 or visible.height() > viewer.viewport().height() * .9:
            QtWidgets.QGraphicsView.fitInView(viewer, bounds.adjusted(-30, -30, 30, 30), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            viewer._scene.update()
        else:
            viewer.centerOn(bounds.center())
        self.schedule()

    def _change_metadata(self, transform):
        if self.busy():
            return
        selected = self.selected_region()
        refs = self.region_refs(*selected) if selected else []
        if not refs:
            return
        before = normalize_region_metadata(refs[0].region_metadata)
        after = transform(copy.deepcopy(before), self.snapshot().get(selected))
        if after is None or before == after:
            return
        command = RegionReviewCommand(self, *selected, before, after)
        stack = self.main.undo_stacks.get(self.path_for_page(selected[0]))
        if stack is not None:
            stack.push(command)

    def set_protection(self, key, enabled):
        if key not in ('keep_source', 'keep_translation', 'keep_layout', 'skip_auto_cleanup'):
            return
        def change(metadata, _snap):
            metadata['protections'][key] = bool(enabled)
            return metadata
        self._change_metadata(change)

    def mark_current_reviewed(self, stage):
        if stage not in ('source', 'translation', 'layout', 'cleanup'):
            return
        self._change_metadata(lambda meta, snap: mark_reviewed(meta, stage, self._fingerprint(snap, stage))
                              if snap and (stage != 'cleanup' or snap['workflow']) else None)

    def ignore_current_issue(self):
        key = self.panel.selected_key()
        issue = next((i for i in getattr(self, '_all_issues', []) if i.key == key), None)
        if issue is None or self.selected_region() != tuple(key[:2]):
            return
        self._change_metadata(lambda meta, _snap: dismiss_issue(meta, issue))

    def fit_current(self):
        selected = self.selected_region()
        if self.busy() or not selected:
            return
        item = self.live_item(self.path_for_page(selected[0]), selected[1])
        if item is not None:
            self.main.image_viewer._scene.clearSelection()
            item.setSelected(True)
            item.selected = True
            self.main.text_ctrl.on_text_item_selected(item)
            self.main.text_ctrl.fit_controller.open_preview()

    def add_context_actions(self, menu, *, item=None):
        if item is not None:
            self.main.text_ctrl.on_text_item_selected(item)
        refs = self.region_refs(*self.selected_region()) if self.selected_region() else []
        if not refs:
            return
        menu.addSeparator()
        labels = {'keep_source': 'Keep source', 'keep_translation': 'Keep translation',
                  'keep_layout': 'Keep layout', 'skip_auto_cleanup': 'Skip automatic cleanup'}
        for key, label in labels.items():
            action = menu.addAction(QtCore.QCoreApplication.translate('ReviewPanel', label))
            action.setCheckable(True)
            action.setChecked(is_protected(refs[0], key))
            action.setEnabled(not self.busy())
            action.triggered.connect(lambda checked, name=key: self.set_protection(name, checked))
        review = menu.addMenu(QtCore.QCoreApplication.translate('ReviewPanel', 'Mark reviewed'))
        for stage, label in (('source', 'Source'), ('translation', 'Translation'), ('layout', 'Layout'), ('cleanup', 'Cleanup')):
            action = review.addAction(QtCore.QCoreApplication.translate('ReviewPanel', label))
            action.setEnabled(not self.busy())
            action.triggered.connect(lambda _checked=False, name=stage: self.mark_current_reviewed(name))

    def retry_ocr(self):
        """Explicitly compare one alternative local recognizer on the selected region."""
        from modules.ocr.retry import retry_recognizers
        if self.busy():
            return
        key = self.selected_region()
        snapshots = self.snapshot()
        snap = snapshots.get(key)
        if snap is None:
            self.panel.set_status(self.tr('Select a text region first.'))
            return
        source = snap['languages']['source']
        source = self.main.lang_mapping.get(source, source)
        settings = self.main.settings_page
        selected = settings.get_tool_selection('ocr')
        default_selected = selected in ('Default', QtCore.QCoreApplication.translate('SettingsPageUI', 'Default'))
        page_size = None
        if source in ('Auto', 'Japanese') and snap['block'].text_class == 'text_free':
            from PIL import Image
            try:
                with Image.open(snap['path']) as original:
                    page_size = original.size
            except OSError:
                pass
        choices = retry_recognizers(snap['block'], source, default_selected=default_selected, page_size=page_size,
            page_blocks=[s['block'] for s in snapshots.values() if s['path'] == snap['path']])
        if not choices:
            QtWidgets.QMessageBox.information(self.main, self.tr('Retry OCR'),
                self.tr('No alternative local recognizer is available for this source language.'))
            return
        epoch = self._project_epoch()
        guard = self._replacement_guard(snap, 'source')
        label, accepted = QtWidgets.QInputDialog.getItem(self.main, self.tr('Retry OCR'),
            self.tr('Local recognizer') + '\n' + self.tr('Missing model files will be downloaded when you continue.'),
            [option.label for option in choices], 0, False)
        if not accepted:
            return
        current = self.snapshot().get(key)
        if (self.busy() or current is None or self.selected_region() != key or self._project_epoch() != epoch
                or self._replacement_guard(current, 'source') != guard):
            self.panel.set_status(self.tr('This region changed while the preview was open. Run the action again.'))
            return
        route = next((option for option in choices if option.label == label), None)
        if route is not None:
            self._start_replacement(key, current, snapshots, 'source', retry_route=route)

    def _project_epoch(self):
        return (getattr(self.main, '_translation_project_epoch', 0), getattr(self.main, '_translation_cancel_epoch', 0))

    def _replacement_guard(self, snap, stage):
        try:
            stat = os.stat(snap['path'])
            stamp = (stat.st_size, stat.st_mtime_ns)
        except OSError:
            stamp = None
        block = snap['block']
        return (self._fingerprint(snap, stage), stamp, tuple(getattr(block, 'tr_origin_point', ()) or ()),
                block.region_metadata.get('protections', {}).copy(),
                current_ocr_evidence(block) if stage == 'source' else None)

    def preview_replacement(self, stage, *, update_lettering=True):
        """Handle an explicit selected retry only when a kept result is at risk."""
        block = self.main.pipeline.get_selected_block()
        if block is None or not (is_protected(block, 'keep_source' if stage == 'source' else 'keep_translation')
                                 or (stage == 'translation' and is_protected(block, 'keep_layout'))):
            return False
        if self.busy():
            return True
        path = self._live_path(block)
        if path is None:
            return True
        snapshots = self.snapshot()
        key = self.main.image_states[path]['page_id'], block.region_id
        snap = snapshots.get(key)
        if snap is None:
            return True
        return self._start_replacement(key, snap, snapshots, stage, update_lettering=update_lettering)

    def _start_replacement(self, key, snap, snapshots, stage, *, update_lettering=True, retry_route=None):
        path = snap['path']
        if stage == 'translation':
            from modules.utils.pipeline_config import validate_translator
            if not validate_translator(self.main, snap['languages']['target']):
                return True
        working = snap['block'].deep_copy()
        fingerprint = self._replacement_guard(snap, stage)
        self._preview_running = True
        self.panel.set_busy(True)
        context = self.main.settings_page.get_llm_settings()['extra_context'] if stage == 'translation' else None
        source = snap['languages']['source']
        source = self.main.lang_mapping.get(source, source)
        target = snap['languages']['target']
        target = self.main.lang_mapping.get(target, target)
        epoch = self._project_epoch()
        use_gpu = bool(self.main.settings_page.is_gpu_enabled()) if retry_route else False
        profile = copy.deepcopy(getattr(self.main, 'translation_profile', {}))
        working._page_id = key[0]
        translator = None
        if stage == 'translation':
            try:
                from modules.translation.processor import Translator
                translator = Translator(self.main, source, target)
            except Exception as error:
                import traceback
                self._preview_running = False
                self.schedule()
                self.main.default_error_handler((type(error), error, traceback.format_exc()))
                return True

        history_request = None

        def generate():
            nonlocal history_request
            from modules.rendering.auto_bubble_fit import load_original_for_fitting
            image = load_original_for_fitting(path)
            if image is None:
                raise RuntimeError(self.tr('The original page is unavailable for this preview.'))
            if retry_route is not None:
                from modules.ocr.retry import recognize_alternative
                return recognize_alternative(image, working, retry_route, use_gpu)
            if stage == 'source':
                handler = self.main.pipeline.ocr_handler
                handler._recognize_selected_block(image, working, source,
                    context_blocks=[s['block'] for s in snapshots.values() if s['path'] == path])
                return working.text, current_ocr_evidence(working)
            working.translation = ''
            from modules.translation.history import ApprovedHistory
            if isinstance(getattr(translator, 'approved_history', None), ApprovedHistory):
                history_request = translator.prepare_request([working], image, context)
                translator.translate([working], image, context, request=history_request)
            else:
                translator.translate([working], image, context)
            return working.translation, None

        def finished():
            self._preview_running = False
            self.schedule()

        def ready(result):
            from modules.translation.history import history_is_current
            proposed, evidence = result
            if self._project_epoch() != epoch or not history_is_current(history_request, self.main):
                return
            if retry_route is not None:
                current = self.snapshot().get(key)
                if (self.selected_region() != key or current is None or self._replacement_guard(current, stage) != fingerprint):
                    self.panel.set_status(self.tr('This region changed while the preview was open. Run the action again.'))
                    return
                if not proposed.strip():
                    QtWidgets.QMessageBox.information(self.main, self.tr('Retry OCR'), self.tr('The recognizer returned no text.'))
                    return
            dialog = QtWidgets.QDialog(self.main)
            dialog.setWindowTitle(self.tr('Review replacement'))
            layout = QtWidgets.QVBoxLayout(dialog)
            layout.addWidget(QtWidgets.QLabel(os.path.basename(path)))
            layout.addWidget(QtWidgets.QLabel(self.tr('Current text')))
            old = QtWidgets.QPlainTextEdit()
            old.setPlainText(snap['block'].text if stage == 'source' else snap['block'].translation)
            old.setReadOnly(True)
            layout.addWidget(old)
            proposed_label = self.tr('Proposed text')
            if retry_route is not None:
                proposed_label += ' (' + retry_route.label + ')'
            layout.addWidget(QtWidgets.QLabel(proposed_label))
            editor = QtWidgets.QPlainTextEdit(proposed or '')
            layout.addWidget(editor)
            hint = QtWidgets.QLabel(self.tr('Apply replaces the source text for this region.') if stage == 'source' else
                self.tr('Apply keeps the protection and existing lettering geometry. Use Clean and Render for missing lettering.'))
            hint.setWordWrap(True)
            layout.addWidget(hint)
            buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Apply | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
            buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Apply).clicked.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            dialog.resize(500, 440)
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return
            current = self.snapshot().get(key)
            if (current is None or self._replacement_guard(current, stage) != fingerprint or self._project_epoch() != epoch
                    or not history_is_current(history_request, self.main)
                    or (retry_route is not None and self.selected_region() != key)
                    or (stage == 'translation' and profile != getattr(self.main, 'translation_profile', {}))):
                self.panel.set_status(self.tr('This region changed while the preview was open. Run the action again.'))
                return
            wording = editor.toPlainText()
            self.apply_replacement(key, stage, wording, current, update_lettering=update_lettering,
                                   evidence=evidence if proposed == wording else None)

        self.main.run_threaded(generate, ready, self.main.default_error_handler, finished)
        return True

    def apply_replacement(self, key, stage, text, snap, *, update_lettering=True, evidence=None):
        before = snap['block'].text if stage == 'source' else snap['block'].translation
        # A selected Translate retry changes target data only. Other reviewed
        # text edits can still update existing lettering without changing layout.
        props = snap['props'] if stage == 'translation' and update_lettering else None
        after_props = copy.deepcopy(props)
        if props is not None:
            from modules.rendering.auto_bubble_fit import _document
            document, _ = _document(TextItemProperties.from_dict(props), None)
            old = document.toPlainText()
            prefix = 0
            while prefix < min(len(old), len(text)) and old[prefix] == text[prefix]:
                prefix += 1
            suffix = 0
            while suffix < min(len(old)-prefix, len(text)-prefix) and old[-suffix-1] == text[-suffix-1]:
                suffix += 1
            utf16 = lambda value: len(value.encode('utf-16-le')) // 2
            cursor = QtGui.QTextCursor(document)
            cursor.setPosition(utf16(old[:prefix]))
            fmt = cursor.charFormat()
            cursor.setPosition(utf16(old[:len(old)-suffix]), QtGui.QTextCursor.MoveMode.KeepAnchor)
            cursor.insertText(text[prefix:len(text)-suffix], fmt)
            after_props['text'] = document.toHtml()
            start, old_end = utf16(old[:prefix]), utf16(old[:len(old)-suffix])
            new_end = start + utf16(text[prefix:len(text)-suffix])
            delta = new_end - old_end
            outlines = []
            for outline in after_props.get('selection_outlines', []):
                outline = copy.deepcopy(outline)
                from app.ui.canvas.text_item import OutlineInfo, OutlineType
                if isinstance(outline, dict):
                    kind = outline.get('type', 'selection')
                    outline = OutlineInfo(outline['start'], outline['end'], QtGui.QColor(outline['color']),
                                          outline['width'], kind if isinstance(kind, OutlineType) else OutlineType(kind))
                if outline.type == OutlineType.Full_Document:
                    outline.start, outline.end = 0, document.characterCount() - 1
                else:
                    def move(position, end=False):
                        if old_end == start:
                            return position if position < start or (position == start and end) else position + delta
                        return position if position <= start else position + delta if position >= old_end else new_end if end else start
                    outline.start = move(outline.start)
                    outline.end = move(outline.end, end=True)
                if outline.end > outline.start:
                    outlines.append(outline)
            after_props['selection_outlines'] = outlines
        before_evidence = current_ocr_evidence(snap['block']) if stage == 'source' else None
        command = ReviewedTextCommand(self, *key, stage, before, text, props, after_props, before_evidence, evidence)
        stack = self.main.undo_stacks.get(self.path_for_page(key[0]))
        if stack is not None:
            self.main.text_ctrl._commit_pending_text_command()
            stack.push(command)
