"""Review panel interaction tests, without page processing or model dependencies."""
import os
from pathlib import Path
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6 import QtCore, QtTest, QtWidgets

from app.ui.review_panel import ReviewPanel
from app.ui.dayu_widgets import dayu_theme


class ReviewPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.panel = ReviewPanel()
        self.addCleanup(self.panel.deleteLater)
        self.addCleanup(self.panel.close)
        self.rows = [
            {'key': ('page-a', 'region-a', 'missing_ocr', 'source'), 'label': 'Page A: missing OCR', 'tooltip': 'First region'},
            {'key': ('page-b', 'region-b', 'overflow', 'layout'), 'label': 'Page B: overflow', 'tooltip': 'Second region'},
        ]

    def test_collapsed_default_and_independent_filters(self):
        self.assertFalse(self.panel.header_button.isChecked())
        self.assertTrue(self.panel.body.isHidden())
        self.assertEqual(self.panel.selected_scope(), 'all')
        self.assertEqual(self.panel.selected_filter(), 'all')
        self.assertFalse(self.panel.review_button.isEnabled())
        self.assertFalse(self.panel.ignore_button.isEnabled())
        self.panel.header_button.click()
        self.assertFalse(self.panel.body.isHidden())
        self.assertEqual(self.panel.header_button.arrowType(), QtCore.Qt.ArrowType.DownArrow)
        changes = QtTest.QSignalSpy(self.panel.filter_changed)
        self.panel.scope_combo.setCurrentIndex(1)
        self.panel.filter_combo.setCurrentIndex(self.panel.filter_combo.findData('overflow'))
        self.assertEqual(changes.count(), 2)
        self.assertEqual(self.panel.selected_scope(), 'current')
        self.assertEqual(self.panel.selected_filter(), 'overflow')
        self.panel.set_counts(4, 12, 2)
        self.assertIn('4', self.panel.counts_label.text())
        self.assertIn('12', self.panel.counts_label.text())
        self.panel.set_status('Current page is still loading.')
        self.assertFalse(self.panel.status_label.isHidden())
        self.assertEqual(self.panel.status_label.text(), 'Current page is still loading.')
        self.panel.set_status('')
        self.assertTrue(self.panel.status_label.isHidden())

    def test_keyboard_selection_and_refresh_preserve_stable_tuple_key(self):
        selected = QtTest.QSignalSpy(self.panel.issue_selected)
        self.panel.set_issues(self.rows)
        self.assertEqual(selected.count(), 0)
        self.assertIsNone(self.panel.selected_key())
        self.assertTrue(self.panel.select_key(self.rows[0]['key']))
        self.assertEqual(selected.at(0), [self.rows[0]['key']])
        self.panel.set_expanded(True)
        self.panel.show()
        self.panel.issue_list.setFocus()
        QtTest.QTest.keyClick(self.panel.issue_list, QtCore.Qt.Key.Key_Down)
        self.assertEqual(self.panel.selected_key(), self.rows[1]['key'])
        self.assertEqual(selected.count(), 2)
        self.panel.set_issues(list(reversed(self.rows)))
        self.assertEqual(selected.count(), 2)
        self.assertEqual(self.panel.selected_key(), self.rows[1]['key'])
        self.assertFalse(self.panel.select_key(('missing', 'key')))
        self.panel.set_issues([])
        self.assertIsNone(self.panel.selected_key())
        self.assertFalse(self.panel.ignore_button.isEnabled())
        self.assertFalse(self.panel.previous_button.isEnabled())
        self.assertFalse(self.panel.empty_label.isHidden())

    def test_protection_updates_are_silent_independent_and_busy_safe(self):
        changed = QtTest.QSignalSpy(self.panel.protection_changed)
        self.panel.set_region_state({'keep_source': True, 'skip_auto_cleanup': True}, True)
        self.assertEqual(changed.count(), 0)
        self.assertTrue(self.panel.protection_checks['keep_source'].isChecked())
        self.assertFalse(self.panel.protection_checks['keep_translation'].isChecked())
        self.panel.protection_checks['keep_translation'].click()
        self.assertEqual(changed.at(0), ['keep_translation', True])
        self.assertTrue(self.panel.protection_checks['keep_source'].isChecked())
        QtTest.QTest.mouseClick(self.panel.protection_labels['keep_layout'], QtCore.Qt.MouseButton.LeftButton)
        self.assertEqual(changed.at(1), ['keep_layout', True])
        self.panel.set_busy(True)
        self.panel.set_region_state({'keep_layout': True}, True)
        self.assertEqual(changed.count(), 2)
        for widget in (self.panel.review_button, self.panel.retry_ocr_button, self.panel.fit_button, self.panel.refresh_button,
                       self.panel.report_button, self.panel.issue_list, *self.panel.protection_checks.values()):
            self.assertFalse(widget.isEnabled())
        self.assertTrue(self.panel.header_button.isEnabled())
        self.panel.set_busy(False)
        self.assertTrue(self.panel.review_button.isEnabled())
        self.assertTrue(self.panel.protection_checks['keep_layout'].isChecked())
        self.panel.set_region_state({}, False)
        self.assertFalse(self.panel.fit_button.isEnabled())
        self.assertFalse(self.panel.retry_ocr_button.isEnabled())

    def test_navigation_review_and_actions_emit_only_their_contract_arguments(self):
        self.panel.set_issues(self.rows)
        self.panel.select_key(self.rows[0]['key'])
        self.panel.set_region_state({}, True)
        navigation = QtTest.QSignalSpy(self.panel.navigate_requested)
        review = QtTest.QSignalSpy(self.panel.reviewed_requested)
        ignored = QtTest.QSignalSpy(self.panel.ignored_requested)
        refresh = QtTest.QSignalSpy(self.panel.refresh_requested)
        fitted = QtTest.QSignalSpy(self.panel.fit_requested)
        report = QtTest.QSignalSpy(self.panel.report_requested)
        self.panel.previous_button.click()
        self.panel.next_button.click()
        self.assertEqual([navigation.at(i) for i in range(navigation.count())], [[-1], [1]])
        self.panel.stage_combo.setCurrentIndex(self.panel.stage_combo.findData('layout'))
        self.panel.review_button.click()
        self.assertEqual(review.at(0), ['layout'])
        for button in (self.panel.ignore_button, self.panel.refresh_button, self.panel.fit_button, self.panel.report_button):
            button.click()
        for spy in (ignored, refresh, fitted, report):
            self.assertEqual(spy.count(), 1)
            self.assertEqual(spy.at(0), [])
        for stage in ('source', 'translation', 'layout', 'cleanup'):
            self.panel.set_stage_available(stage, False)
        self.assertFalse(self.panel.review_button.isEnabled())
        self.panel.set_stage_available('translation', True)
        self.assertEqual(self.panel.selected_stage(), 'translation')
        self.assertTrue(self.panel.review_button.isEnabled())

    def test_compiled_labels_and_stable_values_in_all_interface_languages(self):
        catalogs = Path(__file__).resolve().parents[1] / 'resources/translations/compiled'
        for locale in ('de', 'es', 'fr', 'it', 'ja', 'ko', 'ru', 'tr', 'zh-CN'):
            with self.subTest(locale=locale):
                translator = QtCore.QTranslator()
                self.assertTrue(translator.load(str(catalogs / f'ct_{locale}.qm')))
                self.app.installTranslator(translator)
                panel = None
                try:
                    panel = ReviewPanel()
                    translated = lambda source: translator.translate('ReviewPanel', source)
                    self.assertEqual(panel.header_button.text(), translated('Review'))
                    self.assertNotEqual(panel.header_button.text(), 'Review')
                    for combo, options in (
                        (panel.scope_combo, [('All pages', 'all'), ('Current page', 'current')]),
                        (panel.filter_combo, [('All issues', 'all'), ('Missing OCR', 'missing_ocr'), ('Uncertain OCR', 'uncertain_ocr'),
                                              ('Missing translation', 'missing_translation'), ('Overflow', 'overflow'),
                                              ('Small text', 'small_text'), ('Failed operation', 'failed_operation'),
                                              ('Awaiting review', 'awaiting_review')]),
                        (panel.stage_combo, [('Source', 'source'), ('Translation', 'translation'),
                                             ('Layout', 'layout'), ('Cleanup', 'cleanup')]),
                    ):
                        for index, (label, key) in enumerate(options):
                            self.assertTrue(translated(label))
                            self.assertEqual(combo.itemText(index), translated(label))
                            self.assertEqual(combo.itemData(index), key)
                    for key, source in [('keep_source', 'Keep source'), ('keep_translation', 'Keep translation'),
                                        ('keep_layout', 'Keep layout'), ('skip_auto_cleanup', 'Skip automatic cleanup')]:
                        self.assertEqual(panel.protection_checks[key].accessibleName(), translated(source))
                        self.assertEqual(panel.protection_labels[key].text(), translated(source))
                    for button, source in [(panel.review_button, 'Mark reviewed'), (panel.ignore_button, 'Ignore issue'),
                                           (panel.retry_ocr_button, 'Retry OCR…'), (panel.fit_button, 'Fit Text'), (panel.report_button, 'Report')]:
                        self.assertEqual(button.text(), translated(source))
                    panel.set_counts(7, 19, 3)
                    self.assertEqual(panel.protection_hint.text(), translated(
                        'Page-wide operations keep protected stages. Direct edits remain allowed.'))
                    self.assertEqual(panel.counts_label.text(), translated(
                        'Pages: {pages} · Regions: {regions} · Issues: {issues}'
                    ).format(pages=7, regions=19, issues=3))
                    panel.set_expanded(True)
                    panel.resize(260, 800)
                    panel.show()
                    for _ in range(4):
                        self.app.processEvents()
                    self.assertEqual(panel.width(), 260)
                    for button in (panel.previous_button, panel.next_button, panel.refresh_button,
                                   panel.review_button, panel.ignore_button, panel.fit_button, panel.report_button):
                        self.assertLessEqual(button.minimumSizeHint().width(), button.width())
                finally:
                    if panel is not None:
                        panel.close()
                        panel.deleteLater()
                    self.app.removeTranslator(translator)

    def test_controls_fit_narrow_scroll_viewport_despite_wide_sibling_tools(self):
        catalogs = Path(__file__).resolve().parents[1] / 'resources/translations/compiled'
        for locale in ('de', 'es', 'fr', 'it', 'ja', 'ko', 'ru', 'tr', 'zh-CN'):
            with self.subTest(locale=locale):
                translator = QtCore.QTranslator()
                self.assertTrue(translator.load(str(catalogs / f'ct_{locale}.qm')))
                self.app.installTranslator(translator)
                scroll = QtWidgets.QScrollArea()
                try:
                    scroll.setWidgetResizable(True)
                    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                    scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
                    content = QtWidgets.QWidget()
                    layout = QtWidgets.QVBoxLayout(content)
                    panel = ReviewPanel()
                    panel.set_issues(self.rows)
                    panel.set_region_state({}, True)
                    panel.set_expanded(True)
                    layout.addWidget(panel)
                    wide_tools = QtWidgets.QWidget()
                    wide_tools.setMinimumWidth(600)
                    wide_tools.setFixedHeight(30)
                    layout.addWidget(wide_tools)
                    scroll.setWidget(content)
                    dayu_theme.apply(scroll)
                    scroll.resize(250, 500)
                    scroll.show()
                    for viewport_width in (208, 230):
                        for _ in range(3):
                            self.app.processEvents()
                        scroll.resize(scroll.width() + viewport_width - scroll.viewport().width(), 500)
                        for _ in range(4):
                            self.app.processEvents()
                        self.assertEqual(scroll.viewport().width(), viewport_width)
                        self.assertGreater(panel.width(), viewport_width)
                        controls = (panel.header_button, panel.scope_combo, panel.filter_combo, panel.issue_list,
                                    panel.previous_button, panel.next_button, panel.refresh_button, panel.stage_combo,
                                    panel.review_button, panel.ignore_button, panel.fit_button, panel.report_button,
                                    *panel.protection_checks.values(), *panel.protection_labels.values())
                        for control in controls:
                            left = control.mapTo(scroll.viewport(), QtCore.QPoint()).x()
                            self.assertGreaterEqual(left, 0, (locale, type(control).__name__))
                            self.assertLessEqual(left + control.width(), viewport_width,
                                                 (locale, type(control).__name__, left, control.width()))
                        for button in (panel.previous_button, panel.next_button, panel.refresh_button,
                                       panel.review_button, panel.ignore_button, panel.fit_button, panel.report_button):
                            self.assertLessEqual(button.minimumSizeHint().width(), button.width(),
                                                 (locale, button.text()))
                            self.assertEqual(button.accessibleName(), button._review_full_label)
                            self.assertEqual(button.toolTip(), button._review_full_label)
                finally:
                    scroll.close()
                    scroll.deleteLater()
                    self.app.removeTranslator(translator)


if __name__ == '__main__':
    unittest.main()
