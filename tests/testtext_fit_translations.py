"""Check the actual Fit Text dialog against every shipped Qt catalog."""
import os
from pathlib import Path
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.canvas.text_item import TextBlockItem
from app.ui.dialogs.text_fit import TextFitDialog, boundary_message


class TextFitTranslationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.catalogs = Path(__file__).resolve().parents[1] / 'resources/translations/compiled'

    def test_dialog_controls_and_validation_in_all_interface_languages(self):
        for locale in ('de', 'es', 'fr', 'it', 'ja', 'ko', 'ru', 'tr', 'zh-CN'):
            with self.subTest(locale=locale):
                translator = QtCore.QTranslator()
                self.assertTrue(translator.load(str(self.catalogs / f'ct_{locale}.qm')))
                self.app.installTranslator(translator)
                dialog = None
                try:
                    item = TextBlockItem(font_family='Arial', font_size=12, outline_color=None)
                    item.set_plain_text('Bonjour !')
                    item.setTextWidth(160)
                    dialog = TextFitDialog(item, 160, 90, 10, 16,
                                           boundary_context={'image_loader': lambda: (None, 'missing_image')})
                    translated = lambda source: translator.translate('TextFitDialog', source)
                    self.assertEqual(dialog.windowTitle(), translated('Fit Text'))
                    self.assertNotEqual(dialog.windowTitle(), 'Fit Text')
                    self.assertEqual(dialog.apply_button.text(), translated('Apply'))
                    self.assertEqual(dialog.cancel_button.text(), translated('Cancel'))
                    self.assertEqual(dialog.keep_breaks.text(), translated('Keep manual line breaks'))
                    for index, source in enumerate(('Current', 'Balanced', 'Larger text')):
                        self.assertEqual(dialog.choice.itemText(index), translated(source))
                    for index, (source, key) in enumerate((('Box', 'box'), ('Bubble interior', 'bubble'))):
                        self.assertEqual(dialog.mode.itemText(index), translated(source))
                        self.assertEqual(dialog.mode.itemData(index), key)
                    for index, (source, key) in enumerate((('Estimate from original', 'auto'),
                                                         ('Rectangle', 'rectangle'), ('Ellipse', 'ellipse'))):
                        self.assertEqual(dialog.boundary_shape.itemText(index), translated(source))
                        self.assertEqual(dialog.boundary_shape.itemData(index), key)
                    self.assertEqual(dialog.boundary_left.accessibleName(), translated('Left'))
                    self.assertEqual(dialog.boundary_top.accessibleName(), translated('Top'))
                    self.assertEqual(dialog.area_width.accessibleName(), translated('Width'))
                    self.assertEqual(dialog.area_height.accessibleName(), translated('Height'))
                    self.assertTrue(dialog.keep_breaks.isChecked())
                    self.assertEqual(item.toPlainText(), 'Bonjour !')
                    self.assertNotIn('{size}', dialog.status.text())
                    self.assertNotIn('{status}', dialog.status.text())
                    dialog.minimum.setValue(17)
                    dialog.rebuild()
                    self.assertEqual(dialog.status.text(), translated(
                        'Minimum font size must not exceed maximum font size.'))
                    self.assertFalse(dialog.apply_button.isEnabled())
                    dialog.minimum.setValue(10)
                    dialog.mode.setCurrentIndex(1)
                    dialog.rebuild()
                    fallback = 'Original image unavailable or its dimensions changed. Using Box.'
                    self.assertEqual(dialog.boundary_status.text(), translated(fallback))
                    self.assertIsNone(dialog.boundary)
                    for reason, source in (
                            ('unsupported_transform', 'Bubble fitting requires unrotated horizontal text at normal scale. Using Box.'),
                            ('padding_empty', 'Padding leaves no usable interior. Using Box.'),
                            ('region_too_large', 'Fitting area is too large. Using Box.'),
                            ('invalid_bounds', 'The boundary is outside the page or has no usable area. Using Box.'),
                            ('unreliable', 'No reliable enclosed interior was found. Using Box. Try Rectangle or Ellipse.')):
                        self.assertTrue(translated(source))
                        self.assertEqual(boundary_message(reason), translated(source))
                    for source in ('Text overflows the fitting area.',
                                   'Text is below the minimum readable size.',
                                   'Fit Text requires a text item contained within one page.',
                                   'Wait for page processing to finish before fitting text.',
                                   'The text or page changed. Open Fit Text again to make a new preview.'):
                        expected = translator.translate('TextFitController', source)
                        self.assertTrue(expected)
                        self.assertNotEqual(expected, source)
                        self.assertEqual(QtCore.QCoreApplication.translate('TextFitController', source), expected)
                finally:
                    if dialog is not None:
                        dialog.timer.stop()
                        dialog.close()
                        dialog.deleteLater()
                    self.app.removeTranslator(translator)


if __name__ == '__main__':
    unittest.main()
