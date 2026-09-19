import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtGui, QtWidgets

from app.ui.dayu_widgets import dayu_theme
from app.ui.dayu_widgets.theme import MTheme
from app.ui.settings.settings_ui import SettingsPageUI
from app.ui.startup_home import StartupHomeScreen


def _relative_luminance(color: str) -> float:
    rgb = QtGui.QColor(color).getRgbF()[:3]

    def linearize(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue = (linearize(channel) for channel in rgb)
    return (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)


def _contrast_ratio(first: str, second: str) -> float:
    high, low = sorted((_relative_luminance(first), _relative_luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


class ThemeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.original_theme = dayu_theme.theme_name
        self.original_primary = dayu_theme.primary_color

    def tearDown(self):
        dayu_theme.set_primary_color(self.original_primary)
        dayu_theme.set_theme(self.original_theme)

    def test_new_palettes_are_readable_and_have_expected_brightness(self):
        theme = MTheme("midnight", primary_color="#3aaed8")
        self.assertTrue(theme.is_dark)
        self.assertLess(QtGui.QColor(theme.background_color).lightness(), 128)
        self.assertGreaterEqual(
            _contrast_ratio(theme.primary_text_color, theme.background_color),
            4.5,
        )

        theme.set_theme("parchment")
        self.assertFalse(theme.is_dark)
        self.assertGreater(QtGui.QColor(theme.background_color).lightness(), 128)
        self.assertGreaterEqual(
            _contrast_ratio(theme.primary_text_color, theme.background_color),
            4.5,
        )

    def test_settings_exposes_all_theme_choices(self):
        settings_ui = SettingsPageUI()
        choices = [settings_ui.theme_combo.itemText(index) for index in range(settings_ui.theme_combo.count())]
        self.assertEqual(choices, ["Dark", "Light", "Midnight", "Parchment", "Lavender", "Mint"])
        self.assertEqual(settings_ui.value_mappings["Midnight"], "Midnight")
        self.assertEqual(settings_ui.value_mappings["Parchment"], "Parchment")

    def test_startup_home_uses_active_theme_palette(self):
        dayu_theme.set_primary_color("#3aaed8")
        dayu_theme.set_theme("midnight")
        home = StartupHomeScreen()
        home.apply_theme(dayu_theme.is_dark)
        self.assertIn(dayu_theme.background_in_color, home._search_box.styleSheet())
        self.assertIn(dayu_theme.primary_text_color, home._new_hdr.styleSheet())


if __name__ == "__main__":
    unittest.main()
