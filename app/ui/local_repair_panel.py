"""Small, theme-aware controls for deterministic image repairs."""
from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.dayu_widgets.push_button import MPushButton
from app.ui.dayu_widgets.tool_button import MToolButton
from app.ui.dayu_widgets.check_box import MCheckBox
from app.ui.dayu_widgets.spin_box import MSpinBox


class LocalRepairPanel(QtWidgets.QWidget):
    def __init__(self, parent=None, tool_layout=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = tool_layout if tool_layout is not None else QtWidgets.QHBoxLayout()
        self.restore = MToolButton().svg("restore-brush.svg")
        self.restore.setText(self.tr("Restore brush"))
        self.restore.setAccessibleName(self.tr("Restore brush"))
        self.restore.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.restore.setCheckable(True)
        self.restore.setToolTip(self.tr("Restore brush") + "\n" + self.tr("Brush back the original page pixels in a small area. Original text is restored too. Uses the brush size slider."))
        self.pick = MToolButton().svg("eyedropper.svg")
        self.pick.setText(self.tr("Pick color"))
        self.pick.setAccessibleName(self.tr("Pick color"))
        self.pick.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.pick.setCheckable(True)
        self.pick.setToolTip(self.tr("Pick color") + "\n" + self.tr("Click an intact background area to sample its color."))
        row.addWidget(self.restore)
        row.addWidget(self.pick)
        self.clone = MToolButton().svg("clone-stamp.svg")
        self.clone.setText(self.tr("Clone brush"))
        self.clone.setAccessibleName(self.tr("Clone brush"))
        self.clone.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.clone.setCheckable(True)
        self.clone.setToolTip(self.tr("Clone brush") + "\n" + self.tr("Set the brush size, then Alt-click to capture a fixed sample. Paint to repeat that sample. Dashed cyan marks the source; the solid circle follows the brush. Alt-click again after changing size."))
        row.addWidget(self.clone)
        if tool_layout is None:
            layout.addLayout(row)
        self.fill = MPushButton(self.tr("Fill mask with sampled color"))
        self.fill.setEnabled(False)
        self.fill.setToolTip(self.tr("Fill the painted or segmented mask on this page, without AI inpainting or extra mask expansion."))
        layout.addWidget(self.fill)
        self.clone_options = QtWidgets.QWidget()
        options = QtWidgets.QHBoxLayout(self.clone_options)
        options.setContentsMargins(0, 0, 0, 0)
        self.clone_original = MCheckBox(self.tr("Sample original"))
        self.clone_original.setChecked(True)
        self.clone_original.setToolTip(self.tr("Sample the page before cleanup patches. Turn off to sample the edited page. Alt-click again after changing this option."))
        self.clone_softness = MSpinBox()
        self.clone_softness.setRange(0, 100)
        self.clone_softness.setValue(25)
        self.clone_softness.setSuffix("%")
        self.clone_softness.setAccessibleName(self.tr("Soft edge"))
        label = QtWidgets.QLabel(self.tr("Soft edge"))
        label.setBuddy(self.clone_softness)
        options.addWidget(self.clone_original)
        options.addStretch()
        options.addWidget(label)
        options.addWidget(self.clone_softness)
        layout.addWidget(self.clone_options)
        self.clone_options.hide()

    def set_color(self, color):
        swatch = QtGui.QPixmap(18, 18)
        swatch.fill(color)
        self.fill.setIcon(QtGui.QIcon(swatch))
        self.fill.setIconSize(QtCore.QSize(18, 18))
        self.fill.setEnabled(True)
