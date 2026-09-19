from PySide6 import QtWidgets
from ..dayu_widgets.check_box import MCheckBox
from modules.translation.providers import PROVIDERS


class CredentialsPage(QtWidgets.QWidget):
    """User-owned direct API credentials. Local runtimes live on the LLMs page."""

    def __init__(self, services, value_mappings, parent=None):
        super().__init__(parent)
        self.credential_widgets = {}
        layout = QtWidgets.QVBoxLayout(self)
        notice = QtWidgets.QLabel(self.tr(
            "Optional cloud services use your own API keys and provider billing. "
            "Text and enabled image context are sent to the selected provider. "
            "Local LLM needs no cloud account. Select the translator in Tools."
        ))
        notice.setWordWrap(True)
        layout.addWidget(notice)
        self.save_keys_checkbox = MCheckBox(self.tr("Save Keys"))
        self.save_keys_checkbox.setToolTip(self.tr("Keys are stored in this computer's application settings, not in projects."))
        layout.addWidget(self.save_keys_checkbox)
        self.provider_combo = QtWidgets.QComboBox()
        self.pages = QtWidgets.QStackedWidget()
        layout.addWidget(self.provider_combo)
        layout.addWidget(self.pages)
        suggestions = {service: models for service, models in PROVIDERS.values()}
        for label in services:
            service = value_mappings.get(label, label)
            self.provider_combo.addItem("OpenAI" if service == "Open AI GPT" else label, service)
            page = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(page)
            form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
            key = QtWidgets.QLineEdit()
            key.setEchoMode(QtWidgets.QLineEdit.Password)
            suffix = 'api_key_ocr' if service == 'Microsoft Azure' else 'api_key'
            self.credential_widgets[f'{service}_{suffix}'] = key
            form.addRow(self.tr("API Key"), key)
            if service in suggestions:
                model = QtWidgets.QComboBox()
                model.setEditable(True)
                model.addItems(suggestions[service])
                self.credential_widgets[f'{service}_model'] = model
                form.addRow(self.tr("Model"), model)
            elif service == 'Microsoft Azure':
                endpoint = QtWidgets.QLineEdit()
                self.credential_widgets[f'{service}_endpoint'] = endpoint
                form.addRow(self.tr("Endpoint URL"), endpoint)
            self.pages.addWidget(page)
        self.provider_combo.currentIndexChanged.connect(self.pages.setCurrentIndex)
        layout.addStretch()
