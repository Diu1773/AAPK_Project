"""Light curve mode tabs."""

from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QFormLayout,
)


class BaseLightCurveTab(QWidget):
    """Shared UI for light curve modes."""

    mode_title = "Light Curve"
    help_text = "Configure inputs and generate a light curve."

    def __init__(self):
        super().__init__()
        self.result_dir = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel(self.mode_title)
        header.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(header)

        info = QLabel(self.help_text)
        info.setWordWrap(True)
        layout.addWidget(info)

        form_group = QGroupBox("Inputs")
        form_layout = QFormLayout(form_group)

        self.result_dir_edit = QLineEdit()
        self.result_dir_edit.setReadOnly(True)
        self.result_dir_edit.setPlaceholderText("Select result directory with photometry outputs.")

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._select_result_dir)

        result_row = QHBoxLayout()
        result_row.addWidget(self.result_dir_edit)
        result_row.addWidget(browse_btn)

        result_row_widget = QWidget()
        result_row_widget.setLayout(result_row)

        form_layout.addRow("Result Dir:", result_row_widget)

        self.target_id_edit = QLineEdit()
        self.target_id_edit.setPlaceholderText("Target ID or name")
        form_layout.addRow("Target:", self.target_id_edit)

        layout.addWidget(form_group)

        action_row = QHBoxLayout()
        action_row.addStretch()
        self.run_btn = QPushButton("Build Light Curve")
        self.run_btn.clicked.connect(self._run_light_curve)
        action_row.addWidget(self.run_btn)
        layout.addLayout(action_row)

        layout.addStretch()

    def _select_result_dir(self):
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Result Directory",
            str(Path.cwd()),
        )
        if selected:
            self.result_dir = Path(selected)
            self.result_dir_edit.setText(str(self.result_dir))

    def _run_light_curve(self):
        if not self.result_dir:
            QMessageBox.warning(self, "Missing Input", "Select a result directory first.")
            return
        target = self.target_id_edit.text().strip()
        if not target:
            QMessageBox.warning(self, "Missing Input", "Enter a target ID or name.")
            return
        QMessageBox.information(
            self,
            "Not Implemented",
            f"{self.mode_title} light curve builder is a placeholder.\n"
            "Hook this tab to the light curve analysis module.",
        )


class VariableStarTab(BaseLightCurveTab):
    mode_title = "Variable Star"
    help_text = "Generate variable star light curves (absolute/differential)."


class EclipseTab(BaseLightCurveTab):
    mode_title = "Eclipsing"
    help_text = "Generate eclipse light curves with phase folding support."


class AsteroidTab(BaseLightCurveTab):
    mode_title = "Asteroid"
    help_text = "Generate asteroid light curves with motion-aware extraction."
