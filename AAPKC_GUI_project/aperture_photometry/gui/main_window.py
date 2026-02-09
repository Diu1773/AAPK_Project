"""
Main window for Aperture Photometry GUI
Workflow interface
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QListWidget,
    QListWidgetItem, QSplitter, QTextEdit, QProgressBar,
    QMenuBar, QMenu, QAction, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from pathlib import Path
from typing import Optional

from ..config import Parameters
from ..core import InstrumentConfig
from ..core.file_manager import FileManager


class MainWindow(QMainWindow):
    """Main application window with workflow interface"""

    # Signals
    status_updated = pyqtSignal(str)
    log_message = pyqtSignal(str)

    def __init__(self, param_file: Optional[str] = None):
        """
        Initialize main window

        Args:
            param_file: Path to parameter file (default: parameters.toml)
        """
        super().__init__()

        # Load parameters
        try:
            if param_file:
                self.params = Parameters(param_file)
            else:
                # Try to find parameter file
                default_param = Path("parameters.toml")
                if not default_param.exists():
                    # Ask user to select parameter file
                    param_file, _ = QFileDialog.getOpenFileName(
                        self,
                        "Select Parameter File",
                        str(Path.cwd()),
                        "TOML Files (*.toml);;All Files (*.*)"
                    )
                    if not param_file:
                        raise RuntimeError("No parameter file selected")
                self.params = Parameters(param_file or default_param)

            # Initialize instrument config
            self.instrument = InstrumentConfig(self.params)
            self.file_manager = FileManager(self.params)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Initialization Error",
                f"Failed to load parameters:\n{str(e)}\n\n"
                "Please ensure parameters.toml exists in the working directory."
            )
            raise

        # Current project state
        self.current_project = None
        self.workflow_state = {
            "file_selection": False,
            "crop": False,
            "sky_preview": False,
            "detection": False,
            "star_catalog": False,
            "photometry": False,
            "analysis": False,
        }

        # Setup UI
        self.setup_ui()
        self.setup_menu()
        self.setup_connections()

        # Initial status
        self.update_status("Ready")

    def setup_ui(self):
        """Setup main user interface"""
        self.setWindowTitle("Aperture Photometry Toolkit - KNUEMAO")
        self.setMinimumSize(1200, 800)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # === Project Header ===
        project_group = QGroupBox("Current Project")
        project_layout = QHBoxLayout(project_group)

        self.project_label = QLabel("No project loaded")
        self.project_label.setFont(QFont("Arial", 10, QFont.Bold))
        project_layout.addWidget(self.project_label)

        project_layout.addStretch()

        btn_new_project = QPushButton("New Project")
        btn_edit_project = QPushButton("Edit")
        project_layout.addWidget(btn_new_project)
        project_layout.addWidget(btn_edit_project)

        layout.addWidget(project_group)

        # === Instrument Settings ===
        instrument_group = QGroupBox("Instrument Settings")
        instrument_layout = QVBoxLayout(instrument_group)

        telescope_info = QLabel(
            f"Telescope: {self.instrument.telescope_name} "
            f"({self.instrument.aperture_mm:.0f}mm, f/{self.instrument.focal_ratio:.2f})"
        )
        camera_info = QLabel(
            f"Camera: {self.instrument.camera_name} "
            f"({self.instrument.pix_size_um:.2f}μm, "
            f"{self.instrument.sensor_nx_1x}×{self.instrument.sensor_ny_1x})"
        )
        pixscale_info = QLabel(
            f"Pixel Scale: {self.instrument.pix_scale_bin:.3f} arcsec/px "
            f"(binning {self.instrument.binning}×{self.instrument.binning})"
        )

        instrument_layout.addWidget(telescope_info)
        instrument_layout.addWidget(camera_info)
        instrument_layout.addWidget(pixscale_info)

        btn_configure = QPushButton("Configure Hardware...")
        btn_configure.setEnabled(False)  # TODO: implement
        instrument_layout.addWidget(btn_configure)

        layout.addWidget(instrument_group)

        # === Workflow Progress ===
        workflow_group = QGroupBox("Workflow Progress")
        workflow_layout = QVBoxLayout(workflow_group)

        self.workflow_list = QListWidget()
        self.workflow_items = {}

        steps = [
            ("file_selection", "1. File Selection"),
            ("crop", "2. Image Crop"),
            ("sky_preview", "3. Sky Preview & QC"),
            ("detection", "4. Source Detection"),
            ("star_catalog", "5. Star Catalog Editor"),
            ("photometry", "6. Aperture Photometry"),
            ("analysis", "7. CMD & Analysis"),
        ]

        for key, label in steps:
            item = QListWidgetItem(f"⬜ {label}")
            item.setData(Qt.UserRole, key)
            self.workflow_list.addItem(item)
            self.workflow_items[key] = item

        self.workflow_list.itemDoubleClicked.connect(self.on_workflow_step_clicked)
        workflow_layout.addWidget(self.workflow_list)

        layout.addWidget(workflow_group)

        # === Action Buttons ===
        action_layout = QHBoxLayout()

        self.btn_start = QPushButton("Start New Analysis")
        self.btn_start.setStyleSheet("QPushButton { font-size: 14px; padding: 10px; }")
        self.btn_start.clicked.connect(self.start_new_analysis)

        self.btn_resume = QPushButton("Resume from Last Step")
        self.btn_resume.setStyleSheet("QPushButton { font-size: 14px; padding: 10px; }")
        self.btn_resume.setEnabled(False)

        action_layout.addWidget(self.btn_start)
        action_layout.addWidget(self.btn_resume)

        layout.addLayout(action_layout)

        # === Recent Activity Log ===
        log_group = QGroupBox("Recent Activity")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_group)

        # === Status Bar ===
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def setup_menu(self):
        """Setup menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        action_open = QAction("&Open Project...", self)
        action_open.setShortcut("Ctrl+O")
        file_menu.addAction(action_open)

        action_save = QAction("&Save Project", self)
        action_save.setShortcut("Ctrl+S")
        file_menu.addAction(action_save)

        file_menu.addSeparator()

        action_exit = QAction("E&xit", self)
        action_exit.setShortcut("Ctrl+Q")
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        action_params = QAction("&Parameters...", self)
        edit_menu.addAction(action_params)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        action_logs = QAction("View &Logs...", self)
        tools_menu.addAction(action_logs)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        action_about = QAction("&About", self)
        action_about.triggered.connect(self.show_about)
        help_menu.addAction(action_about)

    def setup_connections(self):
        """Setup signal-slot connections"""
        self.status_updated.connect(self.update_status)
        self.log_message.connect(self.append_log)

    def start_new_analysis(self):
        """Start new analysis workflow"""
        # Import here to avoid circular imports
        from .workflow.step1_file_selection import FileSelectionDialog

        dialog = FileSelectionDialog(self.params, self.file_manager, parent=self)
        if dialog.exec_():
            # File selection successful
            self.workflow_state["file_selection"] = True
            self.update_workflow_display()
            self.append_log("File selection completed")

            # TODO: Move to next step
            QMessageBox.information(
                self,
                "Step Complete",
                "File selection completed!\n\nNext steps will be implemented soon."
            )

    def on_workflow_step_clicked(self, item: QListWidgetItem):
        """Handle workflow step click"""
        step_key = item.data(Qt.UserRole)
        QMessageBox.information(
            self,
            "Workflow Step",
            f"Opening step: {item.text()}\n\n(Not yet implemented)"
        )

    def update_workflow_display(self):
        """Update workflow progress display"""
        for key, item in self.workflow_items.items():
            if self.workflow_state[key]:
                text = item.text()
                text = text.replace("⬜", "✓").replace("⏸", "✓")
                item.setText(text)

    def update_status(self, message: str):
        """Update status bar"""
        self.status_bar.showMessage(message)

    def append_log(self, message: str):
        """Append message to activity log"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Aperture Photometry Toolkit",
            "<h2>Aperture Photometry Toolkit</h2>"
            "<p>KNUEMAO Observatory</p>"
            "<p>CDK500 + Moravian C3-61000</p>"
            "<p>Version 1.0.0</p>"
        )

    def closeEvent(self, event):
        """Handle window close event"""
        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
