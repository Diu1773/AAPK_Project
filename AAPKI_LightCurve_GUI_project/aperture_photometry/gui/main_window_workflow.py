"""
Main Window
Step-by-step workflow with popup windows for each step
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QGroupBox, QFrame, QMessageBox, QFileDialog,
    QMenuBar, QMenu, QAction, QApplication, QLineEdit, QTextEdit,
    QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QEvent
from PyQt5.QtGui import QFont, QColor, QPalette
from pathlib import Path
from typing import Optional, List

from ..config import Parameters
from ..core import InstrumentConfig
from ..core.file_manager import FileManager
from ..core.project_state import ProjectState
from ..utils.step_paths import step9_dir


class StepButton(QPushButton):
    """Step button with status indication"""

    def __init__(self, step_number: int, step_name: str, parent=None):
        super().__init__(parent)

        self.step_number = step_number
        self.step_name = step_name
        self.completed = False
        self.accessible = False

        self.setText(f"Step {step_number}: {step_name}")
        self.setMinimumHeight(50)
        self.setMinimumWidth(300)

        self.update_appearance()

    def set_completed(self, completed: bool):
        """Set completion status"""
        self.completed = completed
        self.update_appearance()

    def set_accessible(self, accessible: bool):
        """Set accessibility status"""
        self.accessible = accessible
        self.update_appearance()

    def update_appearance(self):
        """Update button appearance based on status"""
        if self.completed:
            # Green background for completed
            self.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-size: 14px;
                    font-weight: bold;
                    border: 2px solid #45a049;
                    border-radius: 5px;
                    text-align: left;
                    padding: 10px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            self.setText(f"✓ Step {self.step_number}: {self.step_name}")
            self.setEnabled(True)  # Can re-visit completed steps
        elif self.accessible:
            # Blue background for accessible but not completed
            self.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    font-size: 14px;
                    font-weight: bold;
                    border: 2px solid #1976D2;
                    border-radius: 5px;
                    text-align: left;
                    padding: 10px;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
            """)
            self.setText(f"○ Step {self.step_number}: {self.step_name}")
            self.setEnabled(True)  # Accessible, can click
        else:
            # Gray background for LOCKED (not accessible)
            self.setStyleSheet("""
                QPushButton {
                    background-color: #E0E0E0;
                    color: #999999;
                    font-size: 14px;
                    border: 2px solid #CCCCCC;
                    border-radius: 5px;
                    text-align: left;
                    padding: 10px;
                }
                QPushButton:disabled {
                    background-color: #E0E0E0;
                    color: #999999;
                }
            """)
            self.setText(f"🔒 Step {self.step_number}: {self.step_name} (Locked)")
            self.setEnabled(False)  # COMPLETELY DISABLED


class ShortcutRouter(QObject):
    def __init__(self, main_window: "MainWindowWorkflow"):
        super().__init__(main_window)
        self.main_window = main_window

    @staticmethod
    def _is_text_input(widget) -> bool:
        if isinstance(widget, (QLineEdit, QTextEdit, QPlainTextEdit)):
            return True
        if isinstance(widget, QComboBox) and widget.isEditable():
            return True
        return False

    def eventFilter(self, obj, event):
        if event.type() != QEvent.KeyPress:
            return False
        if self._is_text_input(QApplication.focusWidget()):
            return False
        key = event.key()
        if key not in (Qt.Key_Period, Qt.Key_BracketLeft, Qt.Key_BracketRight):
            return False

        target = self.main_window.current_step_window
        if target is None:
            target = QApplication.activeWindow()
        if target is None:
            return False

        if key == Qt.Key_Period and hasattr(target, "cycle_filter"):
            target.cycle_filter()
            return True
        if key == Qt.Key_BracketLeft:
            if hasattr(target, "navigate_frame"):
                target.navigate_frame(-1)
                return True
            if hasattr(target, "step_frame"):
                target.step_frame(-1)
                return True
        if key == Qt.Key_BracketRight:
            if hasattr(target, "navigate_frame"):
                target.navigate_frame(1)
                return True
            if hasattr(target, "step_frame"):
                target.step_frame(1)
                return True
        return False


class MainWindowWorkflow(QMainWindow):
    """
    Main Window
    Displays step buttons and manages workflow
    """

    # Signals
    step_requested = pyqtSignal(int)  # Step index requested
    log_message = pyqtSignal(str)

    def __init__(self, param_file: Optional[str] = None):
        """
        Initialize main window

        Args:
            param_file: Path to parameter file
        """
        super().__init__()

        # Load parameters
        try:
            if param_file:
                self.params = Parameters(param_file)
            else:
                default_param = Path("parameters.toml")
                if not default_param.exists():
                    param_file, _ = QFileDialog.getOpenFileName(
                        self, "Select Parameter File", str(Path.cwd()),
                        "TOML Files (*.toml);;All Files (*.*)"
                    )
                    if not param_file:
                        raise RuntimeError("No parameter file selected")
                self.params = Parameters(param_file or default_param)

            # Initialize components
            self.instrument = InstrumentConfig(self.params)
            self.file_manager = FileManager(self.params)

            # Project state stored in project folder (not data folder)
            # This keeps each application's state separate
            project_root = Path(__file__).parent.parent.parent
            state_dir = project_root / ".state"
            state_dir.mkdir(exist_ok=True)
            self.project_state = ProjectState(state_dir)
            self._bootstrap_file_selection_state()

        except Exception as e:
            QMessageBox.critical(
                self, "Initialization Error",
                f"Failed to load parameters:\n{str(e)}"
            )
            raise

        # Step definitions (updated workflow)
        self.step_names = [
            "File Selection",
            "Image Crop",
            "Sky Preview & QC",
            "Source Detection",
            "WCS Plate Solving",
            "Reference Build",
            "Star ID Matching",
            "Target/Comparison Selection",
            "Forced Photometry",
            "Aperture Overlay",
            "Light Curve Builder",
            "Detrend & Night Merge",
        ]

        # Step buttons
        self.step_buttons: List[StepButton] = []

        # Current open step window
        self.current_step_window = None

        # Setup UI
        self.setup_ui()
        self.setup_menu()
        self.update_step_buttons()

        # Initial message
        self.append_log("AAPKI Light Curve Toolkit initialized")
        self.append_log(f"Project: {self.project_state.state['project_name']}")

        self._shortcut_router = ShortcutRouter(self)
        QApplication.instance().installEventFilter(self._shortcut_router)

    def _bootstrap_file_selection_state(self) -> None:
        """Restore multi-night/file selection context for downstream steps."""
        state_data = self.project_state.get_step_data("file_selection")
        if not state_data:
            return

        data_dir = state_data.get("data_dir")
        if data_dir:
            self.params.P.data_dir = Path(data_dir)
            self.params.P.result_dir = self.params.P.data_dir / "result"
            self.params.P.result_dir.mkdir(parents=True, exist_ok=True)
            self.params.P.cache_dir = self.params.P.result_dir / "cache"
            self.params.P.cache_dir.mkdir(parents=True, exist_ok=True)

        prefix = state_data.get("filename_prefix")
        if prefix:
            self.params.P.filename_prefix = prefix

        ref_frame = state_data.get("reference_frame")
        if ref_frame:
            self.file_manager.ref_filename = ref_frame

        multi_night = bool(state_data.get("multi_night"))
        root_dir = state_data.get("root_dir") or data_dir
        night_dirs = [Path(p) for p in state_data.get("night_dirs", []) if p]
        if multi_night and night_dirs:
            root_path = Path(root_dir) if root_dir else self.params.P.data_dir
            self.file_manager.set_multi_night_dirs(root_path, night_dirs)
        else:
            self.file_manager.clear_multi_night_dirs()

    def setup_ui(self):
        """Setup user interface"""
        self.setWindowTitle("AAPKI Light Curve GUI")
        self.setMinimumSize(800, 700)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # === Title ===
        title = QLabel("Aperture Photometry Toolkit")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("KNUEMAO Observatory - Step-by-Step Workflow")
        subtitle.setFont(QFont("Arial", 10))
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        # === Instrument Settings Button ===
        settings_layout = QHBoxLayout()
        settings_layout.addStretch()

        btn_settings = QPushButton("⚙ Instrument Settings")
        btn_settings.setFont(QFont("Arial", 11, QFont.Bold))
        btn_settings.setMinimumHeight(40)
        btn_settings.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border: 2px solid #7B1FA2;
                border-radius: 5px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        btn_settings.clicked.connect(self.open_settings)
        settings_layout.addWidget(btn_settings)

        settings_layout.addStretch()
        layout.addLayout(settings_layout)

        # === Progress Summary ===
        progress_group = QGroupBox("Workflow Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_label = QLabel(f"Progress: 0/{len(self.step_names)} steps completed")
        self.progress_label.setFont(QFont("Arial", 10, QFont.Bold))
        progress_layout.addWidget(self.progress_label)

        layout.addWidget(progress_group)

        # === Step Buttons ===
        steps_group = QGroupBox("Processing Steps")
        steps_layout = QVBoxLayout(steps_group)

        for i, step_name in enumerate(self.step_names):
            btn = StepButton(i + 1, step_name)
            btn.clicked.connect(lambda checked, idx=i: self.open_step(idx))
            self.step_buttons.append(btn)
            steps_layout.addWidget(btn)

        layout.addWidget(steps_group)

        # === Action Buttons ===
        action_layout = QHBoxLayout()

        btn_resume = QPushButton("Resume Next Step")
        btn_resume.setFont(QFont("Arial", 11, QFont.Bold))
        btn_resume.setMinimumHeight(40)
        btn_resume.clicked.connect(self.resume_next_step)
        action_layout.addWidget(btn_resume)

        btn_reset = QPushButton("Reset Progress")
        btn_reset.setMinimumHeight(40)
        btn_reset.clicked.connect(self.reset_progress)
        action_layout.addWidget(btn_reset)

        layout.addLayout(action_layout)

        # === Activity Log ===
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout(log_group)

        from PyQt5.QtWidgets import QTextEdit
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setFont(QFont("Courier", 8))
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_group)

        # === Status Bar ===
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

    def setup_menu(self):
        """Setup menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        action_save = QAction("&Save Project State", self)
        action_save.setShortcut("Ctrl+S")
        action_save.triggered.connect(self.save_project_state)
        file_menu.addAction(action_save)

        action_export = QAction("&Export Summary...", self)
        action_export.triggered.connect(self.export_summary)
        file_menu.addAction(action_export)

        file_menu.addSeparator()

        action_exit = QAction("E&xit", self)
        action_exit.setShortcut("Ctrl+Q")
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        action_params = QAction("View &Parameters", self)
        action_params.triggered.connect(self.show_parameters)
        tools_menu.addAction(action_params)

        tools_menu.addSeparator()

        action_qa_full = QAction("QA Reports", self)
        action_qa_full.setShortcut("Ctrl+R")
        action_qa_full.triggered.connect(self.open_qa_report)
        tools_menu.addAction(action_qa_full)

        # IRAF tool (integrated photometry + comparison)
        action_iraf = QAction("IRAF/DAOPHOT Tool", self)
        action_iraf.setShortcut("Ctrl+I")
        action_iraf.triggered.connect(self.open_iraf_tool)
        tools_menu.addAction(action_iraf)

        # Extinction & Zeropoint Tool
        action_extinction = QAction("Extinction && Zeropoint Tool", self)
        action_extinction.setShortcut("Ctrl+E")
        action_extinction.triggered.connect(self.open_extinction_tool)
        tools_menu.addAction(action_extinction)

        # Airmass Header Debug Tool
        action_airmass = QAction("Airmass Header Debug", self)
        action_airmass.triggered.connect(self.open_airmass_debug_tool)
        tools_menu.addAction(action_airmass)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        action_about = QAction("&About", self)
        action_about.triggered.connect(self.show_about)
        help_menu.addAction(action_about)

    def update_step_buttons(self):
        """Update step button states based on project state"""
        completed_count = len(self.project_state.state["completed_steps"])
        self.progress_label.setText(f"Progress: {completed_count}/{len(self.step_names)} steps completed")

        for i, btn in enumerate(self.step_buttons):
            completed = self.project_state.is_step_completed(i)
            accessible = self.project_state.is_step_accessible(i)

            btn.set_completed(completed)
            btn.set_accessible(accessible)
            btn.setEnabled(accessible)

    def open_step(self, step_index: int):
        """
        Open step window

        Args:
            step_index: Index of step to open
        """
        if not self.project_state.is_step_accessible(step_index):
            QMessageBox.warning(
                self, "Step Not Accessible",
                f"Please complete Step {step_index} first."
            )
            return

        self.project_state.set_current_step(step_index)
        self.append_log(f"Opening Step {step_index + 1}: {self.step_names[step_index]}")

        # Close previous step window if open
        if self.current_step_window:
            self.current_step_window.close()

        # Open appropriate step window
        if step_index == 0:
            from .workflow.step1_file_selection_window import FileSelectionWindow
            self.current_step_window = FileSelectionWindow(
                self.params, self.file_manager, self.project_state, self
            )
        elif step_index == 1:
            from .workflow.step2_crop_selector import CropSelectorWindow
            self.current_step_window = CropSelectorWindow(
                self.params, self.file_manager, self.project_state, self
            )
        elif step_index == 2:
            from .workflow.step3_sky_preview import SkyPreviewWindow
            self.current_step_window = SkyPreviewWindow(
                self.params, self.file_manager, self.project_state, self
            )
        elif step_index == 3:
            from .workflow.step4_source_detection import SourceDetectionWindow
            self.current_step_window = SourceDetectionWindow(
                self.params, self.file_manager, self.project_state, self
            )
        elif step_index == 4:
            from .workflow.step5_wcs_plate_solving import WcsPlateSolvingWindow
            self.current_step_window = WcsPlateSolvingWindow(
                self.params, self.file_manager, self.project_state, self
            )
        elif step_index == 5:
            from .workflow.step6_ref_build import RefBuildWindow
            self.current_step_window = RefBuildWindow(
                self.params, self.file_manager, self.project_state, self
            )
        elif step_index == 6:
            from .workflow.step7_star_id_matching import StarIdMatchingWindow
            self.current_step_window = StarIdMatchingWindow(
                self.params, self.file_manager, self.project_state, self
            )
        elif step_index == 7:
            from .workflow.step8_master_id_editor import MasterIdEditorWindow
            self.current_step_window = MasterIdEditorWindow(
                self.params, self.file_manager, self.project_state, self
            )
        elif step_index == 8:
            from .workflow.step9_forced_photometry import ForcedPhotometryWindow
            self.current_step_window = ForcedPhotometryWindow(
                self.params, self.file_manager, self.project_state, self
            )
        elif step_index == 9:
            from .workflow.step10_aperture_overlay import ApertureOverlayWindow
            self.current_step_window = ApertureOverlayWindow(
                self.params, self.file_manager, self.project_state, self
            )
        elif step_index == 10:
            from .workflow.step11_light_curve_builder import LightCurveBuilderWindow
            self.current_step_window = LightCurveBuilderWindow(
                self.params, self.file_manager, self.project_state, self
            )
        elif step_index == 11:
            from .workflow.step12_detrend_merge import DetrendNightMergeWindow
            self.current_step_window = DetrendNightMergeWindow(
                self.params, self.file_manager, self.project_state, self
            )
        else:
            QMessageBox.information(self, "Step Not Implemented",
                                    f"Step {step_index + 1} is not yet implemented.")
            return

        self.current_step_window.show()

    def on_step_completed(self, step_index: int):
        """
        Called when a step is completed

        Args:
            step_index: Index of completed step
        """
        self.project_state.mark_step_completed(step_index)
        self.update_step_buttons()
        self.append_log(f"✓ Step {step_index + 1} completed: {self.step_names[step_index]}")

    def resume_next_step(self):
        """Resume from next incomplete step"""
        next_step = self.project_state.get_next_incomplete_step()
        if next_step is not None:
            # Check if step is accessible
            if not self.project_state.is_step_accessible(next_step):
                QMessageBox.warning(
                    self, "Step Not Accessible",
                    f"Please complete previous steps first.\n"
                    f"Next available step: Step {next_step + 1}"
                )
                return
            self.open_step(next_step)
        else:
            QMessageBox.information(
                self, "All Steps Complete",
                "All workflow steps have been completed!"
            )

    def reset_progress(self):
        """Reset all progress"""
        reply = QMessageBox.question(
            self, "Reset Progress",
            "Are you sure you want to reset all progress?\n"
            "This will clear completion status but keep your data files.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.project_state.reset()
            self.update_step_buttons()
            self.append_log("Progress reset")

    def save_project_state(self):
        """Save project state"""
        self.project_state.save()
        self.append_log("Project state saved")
        QMessageBox.information(self, "Saved", "Project state saved successfully.")

    def export_summary(self):
        """Export progress summary"""
        summary = self.project_state.export_summary()

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Summary",
            str(self.params.P.result_dir / "project_summary.txt"),
            "Text Files (*.txt)"
        )

        if file_path:
            Path(file_path).write_text(summary, encoding='utf-8')
            self.append_log(f"Summary exported to {file_path}")

    def show_parameters(self):
        """Show parameter summary"""
        self.params.print_summary()
        QMessageBox.information(
            self, "Parameters",
            "Parameter summary printed to console.\nCheck the terminal output."
        )

    def open_settings(self):
        """Open instrument settings dialog (fully editable)"""
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                     QLineEdit, QDialogButtonBox, QGroupBox, QFormLayout)

        dialog = QDialog(self)
        dialog.setWindowTitle("Instrument Settings")
        dialog.setMinimumWidth(650)

        layout = QVBoxLayout(dialog)

        # Title
        title = QLabel("Instrument Configuration")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)

        # === Telescope Settings (Editable) ===
        tel_group = QGroupBox("Telescope (Editable)")
        tel_layout = QFormLayout(tel_group)

        tel_name_edit = QLineEdit(self.instrument.telescope_name)
        tel_layout.addRow("Name:", tel_name_edit)

        tel_aperture_edit = QLineEdit(str(self.instrument.aperture_mm))
        tel_aperture_edit.setMaximumWidth(150)
        tel_layout.addRow("Aperture (mm):", tel_aperture_edit)

        tel_focal_edit = QLineEdit(str(self.instrument.focal_length_mm))
        tel_focal_edit.setMaximumWidth(150)
        tel_layout.addRow("Focal Length (mm):", tel_focal_edit)

        # Focal Ratio - Auto-calculated label
        tel_ratio_label = QLabel(f"f/{self.instrument.focal_ratio:.2f} (auto-calculated)")
        tel_ratio_label.setStyleSheet("QLabel { color: blue; font-weight: bold; }")
        tel_layout.addRow("Focal Ratio:", tel_ratio_label)

        layout.addWidget(tel_group)

        # === Camera Settings (Editable) ===
        cam_group = QGroupBox("Camera (Editable)")
        cam_layout = QFormLayout(cam_group)

        cam_name_edit = QLineEdit(self.instrument.camera_name)
        cam_layout.addRow("Name:", cam_name_edit)

        cam_pixsize_edit = QLineEdit(str(self.instrument.pix_size_um))
        cam_pixsize_edit.setMaximumWidth(150)
        cam_layout.addRow("Pixel Size (μm):", cam_pixsize_edit)

        cam_nx_edit = QLineEdit(str(self.instrument.sensor_nx_1x))
        cam_nx_edit.setMaximumWidth(150)
        cam_layout.addRow("Sensor Width (px):", cam_nx_edit)

        cam_ny_edit = QLineEdit(str(self.instrument.sensor_ny_1x))
        cam_ny_edit.setMaximumWidth(150)
        cam_layout.addRow("Sensor Height (px):", cam_ny_edit)

        cam_binning_edit = QLineEdit(str(self.instrument.binning))
        cam_binning_edit.setMaximumWidth(150)
        cam_layout.addRow("Binning:", cam_binning_edit)

        layout.addWidget(cam_group)

        # === Camera Parameters (Editable) ===
        params_group = QGroupBox("Camera Parameters (Editable)")
        params_layout = QFormLayout(params_group)

        # Gain (e-/ADU)
        gain_edit = QLineEdit(str(self.params.P.gain_e_per_adu))
        gain_edit.setMaximumWidth(150)
        params_layout.addRow("Gain (e-/ADU):", gain_edit)

        # Read Noise (e-)
        rdnoise_edit = QLineEdit(str(self.params.P.rdnoise_e))
        rdnoise_edit.setMaximumWidth(150)
        params_layout.addRow("Read Noise (e-):", rdnoise_edit)

        # Saturation (ADU)
        saturation_edit = QLineEdit(str(self.params.P.saturation_adu))
        saturation_edit.setMaximumWidth(150)
        params_layout.addRow("Saturation (ADU):", saturation_edit)

        layout.addWidget(params_group)

        # === Parallel Settings (Global) ===
        parallel_group = QGroupBox("Parallel Processing (Global)")
        parallel_layout = QFormLayout(parallel_group)

        parallel_workers_spin = QSpinBox()
        parallel_workers_spin.setRange(0, 16)
        parallel_workers_spin.setValue(int(getattr(self.params.P, "max_workers", 0)))
        parallel_workers_spin.setToolTip("0 = auto (use ~75% of CPU cores)")
        parallel_layout.addRow("Max Workers (0=auto):", parallel_workers_spin)

        layout.addWidget(parallel_group)

        # === Observatory Location (Editable) ===
        site_group = QGroupBox("Observatory Location (Editable)")
        site_layout = QFormLayout(site_group)

        site_lat_edit = QLineEdit(str(getattr(self.params.P, "site_lat_deg", 0.0)))
        site_lat_edit.setMaximumWidth(150)
        site_layout.addRow("Latitude (deg):", site_lat_edit)

        site_lon_edit = QLineEdit(str(getattr(self.params.P, "site_lon_deg", 0.0)))
        site_lon_edit.setMaximumWidth(150)
        site_layout.addRow("Longitude (deg):", site_lon_edit)

        site_alt_edit = QLineEdit(str(getattr(self.params.P, "site_alt_m", 0.0)))
        site_alt_edit.setMaximumWidth(150)
        site_layout.addRow("Altitude (m):", site_alt_edit)

        site_tz_edit = QLineEdit(str(getattr(self.params.P, "site_tz_offset_hours", 0.0)))
        site_tz_edit.setMaximumWidth(150)
        site_layout.addRow("UTC Offset (hours):", site_tz_edit)

        layout.addWidget(site_group)

        # Note
        note = QLabel("<i>Note: Changes will be saved to core/instrument.py and parameters.toml</i>")
        note.setStyleSheet("color: gray;")
        layout.addWidget(note)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            # Save changes
            try:
                # Read all values
                new_tel_name = tel_name_edit.text().strip()
                new_aperture = float(tel_aperture_edit.text())
                new_focal = float(tel_focal_edit.text())
                new_cam_name = cam_name_edit.text().strip()
                new_pixsize = float(cam_pixsize_edit.text())
                new_nx = int(cam_nx_edit.text())
                new_ny = int(cam_ny_edit.text())
                new_binning = int(cam_binning_edit.text())
                new_gain = float(gain_edit.text())
                new_rdnoise = float(rdnoise_edit.text())
                new_saturation = float(saturation_edit.text())
                new_site_lat = float(site_lat_edit.text())
                new_site_lon = float(site_lon_edit.text())
                new_site_alt = float(site_alt_edit.text())
                new_site_tz = float(site_tz_edit.text())
                new_parallel_workers = int(parallel_workers_spin.value())

                # Update instrument in memory
                self.instrument.telescope_name = new_tel_name
                self.instrument.aperture_mm = new_aperture
                self.instrument.focal_length_mm = new_focal
                self.instrument.focal_ratio = new_focal / new_aperture
                self.instrument.camera_name = new_cam_name
                self.instrument.pix_size_um = new_pixsize
                self.instrument.sensor_nx_1x = new_nx
                self.instrument.sensor_ny_1x = new_ny
                self.instrument.binning = new_binning

                # Update parameters in memory
                self.params.P.gain_e_per_adu = new_gain
                self.params.P.rdnoise_e = new_rdnoise
                self.params.P.saturation_adu = new_saturation
                self.params.P.binning_default = new_binning
                self.params.P.site_lat_deg = new_site_lat
                self.params.P.site_lon_deg = new_site_lon
                self.params.P.site_alt_m = new_site_alt
                self.params.P.site_tz_offset_hours = new_site_tz
                self.params.P.max_workers = new_parallel_workers

                # Save to parameter file using save_toml (saves all parameters)
                if not self.params.save_toml():
                    # Fallback to legacy method if save_toml fails
                    self._update_parameter_file({
                        "gain_e_per_adu": new_gain,
                        "rdnoise_e": new_rdnoise,
                        "saturation_adu": new_saturation,
                        "binning_default": new_binning,
                        "site_lat_deg": new_site_lat,
                        "site_lon_deg": new_site_lon,
                        "site_alt_m": new_site_alt,
                        "site_tz_offset_hours": new_site_tz,
                        "max_workers": new_parallel_workers,
                    })

                # Save instrument settings to instrument.py
                self._update_instrument_file({
                    "telescope_name": new_tel_name,
                    "aperture_mm": new_aperture,
                    "focal_length_mm": new_focal,
                    "camera_name": new_cam_name,
                    "pix_size_um": new_pixsize,
                    "sensor_nx_1x": new_nx,
                    "sensor_ny_1x": new_ny,
                })

                QMessageBox.information(
                    self, "Settings Saved",
                    "Instrument settings have been updated and saved."
                )
                self.append_log("Instrument settings updated")

            except ValueError as e:
                QMessageBox.warning(
                    self, "Invalid Input",
                    f"Please enter valid numeric values.\n{str(e)}"
                )

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About Aperture Photometry Toolkit",
            "<h2>Aperture Photometry Toolkit</h2>"
            "<p><b>Workflow System</b></p>"
            "<p>KNUEMAO Observatory</p>"
            "<p>CDK500 + Moravian C3-61000</p>"
            "<p>Version 1.0.0-alpha</p>"
        )

    def append_log(self, message: str):
        """Append message to activity log"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def open_qa_report(self, tab: int = 0):
        """
        Open QA Report window for publication-quality validation

        Args:
            tab: Initial tab to show (0=Error Model, 1=Centroid, 2=Frame, 3=Background, 4=Publication)
        """
        # Check if photometry data exists
        photometry_index = step9_dir(self.params.P.result_dir) / "photometry_index.csv"
        if not photometry_index.exists():
            photometry_index = self.params.P.result_dir / "photometry_index.csv"
        if not photometry_index.exists():
            QMessageBox.warning(
                self, "No Data",
                "Photometry data not found.\n"
                "Please complete the Forced Photometry step first."
            )
            return

        from .workflow.qa_report_window import QAReportWindow

        self.qa_window = QAReportWindow(
            self.params,
            self.params.P.result_dir,
            parent=None
        )
        self.qa_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        # Switch to requested tab
        if hasattr(self.qa_window, 'tabs') and tab >= 0:
            self.qa_window.tabs.setCurrentIndex(min(tab, self.qa_window.tabs.count() - 1))

        self.qa_window.show()
        self.qa_window.raise_()
        self.qa_window.activateWindow()

        self.append_log("Opened QA Report window")

    def open_iraf_tool(self):
        """Open integrated IRAF/DAOPHOT tool window"""
        from .workflow.iraf_photometry_window import IRAFPhotometryWindow

        self.iraf_window = IRAFPhotometryWindow(
            self.params,
            self.params.P.data_dir,
            self.params.P.result_dir,
            self.project_state,
            parent=None
        )
        self.iraf_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.iraf_window.show()
        self.iraf_window.raise_()
        self.iraf_window.activateWindow()
        self.append_log("Opened IRAF/DAOPHOT Tool")

    def open_extinction_tool(self):
        """Open Extinction & Zeropoint fitting tool window"""
        from .workflow.extinction_zeropoint_tool import ExtinctionZeropointToolWindow

        self.extinction_window = ExtinctionZeropointToolWindow(
            self.params,
            self.project_state,
            parent=None
        )
        self.extinction_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.extinction_window.show()
        self.extinction_window.raise_()
        self.extinction_window.activateWindow()
        self.append_log("Opened Extinction & Zeropoint Tool")

    def open_airmass_debug_tool(self):
        """Open airmass header debug tool window"""
        from .workflow.airmass_header_debug_tool import AirmassHeaderDebugToolWindow

        self.airmass_debug_window = AirmassHeaderDebugToolWindow(
            self.params,
            self.project_state,
            parent=None,
            file_manager=self.file_manager
        )
        self.airmass_debug_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.airmass_debug_window.show()
        self.airmass_debug_window.raise_()
        self.airmass_debug_window.activateWindow()
        self.append_log("Opened Airmass Header Debug Tool")

    def _update_parameter_file(self, updates: dict):
        """
        Update specific parameters in parameters.toml

        Args:
            updates: Dictionary of parameter_name: value pairs to update
        """
        param_file = self.params.param_file
        try:  # Python 3.11+
            import tomllib  # type: ignore
        except Exception:
            import tomli as tomllib  # type: ignore
        try:
            import tomli_w  # type: ignore
        except Exception:
            tomli_w = None

        if param_file.exists():
            with param_file.open("rb") as f:
                data = tomllib.load(f)
        else:
            data = {}

        def set_path(path, value):
            cur = data
            for key in path[:-1]:
                if key not in cur or not isinstance(cur[key], dict):
                    cur[key] = {}
                cur = cur[key]
            cur[path[-1]] = value

        if "gain_e_per_adu" in updates:
            set_path(("instrument", "gain_e_per_adu"), updates["gain_e_per_adu"])
        if "rdnoise_e" in updates:
            set_path(("instrument", "rdnoise_e"), updates["rdnoise_e"])
        if "saturation_adu" in updates:
            set_path(("instrument", "saturation_adu"), updates["saturation_adu"])
        if "binning_default" in updates:
            set_path(("instrument", "binning"), updates["binning_default"])
        if "max_workers" in updates:
            set_path(("parallel", "max_workers"), updates["max_workers"])

        if "site_lat_deg" in updates:
            set_path(("site", "lat_deg"), updates["site_lat_deg"])
        if "site_lon_deg" in updates:
            set_path(("site", "lon_deg"), updates["site_lon_deg"])
        if "site_alt_m" in updates:
            set_path(("site", "alt_m"), updates["site_alt_m"])
        if "site_tz_offset_hours" in updates:
            set_path(("site", "tz_offset_hours"), updates["site_tz_offset_hours"])

        if tomli_w is None:
            raise RuntimeError("tomli_w is required to write parameters.toml")
        with param_file.open("wb") as f:
            tomli_w.dump(data, f)

    def _update_instrument_file(self, updates: dict):
        """
        Update instrument settings in instrument.py

        Args:
            updates: Dictionary of setting_name: value pairs to update
        """
        instrument_file = Path(__file__).parent.parent / "core" / "instrument.py"

        if not instrument_file.exists():
            return

        # Read all lines
        lines = instrument_file.read_text(encoding="utf-8").splitlines()
        new_lines = []

        # Mapping of setting names to their attribute assignments in __init__
        for line in lines:
            modified = False

            # Update telescope_name
            if "self.telescope_name = " in line and "telescope_name" in updates:
                indent = len(line) - len(line.lstrip())
                new_lines.append(f"{' ' * indent}self.telescope_name = \"{updates['telescope_name']}\"")
                modified = True

            # Update aperture_mm
            elif "self.aperture_mm = " in line and "aperture_mm" in updates:
                indent = len(line) - len(line.lstrip())
                new_lines.append(f"{' ' * indent}self.aperture_mm = {updates['aperture_mm']}")
                modified = True

            # Update focal_length_mm
            elif "self.focal_length_mm = " in line and "focal_length_mm" in updates:
                indent = len(line) - len(line.lstrip())
                new_lines.append(f"{' ' * indent}self.focal_length_mm = {updates['focal_length_mm']}")
                modified = True

            # Update camera_name
            elif "self.camera_name = " in line and "camera_name" in updates:
                indent = len(line) - len(line.lstrip())
                new_lines.append(f"{' ' * indent}self.camera_name = \"{updates['camera_name']}\"")
                modified = True

            # Update pix_size_um
            elif "self.pix_size_um = " in line and "pix_size_um" in updates:
                indent = len(line) - len(line.lstrip())
                new_lines.append(f"{' ' * indent}self.pix_size_um = {updates['pix_size_um']}")
                modified = True

            # Update sensor_nx_1x
            elif "self.sensor_nx_1x = " in line and "sensor_nx_1x" in updates:
                indent = len(line) - len(line.lstrip())
                new_lines.append(f"{' ' * indent}self.sensor_nx_1x = {updates['sensor_nx_1x']}")
                modified = True

            # Update sensor_ny_1x
            elif "self.sensor_ny_1x = " in line and "sensor_ny_1x" in updates:
                indent = len(line) - len(line.lstrip())
                new_lines.append(f"{' ' * indent}self.sensor_ny_1x = {updates['sensor_ny_1x']}")
                modified = True

            if not modified:
                new_lines.append(line)

        # Write back to file
        instrument_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    def closeEvent(self, event):
        """Handle window close event"""
        # Save state on close
        self.project_state.save()

        # Close any open step windows
        if self.current_step_window:
            self.current_step_window.close()

        event.accept()
