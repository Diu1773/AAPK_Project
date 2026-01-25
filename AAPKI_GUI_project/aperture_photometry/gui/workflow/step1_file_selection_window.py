"""
Step 1: File Selection Window
Popup window with Previous/Next navigation
"""

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QGroupBox, QLineEdit,
    QFileDialog, QMessageBox, QHeaderView
)
from PyQt5.QtCore import Qt
from pathlib import Path

from .step_window_base import StepWindowBase


class FileSelectionWindow(StepWindowBase):
    """Step 1: File Selection and Header Reading"""

    def __init__(self, params, file_manager, project_state, main_window):
        """
        Initialize file selection window

        Args:
            params: Parameters object
            file_manager: FileManager instance
            project_state: ProjectState object
            main_window: Main window reference
        """
        self.file_manager = file_manager

        # Initialize base class
        super().__init__(
            step_index=0,
            step_name="File Selection",
            params=params,
            project_state=project_state,
            main_window=main_window
        )

        # Setup step-specific UI
        self.setup_step_ui()

        # Restore state AFTER UI is created
        self.restore_state()

    def setup_step_ui(self):
        """Setup step-specific UI components"""

        # === Directory Selection ===
        dir_group = QGroupBox("Data Directory")
        dir_layout = QHBoxLayout(dir_group)

        dir_layout.addWidget(QLabel("Directory:"))

        self.dir_edit = QLineEdit(str(self.params.P.data_dir))
        self.dir_edit.setReadOnly(True)
        dir_layout.addWidget(self.dir_edit)

        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_directory)
        dir_layout.addWidget(btn_browse)

        self.content_layout.addWidget(dir_group)

        # === File Prefix Filter ===
        filter_group = QGroupBox("File Filter")
        filter_layout = QHBoxLayout(filter_group)

        filter_layout.addWidget(QLabel("Filename Prefix:"))

        self.prefix_edit = QLineEdit(self.params.P.filename_prefix)
        self.prefix_edit.setMaximumWidth(150)
        filter_layout.addWidget(self.prefix_edit)

        btn_rescan = QPushButton("Rescan Files")
        btn_rescan.clicked.connect(self.rescan_files)
        filter_layout.addWidget(btn_rescan)

        filter_layout.addStretch()

        self.file_count_label = QLabel("Files: 0")
        filter_layout.addWidget(self.file_count_label)

        self.content_layout.addWidget(filter_group)

        # === SIMBAD Target ===
        target_group = QGroupBox("SIMBAD Target")
        target_layout = QHBoxLayout(target_group)

        target_layout.addWidget(QLabel("Target Name:"))
        self.target_edit = QLineEdit()
        self.target_edit.setPlaceholderText("e.g., M31 or WASP-12")
        self.target_edit.setMaximumWidth(200)
        target_layout.addWidget(self.target_edit)

        btn_resolve = QPushButton("Resolve SIMBAD")
        btn_resolve.clicked.connect(self.resolve_target)
        target_layout.addWidget(btn_resolve)

        self.target_result = QLabel("(not resolved)")
        self.target_result.setStyleSheet("QLabel { font-weight: bold; color: #4CAF50; }")
        target_layout.addWidget(self.target_result)

        target_layout.addStretch()
        self.content_layout.addWidget(target_group)

        # === Header Table ===
        table_group = QGroupBox("FITS Headers")
        table_layout = QVBoxLayout(table_group)

        self.header_table = QTableWidget()
        self.header_table.setColumnCount(6)
        self.header_table.setHorizontalHeaderLabels([
            "Filename", "DATE-OBS", "FILTER", "EXPTIME", "AIRMASS", "IMAGETYP"
        ])
        self.header_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.header_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.header_table.setEditTriggers(QTableWidget.NoEditTriggers)

        table_layout.addWidget(self.header_table)

        self.content_layout.addWidget(table_group)

        # === Reference Frame Selection ===
        ref_group = QGroupBox("Reference Frame")
        ref_layout = QHBoxLayout(ref_group)

        ref_layout.addWidget(QLabel("Selected Reference:"))

        self.ref_label = QLabel("(not selected)")
        self.ref_label.setStyleSheet("QLabel { font-weight: bold; color: blue; }")
        ref_layout.addWidget(self.ref_label)

        ref_layout.addStretch()

        btn_auto_ref = QPushButton("Auto-Select Reference")
        btn_auto_ref.clicked.connect(self.auto_select_reference)
        ref_layout.addWidget(btn_auto_ref)

        btn_use_selected = QPushButton("Use Selected Row")
        btn_use_selected.clicked.connect(self.use_selected_as_reference)
        ref_layout.addWidget(btn_use_selected)

        self.content_layout.addWidget(ref_group)

        # Don't auto-load files - user must click "Rescan Files" button
        # (Files will be loaded from restore_state() if previously scanned)

    def validate_step(self) -> bool:
        """Validate if step can be completed"""
        # Only check if files are loaded
        # Reference frame selection will be done after detection/ID matching in later steps
        has_files = len(self.file_manager.filenames) > 0

        return has_files

    def browse_directory(self):
        """Browse for data directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Data Directory",
            str(self.params.P.data_dir)
        )
        if dir_path:
            data_path = Path(dir_path)
            self.params.P.data_dir = data_path
            self.dir_edit.setText(dir_path)

            # Update result_dir to match new data_dir
            self.params.P.result_dir = data_path / "result"
            self.params.P.result_dir.mkdir(parents=True, exist_ok=True)
            self.params.P.cache_dir = self.params.P.result_dir / "cache"
            self.params.P.cache_dir.mkdir(parents=True, exist_ok=True)

            # Save to TOML
            if hasattr(self.params, 'save_toml'):
                self.params.save_toml()

            # Clear file manager state
            self.file_manager.filenames = []
            self.file_manager.df_headers = None
            self.file_manager.ref_filename = None
            # Don't auto-scan - user must click "Rescan Files" button

    def rescan_files(self):
        """Rescan files with current prefix"""
        self.params.P.filename_prefix = self.prefix_edit.text()

        try:
            self.load_files()
            self.update_navigation_buttons()
        except Exception as e:
            QMessageBox.warning(
                self, "Scan Error",
                f"Failed to scan files:\n{str(e)}"
            )

    def resolve_target(self):
        """Resolve target coordinates via SIMBAD"""
        name = self.target_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Target Missing", "Please enter a target name.")
            return

        try:
            inst = self.main_window.instrument
            inst.targets_resolved = []
            inst.primary_target = None
            inst.primary_coord = None
            inst.resolve_targets([name])
            if inst.primary_coord is None:
                self.target_result.setText("(not resolved)")
                QMessageBox.warning(self, "SIMBAD", f"Target not found: {name}")
                return

            ra_deg = float(inst.primary_coord.ra.deg)
            dec_deg = float(inst.primary_coord.dec.deg)
            ra_hms = inst.primary_coord.ra.to_string(unit="hour", sep=":", precision=2)
            dec_dms = inst.primary_coord.dec.to_string(unit="deg", sep=":", precision=1, alwayssign=True)
            self.target_result.setText(f"{ra_hms}, {dec_dms}")

            # Save to params.P and TOML (primary source of truth)
            self.params.P.target_name = name
            self.params.P.target_ra_deg = ra_deg
            self.params.P.target_dec_deg = dec_deg
            if hasattr(self.params, 'save_toml'):
                self.params.save_toml()

        except Exception as e:
            QMessageBox.warning(self, "SIMBAD Error", str(e))

    def load_files(self):
        """Load files and headers"""
        try:
            # Scan for files
            filenames = self.file_manager.scan_files()
            self.file_count_label.setText(f"Files: {len(filenames)}")

            # Read headers
            df_headers = self.file_manager.read_headers()

            # Populate table
            self.header_table.setRowCount(len(df_headers))

            for i, row in df_headers.iterrows():
                self.header_table.setItem(i, 0, QTableWidgetItem(str(row["Filename"])))
                self.header_table.setItem(i, 1, QTableWidgetItem(str(row["DATE-OBS"])))
                self.header_table.setItem(i, 2, QTableWidgetItem(str(row["FILTER"])))
                self.header_table.setItem(i, 3, QTableWidgetItem(str(row["EXPTIME"])))
                self.header_table.setItem(i, 4, QTableWidgetItem(str(row["AIRMASS"])))
                self.header_table.setItem(i, 5, QTableWidgetItem(str(row["IMAGETYP"])))

            # Don't auto-select reference - this should be done after detection/ID matching
            # User can manually select if needed using "Auto-Select Reference" or "Use Selected Row"

            # Update navigation buttons
            self.update_navigation_buttons()

        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to load files:\n{str(e)}"
            )

    def auto_select_reference(self):
        """Automatically select reference frame"""
        try:
            ref_filename = self.file_manager.select_reference_frame()
            self.ref_label.setText(ref_filename)

            # Highlight reference row in table
            for i in range(self.header_table.rowCount()):
                item = self.header_table.item(i, 0)
                if item and item.text() == ref_filename:
                    self.header_table.selectRow(i)
                    break

            # Save state to persist reference selection
            self.save_state()

            self.update_navigation_buttons()

        except Exception as e:
            QMessageBox.warning(
                self, "Reference Selection Error",
                f"Failed to select reference frame:\n{str(e)}"
            )

    def use_selected_as_reference(self):
        """Use currently selected row as reference frame"""
        current_row = self.header_table.currentRow()
        if current_row >= 0:
            item = self.header_table.item(current_row, 0)
            if item:
                ref_filename = item.text()
                self.file_manager.ref_filename = ref_filename
                self.ref_label.setText(ref_filename)

                # Save state to persist reference selection
                self.save_state()

                self.update_navigation_buttons()
        else:
            QMessageBox.information(
                self, "No Selection",
                "Please select a row in the table first."
            )

    def save_state(self):
        """Save step state to project"""
        state_data = {
            "data_dir": str(self.params.P.data_dir),
            "filename_prefix": self.params.P.filename_prefix,
            "file_count": len(self.file_manager.filenames),
            "reference_frame": self.file_manager.ref_filename,
        }

        self.project_state.store_step_data("file_selection", state_data)

        # Also update ref label if reference is set
        if self.file_manager.ref_filename:
            self.ref_label.setText(self.file_manager.ref_filename)

    def restore_state(self):
        """Restore step state from project"""
        state_data = self.project_state.get_step_data("file_selection")

        if state_data:
            # Restore directory and prefix
            if "data_dir" in state_data:
                self.params.P.data_dir = Path(state_data["data_dir"])
                self.dir_edit.setText(str(state_data["data_dir"]))

            if "filename_prefix" in state_data:
                self.params.P.filename_prefix = state_data["filename_prefix"]
                self.prefix_edit.setText(state_data["filename_prefix"])

            # Restore reference frame
            if "reference_frame" in state_data and state_data["reference_frame"]:
                self.file_manager.ref_filename = state_data["reference_frame"]
                self.ref_label.setText(state_data["reference_frame"])

            # Reload files
            try:
                self.load_files()

                # Re-highlight reference in table if it exists
                if self.file_manager.ref_filename:
                    for i in range(self.header_table.rowCount()):
                        item = self.header_table.item(i, 0)
                        if item and item.text() == self.file_manager.ref_filename:
                            self.header_table.selectRow(i)
                            break
            except:
                pass

        # Load target from params.P (which comes from TOML)
        name = getattr(self.params.P, "target_name", None)
        ra = getattr(self.params.P, "target_ra_deg", None)
        dec = getattr(self.params.P, "target_dec_deg", None)
        if name:
            self.target_edit.setText(str(name))
        if ra is not None and dec is not None:
            try:
                self.target_result.setText(f"{float(ra):.6f}, {float(dec):.6f}")
            except (ValueError, TypeError):
                pass
