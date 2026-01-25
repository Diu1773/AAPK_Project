"""
Step 1: File Selection Window
Popup window with Previous/Next navigation
"""

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QGroupBox, QLineEdit,
    QFileDialog, QMessageBox, QHeaderView, QCheckBox, QComboBox,
    QDialog, QDialogButtonBox, QFormLayout, QSpinBox
)
from PyQt5.QtCore import Qt
from pathlib import Path
import re
from typing import Optional

from .step_window_base import StepWindowBase
try:  # Python 3.11+
    import tomllib  # type: ignore
except Exception:  # Python 3.10 and earlier
    import tomli as tomllib  # type: ignore
try:
    import tomli_w  # type: ignore
except Exception:
    tomli_w = None


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
        self.root_dir = Path(params.P.data_dir)
        self.last_data_dir = Path(params.P.data_dir)
        self.night_entries = []
        self.selected_night_dirs = []
        self._night_table_loading = False
        self.night_parse_mode = str(getattr(params.P, "night_parse_mode", "regex"))
        self.night_parse_regex = str(getattr(params.P, "night_parse_regex", r".*_(\d{8})"))
        self.night_parse_split_delim = str(getattr(params.P, "night_parse_split_delim", "_"))
        self.night_parse_split_index = int(getattr(params.P, "night_parse_split_index", -1))
        self.night_parse_last_digits = int(getattr(params.P, "night_parse_last_digits", 8))
        self.night_parse_include_unmatched = bool(getattr(params.P, "night_parse_include_unmatched", False))

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
        dir_layout = QVBoxLayout(dir_group)

        root_row = QHBoxLayout()
        root_row.addWidget(QLabel("Root:"))

        self.dir_edit = QLineEdit(str(self.root_dir))
        self.dir_edit.setReadOnly(True)
        root_row.addWidget(self.dir_edit)

        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_directory)
        root_row.addWidget(btn_browse)
        dir_layout.addLayout(root_row)

        night_row = QHBoxLayout()
        self.multi_night_check = QCheckBox("멀티나잇(하위 폴더) 사용")
        self.multi_night_check.toggled.connect(self.toggle_multi_night)
        night_row.addWidget(self.multi_night_check)

        self.btn_night_params = QPushButton("파라매터")
        self.btn_night_params.setStyleSheet(
            "QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 6px 12px; }"
        )
        self.btn_night_params.clicked.connect(self.open_step1_parameters_dialog)
        self.btn_night_params.setEnabled(True)
        night_row.addWidget(self.btn_night_params)

        self.btn_scan_nights = QPushButton("폴더 스캔")
        self.btn_scan_nights.clicked.connect(self.scan_subfolders)
        self.btn_scan_nights.setEnabled(False)
        night_row.addWidget(self.btn_scan_nights)

        self.btn_select_all = QPushButton("전체 선택")
        self.btn_select_all.clicked.connect(self.select_all_nights)
        self.btn_select_all.setEnabled(False)
        night_row.addWidget(self.btn_select_all)

        self.btn_clear_select = QPushButton("선택 해제")
        self.btn_clear_select.clicked.connect(self.clear_night_selection)
        self.btn_clear_select.setEnabled(False)
        night_row.addWidget(self.btn_clear_select)

        night_row.addStretch()
        dir_layout.addLayout(night_row)

        self.night_table = QTableWidget()
        self.night_table.setColumnCount(4)
        self.night_table.setHorizontalHeaderLabels(["Use", "Date", "Folder", "Files"])
        self.night_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.night_table.horizontalHeader().setStretchLastSection(True)
        self.night_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.night_table.itemChanged.connect(self.on_night_item_changed)
        self.night_table.setEnabled(False)
        dir_layout.addWidget(self.night_table)

        active_row = QHBoxLayout()
        active_row.addWidget(QLabel("현재 적용:"))
        self.active_dir_label = QLabel(str(self.params.P.data_dir))
        self.active_dir_label.setStyleSheet("QLabel { font-weight: bold; color: #1565C0; }")
        active_row.addWidget(self.active_dir_label)
        self.selected_nights_label = QLabel("선택된 폴더: 0")
        self.selected_nights_label.setStyleSheet("QLabel { font-weight: bold; color: #455A64; }")
        active_row.addWidget(self.selected_nights_label)
        active_row.addStretch()
        dir_layout.addLayout(active_row)

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
        self.header_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.header_table.horizontalHeader().setStretchLastSection(True)
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
            str(self.root_dir)
        )
        if dir_path:
            self.root_dir = Path(dir_path)
            self._set_data_dir(self.root_dir)
            self.dir_edit.setText(dir_path)
            self._update_active_labels()
            self.file_manager.filenames = []
            self.file_manager.df_headers = None
            self.file_manager.ref_filename = None
            self.selected_night_dirs = []
            self.file_manager.clear_multi_night_dirs()
            if self.multi_night_check.isChecked():
                self.scan_subfolders()
            # Don't auto-scan - user must click "Rescan Files" button

    def rescan_files(self):
        """Rescan files with current prefix"""
        self.params.P.filename_prefix = self.prefix_edit.text()
        self._persist_param_file(io_updates={"filename_prefix": self.params.P.filename_prefix})

        try:
            if self.multi_night_check.isChecked():
                if not self.selected_night_dirs:
                    QMessageBox.warning(
                        self, "No Night Selected",
                        "멀티나잇 사용 중입니다. 폴더를 선택해주세요."
                    )
                    return
                self._set_data_dir(self.root_dir)
                self.file_manager.set_multi_night_dirs(self.root_dir, self.selected_night_dirs)
            else:
                self.file_manager.clear_multi_night_dirs()
            self.load_files()
            self.update_navigation_buttons()
        except Exception as e:
            QMessageBox.warning(
                self, "Scan Error",
                f"Failed to scan files:\n{str(e)}"
            )

    def toggle_multi_night(self, checked: bool):
        self.night_table.setEnabled(checked)
        self.btn_scan_nights.setEnabled(checked)
        self.btn_select_all.setEnabled(checked)
        self.btn_clear_select.setEnabled(checked)
        self.btn_night_params.setEnabled(True)

        if checked:
            self._set_data_dir(self.root_dir)
            self.scan_subfolders()
            self._update_selected_nights()
        else:
            self.selected_night_dirs = []
            self.file_manager.clear_multi_night_dirs()
            self._set_data_dir(self.root_dir)
            self._update_active_labels()
            self.file_manager.filenames = []
            self.file_manager.df_headers = None
            self.file_manager.ref_filename = None

    def scan_subfolders(self):
        root = Path(self.root_dir)
        self.night_entries = []

        self._night_table_loading = True
        self.night_table.blockSignals(True)
        self.night_table.setRowCount(0)

        if not root.exists():
            self.night_table.blockSignals(False)
            self._night_table_loading = False
            return

        prefix = self.prefix_edit.text().strip()
        prefix_lower = prefix.lower()
        suffixes = (".fit", ".fits", ".fit.fz", ".fits.fz")
        selected_set = {str(p) for p in self.selected_night_dirs}

        for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
            fits_files = []
            for fpath in sub.iterdir():
                if not fpath.is_file():
                    continue
                name = fpath.name
                lower = name.lower()
                if prefix_lower and not lower.startswith(prefix_lower):
                    continue
                if not lower.endswith(suffixes):
                    continue
                fits_files.append(fpath)
            if not fits_files:
                continue
            date_key = self._parse_night_label(sub.name)
            if date_key is None and not self.night_parse_include_unmatched:
                continue
            self.night_entries.append({
                "name": sub.name,
                "path": sub,
                "date_key": date_key or "",
                "count": len(fits_files),
            })

        self.night_entries.sort(key=lambda e: (e["date_key"] or "99999999", e["name"]))

        for entry in self.night_entries:
            row = self.night_table.rowCount()
            self.night_table.insertRow(row)

            use_item = QTableWidgetItem()
            use_item.setFlags(use_item.flags() | Qt.ItemIsUserCheckable)
            use_item.setCheckState(Qt.Checked if str(entry["path"]) in selected_set else Qt.Unchecked)
            use_item.setData(Qt.UserRole, str(entry["path"]))
            self.night_table.setItem(row, 0, use_item)
            self.night_table.setItem(row, 1, QTableWidgetItem(entry["date_key"] or "-"))
            self.night_table.setItem(row, 2, QTableWidgetItem(entry["name"]))
            self.night_table.setItem(row, 3, QTableWidgetItem(str(entry["count"])))

        self.night_table.blockSignals(False)
        self._night_table_loading = False
        self._update_selected_nights()

    def select_all_nights(self):
        if not self.multi_night_check.isChecked():
            return
        self._night_table_loading = True
        self.night_table.blockSignals(True)
        for row in range(self.night_table.rowCount()):
            item = self.night_table.item(row, 0)
            if item is not None:
                item.setCheckState(Qt.Checked)
        self.night_table.blockSignals(False)
        self._night_table_loading = False
        self._update_selected_nights()

    def clear_night_selection(self):
        if not self.multi_night_check.isChecked():
            return
        self._night_table_loading = True
        self.night_table.blockSignals(True)
        for row in range(self.night_table.rowCount()):
            item = self.night_table.item(row, 0)
            if item is not None:
                item.setCheckState(Qt.Unchecked)
        self.night_table.blockSignals(False)
        self._night_table_loading = False
        self._update_selected_nights()

    def on_night_item_changed(self, item: QTableWidgetItem):
        if self._night_table_loading:
            return
        if item.column() != 0:
            return
        self._update_selected_nights()

    def _update_selected_nights(self):
        if not self.multi_night_check.isChecked():
            self.selected_night_dirs = []
            self.file_manager.clear_multi_night_dirs()
            self._update_active_labels()
            return

        selected = []
        for row in range(self.night_table.rowCount()):
            item = self.night_table.item(row, 0)
            if item and item.checkState() == Qt.Checked:
                path_str = item.data(Qt.UserRole)
                if path_str:
                    selected.append(Path(path_str))
        self.selected_night_dirs = selected
        self.file_manager.set_multi_night_dirs(self.root_dir, self.selected_night_dirs)
        self._update_active_labels()

    def _update_active_labels(self):
        self.active_dir_label.setText(str(self.root_dir))
        self.selected_nights_label.setText(f"선택된 폴더: {len(self.selected_night_dirs)}")

    def _parse_night_label(self, folder_name: str) -> Optional[str]:
        mode = str(self.night_parse_mode or "regex").strip().lower()
        if mode == "split":
            delim = self.night_parse_split_delim
            parts = folder_name.split(delim) if delim else [folder_name]
            idx = self.night_parse_split_index
            if not parts:
                return None
            if idx < 0:
                idx = len(parts) + idx
            if idx < 0 or idx >= len(parts):
                return None
            return parts[idx]
        if mode == "last_digits":
            n_digits = max(1, int(self.night_parse_last_digits))
            m = re.search(rf"(\\d{{{n_digits}}})$", folder_name)
            return m.group(1) if m else None

        try:
            pattern = self.night_parse_regex
            m = re.search(pattern, folder_name)
        except re.error:
            return None
        if not m:
            return None
        if m.groupdict().get("date"):
            return m.group("date")
        if m.groups():
            return m.group(1)
        return m.group(0)

    def open_step1_parameters_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Step 1 파라매터")
        layout = QVBoxLayout(dialog)

        night_group = QGroupBox("폴더 규칙")
        form = QFormLayout(night_group)

        mode_combo = QComboBox()
        mode_combo.addItem("Regex", "regex")
        mode_combo.addItem("Split", "split")
        mode_combo.addItem("Last Digits", "last_digits")
        mode_combo.setCurrentIndex(max(0, mode_combo.findData(self.night_parse_mode)))
        form.addRow("Parse Mode:", mode_combo)

        regex_edit = QLineEdit(self.night_parse_regex)
        form.addRow("Regex Pattern:", regex_edit)

        split_delim = QLineEdit(self.night_parse_split_delim)
        split_idx = QSpinBox()
        split_idx.setRange(-20, 20)
        split_idx.setValue(self.night_parse_split_index)
        split_row = QHBoxLayout()
        split_row.addWidget(split_delim)
        split_row.addWidget(QLabel("Index:"))
        split_row.addWidget(split_idx)
        form.addRow("Split (delim):", split_row)

        last_digits = QSpinBox()
        last_digits.setRange(1, 32)
        last_digits.setValue(self.night_parse_last_digits)
        form.addRow("Last Digits:", last_digits)

        include_unmatched = QCheckBox("규칙에 안 맞는 폴더도 표시")
        include_unmatched.setChecked(self.night_parse_include_unmatched)
        form.addRow("", include_unmatched)

        def _sync_fields():
            mode = mode_combo.currentData()
            regex_edit.setEnabled(mode == "regex")
            split_delim.setEnabled(mode == "split")
            split_idx.setEnabled(mode == "split")
            last_digits.setEnabled(mode == "last_digits")

        mode_combo.currentIndexChanged.connect(_sync_fields)
        _sync_fields()

        layout.addWidget(night_group)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        def on_accept():
            self.night_parse_mode = str(mode_combo.currentData())
            self.night_parse_regex = regex_edit.text().strip() or r".*_(\d{8})"
            self.night_parse_split_delim = split_delim.text()
            self.night_parse_split_index = int(split_idx.value())
            self.night_parse_last_digits = int(last_digits.value())
            self.night_parse_include_unmatched = bool(include_unmatched.isChecked())

            self.params.P.night_parse_mode = self.night_parse_mode
            self.params.P.night_parse_regex = self.night_parse_regex
            self.params.P.night_parse_split_delim = self.night_parse_split_delim
            self.params.P.night_parse_split_index = self.night_parse_split_index
            self.params.P.night_parse_last_digits = self.night_parse_last_digits
            self.params.P.night_parse_include_unmatched = self.night_parse_include_unmatched

            self._persist_param_file(io_updates={
                "night_parse_mode": self.night_parse_mode,
                "night_parse_regex": self.night_parse_regex,
                "night_parse_split_delim": self.night_parse_split_delim,
                "night_parse_split_index": self.night_parse_split_index,
                "night_parse_last_digits": self.night_parse_last_digits,
                "night_parse_include_unmatched": self.night_parse_include_unmatched,
            })

            if self.multi_night_check.isChecked():
                self.scan_subfolders()
            dialog.accept()

        buttons.accepted.connect(on_accept)
        buttons.rejected.connect(dialog.reject)
        dialog.exec_()

    def _set_data_dir(self, path: Path):
        self.params.P.data_dir = Path(path)

        # Always set result_dir = data_dir/result when data_dir changes
        self.params.P.result_dir = self.params.P.data_dir / "result"
        self.params.P.result_dir.mkdir(parents=True, exist_ok=True)
        self.params.P.cache_dir = self.params.P.result_dir / "cache"
        self.params.P.cache_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(self, "file_count_label"):
            self.file_count_label.setText("Files: 0")
        self.last_data_dir = self.params.P.data_dir

        # Save to TOML immediately
        self._persist_param_file(
            io_updates={
                "data_dir": str(self.params.P.data_dir),
                "result_dir": str(self.params.P.result_dir),
                "cache_dir": str(self.params.P.cache_dir.name),
            }
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

            self.params.P.target_name = name
            self.params.P.target_ra_deg = ra_deg
            self.params.P.target_dec_deg = dec_deg

            # Save to TOML (primary source of truth)
            self._persist_param_file(target_updates={
                "name": name,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
            })

        except Exception as e:
            QMessageBox.warning(self, "SIMBAD Error", str(e))

    def _persist_param_file(self, io_updates=None, target_updates=None):
        """Persist key IO/target selections to parameters.toml."""
        if tomli_w is None:
            return
        param_path = Path(getattr(self.params, "param_file", "parameters.toml"))
        if not param_path.exists():
            return
        try:
            with param_path.open("rb") as f:
                data = tomllib.load(f)
        except Exception:
            return

        io_updates = io_updates or {}
        target_updates = target_updates or {}

        if io_updates:
            io_block = data.get("io", {}) if isinstance(data.get("io", {}), dict) else {}
            io_block.update(io_updates)
            data["io"] = io_block

        if target_updates:
            tgt_block = data.get("target", {}) if isinstance(data.get("target", {}), dict) else {}
            tgt_block.update(target_updates)
            data["target"] = tgt_block

        try:
            with param_path.open("wb") as f:
                tomli_w.dump(data, f)
        except Exception:
            return

    def load_files(self):
        """Load files and headers"""
        try:
            # Scan for files
            filenames = self.file_manager.scan_files()

            # Read headers
            df_headers = self.file_manager.read_headers()

            self.file_count_label.setText(f"Files: {len(filenames)}")

            # Populate table
            self._populate_header_table(df_headers)

            # Don't auto-select reference - this should be done after detection/ID matching
            # User can manually select if needed using "Auto-Select Reference" or "Use Selected Row"

            # Update navigation buttons
            self.update_navigation_buttons()

        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to load files:\n{str(e)}"
            )

    def _populate_header_table(self, df_headers):
        self.header_table.setRowCount(len(df_headers))
        for i, row in df_headers.iterrows():
            self.header_table.setItem(i, 0, QTableWidgetItem(str(row["Filename"])))
            self.header_table.setItem(i, 1, QTableWidgetItem(str(row["DATE-OBS"])))
            self.header_table.setItem(i, 2, QTableWidgetItem(str(row["FILTER"])))
            self.header_table.setItem(i, 3, QTableWidgetItem(str(row["EXPTIME"])))
            self.header_table.setItem(i, 4, QTableWidgetItem(str(row["AIRMASS"])))
            self.header_table.setItem(i, 5, QTableWidgetItem(str(row["IMAGETYP"])))

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
            "root_dir": str(self.root_dir),
            "multi_night": bool(self.multi_night_check.isChecked()),
            "night_dirs": [str(p) for p in self.selected_night_dirs],
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
                self._set_data_dir(Path(state_data["data_dir"]))
                self._update_active_labels()

            if "root_dir" in state_data:
                self.root_dir = Path(state_data["root_dir"])
                self.dir_edit.setText(str(self.root_dir))
            elif "data_dir" in state_data:
                self.root_dir = Path(state_data["data_dir"])
                self.dir_edit.setText(str(self.root_dir))

            if state_data.get("multi_night"):
                self.multi_night_check.setChecked(True)
                self.selected_night_dirs = [
                    Path(p) for p in state_data.get("night_dirs", []) if p
                ]
                self.scan_subfolders()
            else:
                self.selected_night_dirs = []
                self.file_manager.clear_multi_night_dirs()

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

        self._update_active_labels()

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
