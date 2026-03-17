"""
Step 1: File Selection Window
Popup window with Previous/Next navigation
"""

import json
from collections import defaultdict
from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QGroupBox, QLineEdit,
    QFileDialog, QMessageBox, QHeaderView, QCheckBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt
from pathlib import Path
from typing import Optional

from .step_window_base import StepWindowBase
from ...utils.step_paths import step1_dir
try:  # Python 3.11+
    import tomllib  # type: ignore
except Exception:  # Python 3.10 and earlier
    import tomli as tomllib  # type: ignore
try:
    import tomli_w  # type: ignore
except Exception:
    tomli_w = None


# =============================================================================
# Module-level night classification helpers
# =============================================================================

def _classify_nights_by_jd_gap(
    records: list,
    gap_hours: float = 8.0,
) -> list:
    """Assign night_id to each file by JD gap.

    Args:
        records: list of dicts with keys "filename", "jd" (float|None),
                 "date_obs" (str), "filter" (str)
        gap_hours: hours gap that triggers a new night

    Returns:
        records with "night_id" key added (1-based int)
    """
    gap_days = gap_hours / 24.0
    valid = [(i, r) for i, r in enumerate(records) if r.get("jd") is not None]
    valid.sort(key=lambda x: x[1]["jd"])

    night_id = 1
    prev_jd = None
    id_map = {}  # index -> night_id

    for i, r in valid:
        if prev_jd is not None and (r["jd"] - prev_jd) > gap_days:
            night_id += 1
        id_map[i] = night_id
        prev_jd = r["jd"]

    # files with no JD get assigned to night_id=1 (or nearest, default 1)
    for i, r in enumerate(records):
        if i not in id_map:
            id_map[i] = 1

    result = []
    for i, r in enumerate(records):
        result.append({**r, "night_id": id_map[i]})
    return result


def _build_night_summary(records_with_night: list, tz_offset_hours: float = 0.0) -> list:
    """Build per-night summary list sorted by night_id.

    Returns:
        list of dicts with keys: night_id, label, date, time_range,
        file_count, filter_str, filenames
    """
    nights = defaultdict(lambda: {
        "jd_min": None, "jd_max": None,
        "files": [], "filters": defaultdict(int),
    })
    for r in records_with_night:
        n = r["night_id"]
        jd = r.get("jd")
        if jd is not None:
            if nights[n]["jd_min"] is None or jd < nights[n]["jd_min"]:
                nights[n]["jd_min"] = jd
            if nights[n]["jd_max"] is None or jd > nights[n]["jd_max"]:
                nights[n]["jd_max"] = jd
        nights[n]["files"].append(r["filename"])
        flt = str(r.get("filter", "") or "").strip()
        if flt:
            nights[n]["filters"][flt] += 1

    tz_days = float(tz_offset_hours) / 24.0

    def _jd_to_timestr(jd):
        if jd is None:
            return "?"
        try:
            from astropy.time import Time
            # Apply local timezone offset for display only
            t = Time(jd + tz_days, format="jd", scale="utc")
            return t.datetime.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return "?"

    result = []
    for nid in sorted(nights.keys()):
        n = nights[nid]
        first_str = _jd_to_timestr(n["jd_min"])
        last_str = _jd_to_timestr(n["jd_max"])
        date_label = first_str.split(" ")[0] if " " in first_str else first_str
        time_range = (
            f"{first_str.split(' ')[-1]} ~ {last_str.split(' ')[-1]}"
            if first_str != "?" else "?"
        )
        filter_str = "  ".join(
            f"{k}:{v}" for k, v in sorted(n["filters"].items())
        )
        result.append({
            "night_id": nid,
            "label": f"N{nid}",
            "date": date_label,
            "time_range": time_range,
            "file_count": len(n["files"]),
            "filter_str": filter_str or "-",
            "filenames": n["files"],
        })
    return result


# =============================================================================
# Step 1 window
# =============================================================================

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

        # Night classification state
        self._night_records: list = []       # records with night_id
        self._excluded_nights: set = set()   # set of int night_ids
        self._night_table_loading = False

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

    # -------------------------------------------------------------------------
    # UI setup
    # -------------------------------------------------------------------------

    def setup_step_ui(self):
        """Setup step-specific UI components"""

        # === Directory Selection ===
        dir_group = QGroupBox("Data Directory")
        dir_layout = QVBoxLayout(dir_group)

        # Root row: label + edit + browse
        root_row = QHBoxLayout()
        root_row.addWidget(QLabel("Root:"))

        self.dir_edit = QLineEdit(str(self.root_dir))
        self.dir_edit.setReadOnly(True)
        root_row.addWidget(self.dir_edit)

        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_directory)
        root_row.addWidget(btn_browse)
        dir_layout.addLayout(root_row)

        # Options row: checkbox + gap spinbox + rescan + file count
        options_row = QHBoxLayout()

        self.include_subfolders_check = QCheckBox("하위폴더 포함")
        options_row.addWidget(self.include_subfolders_check)

        options_row.addWidget(QLabel("Night gap:"))
        self.night_gap_spinbox = QDoubleSpinBox()
        self.night_gap_spinbox.setRange(1.0, 24.0)
        self.night_gap_spinbox.setSingleStep(0.5)
        self.night_gap_spinbox.setDecimals(1)
        self.night_gap_spinbox.setValue(
            float(getattr(self.params.P, "night_gap_hours", 8.0))
        )
        self.night_gap_spinbox.setMaximumWidth(70)
        options_row.addWidget(self.night_gap_spinbox)
        options_row.addWidget(QLabel("h"))

        btn_rescan = QPushButton("Rescan Files")
        btn_rescan.clicked.connect(self.rescan_files)
        options_row.addWidget(btn_rescan)

        options_row.addStretch()

        self.file_count_label = QLabel("Files: 0")
        options_row.addWidget(self.file_count_label)

        dir_layout.addLayout(options_row)

        # Night table header buttons
        night_btn_row = QHBoxLayout()
        btn_select_all_nights = QPushButton("전체 선택")
        btn_select_all_nights.clicked.connect(self._select_all_displayed_nights)
        night_btn_row.addWidget(btn_select_all_nights)

        btn_clear_nights = QPushButton("선택 해제")
        btn_clear_nights.clicked.connect(self._clear_all_displayed_nights)
        night_btn_row.addWidget(btn_clear_nights)

        night_btn_row.addStretch()
        dir_layout.addLayout(night_btn_row)

        # Night classification table
        self.night_table = QTableWidget()
        self.night_table.setColumnCount(6)
        self.night_table.setHorizontalHeaderLabels(
            ["Use", "Night", "관측일", "시간범위", "파일수", "필터"]
        )
        self.night_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.night_table.horizontalHeader().setStretchLastSection(True)
        self.night_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.night_table.itemChanged.connect(self._on_night_item_changed)
        self.night_table.setMaximumHeight(160)
        dir_layout.addWidget(self.night_table)

        self.content_layout.addWidget(dir_group)

        # === File Prefix Filter ===
        filter_group = QGroupBox("File Filter")
        filter_layout = QHBoxLayout(filter_group)

        filter_layout.addWidget(QLabel("Filename Prefix:"))

        self.prefix_edit = QLineEdit(self.params.P.filename_prefix)
        self.prefix_edit.setMaximumWidth(150)
        filter_layout.addWidget(self.prefix_edit)

        filter_layout.addStretch()
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
        self.header_table.setColumnCount(7)
        self.header_table.setHorizontalHeaderLabels([
            "Filename", "Night", "DATE-OBS", "FILTER", "EXPTIME", "AIRMASS", "IMAGETYP"
        ])
        self.header_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
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

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_step(self) -> bool:
        """Validate if step can be completed"""
        return len(self.file_manager.filenames) > 0

    # -------------------------------------------------------------------------
    # Directory / file scanning
    # -------------------------------------------------------------------------

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
            self.file_manager.filenames = []
            self.file_manager.df_headers = None
            self.file_manager.ref_filename = None
            self.file_manager.clear_multi_night_dirs()

    def rescan_files(self):
        """Rescan files then re-classify nights by JD gap."""
        self.params.P.filename_prefix = self.prefix_edit.text()
        self._persist_param_file(
            io_updates={"filename_prefix": self.params.P.filename_prefix}
        )

        # Persist gap to params and TOML
        gap = self.night_gap_spinbox.value()
        self.params.P.night_gap_hours = gap
        self._persist_param_file(io_updates={"night_gap_hours": gap})

        try:
            if self.include_subfolders_check.isChecked():
                self._setup_subdirectory_scan()
            else:
                self.file_manager.clear_multi_night_dirs()

            self.load_files()
            self.update_navigation_buttons()
        except Exception as e:
            QMessageBox.warning(
                self, "Scan Error",
                f"Failed to scan files:\n{str(e)}"
            )

    def _setup_subdirectory_scan(self):
        """When 'include subfolders' is checked, collect all immediate subdirs
        that contain matching FITS files and register them with file_manager."""
        root = Path(self.root_dir)
        if not root.exists():
            return

        prefix = self.prefix_edit.text().strip()
        prefix_lower = prefix.lower()
        suffixes = (".fit", ".fits", ".fit.fz", ".fits.fz")

        sub_dirs = []
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            has_fits = any(
                f.is_file()
                and f.name.lower().startswith(prefix_lower)
                and f.name.lower().endswith(suffixes)
                for f in sub.iterdir()
            )
            if has_fits:
                sub_dirs.append(sub)

        if sub_dirs:
            self.file_manager.set_multi_night_dirs(root, sub_dirs)
        else:
            # No FITS in subdirs — fall back to root
            self.file_manager.clear_multi_night_dirs()

    # -------------------------------------------------------------------------
    # File loading and classification
    # -------------------------------------------------------------------------

    def load_files(self):
        """Load files, read headers, then classify nights by JD gap."""
        try:
            filenames = self.file_manager.scan_files()
            df_headers = self.file_manager.read_headers()

            self.file_count_label.setText(f"Files: {len(filenames)}")

            self._classify_and_display(df_headers)
            self.update_navigation_buttons()

        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to load files:\n{str(e)}"
            )

    def _classify_and_display(self, df_headers):
        """Build records from df_headers, classify by JD gap, refresh tables."""
        import numpy as np

        records = []
        for _, row in df_headers.iterrows():
            filename = str(row["Filename"])
            date_obs = str(row.get("DATE-OBS", "") or "")
            flt = str(row.get("FILTER", "") or "")

            # Prefer JD column if already present and finite
            jd = None
            if "JD" in df_headers.columns:
                try:
                    jd_raw = float(row["JD"])
                    if np.isfinite(jd_raw):
                        jd = jd_raw
                except (TypeError, ValueError):
                    pass

            # Fall back: compute from DATE-OBS
            if jd is None and date_obs and date_obs not in ("N/A", "nan", ""):
                try:
                    from astropy.time import Time
                    jd = float(Time(date_obs.strip(), format="isot", scale="utc").jd)
                except Exception:
                    jd = None

            records.append({
                "filename": filename,
                "jd": jd,
                "date_obs": date_obs,
                "filter": flt,
            })

        gap = self.night_gap_spinbox.value()
        records_with_night = _classify_nights_by_jd_gap(records, gap)
        self._night_records = records_with_night

        # Store in file_manager
        self.file_manager.night_assignments = {
            r["filename"]: r["night_id"] for r in records_with_night
        }

        # Populate night summary table
        tz = float(getattr(self.params.P, "site_tz_offset_hours", 0.0))
        summaries = _build_night_summary(records_with_night, tz_offset_hours=tz)
        self._populate_night_table(summaries)

        # Populate header table with Night column
        self._populate_header_table(df_headers, records_with_night)

    def _populate_night_table(self, summaries: list):
        """Fill night_table from summary list, preserving exclusion state."""
        self._night_table_loading = True
        self.night_table.blockSignals(True)
        self.night_table.setRowCount(0)

        for s in summaries:
            nid = s["night_id"]
            row = self.night_table.rowCount()
            self.night_table.insertRow(row)

            use_item = QTableWidgetItem()
            use_item.setFlags(use_item.flags() | Qt.ItemIsUserCheckable)
            checked = Qt.Unchecked if nid in self._excluded_nights else Qt.Checked
            use_item.setCheckState(checked)
            use_item.setData(Qt.UserRole, nid)
            self.night_table.setItem(row, 0, use_item)
            self.night_table.setItem(row, 1, QTableWidgetItem(s["label"]))
            self.night_table.setItem(row, 2, QTableWidgetItem(s["date"]))
            self.night_table.setItem(row, 3, QTableWidgetItem(s["time_range"]))
            self.night_table.setItem(row, 4, QTableWidgetItem(str(s["file_count"])))
            self.night_table.setItem(row, 5, QTableWidgetItem(s["filter_str"]))

        self.night_table.blockSignals(False)
        self._night_table_loading = False

    def _populate_header_table(self, df_headers, records_with_night: list):
        """Fill header_table; column 1 is Night assignment."""
        night_map = {r["filename"]: r["night_id"] for r in records_with_night}

        self.header_table.setRowCount(len(df_headers))
        for i, row in df_headers.iterrows():
            fn = str(row["Filename"])
            nid = night_map.get(fn, "-")
            self.header_table.setItem(i, 0, QTableWidgetItem(fn))
            self.header_table.setItem(i, 1, QTableWidgetItem(f"N{nid}"))
            self.header_table.setItem(i, 2, QTableWidgetItem(str(row["DATE-OBS"])))
            self.header_table.setItem(i, 3, QTableWidgetItem(str(row["FILTER"])))
            self.header_table.setItem(i, 4, QTableWidgetItem(str(row["EXPTIME"])))
            self.header_table.setItem(i, 5, QTableWidgetItem(str(row["AIRMASS"])))
            self.header_table.setItem(i, 6, QTableWidgetItem(str(row["IMAGETYP"])))

    # -------------------------------------------------------------------------
    # Night table interaction
    # -------------------------------------------------------------------------

    def _on_night_item_changed(self, item: QTableWidgetItem):
        if self._night_table_loading:
            return
        if item.column() != 0:
            return
        self._update_excluded_nights()

    def _update_excluded_nights(self):
        """Recompute excluded_nights from checkbox states and push to file_manager."""
        excluded = set()
        for row in range(self.night_table.rowCount()):
            use_item = self.night_table.item(row, 0)
            if use_item and use_item.checkState() == Qt.Unchecked:
                nid = use_item.data(Qt.UserRole)
                if nid is not None:
                    excluded.add(int(nid))
        self._excluded_nights = excluded
        self.file_manager.excluded_nights = excluded

    def _select_all_displayed_nights(self):
        self._night_table_loading = True
        self.night_table.blockSignals(True)
        for row in range(self.night_table.rowCount()):
            item = self.night_table.item(row, 0)
            if item is not None:
                item.setCheckState(Qt.Checked)
        self.night_table.blockSignals(False)
        self._night_table_loading = False
        self._update_excluded_nights()

    def _clear_all_displayed_nights(self):
        self._night_table_loading = True
        self.night_table.blockSignals(True)
        for row in range(self.night_table.rowCount()):
            item = self.night_table.item(row, 0)
            if item is not None:
                item.setCheckState(Qt.Unchecked)
        self.night_table.blockSignals(False)
        self._night_table_loading = False
        self._update_excluded_nights()

    # -------------------------------------------------------------------------
    # Params / directory helpers
    # -------------------------------------------------------------------------

    def _set_data_dir(self, path: Path):
        self.params.P.data_dir = Path(path)

        self.params.P.result_dir = self.params.P.data_dir / "result"
        self.params.P.cache_dir = self.params.P.result_dir / "cache"
        try:
            self.params.P.result_dir.mkdir(parents=True, exist_ok=True)
            self.params.P.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # Drive not connected; dirs created when user picks a valid path

        if hasattr(self, "file_count_label"):
            self.file_count_label.setText("Files: 0")
        self.last_data_dir = self.params.P.data_dir

        self._persist_param_file(
            io_updates={
                "data_dir": str(self.params.P.data_dir),
                "result_dir": str(self.params.P.result_dir),
                "cache_dir": str(self.params.P.cache_dir.name),
            }
        )

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

    # -------------------------------------------------------------------------
    # Target resolution
    # -------------------------------------------------------------------------

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
            dec_dms = inst.primary_coord.dec.to_string(
                unit="deg", sep=":", precision=1, alwayssign=True
            )
            self.target_result.setText(f"{ra_hms}, {dec_dms}")

            self.params.P.target_name = name
            self.params.P.target_ra_deg = ra_deg
            self.params.P.target_dec_deg = dec_deg

            self._persist_param_file(target_updates={
                "name": name,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
            })

        except Exception as e:
            QMessageBox.warning(self, "SIMBAD Error", str(e))

    # -------------------------------------------------------------------------
    # Reference frame selection
    # -------------------------------------------------------------------------

    def auto_select_reference(self):
        """Automatically select reference frame"""
        try:
            ref_filename = self.file_manager.select_reference_frame()
            self.ref_label.setText(ref_filename)

            for i in range(self.header_table.rowCount()):
                item = self.header_table.item(i, 0)
                if item and item.text() == ref_filename:
                    self.header_table.selectRow(i)
                    break

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
                self.save_state()
                self.update_navigation_buttons()
        else:
            QMessageBox.information(
                self, "No Selection",
                "Please select a row in the table first."
            )

    # -------------------------------------------------------------------------
    # State persistence
    # -------------------------------------------------------------------------

    def save_state(self):
        """Save step state to project"""
        state_data = {
            "data_dir": str(self.params.P.data_dir),
            "root_dir": str(self.root_dir),
            "include_subfolders": bool(self.include_subfolders_check.isChecked()),
            "filename_prefix": self.params.P.filename_prefix,
            "file_count": len(self.file_manager.filenames),
            "reference_frame": self.file_manager.ref_filename,
            "night_gap_hours": self.night_gap_spinbox.value(),
            "excluded_nights": sorted(self._excluded_nights),
            "night_assignments": {
                k: v for k, v in self.file_manager.night_assignments.items()
            },
        }

        self.project_state.store_step_data("file_selection", state_data)

        # Also persist night_assignments to disk so downstream steps can load without file_manager
        try:
            s1_dir = step1_dir(self.params.P.result_dir)
            s1_dir.mkdir(parents=True, exist_ok=True)
            na_path = s1_dir / "night_assignments.json"
            na_data = {
                "night_assignments": {k: v for k, v in self.file_manager.night_assignments.items()},
                "excluded_nights": sorted(self._excluded_nights),
            }
            na_path.write_text(json.dumps(na_data, indent=2), encoding="utf-8")
        except Exception:
            pass

        if self.file_manager.ref_filename:
            self.ref_label.setText(self.file_manager.ref_filename)

    def restore_state(self):
        """Restore step state from project"""
        state_data = self.project_state.get_step_data("file_selection")

        if state_data:
            if "data_dir" in state_data:
                self._set_data_dir(Path(state_data["data_dir"]))

            if "root_dir" in state_data:
                self.root_dir = Path(state_data["root_dir"])
                self.dir_edit.setText(str(self.root_dir))
            elif "data_dir" in state_data:
                self.root_dir = Path(state_data["data_dir"])
                self.dir_edit.setText(str(self.root_dir))

            if "include_subfolders" in state_data:
                self.include_subfolders_check.setChecked(
                    bool(state_data["include_subfolders"])
                )

            if "filename_prefix" in state_data:
                self.params.P.filename_prefix = state_data["filename_prefix"]
                self.prefix_edit.setText(state_data["filename_prefix"])

            if "night_gap_hours" in state_data:
                self.night_gap_spinbox.setValue(float(state_data["night_gap_hours"]))

            if "excluded_nights" in state_data:
                self._excluded_nights = set(
                    int(n) for n in state_data["excluded_nights"]
                )
                self.file_manager.excluded_nights = self._excluded_nights

            if "night_assignments" in state_data:
                self.file_manager.night_assignments = {
                    k: int(v) for k, v in state_data["night_assignments"].items()
                }

            if "reference_frame" in state_data and state_data["reference_frame"]:
                self.file_manager.ref_filename = state_data["reference_frame"]
                self.ref_label.setText(state_data["reference_frame"])

            # Reload files to repopulate tables
            try:
                self.load_files()

                if self.file_manager.ref_filename:
                    for i in range(self.header_table.rowCount()):
                        item = self.header_table.item(i, 0)
                        if item and item.text() == self.file_manager.ref_filename:
                            self.header_table.selectRow(i)
                            break
            except Exception:
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
