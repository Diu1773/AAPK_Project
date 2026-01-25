"""
Step 10: Aperture Overlay
Overlay apertures/annuli and labels on a selected frame.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QMessageBox,
    QTextEdit, QDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox,
    QSpinBox, QCheckBox, QComboBox, QWidget
)
from PyQt5.QtCore import Qt

from .step_window_base import StepWindowBase
from ...utils.step_paths import step2_cropped_dir, step7_dir, step9_dir, crop_is_active


class ApertureOverlayWindow(StepWindowBase):
    """Step 10: Aperture Overlay"""

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        self.file_list = []
        self.use_cropped = False
        self.current_filename = None
        self.image_data = None
        self.header = None
        self.master_df = None
        self.ap_df = None
        self._file_filter_map = {}
        self._file_frame_key_map = {}
        self._frame_key_map = {}
        self._frame_keys_by_filter = {}
        self._filter_order = []

        # Matplotlib
        self.figure = None
        self.canvas = None
        self.ax = None

        super().__init__(
            step_index=9,
            step_name="Aperture Overlay",
            params=params,
            project_state=project_state,
            main_window=main_window
        )

        self.setup_step_ui()
        self.restore_state()

    def setup_step_ui(self):
        info = QLabel(
            "Overlay apertures/annuli on a selected frame using photometry results."
        )
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        self.content_layout.addWidget(info)

        control_layout = QHBoxLayout()
        btn_params = QPushButton("Overlay Parameters")
        btn_params.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 8px 15px; }")
        btn_params.clicked.connect(self.open_parameters_dialog)
        control_layout.addWidget(btn_params)

        self.btn_load = QPushButton("Load Overlay")
        self.btn_load.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 15px; }")
        self.btn_load.clicked.connect(self.load_and_display)
        control_layout.addWidget(self.btn_load)

        btn_log = QPushButton("Open Log")
        btn_log.setStyleSheet("QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 8px 15px; }")
        btn_log.clicked.connect(self.show_log_window)
        control_layout.addWidget(btn_log)

        control_layout.addStretch()
        self.content_layout.addLayout(control_layout)

        select_group = QGroupBox("Frame Selection")
        select_layout = QHBoxLayout(select_group)
        select_layout.addWidget(QLabel("Index:"))
        self.index_spin = QSpinBox()
        self.index_spin.setRange(0, 0)
        self.index_spin.valueChanged.connect(self.on_index_changed)
        select_layout.addWidget(self.index_spin)

        select_layout.addWidget(QLabel("File:"))
        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self.on_file_changed)
        select_layout.addWidget(self.file_combo)
        self.content_layout.addWidget(select_group)

        viewer_group = QGroupBox("Overlay Viewer")
        viewer_layout = QVBoxLayout(viewer_group)
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(600, 500)
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.95)
        viewer_layout.addWidget(self.canvas)
        self.content_layout.addWidget(viewer_group, stretch=1)

        self.log_window = QWidget(self, Qt.Window)
        self.log_window.setWindowTitle("Aperture Overlay Log")
        self.log_window.resize(700, 350)
        log_layout = QVBoxLayout(self.log_window)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        log_layout.addWidget(self.log_text)

        self.populate_file_list()
        self.load_master_catalog()
        self.load_aperture_by_frame()

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def show_log_window(self):
        self.log_window.show()
        self.log_window.raise_()
        self.log_window.activateWindow()

    def populate_file_list(self):
        crop_active = crop_is_active(self.params.P.result_dir)
        cropped_dir = step2_cropped_dir(self.params.P.result_dir)
        legacy_cropped = self.params.P.result_dir / "cropped"
        if crop_active and cropped_dir.exists() and list(cropped_dir.glob("*.fit*")):
            files = sorted([f.name for f in cropped_dir.glob("*.fit*")])
            self.use_cropped = True
        elif crop_active and legacy_cropped.exists() and list(legacy_cropped.glob("*.fit*")):
            files = sorted([f.name for f in legacy_cropped.glob("*.fit*")])
            self.use_cropped = True
            cropped_dir = legacy_cropped
        else:
            if not self.file_manager.filenames:
                try:
                    self.file_manager.scan_files()
                except Exception:
                    pass
            files = self.file_manager.filenames
            self.use_cropped = False
        self.file_list = list(files)
        self.file_combo.clear()
        self.file_combo.addItems(self.file_list)
        self._file_filter_map = {}
        self._file_frame_key_map = {}
        self._frame_key_map = {}
        self._frame_keys_by_filter = {}
        self._filter_order = []
        self._load_filter_map_from_index()
        if self.file_list:
            self.index_spin.setRange(0, max(0, len(self.file_list) - 1))
            self.index_spin.setValue(int(getattr(self.params.P, "inspect_index", 0)) if self.file_list else 0)
        if hasattr(self, "log_text"):
            self.log(f"Frames loaded: {len(self.file_list)} | use_cropped={self.use_cropped}")

    def on_file_changed(self, index):
        if index >= 0 and index < len(self.file_list):
            self.index_spin.blockSignals(True)
            self.index_spin.setValue(index)
            self.index_spin.blockSignals(False)
            self.params.P.inspect_index = index
            self.save_state()

    def on_index_changed(self, index):
        if index >= 0 and index < len(self.file_list):
            self.file_combo.blockSignals(True)
            self.file_combo.setCurrentIndex(index)
            self.file_combo.blockSignals(False)
            self.params.P.inspect_index = index
            self.save_state()

    def load_master_catalog(self):
        master_path = step7_dir(self.params.P.result_dir) / "master_catalog.tsv"
        if not master_path.exists():
            master_path = self.params.P.result_dir / "master_catalog.tsv"
        if master_path.exists():
            try:
                self.master_df = pd.read_csv(master_path, sep="\t")
            except Exception:
                self.master_df = None
        if hasattr(self, "log_text"):
            n = len(self.master_df) if isinstance(self.master_df, pd.DataFrame) else 0
            self.log(f"Master catalog: {n} rows")

    def load_aperture_by_frame(self):
        ap_path = step9_dir(self.params.P.result_dir) / "aperture_by_frame.csv"
        if not ap_path.exists():
            ap_path = self.params.P.result_dir / "aperture_by_frame.csv"
        if ap_path.exists():
            try:
                self.ap_df = pd.read_csv(ap_path)
            except Exception:
                self.ap_df = None
        if hasattr(self, "log_text"):
            n = len(self.ap_df) if isinstance(self.ap_df, pd.DataFrame) else 0
            self.log(f"Aperture by frame: {n} rows")

    @staticmethod
    def _pick_col(cols, cands):
        for c in cands:
            if c in cols:
                return c
        return None

    @staticmethod
    def _as_bool(x, default=False):
        if isinstance(x, bool):
            return x
        if x is None:
            return default
        if isinstance(x, (int, np.integer)):
            return bool(x)
        s = str(x).strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            return True
        if s in ("0", "false", "f", "no", "n", "off"):
            return False
        return default

    def _build_lab_frame(self, fname: str) -> pd.DataFrame | None:
        result_dir = self.params.P.result_dir
        cache_dir = self.params.P.cache_dir

        phot_path = step9_dir(result_dir) / f"{fname}_photometry.tsv"
        if not phot_path.exists():
            phot_path = result_dir / f"{fname}_photometry.tsv"
        phot = None
        if phot_path.exists():
            try:
                phot = pd.read_csv(phot_path, sep="\t")
            except Exception:
                try:
                    phot = pd.read_csv(phot_path)
                except Exception:
                    phot = None

        if phot is not None and ("ID" in phot.columns):
            cols = phot.columns
            cx = "xcenter" if "xcenter" in cols else None
            cy = "ycenter" if "ycenter" in cols else None
            ix = "x_init" if "x_init" in cols else None
            iy = "y_init" if "y_init" in cols else None

            use_centroid = self._as_bool(getattr(self.params.P, "overlay_use_phot_centroid", True), True)
            if use_centroid and (cx is not None) and (cy is not None):
                x_use = phot[cx].to_numpy(float)
                y_use = phot[cy].to_numpy(float)
                if (ix is not None) and (iy is not None):
                    mnan = ~np.isfinite(x_use) | ~np.isfinite(y_use)
                    x_use[mnan] = phot[ix].to_numpy(float)[mnan]
                    y_use[mnan] = phot[iy].to_numpy(float)[mnan]
            else:
                if (ix is None) or (iy is None):
                    if (cx is None) or (cy is None):
                        raise RuntimeError("photometry file missing x/y columns")
                    x_use = phot[cx].to_numpy(float)
                    y_use = phot[cy].to_numpy(float)
                else:
                    x_use = phot[ix].to_numpy(float)
                    y_use = phot[iy].to_numpy(float)

            lab = pd.DataFrame({
                "ID": phot["ID"].astype(int).to_numpy(),
                "x_frame": x_use,
                "y_frame": y_use,
                "mag": phot["mag"].to_numpy(float) if "mag" in cols else np.nan,
                "mag_err": phot["mag_err"].to_numpy(float) if "mag_err" in cols else np.nan,
            })
            return lab

        fm_path = step7_dir(result_dir) / "frame_sourceid_to_ID.tsv"
        if not fm_path.exists():
            fm_path = result_dir / "frame_sourceid_to_ID.tsv"
        if fm_path.exists():
            try:
                fm = pd.read_csv(fm_path, sep="\t")
            except Exception:
                fm = pd.read_csv(fm_path)
            c_file = self._pick_col(fm.columns, ["file", "fname", "frame"])
            c_id = self._pick_col(fm.columns, ["ID", "id"])
            c_x = self._pick_col(fm.columns, ["x", "x_det", "x_pix", "x0"])
            c_y = self._pick_col(fm.columns, ["y", "y_det", "y_pix", "y0"])
            if c_file and c_id and c_x and c_y:
                sub = fm[fm[c_file].astype(str) == str(fname)].copy()
                if len(sub):
                    return pd.DataFrame({
                        "ID": sub[c_id].astype(int).to_numpy(),
                        "x_frame": sub[c_x].astype(float).to_numpy(),
                        "y_frame": sub[c_y].astype(float).to_numpy(),
                        "mag": np.nan,
                        "mag_err": np.nan,
                    })

        idm_path = cache_dir / "idmatch" / f"idmatch_{fname}.csv"
        if idm_path.exists():
            idm = pd.read_csv(idm_path)
            c_sid = self._pick_col(idm.columns, ["source_id", "sourceid", "sid"])
            c_x = self._pick_col(idm.columns, ["x", "x_det", "x_pix", "x0"])
            c_y = self._pick_col(idm.columns, ["y", "y_det", "y_pix", "y0"])
            if c_sid and c_x and c_y:
                sid = idm[c_sid].astype(np.int64).to_numpy()
                x = idm[c_x].astype(float).to_numpy()
                y = idm[c_y].astype(float).to_numpy()

                map_path = step7_dir(result_dir) / "sourceid_to_ID.csv"
                if not map_path.exists():
                    map_path = result_dir / "sourceid_to_ID.csv"
                if map_path.exists():
                    mp = pd.read_csv(map_path)
                    if ("source_id" in mp.columns) and ("ID" in mp.columns):
                        sid2id = dict(zip(mp["source_id"].astype(np.int64), mp["ID"].astype(int)))
                    else:
                        sid2id = {}
                else:
                    sid2id = {}
                if not sid2id and (self.master_df is not None) and ("source_id" in self.master_df.columns):
                    sid2id = dict(zip(self.master_df["source_id"].astype(np.int64), self.master_df["ID"].astype(int)))

                ID = np.array([sid2id.get(s, -1) for s in sid], dtype=int)
                ok = ID >= 0
                return pd.DataFrame({
                    "ID": ID[ok],
                    "x_frame": x[ok],
                    "y_frame": y[ok],
                    "mag": np.nan,
                    "mag_err": np.nan,
                }).drop_duplicates("ID", keep="first").reset_index(drop=True)

        return None

    def load_and_display(self):
        if not self.file_list:
            QMessageBox.warning(self, "Warning", "No files found to display")
            return
        if self.master_df is None:
            self.load_master_catalog()
        if self.ap_df is None:
            self.load_aperture_by_frame()
        if self.master_df is None or self.ap_df is None:
            QMessageBox.warning(self, "Missing Data", "master_catalog.tsv or aperture_by_frame.csv not found")
            return

        fname = self.file_combo.currentText()
        if not fname:
            return
        self.current_filename = fname
        self._file_filter_map.setdefault(fname, self._extract_filter_from_header(self.header))
        self._file_frame_key_map.setdefault(
            fname,
            self._extract_frame_key(fname, self._file_filter_map.get(fname, ""))
        )

        if self.use_cropped:
            cropped_dir = step2_cropped_dir(self.params.P.result_dir)
            if not cropped_dir.exists():
                cropped_dir = self.params.P.result_dir / "cropped"
            file_path = cropped_dir / fname
        else:
            file_path = self.params.P.data_dir / fname
        if not file_path.exists():
            QMessageBox.warning(self, "Missing File", f"File not found: {file_path}")
            return

        try:
            with fits.open(file_path) as hdul:
                self.image_data = hdul[0].data.astype(float)
                self.header = hdul[0].header
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))
            return

        row = self.ap_df[self.ap_df["file"].astype(str) == str(fname)]
        if row.empty:
            QMessageBox.warning(self, "Missing Aperture", f"No aperture_by_frame entry for {fname}")
            return
        r_ap = float(row["r_ap"].values[0])
        r_in = float(row["r_in"].values[0])
        r_out = float(row["r_out"].values[0])

        lab_frame = self._build_lab_frame(fname)
        if lab_frame is None or len(lab_frame) == 0:
            QMessageBox.warning(self, "Missing Positions", "Could not build frame positions from photometry/idmatch")
            return

        lab = lab_frame.merge(self.master_df[["ID", "x_ref", "y_ref"]], on="ID", how="left")

        label_limit = int(getattr(self.params.P, "overlay_max_labels", 2000))
        label_font = float(getattr(self.params.P, "overlay_label_fontsize", 6.0))
        label_offset = float(getattr(self.params.P, "overlay_label_offset_px", 3.0))
        show_id_no_mag = self._as_bool(getattr(self.params.P, "overlay_show_id_when_no_mag", False), False)
        show_ref = self._as_bool(getattr(self.params.P, "overlay_show_ref_pos", True), True)
        show_shift = self._as_bool(getattr(self.params.P, "overlay_show_shift_vectors", False), False)
        shift_max = int(getattr(self.params.P, "overlay_shift_max_vectors", 300))
        shift_min = float(getattr(self.params.P, "overlay_shift_min_px", 1.5))

        lab["_has_mag"] = np.isfinite(lab.get("mag", np.nan))
        lab["_mag_for_sort"] = lab["mag"].fillna(99.0) if "mag" in lab.columns else 99.0
        lab = lab.sort_values(by=["_has_mag", "_mag_for_sort", "ID"], ascending=[False, True, True])
        lab_sel = lab.head(label_limit).copy()

        self.ax.clear()
        vmin, vmax = ZScaleInterval().get_limits(self.image_data)
        norm = ImageNormalize(vmin=vmin, vmax=vmax)
        self.ax.imshow(self.image_data, origin="lower", cmap="gray", norm=norm)

        xy_frame = lab[["x_frame", "y_frame"]].to_numpy(float)
        xy_frame = xy_frame[np.isfinite(xy_frame).all(axis=1)]
        for (x, y) in xy_frame:
            self.ax.add_patch(Circle((x, y), r_ap, ec="gold", fc="none", lw=0.9, alpha=0.95))
            self.ax.add_patch(Circle((x, y), r_in, ec="cyan", fc="none", lw=0.6, alpha=0.70))
            self.ax.add_patch(Circle((x, y), r_out, ec="cyan", fc="none", lw=0.6, alpha=0.50))

        if show_ref:
            xy_ref = lab[["x_ref", "y_ref"]].to_numpy(float)
            xy_ref = xy_ref[np.isfinite(xy_ref).all(axis=1)]
            for (x, y) in xy_ref:
                self.ax.add_patch(Circle((x, y), r_ap, ec="orange", fc="none", lw=0.4, alpha=0.35))

        if show_shift and show_ref:
            sub = lab.copy()
            m = np.isfinite(sub["x_ref"]) & np.isfinite(sub["y_ref"]) & np.isfinite(sub["x_frame"]) & np.isfinite(sub["y_frame"])
            sub = sub.loc[m].copy()
            sub["dx"] = sub["x_frame"] - sub["x_ref"]
            sub["dy"] = sub["y_frame"] - sub["y_ref"]
            sub["dr"] = np.hypot(sub["dx"], sub["dy"])
            sub = sub[sub["dr"] >= shift_min].sort_values("dr", ascending=False).head(shift_max)

            for _, r in sub.iterrows():
                x0, y0 = float(r["x_ref"]), float(r["y_ref"])
                x1, y1 = float(r["x_frame"]), float(r["y_frame"])
                arr = FancyArrowPatch((x0, y0), (x1, y1),
                                      arrowstyle='->', mutation_scale=8,
                                      lw=0.6, color='magenta', alpha=0.65)
                self.ax.add_patch(arr)

        for _, r in lab_sel.iterrows():
            x = float(r["x_frame"])
            y = float(r["y_frame"])
            if (not np.isfinite(x)) or (not np.isfinite(y)):
                continue
            ID = int(r["ID"])
            if ("mag" in r) and np.isfinite(r["mag"]):
                txt = f"{ID}  m={float(r['mag']):.2f}"
            else:
                if not show_id_no_mag:
                    continue
                txt = f"{ID}"

            self.ax.text(
                x + label_offset, y + label_offset, txt,
                color="yellow", fontsize=label_font, ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.12", fc=(0, 0, 0, 0.45), ec="none")
            )

        self.ax.set_xlim(0, self.image_data.shape[1])
        self.ax.set_ylim(0, self.image_data.shape[0])
        self.canvas.draw_idle()

        n_ref_ok = int(np.isfinite(lab["x_ref"]).sum()) if "x_ref" in lab.columns else 0
        n_fr_ok = int(np.isfinite(lab["x_frame"]).sum()) if "x_frame" in lab.columns else 0
        self.log(f"Overlay {fname}: frame_xy_ok={n_fr_ok} ref_xy_ok={n_ref_ok}")
        if show_ref and ("x_ref" in lab.columns):
            m = np.isfinite(lab["x_ref"]) & np.isfinite(lab["y_ref"]) & np.isfinite(lab["x_frame"]) & np.isfinite(lab["y_frame"])
            if m.any():
                dr = np.hypot(lab.loc[m, "x_frame"] - lab.loc[m, "x_ref"], lab.loc[m, "y_frame"] - lab.loc[m, "y_ref"])
                self.log(f"Shift px: min/med/max = {dr.min():.3f}/{np.median(dr):.3f}/{dr.max():.3f}")

    def navigate_frame(self, direction: int):
        """Navigate to previous/next frame within the SAME filter"""
        if not self.file_list:
            return
        self._build_frame_key_map()
        current_filter = self._file_filter_map.get(self.current_filename, "")
        current_index = self.index_spin.value()

        # Build list of file indices for current filter
        filter_indices = []
        for idx, fname in enumerate(self.file_list):
            fkey = self._file_filter_map.get(fname, "")
            if fkey == current_filter:
                filter_indices.append(idx)

        if not filter_indices:
            # Fallback: cycle through all files
            new_index = (current_index + direction) % len(self.file_list)
        else:
            # Find current position within filter's files
            try:
                pos = filter_indices.index(current_index)
            except ValueError:
                pos = 0
            # Move within filter only
            pos = (pos + direction) % len(filter_indices)
            new_index = filter_indices[pos]

        self.index_spin.setValue(new_index)

    def cycle_filter(self):
        """Cycle to next filter, keeping the same frame position (index within filter)"""
        if not self.file_list:
            return
        self._build_frame_key_map()
        current_filter = self._file_filter_map.get(self.current_filename, "")
        current_index = self.index_spin.value()
        filters = [f for f in self._filter_order if f]
        if len(filters) <= 1:
            self.log("Filter cycle skipped: only one filter found")
            return

        # Find next filter
        try:
            current_filter_idx = filters.index(current_filter)
            next_filter = filters[(current_filter_idx + 1) % len(filters)]
        except ValueError:
            next_filter = filters[0]

        # Build file indices for current and next filter
        current_filter_indices = []
        next_filter_indices = []
        for idx, fname in enumerate(self.file_list):
            fkey = self._file_filter_map.get(fname, "")
            if fkey == current_filter:
                current_filter_indices.append(idx)
            if fkey == next_filter:
                next_filter_indices.append(idx)

        if not next_filter_indices:
            return

        # Find current position within current filter's files
        try:
            pos_in_filter = current_filter_indices.index(current_index)
        except ValueError:
            pos_in_filter = 0

        # Go to same position in next filter (or last if shorter)
        target_pos = min(pos_in_filter, len(next_filter_indices) - 1)
        new_index = next_filter_indices[target_pos]

        self.index_spin.setValue(new_index)
        self.log(f"Filter cycle: {next_filter}")

    @staticmethod
    def _normalize_filter_key(value: str | None) -> str:
        if value is None:
            return ""
        return str(value).strip().upper()

    def _extract_filter_from_header(self, header) -> str:
        if header is None:
            return ""
        for key in ("FILTER", "FILTER1", "FILTER2", "FILTNAM"):
            val = header.get(key)
            if val:
                return self._normalize_filter_key(val)
        return ""

    @staticmethod
    def _infer_filter_from_filename(fname: str) -> str:
        base = Path(fname).name
        for ext in (".fits", ".fit", ".fts", ".fz", ".gz"):
            if base.lower().endswith(ext):
                base = base[: -len(ext)]
        parts = [p for p in base.replace(".", "_").replace("-", "_").split("_") if p]
        for token in reversed(parts):
            cand = token.lower()
            if 1 <= len(cand) <= 3 and cand.isalpha():
                return cand.upper()
        return ""

    def _extract_frame_key(self, fname: str, filter_key: str) -> str:
        name = Path(fname).name
        base = name
        for ext in (".fits", ".fit", ".fts", ".fz", ".gz"):
            if base.lower().endswith(ext):
                base = base[: -len(ext)]
        base_lower = base.lower()
        filt = str(filter_key or "").lower() or self._infer_filter_from_filename(fname).lower()
        if filt:
            for sep in ("-", "_", "."):
                suffix = f"{sep}{filt}"
                if base_lower.endswith(suffix):
                    return base[: -len(suffix)]
        return base

    def _load_filter_map_from_index(self):
        idx_path = step9_dir(self.params.P.result_dir) / "photometry_index.csv"
        if not idx_path.exists():
            idx_path = self.params.P.result_dir / "photometry_index.csv"
        if not idx_path.exists():
            return
        try:
            df = pd.read_csv(idx_path)
        except Exception:
            return
        if "file" not in df.columns or "filter" not in df.columns:
            return
        for fname, filt in zip(df["file"].astype(str), df["filter"].astype(str)):
            fkey = self._normalize_filter_key(filt)
            if not fkey:
                fkey = self._infer_filter_from_filename(fname)
            self._file_filter_map[fname] = fkey
            if fname.lower().startswith("crop_"):
                self._file_filter_map[fname[5:]] = fkey
            if fname.lower().startswith("cropped_"):
                self._file_filter_map[fname[8:]] = fkey

    def _build_frame_key_map(self):
        if self._frame_key_map:
            return
        for idx, fname in enumerate(self.file_list):
            fkey = self._file_filter_map.get(fname)
            if not fkey:
                try:
                    if self.use_cropped:
                        cropped_dir = step2_cropped_dir(self.params.P.result_dir)
                        if not cropped_dir.exists():
                            cropped_dir = self.params.P.result_dir / "cropped"
                        fpath = cropped_dir / fname
                    else:
                        fpath = self.params.P.data_dir / fname
                    hdr = fits.getheader(fpath)
                    fkey = self._extract_filter_from_header(hdr)
                except Exception:
                    fkey = ""
                if not fkey:
                    fkey = self._infer_filter_from_filename(fname)
                self._file_filter_map[fname] = fkey
            frame_key = self._extract_frame_key(fname, fkey)
            self._file_frame_key_map[fname] = frame_key
            self._frame_key_map[(frame_key, fkey)] = idx
            self._frame_keys_by_filter.setdefault(fkey, []).append(frame_key)
        self._filter_order = [f for f in sorted(self._frame_keys_by_filter.keys()) if f]

    def open_parameters_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Overlay Parameters")
        dialog.resize(460, 420)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        self.param_labels = QSpinBox()
        self.param_labels.setRange(0, 20000)
        self.param_labels.setValue(int(getattr(self.params.P, "overlay_max_labels", 2000)))
        form.addRow("Max labels:", self.param_labels)

        self.param_label_fs = QDoubleSpinBox()
        self.param_label_fs.setRange(1.0, 20.0)
        self.param_label_fs.setValue(float(getattr(self.params.P, "overlay_label_fontsize", 6.0)))
        form.addRow("Label font size:", self.param_label_fs)

        self.param_label_offset = QDoubleSpinBox()
        self.param_label_offset.setRange(0.0, 20.0)
        self.param_label_offset.setValue(float(getattr(self.params.P, "overlay_label_offset_px", 3.0)))
        form.addRow("Label offset px:", self.param_label_offset)

        self.param_show_id = QCheckBox("Show ID when mag missing")
        self.param_show_id.setChecked(bool(getattr(self.params.P, "overlay_show_id_when_no_mag", False)))
        form.addRow("", self.param_show_id)

        self.param_use_centroid = QCheckBox("Use photometry centroid")
        self.param_use_centroid.setChecked(bool(getattr(self.params.P, "overlay_use_phot_centroid", True)))
        form.addRow("", self.param_use_centroid)

        self.param_show_ref = QCheckBox("Show ref positions")
        self.param_show_ref.setChecked(bool(getattr(self.params.P, "overlay_show_ref_pos", True)))
        form.addRow("", self.param_show_ref)

        self.param_show_shift = QCheckBox("Show shift vectors")
        self.param_show_shift.setChecked(bool(getattr(self.params.P, "overlay_show_shift_vectors", False)))
        form.addRow("", self.param_show_shift)

        self.param_shift_max = QSpinBox()
        self.param_shift_max.setRange(0, 10000)
        self.param_shift_max.setValue(int(getattr(self.params.P, "overlay_shift_max_vectors", 300)))
        form.addRow("Shift max vectors:", self.param_shift_max)

        self.param_shift_min = QDoubleSpinBox()
        self.param_shift_min.setRange(0.0, 50.0)
        self.param_shift_min.setValue(float(getattr(self.params.P, "overlay_shift_min_px", 1.5)))
        form.addRow("Shift min px:", self.param_shift_min)

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: self.save_parameters(dialog))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec_()

    def save_parameters(self, dialog):
        self.params.P.overlay_max_labels = self.param_labels.value()
        self.params.P.overlay_label_fontsize = self.param_label_fs.value()
        self.params.P.overlay_label_offset_px = self.param_label_offset.value()
        self.params.P.overlay_show_id_when_no_mag = self.param_show_id.isChecked()
        self.params.P.overlay_use_phot_centroid = self.param_use_centroid.isChecked()
        self.params.P.overlay_show_ref_pos = self.param_show_ref.isChecked()
        self.params.P.overlay_show_shift_vectors = self.param_show_shift.isChecked()
        self.params.P.overlay_shift_max_vectors = self.param_shift_max.value()
        self.params.P.overlay_shift_min_px = self.param_shift_min.value()
        self.save_state()
        QMessageBox.information(dialog, "Success", "Parameters saved!")
        dialog.accept()

    def validate_step(self) -> bool:
        ap_path = step9_dir(self.params.P.result_dir) / "aperture_by_frame.csv"
        if not ap_path.exists():
            ap_path = self.params.P.result_dir / "aperture_by_frame.csv"
        master_path = step7_dir(self.params.P.result_dir) / "master_catalog.tsv"
        if not master_path.exists():
            master_path = self.params.P.result_dir / "master_catalog.tsv"
        return ap_path.exists() and master_path.exists()

    def save_state(self):
        state_data = {
            "inspect_index": getattr(self.params.P, "inspect_index", 0),
            "overlay_max_labels": getattr(self.params.P, "overlay_max_labels", 2000),
            "overlay_label_fontsize": getattr(self.params.P, "overlay_label_fontsize", 6.0),
            "overlay_label_offset_px": getattr(self.params.P, "overlay_label_offset_px", 3.0),
            "overlay_show_id_when_no_mag": getattr(self.params.P, "overlay_show_id_when_no_mag", False),
            "overlay_use_phot_centroid": getattr(self.params.P, "overlay_use_phot_centroid", True),
            "overlay_show_ref_pos": getattr(self.params.P, "overlay_show_ref_pos", True),
            "overlay_show_shift_vectors": getattr(self.params.P, "overlay_show_shift_vectors", False),
            "overlay_shift_max_vectors": getattr(self.params.P, "overlay_shift_max_vectors", 300),
            "overlay_shift_min_px": getattr(self.params.P, "overlay_shift_min_px", 1.5),
        }
        self.project_state.store_step_data("aperture_overlay", state_data)

    def restore_state(self):
        state_data = self.project_state.get_step_data("aperture_overlay")
        if state_data:
            for key, val in state_data.items():
                if hasattr(self.params.P, key):
                    setattr(self.params.P, key, val)
            idx = int(getattr(self.params.P, "inspect_index", 0))
            if self.file_list and 0 <= idx < len(self.file_list):
                self.file_combo.setCurrentIndex(idx)
                self.index_spin.setValue(idx)
