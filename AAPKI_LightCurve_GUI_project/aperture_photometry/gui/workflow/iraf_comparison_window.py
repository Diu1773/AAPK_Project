"""
IRAF Comparison Tool Window
Compares AAPKI photometry results with IRAF photometry for quality validation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QGroupBox, QFormLayout, QLineEdit, QDoubleSpinBox, QComboBox,
    QFileDialog, QMessageBox, QProgressBar, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QHeaderView
)


def _read_iraf_txt(path: Path) -> pd.DataFrame:
    """Read IRAF txdump output (.txt)."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        names=["ID", "x", "y", "mag", "merr", "msky"],
        engine="python",
    )
    for col in ["ID", "x", "y", "mag", "merr", "msky"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["x"] = df["x"] - 1.0
    df["y"] = df["y"] - 1.0
    return df


def _read_iraf_coo(path: Path) -> pd.DataFrame:
    """Read IRAF daofind output (.coo)."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=["x", "y", "mag", "sharp", "sround", "ground", "ID"],
        engine="python",
    )
    for col in ["ID", "x", "y", "mag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["ID", "x", "y", "mag"]].copy()
    df["x"] = df["x"] - 1.0
    df["y"] = df["y"] - 1.0
    return df


def _read_iraf_mag(path: Path) -> pd.DataFrame:
    """Read IRAF phot output (.mag)."""
    def _parse_segments(rows):
        data = []
        seq_id = 1
        for segments in rows:
            x_val, y_val, mag_val = np.nan, np.nan, np.nan
            obj_id = None

            if len(segments) > 0:
                seg0 = segments[0].split()
                if len(seg0) >= 3:
                    try:
                        obj_id = int(float(seg0[2]))
                    except:
                        pass

            if len(segments) > 1:
                seg1 = segments[1].split()
                if len(seg1) >= 2:
                    try:
                        x_val = float(seg1[0])
                        y_val = float(seg1[1])
                    except:
                        pass

            if len(segments) > 4:
                seg4 = segments[4].split()
                if len(seg4) >= 5:
                    try:
                        mag_val = float(seg4[4])
                    except:
                        pass

            if obj_id is None:
                obj_id = seq_id
            data.append({"ID": obj_id, "x": x_val, "y": y_val, "mag": mag_val})
            seq_id += 1
        return pd.DataFrame(data, columns=["ID", "x", "y", "mag"])

    rows = []
    acc_lines = []
    data_lines = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            data_lines.append(line)
            cont = line.rstrip().endswith("\\")
            text = line.rstrip()
            if cont:
                text = text[:-1].rstrip()
            if text.strip():
                acc_lines.append(text)
            if not cont:
                if acc_lines:
                    rows.append(acc_lines)
                acc_lines = []
    if acc_lines:
        rows.append(acc_lines)

    df = _parse_segments(rows)
    if df.empty or not np.isfinite(df[["x", "y"]].to_numpy(float)).any():
        grouped = []
        chunk = []
        for line in data_lines:
            chunk.append(line.rstrip())
            if len(chunk) == 5:
                grouped.append(chunk)
                chunk = []
        if chunk:
            grouped.append(chunk)
        df = _parse_segments(grouped)
    if "x" in df.columns:
        df["x"] = df["x"] - 1.0
    if "y" in df.columns:
        df["y"] = df["y"] - 1.0
    return df


def _read_iraf_file(path: Path) -> pd.DataFrame:
    """Read IRAF output file based on extension."""
    suffix = path.suffix.lower()
    if suffix == ".mag":
        return _read_iraf_mag(path)
    if suffix == ".coo":
        return _read_iraf_coo(path)
    return _read_iraf_txt(path)


def _read_aapki_tsv(path: Path) -> pd.DataFrame:
    """Read AAPKI photometry TSV file."""
    try:
        return pd.read_csv(path, sep="\t")
    except:
        return pd.read_csv(path)


def _pick_first(cols, candidates):
    """Pick first matching column from candidates."""
    for c in candidates:
        if c in cols:
            return c
    return None


def _normalize_frame_key(stem: str) -> str:
    """Normalize frame filename for matching."""
    key = stem
    if key.endswith("_photometry"):
        key = key[:-len("_photometry")]
    if key.endswith(".fit") or key.endswith(".fits"):
        key = key.rsplit(".", 1)[0]
    if key.startswith("Crop_"):
        key = key[len("Crop_"):]
    return key


BASE_IRAF_SHIFT = -1.0
AUTO_SHIFT_THRESHOLD = 0.6


def _auto_axis_shift(delta_med: float) -> float:
    if not np.isfinite(delta_med):
        return 0.0
    if abs(delta_med - 1.0) <= AUTO_SHIFT_THRESHOLD:
        return 1.0
    if abs(delta_med + 1.0) <= AUTO_SHIFT_THRESHOLD:
        return -1.0
    return 0.0


class IRAFComparisonWorker(QThread):
    """Worker thread for comparing AAPKI and IRAF photometry."""

    progress = pyqtSignal(int, int, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        aapki_dir: Path,
        iraf_dir: Path,
        tol_px: float = 1.5,
        pos_mode: str = "xcenter/ycenter",
        iraf_mode: str = "auto",
    ):
        super().__init__()
        self.aapki_dir = Path(aapki_dir)
        self.iraf_dir = Path(iraf_dir)
        self.tol_px = tol_px
        self.pos_mode = pos_mode
        self.iraf_mode = iraf_mode
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def _log(self, msg: str):
        self.log.emit(msg)

    def run(self):
        try:
            # Collect files
            aapki_map = self._collect_aapki(self.aapki_dir)
            iraf_map = self._collect_iraf(self.iraf_dir, self.iraf_mode)

            self._log(f"AAPKI files: {len(aapki_map)}")
            self._log(f"IRAF files: {len(iraf_map)}")

            frames = sorted(set(aapki_map) & set(iraf_map))
            if not frames:
                self.error.emit("No matching frames found between AAPKI and IRAF.")
                return

            self._log(f"Matched frames: {len(frames)}")
            self._log(f"Tolerance: {self.tol_px:.2f} px")

            frame_rows = []
            frame_matches = {}
            all_matches = []

            for idx, frame in enumerate(frames):
                if self._stop_requested:
                    self._log("Stopped by user")
                    break

                self.progress.emit(idx + 1, len(frames), f"Comparing {frame}")

                match_df, info = self._match_frame(
                    aapki_map[frame], iraf_map[frame]
                )
                frame_matches[frame] = match_df

                if not match_df.empty:
                    all_matches.append(match_df.assign(frame=frame))

                dmag = match_df["dmag"] if len(match_df) else pd.Series(dtype=float)
                dx = match_df["dx"] if len(match_df) else pd.Series(dtype=float)
                dy = match_df["dy"] if len(match_df) else pd.Series(dtype=float)
                dist = match_df["dist_px"] if len(match_df) else pd.Series(dtype=float)

                dmag_med, dmag_std = self._nanstats(dmag)
                dx_med, _ = self._nanstats(dx)
                dy_med, _ = self._nanstats(dy)
                dist_med, _ = self._nanstats(dist)
                dist_p95 = float(np.nanpercentile(dist, 95)) if len(dist) else np.nan
                frac_within = float(np.mean(dist <= self.tol_px)) if len(dist) else 0.0

                stats = {
                    "frame": frame,
                    "n": len(match_df),
                    "dmag_med": dmag_med,
                    "dmag_std": dmag_std,
                    "dx_med": dx_med,
                    "dy_med": dy_med,
                    "dist_med": dist_med,
                    "dist_p95": dist_p95,
                    "frac_within_tol": frac_within,
                    "best_shift_x": info.get("best_shift_x", np.nan),
                    "best_shift_y": info.get("best_shift_y", np.nan),
                    "n_iraf_total": info.get("n_iraf_total", np.nan),
                    "n_aapki_total": info.get("n_aapki_total", np.nan),
                }
                frame_rows.append(stats)

                self._log(
                    f"{frame}: matched {len(match_df)}, "
                    f"dmag={dmag_med:.4f}+-{dmag_std:.4f} "
                    f"dx={dx_med:.3f} dy={dy_med:.3f} "
                    f"dist_med={dist_med:.3f} p95={dist_p95:.3f} "
                    f"shift=({stats['best_shift_x']:.1f},{stats['best_shift_y']:.1f}) "
                    f"n_iraf={stats['n_iraf_total']} n_aapki={stats['n_aapki_total']}"
                )

            # Combine all matches
            matched_all = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()

            # Compute global statistics
            if not matched_all.empty:
                global_dmag_med, global_dmag_std = self._nanstats(matched_all["dmag"])
                global_dx_med, _ = self._nanstats(matched_all["dx"])
                global_dy_med, _ = self._nanstats(matched_all["dy"])
            else:
                global_dmag_med = global_dmag_std = np.nan
                global_dx_med = global_dy_med = np.nan

            self._log(f"\n=== Global Statistics ===")
            self._log(f"Total matched: {len(matched_all)}")
            self._log(f"dmag median: {global_dmag_med:.4f}")
            self._log(f"dmag std: {global_dmag_std:.4f}")

            self.finished.emit({
                "frame_rows": frame_rows,
                "frame_matches": frame_matches,
                "matched_all": matched_all,
                "global_stats": {
                    "dmag_med": global_dmag_med,
                    "dmag_std": global_dmag_std,
                    "dx_med": global_dx_med,
                    "dy_med": global_dy_med,
                    "n_total": len(matched_all),
                    "n_frames": len(frames),
                }
            })

        except Exception as e:
            self.error.emit(str(e))

    def _nanstats(self, series):
        values = np.asarray(series, float)
        finite = np.isfinite(values)
        if not np.any(finite):
            return np.nan, np.nan
        return float(np.nanmedian(values)), float(np.nanstd(values))

    def _collect_aapki(self, base_dir: Path):
        mapping = {}
        for path in base_dir.rglob("*_photometry.tsv"):
            base = _normalize_frame_key(path.stem)
            mapping.setdefault(base, path)
        return mapping

    def _collect_iraf(self, base_dir: Path, mode: str):
        mapping = {}
        if mode == ".txt":
            patterns = ["*.txt"]
        elif mode == ".mag":
            patterns = ["*.mag"]
        elif mode == ".coo":
            patterns = ["*.coo"]
        else:
            patterns = ["*.mag", "*.coo", "*.txt"]
        for pattern in patterns:
            for path in base_dir.rglob(pattern):
                base = _normalize_frame_key(path.stem)
                mapping.setdefault(base, path)
        return mapping

    def _match_frame(self, aapki_path: Path, iraf_path: Path):
        aapki = _read_aapki_tsv(aapki_path)
        iraf = _read_iraf_file(iraf_path)
        n_aapki_total = len(aapki)
        n_iraf_total = len(iraf)

        mag_col = _pick_first(aapki.columns, ["mag_inst", "mag", "MAG", "mag_raw"])
        if mag_col is None:
            raise ValueError(f"Missing mag column in {aapki_path}")

        if self.pos_mode.startswith("xcenter"):
            x_col = _pick_first(aapki.columns, ["xcenter", "x", "x_init"])
            y_col = _pick_first(aapki.columns, ["ycenter", "y", "y_init"])
        else:
            x_col = _pick_first(aapki.columns, ["x_init", "xcenter", "x"])
            y_col = _pick_first(aapki.columns, ["y_init", "ycenter", "y"])

        if x_col is None or y_col is None:
            raise ValueError(f"Missing XY columns in {aapki_path}")

        def _match_with_iraf(iraf_df):
            axy = aapki[[x_col, y_col]].to_numpy(float)
            ixy = iraf_df[["x", "y"]].to_numpy(float)

            if axy.size == 0 or ixy.size == 0:
                return pd.DataFrame()

            tree = cKDTree(axy)
            dist, idx = tree.query(ixy, distance_upper_bound=self.tol_px)
            mask = np.isfinite(dist) & (dist <= self.tol_px)

            if not np.any(mask):
                return pd.DataFrame()

            match = pd.DataFrame({
                "iraf_id": iraf_df.loc[mask, "ID"].to_numpy(),
                "iraf_x": iraf_df.loc[mask, "x"].to_numpy(),
                "iraf_y": iraf_df.loc[mask, "y"].to_numpy(),
                "iraf_mag": iraf_df.loc[mask, "mag"].to_numpy(),
                "aapki_x": aapki.loc[idx[mask], x_col].to_numpy(),
                "aapki_y": aapki.loc[idx[mask], y_col].to_numpy(),
                "aapki_mag": aapki.loc[idx[mask], mag_col].to_numpy(),
                "dist_px": dist[mask],
            })
            match["dx"] = match["aapki_x"] - match["iraf_x"]
            match["dy"] = match["aapki_y"] - match["iraf_y"]
            match["dmag"] = match["aapki_mag"] - match["iraf_mag"]
            return match

        match = _match_with_iraf(iraf)
        if not match.empty:
            dx_med = float(np.nanmedian(match["dx"]))
            dy_med = float(np.nanmedian(match["dy"]))
        else:
            dx_med = np.nan
            dy_med = np.nan

        shift_x = _auto_axis_shift(dx_med)
        shift_y = _auto_axis_shift(dy_med)
        if (shift_x != 0.0) or (shift_y != 0.0):
            iraf_adj = iraf.copy()
            iraf_adj["x"] = iraf_adj["x"] + shift_x
            iraf_adj["y"] = iraf_adj["y"] + shift_y
            match = _match_with_iraf(iraf_adj)

        if match.empty:
            return pd.DataFrame(), {
                "x_col": x_col,
                "y_col": y_col,
                "mag_col": mag_col,
                "best_shift_x": np.nan,
                "best_shift_y": np.nan,
                "dist_med": np.nan,
                "dist_p95": np.nan,
                "frac_within_tol": 0.0,
                "n_iraf_total": n_iraf_total,
                "n_aapki_total": n_aapki_total,
            }

        dist_vals = match["dist_px"].to_numpy(float)
        dist_med = float(np.nanmedian(dist_vals)) if dist_vals.size else np.nan
        dist_p95 = float(np.nanpercentile(dist_vals, 95)) if dist_vals.size else np.nan
        frac_within = float(np.mean(dist_vals <= self.tol_px)) if dist_vals.size else 0.0

        best_shift_x = BASE_IRAF_SHIFT + shift_x
        best_shift_y = BASE_IRAF_SHIFT + shift_y

        return match, {
            "x_col": x_col,
            "y_col": y_col,
            "mag_col": mag_col,
            "best_shift_x": best_shift_x,
            "best_shift_y": best_shift_y,
            "dist_med": dist_med,
            "dist_p95": dist_p95,
            "frac_within_tol": frac_within,
            "n_iraf_total": n_iraf_total,
            "n_aapki_total": n_aapki_total,
        }


class IRAFComparisonWindow(QMainWindow):
    """Main window for IRAF vs AAPKI photometry comparison."""

    def __init__(self, params, data_dir: Path, result_dir: Path, parent=None):
        super().__init__(parent)
        self.params = params
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.worker = None

        # Results storage
        self.frame_rows = []
        self.frame_matches = {}
        self.matched_all = None

        self.setWindowTitle("IRAF vs AAPKI Photometry Comparison")
        self.setMinimumSize(1200, 800)
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # === Paths and Settings ===
        settings_group = QGroupBox("Paths and Settings")
        settings_layout = QFormLayout()

        # AAPKI result directory
        aapki_row = QHBoxLayout()
        self.aapki_edit = QLineEdit(str(self.result_dir))
        aapki_btn = QPushButton("Browse")
        aapki_btn.clicked.connect(lambda: self._browse_dir(self.aapki_edit))
        aapki_row.addWidget(self.aapki_edit)
        aapki_row.addWidget(aapki_btn)
        aapki_widget = QWidget()
        aapki_widget.setLayout(aapki_row)
        settings_layout.addRow("AAPKI Result Dir:", aapki_widget)

        # IRAF result directory
        iraf_row = QHBoxLayout()
        default_iraf = self.result_dir / "iraf_phot"
        self.iraf_edit = QLineEdit(str(default_iraf))
        iraf_btn = QPushButton("Browse")
        iraf_btn.clicked.connect(lambda: self._browse_dir(self.iraf_edit))
        iraf_row.addWidget(self.iraf_edit)
        iraf_row.addWidget(iraf_btn)
        iraf_widget = QWidget()
        iraf_widget.setLayout(iraf_row)
        settings_layout.addRow("IRAF Result Dir:", iraf_widget)

        # Tolerance
        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setRange(0.1, 10.0)
        self.tol_spin.setSingleStep(0.1)
        self.tol_spin.setValue(1.5)
        settings_layout.addRow("Match Tolerance (px):", self.tol_spin)

        # Position mode
        self.pos_combo = QComboBox()
        self.pos_combo.addItems(["xcenter/ycenter", "x_init/y_init"])
        settings_layout.addRow("AAPKI Position:", self.pos_combo)

        # IRAF file mode
        self.iraf_combo = QComboBox()
        self.iraf_combo.addItems(["auto", ".txt", ".mag", ".coo"])
        settings_layout.addRow("IRAF File Type:", self.iraf_combo)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # === Control Buttons ===
        btn_layout = QHBoxLayout()

        self.run_btn = QPushButton("Compare")
        self.run_btn.clicked.connect(self.run_comparison)
        btn_layout.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_comparison)
        btn_layout.addWidget(self.stop_btn)

        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self.export_csv)
        btn_layout.addWidget(self.export_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # === Progress ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # === Tabs ===
        tabs = QTabWidget()

        # Tab 1: Summary
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)

        self.summary_label = QLabel("No comparison run yet.")
        self.summary_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        summary_layout.addWidget(self.summary_label)

        # Frame table
        self.table = QTableWidget(0, 13)
        self.table.setHorizontalHeaderLabels([
            "Frame", "Matched", "dmag_med", "dmag_std", "dx_med", "dy_med",
            "dist_med", "dist_p95", "frac<=tol", "shift_x", "shift_y",
            "N_iraf", "N_aapki"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._plot_selected)
        summary_layout.addWidget(self.table)

        tabs.addTab(summary_tab, "Summary")

        # Tab 2: Plot
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)

        self.fig = Figure(figsize=(10, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        tabs.addTab(plot_tab, "Plot")

        # Tab 3: Global Plot
        global_tab = QWidget()
        global_layout = QVBoxLayout(global_tab)

        self.global_fig = Figure(figsize=(10, 6), tight_layout=True)
        self.global_canvas = FigureCanvas(self.global_fig)
        self.global_toolbar = NavigationToolbar(self.global_canvas, self)
        global_layout.addWidget(self.global_toolbar)
        global_layout.addWidget(self.global_canvas)

        tabs.addTab(global_tab, "Global Analysis")

        # Tab 4: Log
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        tabs.addTab(log_tab, "Log")

        layout.addWidget(tabs)

    def _browse_dir(self, line_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(
            self, "Select Directory", line_edit.text()
        )
        if path:
            line_edit.setText(path)

    def run_comparison(self):
        aapki_dir = Path(self.aapki_edit.text())
        iraf_dir = Path(self.iraf_edit.text())

        if not aapki_dir.exists():
            QMessageBox.warning(self, "Error", f"AAPKI directory not found:\n{aapki_dir}")
            return
        if not iraf_dir.exists():
            QMessageBox.warning(self, "Error", f"IRAF directory not found:\n{iraf_dir}")
            return

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()

        self.worker = IRAFComparisonWorker(
            aapki_dir=aapki_dir,
            iraf_dir=iraf_dir,
            tol_px=self.tol_spin.value(),
            pos_mode=self.pos_combo.currentText(),
            iraf_mode=self.iraf_combo.currentText(),
        )

        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._on_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)

        self.worker.start()

    def stop_comparison(self):
        if self.worker:
            self.worker.stop()

    def _on_progress(self, current, total, message):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_log(self, msg):
        self.log_text.append(msg)

    def _on_finished(self, result):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        self.frame_rows = result["frame_rows"]
        self.frame_matches = result["frame_matches"]
        self.matched_all = result["matched_all"]
        global_stats = result["global_stats"]

        # Update summary
        self.summary_label.setText(
            f"Total matched: {global_stats['n_total']} stars from {global_stats['n_frames']} frames | "
            f"dmag median: {global_stats['dmag_med']:.4f} | "
            f"dmag std: {global_stats['dmag_std']:.4f}"
        )

        # Update table
        self._refresh_table()

        # Update global plot
        self._plot_global()

        if self.table.rowCount() > 0:
            self.table.selectRow(0)

    def _on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        QMessageBox.critical(self, "Error", msg)

    def _refresh_table(self):
        self.table.setRowCount(len(self.frame_rows))
        for row, stats in enumerate(self.frame_rows):
            items = [
                stats["frame"],
                f"{stats['n']}",
                f"{stats['dmag_med']:.4f}" if np.isfinite(stats["dmag_med"]) else "nan",
                f"{stats['dmag_std']:.4f}" if np.isfinite(stats["dmag_std"]) else "nan",
                f"{stats['dx_med']:.3f}" if np.isfinite(stats["dx_med"]) else "nan",
                f"{stats['dy_med']:.3f}" if np.isfinite(stats["dy_med"]) else "nan",
                f"{stats['dist_med']:.3f}" if np.isfinite(stats["dist_med"]) else "nan",
                f"{stats['dist_p95']:.3f}" if np.isfinite(stats["dist_p95"]) else "nan",
                f"{stats['frac_within_tol']:.3f}" if np.isfinite(stats["frac_within_tol"]) else "0.000",
                f"{stats['best_shift_x']:.1f}" if np.isfinite(stats["best_shift_x"]) else "nan",
                f"{stats['best_shift_y']:.1f}" if np.isfinite(stats["best_shift_y"]) else "nan",
                str(int(stats["n_iraf_total"])) if np.isfinite(stats["n_iraf_total"]) else "0",
                str(int(stats["n_aapki_total"])) if np.isfinite(stats["n_aapki_total"]) else "0",
            ]
            for col, text in enumerate(items):
                self.table.setItem(row, col, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()

    def _plot_selected(self):
        items = self.table.selectedItems()
        if not items:
            return

        row = items[0].row()
        frame = self.table.item(row, 0).text()
        match = self.frame_matches.get(frame)

        self.fig.clear()

        if match is None or match.empty:
            ax = self.fig.add_subplot(111)
            ax.set_title(f"{frame} (no matches)")
            self.canvas.draw_idle()
            return

        # Create 2x2 subplot
        ax1 = self.fig.add_subplot(221)
        ax2 = self.fig.add_subplot(222)
        ax3 = self.fig.add_subplot(223)
        ax4 = self.fig.add_subplot(224)

        # Plot 1: dmag vs AAPKI mag
        ax1.scatter(match["aapki_mag"], match["dmag"], s=10, alpha=0.6)
        ax1.axhline(0.0, color="red", lw=1, ls="--")
        dmag_med = np.nanmedian(match["dmag"])
        ax1.axhline(dmag_med, color="blue", lw=1, ls=":")
        ax1.set_xlabel("AAPKI mag")
        ax1.set_ylabel("AAPKI - IRAF (mag)")
        ax1.set_title(f"dmag vs mag (med={dmag_med:.4f})")

        # Plot 2: dmag histogram
        dmag_finite = match["dmag"].dropna()
        if len(dmag_finite) > 0:
            ax2.hist(dmag_finite, bins=30, alpha=0.7, edgecolor="black")
            ax2.axvline(0.0, color="red", lw=1, ls="--")
            ax2.axvline(dmag_med, color="blue", lw=1, ls=":")
            ax2.set_xlabel("dmag (AAPKI - IRAF)")
            ax2.set_ylabel("Count")
            ax2.set_title(f"dmag distribution (std={np.nanstd(dmag_finite):.4f})")

        # Plot 3: Position offset
        ax3.scatter(match["dx"], match["dy"], s=10, alpha=0.6)
        ax3.axhline(0.0, color="gray", lw=0.5)
        ax3.axvline(0.0, color="gray", lw=0.5)
        ax3.set_xlabel("dx (AAPKI - IRAF) [px]")
        ax3.set_ylabel("dy (AAPKI - IRAF) [px]")
        ax3.set_title("Position offset")
        ax3.set_aspect("equal")

        # Plot 4: AAPKI vs IRAF mag
        ax4.scatter(match["iraf_mag"], match["aapki_mag"], s=10, alpha=0.6)
        mag_range = [
            min(match["iraf_mag"].min(), match["aapki_mag"].min()),
            max(match["iraf_mag"].max(), match["aapki_mag"].max())
        ]
        ax4.plot(mag_range, mag_range, "r--", lw=1)
        ax4.set_xlabel("IRAF mag")
        ax4.set_ylabel("AAPKI mag")
        ax4.set_title("1:1 comparison")

        self.fig.suptitle(f"{frame} (N={len(match)})", fontsize=12)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _plot_global(self):
        self.global_fig.clear()

        if self.matched_all is None or self.matched_all.empty:
            ax = self.global_fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
            self.global_canvas.draw_idle()
            return

        df = self.matched_all

        ax1 = self.global_fig.add_subplot(231)
        ax2 = self.global_fig.add_subplot(232)
        ax3 = self.global_fig.add_subplot(233)
        ax4 = self.global_fig.add_subplot(234)
        ax5 = self.global_fig.add_subplot(235)
        ax6 = self.global_fig.add_subplot(236)

        # Plot 1: Global dmag vs mag
        ax1.scatter(df["aapki_mag"], df["dmag"], s=3, alpha=0.3)
        ax1.axhline(0.0, color="red", lw=1, ls="--")
        dmag_med = np.nanmedian(df["dmag"])
        ax1.axhline(dmag_med, color="blue", lw=1, ls=":")
        ax1.set_xlabel("AAPKI mag")
        ax1.set_ylabel("dmag")
        ax1.set_title(f"All frames (med={dmag_med:.4f})")

        # Plot 2: dmag histogram
        dmag_finite = df["dmag"].dropna()
        ax2.hist(dmag_finite, bins=50, alpha=0.7, edgecolor="black")
        ax2.axvline(0.0, color="red", lw=1, ls="--")
        ax2.axvline(dmag_med, color="blue", lw=1, ls=":")
        ax2.set_xlabel("dmag")
        ax2.set_ylabel("Count")
        ax2.set_title(f"std={np.nanstd(dmag_finite):.4f}")

        # Plot 3: Position offset
        ax3.scatter(df["dx"], df["dy"], s=3, alpha=0.3)
        ax3.axhline(0.0, color="gray", lw=0.5)
        ax3.axvline(0.0, color="gray", lw=0.5)
        ax3.set_xlabel("dx [px]")
        ax3.set_ylabel("dy [px]")
        ax3.set_title("Position offset")

        # Plot 4: dmag vs frame
        frame_dmag = df.groupby("frame")["dmag"].median()
        ax4.bar(range(len(frame_dmag)), frame_dmag.values, alpha=0.7)
        ax4.axhline(0.0, color="red", lw=1, ls="--")
        ax4.set_xlabel("Frame index")
        ax4.set_ylabel("Median dmag")
        ax4.set_title("Per-frame median dmag")

        # Plot 5: dmag std per frame
        frame_std = df.groupby("frame")["dmag"].std()
        ax5.bar(range(len(frame_std)), frame_std.values, alpha=0.7, color="orange")
        ax5.set_xlabel("Frame index")
        ax5.set_ylabel("dmag std")
        ax5.set_title("Per-frame dmag std")

        # Plot 6: Match count per frame
        frame_n = df.groupby("frame").size()
        ax6.bar(range(len(frame_n)), frame_n.values, alpha=0.7, color="green")
        ax6.set_xlabel("Frame index")
        ax6.set_ylabel("N matched")
        ax6.set_title("Stars per frame")

        self.global_fig.suptitle(
            f"Global Analysis: {len(df)} matched stars from {df['frame'].nunique()} frames",
            fontsize=12
        )
        self.global_fig.tight_layout()
        self.global_canvas.draw_idle()

    def export_csv(self):
        if self.matched_all is None or self.matched_all.empty:
            QMessageBox.information(self, "Export", "No data to export.")
            return

        out_dir = Path(self.aapki_edit.text()) / "iraf_comparison"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Export all matches
        all_path = out_dir / "iraf_compare_all.csv"
        self.matched_all.to_csv(all_path, index=False)

        # Export frame summary
        summary_path = out_dir / "iraf_compare_summary.csv"
        pd.DataFrame(self.frame_rows).to_csv(summary_path, index=False)

        QMessageBox.information(
            self, "Export Complete",
            f"Exported to:\n{all_path}\n{summary_path}"
        )
