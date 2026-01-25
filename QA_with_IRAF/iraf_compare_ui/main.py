import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
    QComboBox,
    QTextEdit,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def _read_iraf_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        names=["ID", "x", "y", "mag", "merr", "msky"],
        engine="python",
    )
    for col in ["ID", "x", "y", "mag", "merr", "msky"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _read_iraf_coo(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=["x", "y", "mag", "sharp", "sround", "ground", "ID"],
        engine="python",
    )
    for col in ["ID", "x", "y", "mag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["ID", "x", "y", "mag"]]


def _read_iraf_mag(path: Path) -> pd.DataFrame:
    def _parse_segments(rows):
        data = []
        seq_id = 1
        for segments in rows:
            x_val = np.nan
            y_val = np.nan
            mag_val = np.nan
            obj_id = None

            if len(segments) > 0:
                seg0 = segments[0].split()
                if len(seg0) >= 3:
                    try:
                        obj_id = int(float(seg0[2]))
                    except Exception:
                        obj_id = None

            if len(segments) > 1:
                seg1 = segments[1].split()
                if len(seg1) >= 2:
                    try:
                        x_val = float(seg1[0])
                        y_val = float(seg1[1])
                    except Exception:
                        x_val = np.nan
                        y_val = np.nan

            if len(segments) > 4:
                seg4 = segments[4].split()
                if len(seg4) >= 5:
                    try:
                        mag_val = float(seg4[4])
                    except Exception:
                        mag_val = np.nan

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
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
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
    return df


def _read_iraf_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".mag":
        return _read_iraf_mag(path)
    if suffix == ".coo":
        return _read_iraf_coo(path)
    return _read_iraf_txt(path)


def _read_aapki_tsv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.read_csv(path)


def _pick_first(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def _normalize_frame_key(stem: str) -> str:
    key = stem
    if key.endswith("_photometry"):
        key = key[: -len("_photometry")]
    if key.endswith(".fit") or key.endswith(".fits"):
        key = key.rsplit(".", 1)[0]
    if key.startswith("Crop_"):
        key = key[len("Crop_") :]
    return key


class CompareUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IRAF vs AAPKI Photometry QA")
        self.resize(1500, 900)

        self.frame_rows = []
        self.frame_matches = {}
        self.matched_all = None

        root = Path(__file__).resolve().parents[2]
        default_aapki = root / "data"
        default_iraf = root / "QA_with_IRAF" / "PyRAF" / "M38" / "result"
        default_out = root / "QA_with_IRAF" / "iraf_compare_ui" / "output"

        path_box = QGroupBox("Paths and Settings")
        path_form = QFormLayout()

        self.aapki_edit = QLineEdit(str(default_aapki))
        self.iraf_edit = QLineEdit(str(default_iraf))
        self.out_edit = QLineEdit(str(default_out))

        aapki_btn = QPushButton("Browse")
        iraf_btn = QPushButton("Browse")
        out_btn = QPushButton("Browse")

        aapki_btn.clicked.connect(lambda: self._pick_dir(self.aapki_edit))
        iraf_btn.clicked.connect(lambda: self._pick_dir(self.iraf_edit))
        out_btn.clicked.connect(lambda: self._pick_dir(self.out_edit))

        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setRange(0.1, 10.0)
        self.tol_spin.setSingleStep(0.1)
        self.tol_spin.setValue(1.5)

        self.pos_combo = QComboBox()
        self.pos_combo.addItems(["xcenter/ycenter", "x_init/y_init"])

        self.iraf_combo = QComboBox()
        self.iraf_combo.addItems([".txt", ".mag", ".coo", "auto (.mag > .coo > .txt)"])

        self.compare_btn = QPushButton("Compare")
        self.export_btn = QPushButton("Export CSV")

        self.compare_btn.clicked.connect(self.compare_all)
        self.export_btn.clicked.connect(self.export_csv)

        def row_with_btn(line_edit, btn):
            row = QHBoxLayout()
            row.addWidget(line_edit)
            row.addWidget(btn)
            w = QWidget()
            w.setLayout(row)
            return w

        path_form.addRow("AAPKI result dir:", row_with_btn(self.aapki_edit, aapki_btn))
        path_form.addRow("IRAF result dir:", row_with_btn(self.iraf_edit, iraf_btn))
        path_form.addRow("Output dir:", row_with_btn(self.out_edit, out_btn))
        path_form.addRow("Match tolerance (px):", self.tol_spin)
        path_form.addRow("AAPKI position:", self.pos_combo)
        path_form.addRow("IRAF input:", self.iraf_combo)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.compare_btn)
        btn_row.addWidget(self.export_btn)
        btn_row.addStretch(1)
        btn_wrap = QWidget()
        btn_wrap.setLayout(btn_row)
        path_form.addRow("", btn_wrap)

        path_box.setLayout(path_form)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Frame", "Matched", "dmag_med", "dmag_std", "dx_med", "dy_med"]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.itemSelectionChanged.connect(self._plot_selected)

        self.summary_label = QLabel("No data loaded.")
        self.summary_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(160)

        self.fig = Figure(figsize=(5, 4), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        right = QVBoxLayout()
        right.addWidget(self.summary_label)
        right.addWidget(self.log_box)
        right.addWidget(self.canvas)
        right_w = QWidget()
        right_w.setLayout(right)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.table)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        layout = QVBoxLayout()
        layout.addWidget(path_box)
        layout.addWidget(splitter)
        self.setLayout(layout)

    def _pick_dir(self, line_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Select directory", line_edit.text())
        if path:
            line_edit.setText(path)

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

    def _match_frame(self, aapki_path: Path, iraf_path: Path, tol_px: float, pos_mode: str):
        aapki = _read_aapki_tsv(aapki_path)
        iraf = _read_iraf_file(iraf_path)

        mag_col = _pick_first(aapki.columns, ["mag_inst", "mag", "MAG", "mag_raw"])
        if mag_col is None:
            raise ValueError(f"Missing mag column in {aapki_path}")

        if pos_mode.startswith("xcenter"):
            x_col = _pick_first(aapki.columns, ["xcenter", "x", "x_init"])
            y_col = _pick_first(aapki.columns, ["ycenter", "y", "y_init"])
        else:
            x_col = _pick_first(aapki.columns, ["x_init", "xcenter", "x"])
            y_col = _pick_first(aapki.columns, ["y_init", "ycenter", "y"])

        if x_col is None or y_col is None:
            raise ValueError(f"Missing XY columns in {aapki_path}")

        axy = aapki[[x_col, y_col]].to_numpy(float)
        ixy = iraf[["x", "y"]].to_numpy(float)
        axy_finite = np.isfinite(axy[:, 0]) & np.isfinite(axy[:, 1])
        ixy_finite = np.isfinite(ixy[:, 0]) & np.isfinite(ixy[:, 1])

        if axy.size == 0 or ixy.size == 0:
            return pd.DataFrame(), {
                "x_col": x_col,
                "y_col": y_col,
                "mag_col": mag_col,
                "iraf_suffix": iraf_path.suffix.lower(),
                "nearest_med": np.nan,
                "offset_dx_med": np.nan,
                "offset_dy_med": np.nan,
                "aapki_rows": int(len(axy)),
                "aapki_finite_xy": int(np.sum(axy_finite)),
                "iraf_rows": int(len(ixy)),
                "iraf_finite_xy": int(np.sum(ixy_finite)),
            }

        tree = cKDTree(axy)
        dist_all, idx_all = tree.query(ixy)
        finite_all = np.isfinite(dist_all)
        if np.any(finite_all):
            dx_all = axy[idx_all[finite_all], 0] - ixy[finite_all, 0]
            dy_all = axy[idx_all[finite_all], 1] - ixy[finite_all, 1]
            nearest_med = float(np.nanmedian(dist_all[finite_all]))
            offset_dx_med = float(np.nanmedian(dx_all))
            offset_dy_med = float(np.nanmedian(dy_all))
        else:
            nearest_med = np.nan
            offset_dx_med = np.nan
            offset_dy_med = np.nan

        dist, idx = tree.query(ixy, distance_upper_bound=tol_px)
        mask = np.isfinite(dist) & (dist <= tol_px)
        if not np.any(mask):
            return pd.DataFrame(), {
                "x_col": x_col,
                "y_col": y_col,
                "mag_col": mag_col,
                "iraf_suffix": iraf_path.suffix.lower(),
                "nearest_med": nearest_med,
                "offset_dx_med": offset_dx_med,
                "offset_dy_med": offset_dy_med,
                "aapki_rows": int(len(axy)),
                "aapki_finite_xy": int(np.sum(axy_finite)),
                "iraf_rows": int(len(ixy)),
                "iraf_finite_xy": int(np.sum(ixy_finite)),
            }

        match = pd.DataFrame(
            {
                "iraf_id": iraf.loc[mask, "ID"].to_numpy(),
                "iraf_x": iraf.loc[mask, "x"].to_numpy(),
                "iraf_y": iraf.loc[mask, "y"].to_numpy(),
                "iraf_mag": iraf.loc[mask, "mag"].to_numpy(),
                "aapki_x": aapki.loc[idx[mask], x_col].to_numpy(),
                "aapki_y": aapki.loc[idx[mask], y_col].to_numpy(),
                "aapki_mag": aapki.loc[idx[mask], mag_col].to_numpy(),
                "dist_px": dist[mask],
            }
        )
        match["dx"] = match["aapki_x"] - match["iraf_x"]
        match["dy"] = match["aapki_y"] - match["iraf_y"]
        match["dmag"] = match["aapki_mag"] - match["iraf_mag"]
        return match, {
            "x_col": x_col,
            "y_col": y_col,
            "mag_col": mag_col,
            "iraf_suffix": iraf_path.suffix.lower(),
            "nearest_med": nearest_med,
            "offset_dx_med": offset_dx_med,
            "offset_dy_med": offset_dy_med,
            "aapki_rows": int(len(axy)),
            "aapki_finite_xy": int(np.sum(axy_finite)),
            "iraf_rows": int(len(ixy)),
            "iraf_finite_xy": int(np.sum(ixy_finite)),
        }

    def _append_log(self, text: str):
        if self.log_box.toPlainText():
            self.log_box.append(text)
        else:
            self.log_box.setPlainText(text)

    def _nan_stats(self, series: pd.Series):
        values = series.to_numpy(float)
        finite = np.isfinite(values)
        if not np.any(finite):
            return np.nan, np.nan, 0
        return float(np.nanmedian(values)), float(np.nanstd(values)), int(np.sum(finite))

    def _range_text(self, values: np.ndarray) -> str:
        finite = np.isfinite(values)
        if not np.any(finite):
            return "nan..nan"
        return f"{float(np.nanmin(values)):.1f}..{float(np.nanmax(values)):.1f}"

    def compare_all(self):
        try:
            aapki_dir = Path(self.aapki_edit.text()).expanduser()
            iraf_dir = Path(self.iraf_edit.text()).expanduser()
            if not aapki_dir.exists():
                raise FileNotFoundError(f"AAPKI dir not found: {aapki_dir}")
            if not iraf_dir.exists():
                raise FileNotFoundError(f"IRAF dir not found: {iraf_dir}")

            aapki_map = self._collect_aapki(aapki_dir)
            iraf_mode = self.iraf_combo.currentText()
            iraf_map = self._collect_iraf(iraf_dir, iraf_mode)
            frames = sorted(set(aapki_map) & set(iraf_map))
            if not frames:
                raise RuntimeError("No matching frames found.")

            tol_px = float(self.tol_spin.value())
            pos_mode = self.pos_combo.currentText()

            self.frame_rows = []
            self.frame_matches = {}
            all_matches = []
            self.log_box.clear()
            self._append_log(
                f"AAPKI files: {len(aapki_map)} | IRAF files: {len(iraf_map)} | "
                f"Matched frames: {len(frames)}"
            )
            self._append_log(f"Tolerance: {tol_px:.2f} px | Position mode: {pos_mode}")
            self._append_log(f"IRAF input mode: {iraf_mode}")
            self._append_log(f"Example frame key: {frames[0]}")

            for frame in frames:
                match, info = self._match_frame(
                    aapki_map[frame], iraf_map[frame], tol_px, pos_mode
                )
                self.frame_matches[frame] = match
                if not match.empty:
                    all_matches.append(match.assign(frame=frame))

                dmag_med, dmag_std, dmag_n = self._nan_stats(match["dmag"]) if len(match) else (np.nan, np.nan, 0)
                dx_med, _, _ = self._nan_stats(match["dx"]) if len(match) else (np.nan, np.nan, 0)
                dy_med, _, _ = self._nan_stats(match["dy"]) if len(match) else (np.nan, np.nan, 0)

                stats = {
                    "frame": frame,
                    "n": int(len(match)),
                    "dmag_med": dmag_med,
                    "dmag_std": dmag_std,
                    "dx_med": dx_med,
                    "dy_med": dy_med,
                }
                self.frame_rows.append(stats)
                aapki_xy = _read_aapki_tsv(aapki_map[frame])[[info["x_col"], info["y_col"]]].to_numpy(float)
                iraf_xy = _read_iraf_file(iraf_map[frame])[["x", "y"]].to_numpy(float)
                self._append_log(
                    f"{frame}: AAPKI({info['x_col']},{info['y_col']},{info['mag_col']}) "
                    f"IRAF({info['iraf_suffix']}) match {len(match)} (finite dmag {dmag_n})"
                )
                self._append_log(
                    f"  AAPKI x/y range {self._range_text(aapki_xy[:, 0])}, "
                    f"{self._range_text(aapki_xy[:, 1])} | "
                    f"IRAF x/y range {self._range_text(iraf_xy[:, 0])}, "
                    f"{self._range_text(iraf_xy[:, 1])}"
                )
                self._append_log(
                    f"  AAPKI rows {info['aapki_rows']} (finite {info['aapki_finite_xy']}) | "
                    f"IRAF rows {info['iraf_rows']} (finite {info['iraf_finite_xy']})"
                )
                self._append_log(
                    f"  Nearest dist med {info['nearest_med']:.2f} px | "
                    f"offset med dx {info['offset_dx_med']:.2f}, dy {info['offset_dy_med']:.2f}"
                )

            self.matched_all = pd.concat(all_matches, ignore_index=True) if all_matches else None
            self._refresh_table()
            self._update_summary()
            if self.table.rowCount() > 0:
                self.table.selectRow(0)
        except Exception as exc:
            QMessageBox.critical(self, "Compare failed", str(exc))

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
            ]
            for col, text in enumerate(items):
                self.table.setItem(row, col, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()

    def _update_summary(self):
        if self.matched_all is None or self.matched_all.empty:
            self.summary_label.setText("No matches found.")
            self.ax.clear()
            self.canvas.draw_idle()
            return
        dmag_med, dmag_std, dmag_n = self._nan_stats(self.matched_all["dmag"])
        msg = (
            f"Total matched: {len(self.matched_all)} | "
            f"dmag median: {dmag_med:.4f} | "
            f"dmag std: {dmag_std:.4f} | "
            f"dmag finite: {dmag_n}"
        )
        self.summary_label.setText(msg)

    def _plot_selected(self):
        items = self.table.selectedItems()
        if not items:
            return
        row = items[0].row()
        frame = self.table.item(row, 0).text()
        match = self.frame_matches.get(frame)
        if match is None or match.empty:
            self.ax.clear()
            self.ax.set_title(f"{frame} (no matches)")
            self.canvas.draw_idle()
            return

        self.ax.clear()
        self.ax.scatter(match["aapki_mag"], match["dmag"], s=12, alpha=0.6)
        self.ax.axhline(0.0, color="gray", lw=1)
        self.ax.set_xlabel("AAPKI mag")
        self.ax.set_ylabel("AAPKI - IRAF (mag)")
        self.ax.set_title(f"{frame} (N={len(match)})")
        self.canvas.draw_idle()

    def export_csv(self):
        if self.matched_all is None or self.matched_all.empty:
            QMessageBox.information(self, "Export", "No matched data to export.")
            return
        out_dir = Path(self.out_edit.text()).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "iraf_compare_all.csv"
        self.matched_all.to_csv(out_path, index=False)
        QMessageBox.information(self, "Export", f"Saved: {out_path}")


def main():
    app = QApplication(sys.argv)
    ui = CompareUI()
    ui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
