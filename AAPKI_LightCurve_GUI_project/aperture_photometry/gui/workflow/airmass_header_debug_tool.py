"""
Airmass Header Debug Tool
Compare header AIRMASS to values computed from OBJCTALT or RA/Dec+time.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QCheckBox, QGroupBox, QFileDialog, QMessageBox,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from ...core.file_manager import FileManager
from ...utils.astro_utils import (
    AIRMASS_FORMULAS,
    DEFAULT_AIRMASS_FORMULA,
    airmass_from_alt,
    compute_airmass_from_header,
)
from ...utils.step_paths import step1_dir, step5_dir


_DATE_RE = re.compile(r"(\\d{8})")


def _safe_float(value, default=np.nan) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _parse_angle_deg(value) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return np.nan
    s_up = s.upper()
    sign = -1.0 if s_up.startswith("-") else 1.0
    if "S" in s_up or "W" in s_up:
        sign = -1.0
    s_clean = s_up.replace(":", " ")
    s_clean = re.sub(r"[A-Z]", " ", s_clean)
    parts = [p for p in re.split(r"[ ,]+", s_clean) if p]
    try:
        if len(parts) == 1:
            return sign * float(parts[0])
        deg = float(parts[0])
        minutes = float(parts[1]) if len(parts) > 1 else 0.0
        seconds = float(parts[2]) if len(parts) > 2 else 0.0
        mag = abs(deg) + (minutes / 60.0) + (seconds / 3600.0)
        if deg < 0:
            sign = -1.0
        return sign * mag
    except Exception:
        return np.nan


def _parse_time_jd(header: fits.Header) -> float:
    for key in ("JD", "JULIAN", "BJD", "HJD", "MJD-OBS", "MJD"):
        if key in header:
            val = _safe_float(header.get(key), np.nan)
            if np.isfinite(val):
                if key.startswith("MJD"):
                    return float(val + 2400000.5)
                return float(val)
    date_obs = header.get("DATE-OBS") or header.get("DATE")
    time_obs = header.get("TIME-OBS") or header.get("UTC") or header.get("UT")
    if date_obs:
        dt_str = str(date_obs).strip()
        if "T" not in dt_str and time_obs:
            dt_str = f"{dt_str}T{str(time_obs).strip()}"
        try:
            t = Time(dt_str, format="isot", scale="utc")
            return float(t.jd)
        except Exception:
            pass
        try:
            t = Time(dt_str, scale="utc")
            return float(t.jd)
        except Exception:
            pass
    return np.nan




def _parse_date_key(value: str, params) -> str:
    mode = str(getattr(params.P, "night_parse_mode", "regex") or "regex").strip().lower()
    if mode == "split":
        delim = str(getattr(params.P, "night_parse_split_delim", "_"))
        parts = value.split(delim) if delim else [value]
        idx = int(getattr(params.P, "night_parse_split_index", -1))
        if idx < 0:
            idx = len(parts) + idx
        if idx < 0 or idx >= len(parts):
            return "unknown_date"
        return parts[idx]
    if mode == "last_digits":
        n_digits = max(1, int(getattr(params.P, "night_parse_last_digits", 8)))
        m = re.search(rf"(\\d{{{n_digits}}})$", value)
        return m.group(1) if m else "unknown_date"
    try:
        pattern = str(getattr(params.P, "night_parse_regex", r".*_(\\d{8})"))
        m = re.search(pattern, value)
    except re.error:
        return "unknown_date"
    if not m:
        return "unknown_date"
    if m.groupdict().get("date"):
        return m.group("date")
    if m.groups():
        return m.group(1)
    return m.group(0)


def _extract_date_key(filename: str, params=None) -> str:
    if params is None or not hasattr(params, "P"):
        match = _DATE_RE.search(str(filename))
        return match.group(1) if match else "unknown_date"
    date_key = None
    try:
        data_dir = Path(getattr(params.P, "data_dir", "."))
        file_path = Path(params.get_file_path(filename))
        if file_path.parent != data_dir:
            date_key = _parse_date_key(file_path.parent.name, params)
        if not date_key:
            date_key = _parse_date_key(file_path.name, params)
    except Exception:
        date_key = None
    if not date_key:
        date_key = _parse_date_key(str(filename), params)
    return date_key or "unknown_date"


class AirmassHeaderDebugToolWindow(QWidget):
    """Plot header AIRMASS vs computed values by date."""

    def __init__(self, params, project_state, parent=None, file_manager: FileManager | None = None):
        super().__init__(parent)
        self.params = params
        self.project_state = project_state
        self.file_manager = file_manager or FileManager(params)
        self._init_path_map()
        self.data_df = pd.DataFrame()

        self.setWindowTitle("Airmass Header Debug")
        self.setWindowFlag(Qt.Window, True)
        self.resize(1100, 700)
        self.setMinimumSize(900, 600)

        self._setup_ui()
        self._load_data()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        controls = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls)

        controls_layout.addWidget(QLabel("Date:"))
        self.date_combo = QComboBox()
        controls_layout.addWidget(self.date_combo)

        controls_layout.addWidget(QLabel("X axis:"))
        self.xaxis_combo = QComboBox()
        self.xaxis_combo.addItems(["Time (JD)", "Index"])
        controls_layout.addWidget(self.xaxis_combo)

        controls_layout.addWidget(QLabel("Formula:"))
        self.formula_combo = QComboBox()
        self.formula_combo.addItems(list(AIRMASS_FORMULAS.keys()))
        default_formula = str(getattr(self.params.P, "airmass_formula", DEFAULT_AIRMASS_FORMULA) or DEFAULT_AIRMASS_FORMULA)
        if default_formula in AIRMASS_FORMULAS:
            self.formula_combo.setCurrentText(default_formula)
        controls_layout.addWidget(self.formula_combo)

        self.chk_objalt = QCheckBox("OBJCTALT airmass")
        self.chk_objalt.setChecked(True)
        controls_layout.addWidget(self.chk_objalt)

        self.chk_radec = QCheckBox("RA/Dec airmass")
        self.chk_radec.setChecked(True)
        controls_layout.addWidget(self.chk_radec)

        controls_layout.addWidget(QLabel("Write source:"))
        self.write_source_combo = QComboBox()
        self.write_source_combo.addItems(["Auto (RA/Dec → ALT)", "RA/Dec only", "OBJCTALT only"])
        self.write_source_combo.setItemData(0, "auto")
        self.write_source_combo.setItemData(1, "radec")
        self.write_source_combo.setItemData(2, "alt")
        current_source = str(getattr(self.params.P, "airmass_update_source", "auto") or "auto")
        idx = self.write_source_combo.findData(current_source)
        if idx >= 0:
            self.write_source_combo.setCurrentIndex(idx)
        controls_layout.addWidget(self.write_source_combo)

        controls_layout.addStretch()

        self.btn_reload = QPushButton("Reload")
        controls_layout.addWidget(self.btn_reload)
        self.btn_export = QPushButton("Export CSV")
        controls_layout.addWidget(self.btn_export)
        self.btn_write = QPushButton("Write AIRMASS")
        controls_layout.addWidget(self.btn_write)

        layout.addWidget(controls)

        plot_group = QGroupBox("Airmass Comparison (Header vs Computed)")
        plot_layout = QVBoxLayout(plot_group)
        self.figure = Figure(figsize=(9, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        layout.addWidget(plot_group)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("QLabel { color: #555; }")
        layout.addWidget(self.status_label)

        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("QLabel { color: #333; }")
        self.stats_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)

        self.btn_reload.clicked.connect(self._load_data)
        self.btn_export.clicked.connect(self._export_csv)
        self.btn_write.clicked.connect(self._write_airmass_headers)
        self.date_combo.currentIndexChanged.connect(self._update_plot)
        self.xaxis_combo.currentIndexChanged.connect(self._update_plot)
        self.formula_combo.currentIndexChanged.connect(self._update_plot)
        self.chk_objalt.stateChanged.connect(self._update_plot)
        self.chk_radec.stateChanged.connect(self._update_plot)

    def _init_path_map(self) -> None:
        path_map = getattr(self.params.P, "file_path_map", None)
        if isinstance(path_map, dict) and path_map:
            self.file_manager.path_map = {k: Path(v) for k, v in path_map.items() if v}

    def _make_file_key(self, rel_path: Path) -> str:
        parts = [p for p in rel_path.parts if p not in (".", "")]
        return "__".join(parts)

    def _ensure_unique_key(self, key: str, existing: dict) -> str:
        if key not in existing:
            return key
        base = key
        idx = 2
        while f"{base}__dup{idx}" in existing:
            idx += 1
        return f"{base}__dup{idx}"

    def _load_target_list(self) -> list[str]:
        self._restore_file_selection_state()
        try:
            files = self.file_manager.scan_files()
            if files:
                return files
        except Exception:
            pass
        step1_out = step1_dir(self.params.P.result_dir)
        target_list = step1_out / "target_list.txt"
        if target_list.exists():
            lines = target_list.read_text(encoding="utf-8").splitlines()
            cached = [ln.strip() for ln in lines if ln.strip()]
            if cached:
                return cached
        return self._scan_all_fits()

    def _restore_file_selection_state(self) -> None:
        state_data = self.project_state.get_step_data("file_selection") if self.project_state else {}
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
        multi_night = bool(state_data.get("multi_night"))
        root_dir = state_data.get("root_dir") or data_dir
        night_dirs = [Path(p) for p in state_data.get("night_dirs", []) if p]
        if multi_night and night_dirs:
            root_path = Path(root_dir) if root_dir else Path(self.params.P.data_dir)
            self.file_manager.set_multi_night_dirs(root_path, night_dirs)
        else:
            self.file_manager.clear_multi_night_dirs()

    def _scan_all_fits(self) -> list[str]:
        data_dir = Path(getattr(self.params.P, "data_dir", "."))
        prefix = str(getattr(self.params.P, "filename_prefix", "")).lower()
        suffixes = (".fit", ".fits", ".fit.fz", ".fits.fz")

        file_items = []
        seen_keys = {}
        selected_dirs = [Path(p) for p in getattr(self.file_manager, "selected_dirs", []) if p]
        root_dir = Path(getattr(self.file_manager, "root_dir", data_dir))

        if selected_dirs:
            for subdir in selected_dirs:
                if not subdir.exists():
                    continue
                for fpath in sorted([p for p in subdir.iterdir() if p.is_file()]):
                    name = fpath.name
                    lower = name.lower()
                    if prefix and not lower.startswith(prefix):
                        continue
                    if not lower.endswith(suffixes):
                        continue
                    try:
                        rel = fpath.relative_to(root_dir)
                    except ValueError:
                        rel = Path(subdir.name) / fpath.name
                    key = self._make_file_key(rel)
                    key = self._ensure_unique_key(key, seen_keys)
                    seen_keys[key] = True
                    file_items.append((key, fpath))
        else:
            for fpath in sorted([p for p in data_dir.rglob("*") if p.is_file()]):
                name = fpath.name
                lower = name.lower()
                if not lower.endswith(suffixes):
                    continue
                if prefix and not lower.startswith(prefix):
                    continue
                try:
                    rel = fpath.relative_to(root_dir)
                except ValueError:
                    rel = fpath.relative_to(data_dir)
                key = self._make_file_key(rel)
                key = self._ensure_unique_key(key, seen_keys)
                seen_keys[key] = True
                file_items.append((key, fpath))

        self.file_manager.path_map = {k: v for k, v in file_items}
        self.params.P.file_path_map = {k: str(v) for k, v in self.file_manager.path_map.items()}

        return [k for k, _ in file_items]

    def _resolve_fits_path(self, fname: str) -> Path | None:
        if fname in self.file_manager.path_map:
            return Path(self.file_manager.path_map[fname])
        try:
            return Path(self.params.get_file_path(fname))
        except Exception:
            data_dir = Path(getattr(self.params.P, "data_dir", "."))
            key = re.sub(r"__dup\\d+$", "", str(fname))
            if "__" in key:
                rel = Path(*key.split("__"))
                root_dir = getattr(self.file_manager, "root_dir", None)
                base_dir = Path(root_dir) if root_dir else data_dir
                cand_rel = base_dir / rel
                if cand_rel.exists():
                    return cand_rel
                cand_rel = data_dir / rel
                if cand_rel.exists():
                    return cand_rel
            cand = data_dir / fname
            if cand.exists():
                return cand
            for pat in ("*.fit", "*.fits", "*.fit.fz", "*.fits.fz"):
                for p in data_dir.rglob(pat):
                    if p.name == fname:
                        return p
            return None

    def _resolve_site(self, header: fits.Header) -> tuple[float, float, float]:
        lat_val = header.get("SITELAT") or header.get("SITE_LAT") or header.get("OBSLAT") or header.get("LATITUDE")
        lon_val = header.get("SITELONG") or header.get("SITE_LON") or header.get("OBSLONG") \
            or header.get("LONGITUD") or header.get("LONGITUDE")
        alt_val = header.get("SITEALT") or header.get("OBSALT") or header.get("ALTITUDE") \
            or header.get("ELEVATION") or header.get("SITEELEV")
        lat = _parse_angle_deg(lat_val)
        lon = _parse_angle_deg(lon_val)
        alt = _safe_float(alt_val, np.nan)
        if np.isfinite(lat) and np.isfinite(lon) and np.isfinite(alt):
            return lat, lon, alt
        lat_p = _safe_float(getattr(self.params.P, "site_lat_deg", np.nan), np.nan)
        lon_p = _safe_float(getattr(self.params.P, "site_lon_deg", np.nan), np.nan)
        alt_p = _safe_float(getattr(self.params.P, "site_alt_m", np.nan), np.nan)
        return lat_p, lon_p, alt_p

    def _load_data(self):
        files = self._load_target_list()
        if not files:
            QMessageBox.warning(self, "Airmass Debug", "No input files found.")
            return

        rows = []
        missing = 0
        read_fail = 0
        for idx, fname in enumerate(files):
            path = self._resolve_fits_path(fname)
            if path is None or not path.exists():
                missing += 1
                continue
            try:
                with fits.open(path) as hdul:
                    h = hdul[0].header
            except Exception:
                read_fail += 1
                continue

            hdr_airmass = _safe_float(h.get("AIRMASS"), np.nan)
            obj_alt = _parse_angle_deg(h.get("OBJCTALT") or h.get("OBJALT"))
            obj_az = _parse_angle_deg(h.get("OBJCTAZ") or h.get("OBJAZ"))
            lat, lon, alt = self._resolve_site(h)
            alt_radec = np.nan
            if np.isfinite(lat) and np.isfinite(lon) and np.isfinite(alt):
                info = compute_airmass_from_header(h, lat, lon, alt, 0.0)
                alt_radec = _safe_float(info.get("alt_deg"), np.nan)

            jd_val = _parse_time_jd(h)
            date_key = _extract_date_key(fname, self.params)

            rows.append({
                "file": fname,
                "index": idx,
                "date_key": date_key,
                "jd": jd_val,
                "airmass_hdr": hdr_airmass,
                "obj_alt": obj_alt,
                "obj_az": obj_az,
                "alt_radec": alt_radec,
            })

        self.data_df = pd.DataFrame(rows)
        self._refresh_date_combo()
        self._update_plot()
        if rows:
            self.status_label.setText(
                f"Loaded={len(rows)} | missing={missing} | read_fail={read_fail}"
            )

    def _refresh_date_combo(self):
        self.date_combo.blockSignals(True)
        self.date_combo.clear()
        if self.data_df.empty:
            self.date_combo.addItem("All Dates")
            self.date_combo.blockSignals(False)
            return
        dates = sorted(self.data_df["date_key"].fillna("unknown_date").astype(str).unique().tolist())
        self.date_combo.addItem("All Dates")
        for d in dates:
            self.date_combo.addItem(d)
        if dates:
            self.date_combo.setCurrentText(dates[0])
        self.date_combo.blockSignals(False)

    def _subset_df(self) -> pd.DataFrame:
        if self.data_df.empty:
            return self.data_df
        sel = self.date_combo.currentText()
        if sel and sel != "All Dates":
            return self.data_df[self.data_df["date_key"] == sel].copy()
        return self.data_df.copy()

    def _update_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(2, 1, 1)
        ax_diff = self.figure.add_subplot(2, 1, 2, sharex=ax)

        df = self._subset_df()
        if df.empty:
            self.canvas.draw_idle()
            return

        use_time = self.xaxis_combo.currentIndex() == 0
        x_vals = df["jd"].to_numpy(float)
        x_label = "JD"
        if not np.isfinite(x_vals).any() or not use_time:
            x_vals = df["index"].to_numpy(float)
            x_label = "Index"
        else:
            x_vals = x_vals - np.nanmin(x_vals)
            x_label = "JD - JD0"

        hdr = df["airmass_hdr"].to_numpy(float)
        alt_obj = df["obj_alt"].to_numpy(float)
        alt_rd = df["alt_radec"].to_numpy(float)

        formula_name = self.formula_combo.currentText()
        objalt = airmass_from_alt(alt_obj, formula_name)
        radec = airmass_from_alt(alt_rd, formula_name)

        ax.scatter(x_vals, hdr, s=22, color="#212121", alpha=0.8, label="Header AIRMASS")

        if self.chk_objalt.isChecked():
            ax.scatter(x_vals, objalt, s=20, color="#E53935", alpha=0.8, label="OBJCTALT AIRMASS")
            ax_diff.scatter(x_vals, hdr - objalt, s=16, color="#E53935", alpha=0.7, label="HDR-OBJCTALT")

        if self.chk_radec.isChecked():
            order = np.argsort(x_vals)
            x_sorted = x_vals[order]
            r_sorted = radec[order]
            ax.plot(x_sorted, r_sorted, color="#1E88E5", alpha=0.9, linewidth=1.6, label="RA/Dec AIRMASS")
            ax_diff.plot(x_sorted, (hdr - radec)[order], color="#1E88E5", alpha=0.9, linewidth=1.2, label="HDR-RA/Dec")

        ax.set_ylabel("Airmass")
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=8, frameon=False)

        ax_diff.axhline(0.0, color="#9E9E9E", linewidth=1.0, linestyle="--")
        ax_diff.set_ylabel("Diff")
        ax_diff.set_xlabel(x_label)
        ax_diff.grid(True, alpha=0.2)
        ax_diff.legend(loc="best", fontsize=8, frameon=False)

        sel = self.date_combo.currentText()
        n_total = len(df)
        n_hdr = int(np.isfinite(hdr).sum())
        n_obj = int(np.isfinite(objalt).sum())
        n_rd = int(np.isfinite(radec).sum())
        alt_std = np.nanstd(objalt) if np.isfinite(objalt).any() else np.nan
        alt_note = ""
        if np.isfinite(alt_std) and alt_std < 1e-3:
            alt_note = " | OBJCTALT looks constant"
        self.status_label.setText(
            f"Date={sel} | frames={n_total} | hdr={n_hdr} | objalt={n_obj} | radec={n_rd} | "
            f"formula={formula_name}{alt_note}"
        )

        self.stats_label.setText(self._build_residual_summary(df))

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _build_residual_summary(self, df: pd.DataFrame) -> str:
        if df.empty:
            return ""
        hdr = df["airmass_hdr"].to_numpy(float)
        alt_obj = df["obj_alt"].to_numpy(float)
        alt_rd = df["alt_radec"].to_numpy(float)

        lines = []
        best_obj = ("", np.nan)
        best_rd = ("", np.nan)

        def _rms(x):
            x = x[np.isfinite(x)]
            if len(x) == 0:
                return np.nan
            return float(np.sqrt(np.nanmean(x * x)))

        for name in AIRMASS_FORMULAS.keys():
            am_obj = airmass_from_alt(alt_obj, name)
            am_rd = airmass_from_alt(alt_rd, name)
            diff_obj = hdr - am_obj
            diff_rd = hdr - am_rd
            rms_obj = _rms(diff_obj)
            rms_rd = _rms(diff_rd)
            if np.isfinite(rms_obj):
                if not np.isfinite(best_obj[1]) or rms_obj < best_obj[1]:
                    best_obj = (name, rms_obj)
            if np.isfinite(rms_rd):
                if not np.isfinite(best_rd[1]) or rms_rd < best_rd[1]:
                    best_rd = (name, rms_rd)
            lines.append(
                f"{name}: RMS(HDR-OBJ)={rms_obj:.5f} | RMS(HDR-RD)={rms_rd:.5f}"
            )

        best_lines = []
        if best_obj[0]:
            best_lines.append(f"Best OBJCTALT match: {best_obj[0]} (RMS={best_obj[1]:.5f})")
        if best_rd[0]:
            best_lines.append(f"Best RA/Dec match: {best_rd[0]} (RMS={best_rd[1]:.5f})")

        return "Residual RMS by formula:\n" + "\n".join(lines) + "\n" + "\n".join(best_lines)

    def _export_csv(self):
        if self.data_df.empty:
            return
        out_dir = step5_dir(self.params.P.result_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        default_path = out_dir / "airmass_header_debug.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", str(default_path), "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            self.data_df.to_csv(path, index=False)
            QMessageBox.information(self, "Airmass Debug", f"Saved: {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Airmass Debug", f"Save failed: {exc}")

    def _write_airmass_headers(self):
        source = self.write_source_combo.currentData() or "auto"
        formula = self.formula_combo.currentText()
        mode = str(getattr(self.params.P, "airmass_update_mode", "overwrite") or "overwrite")
        try:
            stats = self.file_manager.update_airmass_headers(formula, mode, source)
            self._load_data()
            if stats:
                msg = (
                    f"Updated: {stats.get('updated', 0)}\n"
                    f"Skipped (has AIRMASS): {stats.get('skipped_has_airmass', 0)}\n"
                    f"Skipped (missing alt/time/ra/dec): {stats.get('skipped_missing_alt', 0)}\n"
                    f"Skipped (compressed .fz): {stats.get('skipped_compressed', 0)}\n"
                    f"Failed: {stats.get('failed', 0)}"
                )
                QMessageBox.information(self, "Airmass Update", msg)
        except Exception as exc:
            QMessageBox.warning(self, "Airmass Update", f"Failed: {exc}")
