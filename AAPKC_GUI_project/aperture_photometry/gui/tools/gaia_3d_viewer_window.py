"""
Gaia 3D Cluster Viewer
Visualize cluster structure in Gaia astrometric 3D space.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D, proj3d  # noqa: F401

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox,
    QFileDialog, QLineEdit, QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox,
    QTextEdit, QGroupBox, QDialog, QDialogButtonBox, QFormLayout, QProgressDialog,
    QApplication,
)

from ...utils.step_paths import step5_dir, step6_dir
from ...utils.io_utils import parse_int64_series, read_ecsv_int64_source_id


# ---------------------------------------------------------------------------
# Stellar color system (same as CmdViewerWindow in step10)
# ---------------------------------------------------------------------------
_TEFF_VMIN = 2400.0
_TEFF_VMAX = 40000.0

_TEFF_COLOR_ANCHORS: dict[str, dict] = {
    "g-r": {
        "x": np.array([-0.40, -0.20, 0.00, 0.30, 0.45, 0.80, 1.40, 1.80], float),
        "t": np.array([35000, 20000, 10000, 7500, 6000, 4500, 3200, 2400], float),
    },
    "BP-RP": {
        "x": np.array([-0.40, -0.20, 0.00, 0.40, 0.80, 1.30, 2.30, 3.50], float),
        "t": np.array([35000, 20000, 10000, 7500, 6000, 4500, 3200, 2400], float),
    },
}


def _color_to_teff(color_arr: np.ndarray, mode: str) -> np.ndarray:
    """Interpolate color index → Teff (K), matching step10 logic."""
    a = _TEFF_COLOR_ANCHORS.get(mode)
    if a is None:
        return np.full(len(color_arr), np.nan, float)
    teff = np.interp(color_arr, a["x"], a["t"])
    return np.clip(teff, _TEFF_VMIN, _TEFF_VMAX)


def _make_stellar_cmap() -> LinearSegmentedColormap:
    """OBAFGKM-like colormap keyed to Teff (same anchors as step10)."""
    anchors = [
        (2400,  "#E53935"),
        (3200,  "#FF6A3D"),
        (4500,  "#FFB84D"),
        (5800,  "#FFE36A"),
        (6500,  "#FFF6C7"),
        (8000,  "#FFFFFF"),
        (10000, "#FFFFFF"),
        (20000, "#2D5BFF"),
        (40000, "#7A3CFF"),
    ]
    anchors = sorted(anchors, key=lambda a: a[0])
    pos = [(t - _TEFF_VMIN) / (_TEFF_VMAX - _TEFF_VMIN) for t, _ in anchors]
    pos[0], pos[-1] = 0.0, 1.0
    cmap = LinearSegmentedColormap.from_list(
        "stellar_teff",
        list(zip(pos, [c for _, c in anchors])),
        N=256,
    )
    cmap.set_bad("#777777")
    return cmap


_STELLAR_CMAP = _make_stellar_cmap()
_STELLAR_NORM = Normalize(vmin=_TEFF_VMIN, vmax=_TEFF_VMAX, clip=True)

_MAX_PM_ARROWS = 3000   # quiver arrow limit to keep interactive performance


# ---------------------------------------------------------------------------
# Main viewer window
# ---------------------------------------------------------------------------
class Gaia3DViewerWindow(QWidget):
    """Tool window for Gaia 3D visualization."""

    def __init__(self, params, result_dir: Path | None = None, parent=None):
        super().__init__(parent)
        self.params = params
        self.result_dir = Path(result_dir or getattr(params.P, "result_dir", Path.cwd()))

        self.df = pd.DataFrame()
        self.loaded_from = ""
        self.membership_col = None

        self.figure = None
        self.canvas = None
        self.toolbar = None
        self.ax = None
        self._plot_cache = None
        self._highlight = None
        self._auto_rotate_timer = QTimer(self)
        self._auto_rotate_timer.timeout.connect(self._on_auto_rotate_tick)
        self._anim_phase = 0.0          # 0.0–1.0, one full GIF cycle
        self._anim_azim0 = 0.0          # azimuth at the moment auto-rotate started
        self._anim_xc = self._anim_xr = 0.0
        self._anim_yc = self._anim_yr = 0.0
        self._anim_zc = self._anim_zr = 0.0

        _tname = getattr(getattr(params, "P", None), "target_name", None) or ""
        win_title = f"{_tname} — Gaia 3D Cluster Viewer" if _tname else "Gaia 3D Cluster Viewer"
        self.setWindowTitle(win_title)
        self.setMinimumSize(1150, 780)
        self._build_ui()
        self.reload_data()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(4)

        # ── path row ──────────────────────────────────────────
        row_path = QHBoxLayout()
        row_path.addWidget(QLabel("Folder:"))
        self.path_edit = QLineEdit(str(self.result_dir))
        self.path_edit.setReadOnly(True)
        row_path.addWidget(self.path_edit)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.browse_result_dir)
        row_path.addWidget(btn_browse)
        btn_reload = QPushButton("Reload")
        btn_reload.clicked.connect(self.reload_data)
        row_path.addWidget(btn_reload)
        root.addLayout(row_path)

        # ── body: canvas (left) + sidebar (right) ─────────────
        body = QHBoxLayout()
        body.setSpacing(6)

        # Canvas column
        canvas_col = QVBoxLayout()
        canvas_col.setSpacing(2)
        self.figure = Figure(figsize=(9, 7))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        self.toolbar = NavigationToolbar(self.canvas, self)
        canvas_col.addWidget(self.toolbar)
        canvas_col.addWidget(self.canvas, stretch=1)
        self.info_label = QLabel("Ready")
        self.info_label.setStyleSheet("QLabel { color:#555; font-size:8pt; }")
        canvas_col.addWidget(self.info_label)
        self.pick_label = QLabel("Selected: (none)")
        self.pick_label.setWordWrap(True)
        self.pick_label.setStyleSheet("QLabel { color:#2E7D32; font-size:8pt; }")
        canvas_col.addWidget(self.pick_label)
        body.addLayout(canvas_col, stretch=1)

        # ── Sidebar ───────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(2, 0, 2, 0)
        sb.setSpacing(5)

        # ─ Data ─
        grp_data = QGroupBox("Data")
        gd = QVBoxLayout(grp_data)
        gd.setSpacing(4)

        hl = QHBoxLayout()
        hl.addWidget(QLabel("Space:"))
        self.space_combo = QComboBox()
        self.space_combo.addItems(["PM Space", "Sky 3D"])
        self.space_combo.currentIndexChanged.connect(self._on_space_changed)
        hl.addWidget(self.space_combo, stretch=1)
        gd.addLayout(hl)

        hl = QHBoxLayout()
        hl.addWidget(QLabel("Color:"))
        self.color_combo = QComboBox()
        self.color_combo.addItems([
            "Membership", "G mag", "Parallax",
            "Teff (BP-RP)", "Teff (g-r std)", "Teff (g-r inst)",
            "Single color",
        ])
        self.color_combo.currentIndexChanged.connect(self.redraw_plot)
        hl.addWidget(self.color_combo, stretch=1)
        gd.addLayout(hl)

        hl = QHBoxLayout()
        self.member_only_check = QCheckBox("Members only")
        self.member_only_check.toggled.connect(self.redraw_plot)
        hl.addWidget(self.member_only_check)
        self.member_thr = QDoubleSpinBox()
        self.member_thr.setRange(0.0, 1.0); self.member_thr.setDecimals(2)
        self.member_thr.setSingleStep(0.05); self.member_thr.setValue(0.50)
        self.member_thr.setFixedWidth(58)
        self.member_thr.valueChanged.connect(self.redraw_plot)
        hl.addWidget(self.member_thr)
        gd.addLayout(hl)

        hl = QHBoxLayout()
        hl.addWidget(QLabel("Max pts:"))
        self.max_points_spin = QSpinBox()
        self.max_points_spin.setRange(500, 500000)
        self.max_points_spin.setSingleStep(1000)
        self.max_points_spin.setValue(30000)
        hl.addWidget(self.max_points_spin, stretch=1)
        gd.addLayout(hl)
        sb.addWidget(grp_data)

        # ─ Display ─
        grp_disp = QGroupBox("Display")
        dd = QVBoxLayout(grp_disp)
        dd.setSpacing(4)

        self.glow_check = QCheckBox("Glow  (size ∝ G brightness)")
        self.glow_check.setChecked(True)
        self.glow_check.toggled.connect(self.redraw_plot)
        dd.addWidget(self.glow_check)
        sb.addWidget(grp_disp)

        # ─ PM Arrows (Sky 3D only) ─
        grp_pm = QGroupBox("PM Arrows  (Sky 3D only)")
        dp = QVBoxLayout(grp_pm)
        dp.setSpacing(4)

        self.pm_arrows_check = QCheckBox("Show PM arrows")
        self.pm_arrows_check.setChecked(False)
        self.pm_arrows_check.setToolTip(
            "u=pmra along RA, v=pmdec along Dec, w=0.\n"
            "Gold = members, gray = non-members."
        )
        self.pm_arrows_check.toggled.connect(self.redraw_plot)
        dp.addWidget(self.pm_arrows_check)

        hl = QHBoxLayout()
        hl.addWidget(QLabel("Scale:"))
        self.pm_scale_spin = QDoubleSpinBox()
        self.pm_scale_spin.setRange(0.05, 20.0); self.pm_scale_spin.setDecimals(2)
        self.pm_scale_spin.setSingleStep(0.1);   self.pm_scale_spin.setValue(1.0)
        self.pm_scale_spin.valueChanged.connect(self.redraw_plot)
        hl.addWidget(self.pm_scale_spin, stretch=1)
        dp.addLayout(hl)

        hl = QHBoxLayout()
        hl.addWidget(QLabel("Pmem thr:"))
        self.pm_member_thr = QDoubleSpinBox()
        self.pm_member_thr.setRange(0.0, 1.0); self.pm_member_thr.setDecimals(2)
        self.pm_member_thr.setSingleStep(0.05); self.pm_member_thr.setValue(0.50)
        self.pm_member_thr.setToolTip("Pmem >= this → gold arrow")
        self.pm_member_thr.valueChanged.connect(self.redraw_plot)
        hl.addWidget(self.pm_member_thr, stretch=1)
        dp.addLayout(hl)
        sb.addWidget(grp_pm)

        # ─ Rotation ─
        grp_rot = QGroupBox("Rotation")
        dr = QVBoxLayout(grp_rot)
        dr.setSpacing(4)

        self.auto_rotate_check = QCheckBox("Auto Rotate")
        self.auto_rotate_check.toggled.connect(self._toggle_auto_rotate)
        dr.addWidget(self.auto_rotate_check)

        hl = QHBoxLayout()
        hl.addWidget(QLabel("Speed:"))
        self.rot_speed_spin = QDoubleSpinBox()
        self.rot_speed_spin.setRange(0.1, 20.0); self.rot_speed_spin.setDecimals(1)
        self.rot_speed_spin.setSingleStep(0.2);  self.rot_speed_spin.setValue(1.2)
        hl.addWidget(self.rot_speed_spin, stretch=1)
        hl.addWidget(QLabel("°/tick"))
        dr.addLayout(hl)

        hl = QHBoxLayout()
        hl.addWidget(QLabel("Zoom in:"))
        self.zoom_in_spin = QDoubleSpinBox()
        self.zoom_in_spin.setRange(0.05, 1.0); self.zoom_in_spin.setDecimals(2)
        self.zoom_in_spin.setSingleStep(0.05); self.zoom_in_spin.setValue(0.2)
        self.zoom_in_spin.setToolTip("축 범위 배율 최솟값 (0.2 = 원래의 20%까지 줌인)")
        hl.addWidget(self.zoom_in_spin, stretch=1)
        dr.addLayout(hl)

        hl = QHBoxLayout()
        hl.addWidget(QLabel("Zoom out:"))
        self.zoom_out_spin = QDoubleSpinBox()
        self.zoom_out_spin.setRange(1.0, 5.0); self.zoom_out_spin.setDecimals(2)
        self.zoom_out_spin.setSingleStep(0.1); self.zoom_out_spin.setValue(1.5)
        self.zoom_out_spin.setToolTip("축 범위 배율 최댓값 (1.5 = 원래의 150%까지 줌아웃)")
        hl.addWidget(self.zoom_out_spin, stretch=1)
        dr.addLayout(hl)

        btn_anim = QPushButton("🎬 Export Animation…")
        btn_anim.setToolTip("Render a rotating 3-D GIF or MP4")
        btn_anim.clicked.connect(self._export_animation)
        dr.addWidget(btn_anim)
        sb.addWidget(grp_rot)

        # ─ Actions ─
        grp_act = QGroupBox("Actions")
        da = QVBoxLayout(grp_act)
        da.setSpacing(4)

        self.btn_earth_view = QPushButton("🌍 Earth View  (Sky 3D)")
        self.btn_earth_view.setToolTip("Look from Earth toward the cluster centre")
        self.btn_earth_view.clicked.connect(self._snap_earth_view)
        da.addWidget(self.btn_earth_view)

        hl = QHBoxLayout()
        btn_plot = QPushButton("Plot")
        btn_plot.clicked.connect(self.redraw_plot)
        hl.addWidget(btn_plot)
        btn_save = QPushButton("Save PNG")
        btn_save.clicked.connect(self.save_png)
        hl.addWidget(btn_save)
        da.addLayout(hl)
        sb.addWidget(grp_act)

        sb.addStretch()
        body.addWidget(sidebar)
        root.addLayout(body, stretch=1)

        # ── log ───────────────────────────────────────────────
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(80)
        self.log_text.setStyleSheet("QTextEdit { font-family:monospace; font-size:8pt; }")
        root.addWidget(self.log_text)

    def _on_space_changed(self):
        """Enable/disable Sky-3D-only controls."""
        is_sky = self.space_combo.currentIndex() == 1
        for w in (self.pm_arrows_check, self.pm_scale_spin, self.pm_member_thr,
                  self.btn_earth_view):
            w.setEnabled(is_sky)
        self.redraw_plot()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {msg}")

    # ------------------------------------------------------------------
    # Column helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pick_membership_col(cols) -> str | None:
        for c in ("gaia_pmem", "pmem_gaia", "membership_prob_gaia", "membership_prob", "pmem"):
            if c in cols:
                return c
        return None

    @staticmethod
    def _pick_col(cols, cands):
        for c in cands:
            if c in cols:
                return c
        return None

    @staticmethod
    def _first_numeric_array(df: pd.DataFrame, cands):
        for c in cands:
            if c in df.columns:
                return pd.to_numeric(df[c], errors="coerce").to_numpy(float), c
        return None, None

    def _compute_bp_rp(self, df: pd.DataFrame) -> np.ndarray:
        bp_rp = np.full(len(df), np.nan, dtype=float)
        if "bp_rp" in df.columns:
            bp_rp = pd.to_numeric(df["bp_rp"], errors="coerce").to_numpy(float)
        bp, _ = self._first_numeric_array(df, ("phot_bp_mean_mag", "gaia_BP", "bpmag"))
        rp, _ = self._first_numeric_array(df, ("phot_rp_mean_mag", "gaia_RP", "rpmag"))
        if bp is not None and rp is not None:
            calc = bp - rp
            bp_rp = np.where(np.isfinite(bp_rp), bp_rp, calc)
        return bp_rp

    @staticmethod
    def _key_series(df: pd.DataFrame, key: str):
        if key not in df.columns:
            return None
        if key in ("source_id", "gaia_source_id"):
            return parse_int64_series(df[key]).astype("Int64")
        vals = pd.to_numeric(df[key], errors="coerce")
        return pd.Series(pd.array(vals, dtype="Int64"), index=df.index)

    def _merge_aux_columns(self, df: pd.DataFrame, aux: pd.DataFrame, value_cols, tag: str):
        if df is None or df.empty or aux is None or aux.empty:
            return df
        best_key = None
        best_overlap = -1
        for key in ("source_id", "gaia_source_id", "ID"):
            if key not in df.columns or key not in aux.columns:
                continue
            lk = self._key_series(df, key)
            rk = self._key_series(aux, key)
            if lk is None or rk is None:
                continue
            left_vals = {int(v) for v in lk.dropna().tolist()}
            right_vals = {int(v) for v in rk.dropna().tolist()}
            overlap = len(left_vals & right_vals)
            if overlap > best_overlap:
                best_overlap = overlap
                best_key = key
        if best_key is None:
            return df

        add_cols = [c for c in value_cols if c in aux.columns and c != best_key]
        if not add_cols:
            return df

        left = df.copy()
        right = aux.copy()
        left[best_key] = self._key_series(left, best_key)
        right[best_key] = self._key_series(right, best_key)
        right = right[[best_key] + add_cols].drop_duplicates(subset=[best_key], keep="first")
        merged = left.merge(right, on=best_key, how="left", suffixes=("", "__aux"))

        n_filled = 0
        for c in add_cols:
            ca = f"{c}__aux"
            if ca not in merged.columns:
                continue
            if c in merged.columns:
                miss = merged[c].isna()
                n_filled += int((miss & merged[ca].notna()).sum())
                merged.loc[miss, c] = merged.loc[miss, ca]
            else:
                n_filled += int(merged[ca].notna().sum())
                merged[c] = merged[ca]
            merged.drop(columns=[ca], inplace=True)

        if n_filled > 0:
            self.log(f"{tag} merged on {best_key}: filled {n_filled} values")
        return merged

    @staticmethod
    def _format_sid(v):
        try:
            if pd.isna(v):
                return "-"
            return str(int(v))
        except Exception:
            try:
                return str(v)
            except Exception:
                return "-"

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def browse_result_dir(self):
        selected = QFileDialog.getExistingDirectory(
            self, "Select Result Folder", str(self.result_dir)
        )
        if not selected:
            return
        self.result_dir = Path(selected)
        self.path_edit.setText(str(self.result_dir))
        self.reload_data()

    def _load_membership_aux(self):
        result_dir = self.result_dir
        cands = [
            step5_dir(result_dir) / "gaia_derived.csv",
            result_dir / "gaia_derived.csv",
            result_dir / "cmd_with_gaia_membership.csv",
            result_dir / "step11" / "cmd_with_gaia_membership.csv",
            result_dir / "median_by_ID_filter_wide_cmd.csv",
            result_dir / "step11" / "median_by_ID_filter_wide_cmd.csv",
        ]
        for p in cands:
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            if df.empty:
                continue
            pcol = self._pick_membership_col(df.columns)
            if pcol is None:
                continue
            out = df.copy()
            if "source_id" in out.columns:
                out["source_id"] = parse_int64_series(out["source_id"]).astype("Int64")
            if "gaia_source_id" in out.columns:
                out["gaia_source_id"] = parse_int64_series(out["gaia_source_id"]).astype("Int64")
            return out, pcol, p.name
        return None, None, ""

    def _load_cmd_color_aux(self):
        result_dir = self.result_dir
        cands = [
            result_dir / "median_by_ID_filter_wide_cmd.csv",
            result_dir / "step11" / "median_by_ID_filter_wide_cmd.csv",
            result_dir / "median_by_ID_filter_wide.csv",
            result_dir / "step11" / "median_by_ID_filter_wide.csv",
        ]
        value_cols = ("mag_std_g", "mag_std_r", "mag_inst_g", "mag_inst_r", "color_gr", "color_ri")
        for p in cands:
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            if df.empty:
                continue
            key_cols = [k for k in ("ID", "source_id", "gaia_source_id") if k in df.columns]
            add_cols = [c for c in value_cols if c in df.columns]
            if not key_cols or not add_cols:
                continue
            out = df[key_cols + add_cols].copy()
            for k in ("source_id", "gaia_source_id"):
                if k in out.columns:
                    out[k] = parse_int64_series(out[k]).astype("Int64")
            if "ID" in out.columns:
                out["ID"] = pd.Series(pd.array(pd.to_numeric(out["ID"], errors="coerce"), dtype="Int64"), index=out.index)
            return out, p.name
        return None, ""

    def _load_base_df(self):
        result_dir = self.result_dir
        for p in [step6_dir(result_dir) / "master_catalog.tsv",
                  result_dir / "master_catalog.tsv"]:
            if not p.exists():
                continue
            try:
                m = pd.read_csv(p, sep="\t")
            except Exception:
                continue
            if len(m) > 0:
                return m, p.name

        for p in [step5_dir(result_dir) / "gaia_derived.csv",
                  result_dir / "gaia_derived.csv",
                  step5_dir(result_dir) / "gaia_fov.ecsv",
                  result_dir / "gaia_fov.ecsv"]:
            if not p.exists():
                continue
            try:
                g = (read_ecsv_int64_source_id(p) if p.suffix.lower() == ".ecsv"
                     else pd.read_csv(p))
            except Exception:
                continue
            if len(g) == 0:
                continue
            if "source_id" in g.columns:
                g["source_id"] = parse_int64_series(g["source_id"]).astype("Int64")
            return g, p.name

        return pd.DataFrame(), ""

    def reload_data(self):
        base, base_name = self._load_base_df()
        if base.empty:
            QMessageBox.warning(
                self, "No Gaia Data",
                "Could not find Gaia-enabled catalogs.\n"
                "Expected: step6_refbuild/master_catalog.tsv or step5_wcs/gaia_derived.csv"
            )
            self.df = pd.DataFrame()
            self.loaded_from = ""
            self.membership_col = None
            self.redraw_plot()
            return

        df = base.copy()
        for col in ("source_id", "gaia_source_id"):
            if col in df.columns:
                df[col] = parse_int64_series(df[col]).astype("Int64")
        if "ID" in df.columns:
            df["ID"] = pd.Series(pd.array(pd.to_numeric(df["ID"], errors="coerce"), dtype="Int64"), index=df.index)

        aux, pcol, aux_name = self._load_membership_aux()
        self.membership_col = self._pick_membership_col(df.columns)
        if aux is not None and pcol is not None and self.membership_col is None:
            joined = False
            for key in ("source_id", "gaia_source_id"):
                if key in df.columns and key in aux.columns:
                    keep = (aux[[key, pcol]].dropna(subset=[key])
                            .drop_duplicates(key))
                    df = df.merge(keep, on=key, how="left")
                    joined = True
                    break
            if joined:
                self.membership_col = pcol
                self.log(f"Membership merged from {aux_name} ({pcol})")

        cmd_aux, cmd_name = self._load_cmd_color_aux()
        if cmd_aux is not None:
            df = self._merge_aux_columns(
                df, cmd_aux,
                ("mag_std_g", "mag_std_r", "mag_inst_g", "mag_inst_r", "color_gr", "color_ri"),
                tag=f"CMD colors ({cmd_name})",
            )

        if self.membership_col is not None:
            df[self.membership_col] = pd.to_numeric(df[self.membership_col], errors="coerce")

        # Report parallax quality
        if "parallax" in df.columns:
            plx = pd.to_numeric(df["parallax"], errors="coerce").to_numpy(float)
            n_total = len(plx)
            n_neg = int(np.sum(np.isfinite(plx) & (plx <= 0)))
            n_nan = int(np.sum(~np.isfinite(plx)))
            if n_neg > 0:
                self.log(f"WARNING: {n_neg}/{n_total} stars have non-positive parallax (excluded from distance)")
            if n_nan > 0:
                self.log(f"INFO: {n_nan}/{n_total} stars have NaN parallax")

        self.df = df
        self.loaded_from = base_name
        self.log(f"Loaded: {base_name} | rows={len(df)}")
        self.redraw_plot()

    # ------------------------------------------------------------------
    # Point preparation
    # ------------------------------------------------------------------
    def _prepare_points(self):
        if self.df is None or self.df.empty:
            return None

        df = self.df.copy()
        pmem = (pd.to_numeric(df[self.membership_col], errors="coerce").to_numpy(float)
                if self.membership_col is not None and self.membership_col in df.columns
                else np.full(len(df), np.nan, float))

        is_sky = self.space_combo.currentIndex() == 1
        earth_elev: float | None = None
        earth_azim: float | None = None

        if not is_sky:
            # --- PM space ---
            c_pmra = self._pick_col(df.columns, ("pmra",))
            c_pmdec = self._pick_col(df.columns, ("pmdec",))
            c_plx = self._pick_col(df.columns, ("parallax",))
            if c_pmra is None or c_pmdec is None or c_plx is None:
                return {"err": "pmra/pmdec/parallax columns required for PM Space."}
            x = pd.to_numeric(df[c_pmra],   errors="coerce").to_numpy(float)
            y = pd.to_numeric(df[c_pmdec],  errors="coerce").to_numpy(float)
            z = pd.to_numeric(df[c_plx],    errors="coerce").to_numpy(float)
            xlab, ylab, zlab = "pmra (mas/yr)", "pmdec (mas/yr)", "parallax (mas)"
            title = "Gaia PM Space"
            # pm arrays not used as arrows in this mode
            ra_rad_full  = np.full(len(df), np.nan, float)
            dec_rad_full = np.full(len(df), np.nan, float)
            pmra_full    = x.copy()
            pmdec_full   = y.copy()
        else:
            # --- Sky 3D: X=RA, Y=Dec, Z=distance ---
            c_ra  = self._pick_col(df.columns, ("gaia_ra_deg", "ra"))
            c_dec = self._pick_col(df.columns, ("gaia_dec_deg", "dec"))
            c_plx = self._pick_col(df.columns, ("parallax",))
            if c_ra is None or c_dec is None or c_plx is None:
                return {"err": "ra/dec/parallax columns required for Sky 3D."}
            ra_deg  = pd.to_numeric(df[c_ra],  errors="coerce").to_numpy(float)
            dec_deg = pd.to_numeric(df[c_dec], errors="coerce").to_numpy(float)
            plx_raw = pd.to_numeric(df[c_plx], errors="coerce").to_numpy(float)
            dist_pc_full = np.where(np.isfinite(plx_raw) & (plx_raw > 0),
                                    1000.0 / plx_raw, np.nan)

            # Cluster centre (median of finite values)
            ra0  = float(np.nanmedian(ra_deg[np.isfinite(ra_deg)])) if np.isfinite(ra_deg).any() else 0.0
            dec0 = float(np.nanmedian(dec_deg[np.isfinite(dec_deg)])) if np.isfinite(dec_deg).any() else 0.0
            dist0 = float(np.nanmedian(dist_pc_full[np.isfinite(dist_pc_full)])) if np.isfinite(dist_pc_full).any() else 0.0

            x = ra_deg  - ra0
            y = dec_deg - dec0
            z = dist_pc_full - dist0
            xlab = f"ΔRA (°,  RA₀={ra0:.3f}°)"
            ylab = f"ΔDec (°,  Dec₀={dec0:.3f}°)"
            zlab = f"Δdist (pc,  d₀={dist0:.0f} pc)"
            title = "Gaia Sky 3D"

            # Earth is at -Z (behind the cluster): look from -Z toward +Z
            earth_elev = -90.0
            earth_azim =   0.0

            ra_rad_full  = np.deg2rad(ra_deg)   # kept for completeness
            dec_rad_full = np.deg2rad(dec_deg)
            pmra_col  = self._pick_col(df.columns, ("pmra",))
            pmdec_col = self._pick_col(df.columns, ("pmdec",))
            pmra_full  = (pd.to_numeric(df[pmra_col],  errors="coerce").to_numpy(float)
                          if pmra_col  else np.full(len(df), np.nan, float))
            pmdec_full = (pd.to_numeric(df[pmdec_col], errors="coerce").to_numpy(float)
                          if pmdec_col else np.full(len(df), np.nan, float))

        # --- filter finite ---
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if self.member_only_check.isChecked():
            thr = float(self.member_thr.value())
            finite &= np.isfinite(pmem) & (pmem >= thr)
        if not np.any(finite):
            return {"err": "No valid points after filtering."}

        idx = np.where(finite)[0]
        max_n = int(self.max_points_spin.value())
        if len(idx) > max_n:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(idx, size=max_n, replace=False))

        x    = x[idx];    y    = y[idx];    z    = z[idx]
        pmem = pmem[idx]
        ra_arr    = ra_rad_full[idx]
        dec_arr   = dec_rad_full[idx]
        pmra_arr  = pmra_full[idx]
        pmdec_arr = pmdec_full[idx]

        # --- photometric columns ---
        g_col = self._pick_col(df.columns, ("phot_g_mean_mag", "gaia_G", "gmag"))
        gmag = (pd.to_numeric(df[g_col], errors="coerce").to_numpy(float)[idx]
                if g_col else np.full(len(idx), np.nan))
        plx_arr = (pd.to_numeric(df["parallax"], errors="coerce").to_numpy(float)[idx]
                   if "parallax" in df.columns else np.full(len(idx), np.nan))
        dist_pc = np.where(np.isfinite(plx_arr) & (plx_arr > 0), 1000.0 / plx_arr, np.nan)
        bp_rp   = self._compute_bp_rp(df)[idx]

        if "mag_std_g" in df.columns and "mag_std_r" in df.columns:
            gr_std = (pd.to_numeric(df["mag_std_g"], errors="coerce").to_numpy(float)
                      - pd.to_numeric(df["mag_std_r"], errors="coerce").to_numpy(float))[idx]
        elif "color_gr" in df.columns:
            gr_std = pd.to_numeric(df["color_gr"], errors="coerce").to_numpy(float)[idx]
        else:
            gr_std = np.full(len(idx), np.nan, float)
        if "mag_inst_g" in df.columns and "mag_inst_r" in df.columns:
            gr_inst = (pd.to_numeric(df["mag_inst_g"], errors="coerce").to_numpy(float)
                       - pd.to_numeric(df["mag_inst_r"], errors="coerce").to_numpy(float))[idx]
        elif "color_gr" in df.columns:
            gr_inst = pd.to_numeric(df["color_gr"], errors="coerce").to_numpy(float)[idx]
        else:
            gr_inst = np.full(len(idx), np.nan, float)

        # --- color mode ---
        c_mode = self.color_combo.currentIndex()
        vmin: float | None = None
        vmax: float | None = None

        if c_mode == 0 and np.isfinite(pmem).sum() > 0:
            c = pmem;     cmap = "viridis";   cbar = "Pmem"
        elif c_mode == 1 and np.isfinite(gmag).sum() > 0:
            c = gmag;     cmap = "plasma_r";  cbar = "Gaia G (mag)"
        elif c_mode == 2 and np.isfinite(plx_arr).sum() > 0:
            c = plx_arr;  cmap = "turbo";     cbar = "Parallax (mas)"
        elif c_mode == 3 and np.isfinite(bp_rp).sum() > 0:
            c = _color_to_teff(bp_rp, "BP-RP")
            cmap = _STELLAR_CMAP; cbar = "Teff (K) from BP-RP"
            vmin, vmax = _TEFF_VMIN, _TEFF_VMAX
        elif c_mode == 4 and np.isfinite(gr_std).sum() > 0:
            c = _color_to_teff(gr_std, "g-r")
            cmap = _STELLAR_CMAP; cbar = "Teff (K) from g-r std"
            vmin, vmax = _TEFF_VMIN, _TEFF_VMAX
        elif c_mode == 5 and np.isfinite(gr_inst).sum() > 0:
            c = _color_to_teff(gr_inst, "g-r")
            cmap = _STELLAR_CMAP; cbar = "Teff (K) from g-r inst"
            vmin, vmax = _TEFF_VMIN, _TEFF_VMAX
        else:
            c = "#4FC3F7"; cmap = None; cbar = None

        return dict(
            x=x, y=y, z=z,
            c=c, cmap=cmap, cbar=cbar, vmin=vmin, vmax=vmax,
            n=len(x), xlab=xlab, ylab=ylab, zlab=zlab, title=title,
            pmem_n=int(np.isfinite(pmem).sum()),
            row_idx=idx, pmem=pmem,
            gmag=gmag, plx=plx_arr, dist_pc=dist_pc,
            bp_rp=bp_rp, gr_std=gr_std, gr_inst=gr_inst,
            ra_arr=ra_arr, dec_arr=dec_arr,
            pmra_arr=pmra_arr, pmdec_arr=pmdec_arr,
            is_sky=is_sky,
            earth_elev=earth_elev, earth_azim=earth_azim,
        )

    # ------------------------------------------------------------------
    # PM arrow overlay
    # ------------------------------------------------------------------
    def _plot_pm_arrows(self, info: dict, sx: float, sy: float, sz: float):
        """
        Overlay proper-motion vectors as quiver arrows in Sky 3D space.

        Since Sky 3D uses X=RA(°), Y=Dec(°), Z=dist(pc):
            u = pmra  (mas/yr → shown along RA axis)
            v = pmdec (mas/yr → shown along Dec axis)
            w = 0     (no line-of-sight PM component)

        Arrows are scaled so the 95th-percentile PM spans ~12% of the
        RA/Dec plot range, then multiplied by the user PM scale factor.
        Members (Pmem ≥ threshold): gold  |  others: dim gray
        """
        pmra_a  = info["pmra_arr"]
        pmdec_a = info["pmdec_arr"]
        x, y, z = info["x"], info["y"], info["z"]
        pmem    = info["pmem"]

        valid = np.isfinite(pmra_a) & np.isfinite(pmdec_a)
        if not np.any(valid):
            self.log("PM Arrows: no valid pmra/pmdec data found.")
            return

        # Arrow direction: along RA (X) and Dec (Y) axes, no Z component
        u = pmra_a
        v = pmdec_a
        w = np.zeros_like(u)

        # Scale: 95th-percentile |PM| → 12% of RA/Dec plot range
        pm_mag = np.sqrt(u**2 + v**2)
        finite_mag = pm_mag[valid & np.isfinite(pm_mag)]
        pm95 = float(np.percentile(finite_mag, 95)) if len(finite_mag) > 0 else 1.0
        if pm95 <= 0:
            pm95 = 1.0
        ra_dec_range = max(sx, sy)   # RA/Dec axes (degrees)
        scale = ra_dec_range * 0.12 * float(self.pm_scale_spin.value()) / pm95

        # Subsample if too many arrows — keep all members, fill remainder with non-members
        vi = np.where(valid)[0]
        if len(vi) > _MAX_PM_ARROWS:
            pmem_thr_sub = float(self.pm_member_thr.value())
            is_mem_sub = np.isfinite(pmem[vi]) & (pmem[vi] >= pmem_thr_sub)
            mem_vi  = vi[is_mem_sub]
            nonm_vi = vi[~is_mem_sub]
            rng = np.random.default_rng(0)
            n_nonm = max(0, _MAX_PM_ARROWS - len(mem_vi))
            if n_nonm < len(nonm_vi):
                nonm_vi = np.sort(rng.choice(nonm_vi, size=n_nonm, replace=False))
            vi = np.sort(np.concatenate([mem_vi, nonm_vi]))

        # Split by membership
        pmem_thr = float(self.pm_member_thr.value())
        is_member = np.isfinite(pmem[vi]) & (pmem[vi] >= pmem_thr)

        def _quiver(mask, color, alpha, lw):
            sel = vi[mask]
            if len(sel) == 0:
                return
            self.ax.quiver(
                x[sel], y[sel], z[sel],
                u[sel] * scale, v[sel] * scale, w[sel] * scale,
                color=color, alpha=alpha, linewidth=lw,
                arrow_length_ratio=0.25, normalize=False,
            )

        # Non-members first (behind), members on top
        _quiver(~is_member, color="#999999", alpha=0.30, lw=0.5)
        _quiver( is_member, color="#FFD700", alpha=0.85, lw=1.0)

        n_mem = int(is_member.sum())
        self.log(f"PM Arrows: {len(vi)} arrows ({n_mem} members, "
                 f"{len(vi)-n_mem} non-members), scale={scale:.4f} °/(mas/yr)")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def redraw_plot(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111, projection="3d")
        self._plot_cache = None
        self._highlight  = None

        info = self._prepare_points()
        if info is None or "err" in info:
            msg = "No data loaded." if info is None else info["err"]
            self.ax.text2D(0.05, 0.95, msg, transform=self.ax.transAxes, color="#E53935")
            self.ax.set_title("Gaia 3D Cluster Viewer")
            self.canvas.draw_idle()
            self.info_label.setText(msg)
            self.pick_label.setText("Selected star: (none)")
            return

        x3, y3, z3 = info["x"], info["y"], info["z"]
        c_val, cmap_val = info["c"], info["cmap"]
        vmin, vmax = info.get("vmin"), info.get("vmax")
        gmag = info["gmag"]

        # --- marker size from G brightness ---
        use_glow = self.glow_check.isChecked()
        if np.isfinite(gmag).sum() > 2:
            mag_med = float(np.nanmedian(gmag))
            flux = np.where(np.isfinite(gmag), 10.0 ** ((mag_med - gmag) / 2.5), 1.0)
            sizes = np.clip(flux * 6.0, 0.5, 60.0)
        else:
            sizes = np.full(len(x3), 6.0)

        ckw: dict = dict(linewidths=0)
        if vmin is not None:
            ckw["vmin"] = vmin
        if vmax is not None:
            ckw["vmax"] = vmax

        if cmap_val is None:
            if use_glow:
                self.ax.scatter(x3, y3, z3, s=sizes * 6, alpha=0.08, c=c_val,
                                depthshade=False, **ckw)
            self.ax.scatter(x3, y3, z3, s=sizes, alpha=0.82, c=c_val,
                            depthshade=True, **ckw)
        else:
            if use_glow:
                self.ax.scatter(x3, y3, z3, s=sizes * 6, alpha=0.08, c=c_val,
                                cmap=cmap_val, depthshade=False, **ckw)
            sc = self.ax.scatter(x3, y3, z3, s=sizes, alpha=0.82, c=c_val,
                                 cmap=cmap_val, depthshade=True, **ckw)
            cbar = self.figure.colorbar(sc, ax=self.ax, shrink=0.75, pad=0.08)
            cbar.set_label(info["cbar"])

        # --- center axes on cluster (3σ per axis) ---
        cx = float(np.nanmedian(x3));  cy = float(np.nanmedian(y3));  cz = float(np.nanmedian(z3))
        sx = max(float(np.nanstd(x3)) * 3.0, 1e-9)
        sy = max(float(np.nanstd(y3)) * 3.0, 1e-9)
        sz = max(float(np.nanstd(z3)) * 3.0, 1e-9)
        self.ax.set_xlim3d(cx - sx, cx + sx)
        self.ax.set_ylim3d(cy - sy, cy + sy)
        self.ax.set_zlim3d(cz - sz, cz + sz)

        # --- PM arrows ---
        if (info.get("is_sky") and self.pm_arrows_check.isChecked()
                and self.pm_arrows_check.isEnabled()):
            self._plot_pm_arrows(info, sx, sy, sz)

        self.ax.set_xlabel(info["xlab"])
        self.ax.set_ylabel(info["ylab"])
        self.ax.set_zlabel(info["zlab"])
        target = (getattr(getattr(self.params, "P", None), "target_name", None)
                  or self.result_dir.parent.name
                  or "")
        target_str = f"{target}  " if target else ""
        self.ax.set_title(f"{target_str}{info['title']} | N={info['n']}")
        self.ax.grid(True, alpha=0.25)
        self.ax.view_init(elev=22.0, azim=38.0)
        info["cx"] = cx; info["cy"] = cy; info["cz"] = cz
        info["sx"] = sx; info["sy"] = sy; info["sz"] = sz
        self._plot_cache = info
        # Store pristine home state for animation (only here, never during animation)
        self._anim_azim0 = 38.0
        self._anim_xc = cx; self._anim_xr = sx
        self._anim_yc = cy; self._anim_yr = sy
        self._anim_zc = cz; self._anim_zr = sz

        member_msg = (f"membership={self.membership_col}, finite={info['pmem_n']}"
                      if self.membership_col is not None else "membership=none")
        earth_hint = " | Earth View available" if info.get("earth_elev") is not None else ""
        self.info_label.setText(
            f"{self.loaded_from or '?'} | pts={info['n']} | {member_msg}"
            f"{earth_hint} | wheel=zoom, click=inspect"
        )
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Zoom / rotate helpers
    # ------------------------------------------------------------------
    def _zoom_axes(self, scale: float):
        if self.ax is None:
            return
        try:
            x0, x1 = self.ax.get_xlim3d()
            y0, y1 = self.ax.get_ylim3d()
            z0, z1 = self.ax.get_zlim3d()
            xc, yc, zc = (x0+x1)*0.5, (y0+y1)*0.5, (z0+z1)*0.5
            self.ax.set_xlim3d(xc - (x1-x0)*0.5*scale, xc + (x1-x0)*0.5*scale)
            self.ax.set_ylim3d(yc - (y1-y0)*0.5*scale, yc + (y1-y0)*0.5*scale)
            self.ax.set_zlim3d(zc - (z1-z0)*0.5*scale, zc + (z1-z0)*0.5*scale)
            self.canvas.draw_idle()
        except Exception:
            pass

    def _on_scroll_zoom(self, event):
        if event is None or event.inaxes != self.ax:
            return
        if str(getattr(event, "button", "")) == "up":
            self._zoom_axes(0.93)
        elif str(getattr(event, "button", "")) == "down":
            self._zoom_axes(1.075)

    def _rotate_by(self, deg: float):
        if self.ax is None:
            return
        try:
            self.ax.view_init(elev=self.ax.elev, azim=self.ax.azim + float(deg))
            self.canvas.draw_idle()
        except Exception:
            pass

    def _toggle_auto_rotate(self, checked: bool):
        if checked:
            self._anim_phase = 0.0   # home state already set by last redraw_plot
            self._auto_rotate_timer.start(40)
        else:
            self._auto_rotate_timer.stop()

    def _on_auto_rotate_tick(self):
        if self.ax is None:
            return
        # Same path as GIF export: full 360° orbit + elevation wave + zoom pulse
        # rot_speed_spin (°/tick) controls how fast azimuth advances → phase rate
        deg_per_tick = float(self.rot_speed_spin.value())
        self._anim_phase = (self._anim_phase + deg_per_tick / 360.0) % 1.0
        t = self._anim_phase
        azim = self._anim_azim0 + 360.0 * t
        elev = 20.0 + 20.0 * np.sin(2.0 * np.pi * 1.5 * t)
        z_in  = float(self.zoom_in_spin.value())
        z_out = float(self.zoom_out_spin.value())
        zoom  = z_in + (z_out - z_in) * 0.5 * (1.0 + np.cos(2.0 * np.pi * 2.0 * t))
        try:
            self.ax.view_init(elev=float(elev), azim=float(azim))
            self.ax.set_xlim3d(self._anim_xc - self._anim_xr * zoom,
                               self._anim_xc + self._anim_xr * zoom)
            self.ax.set_ylim3d(self._anim_yc - self._anim_yr * zoom,
                               self._anim_yc + self._anim_yr * zoom)
            self.ax.set_zlim3d(self._anim_zc - self._anim_zr * zoom,
                               self._anim_zc + self._anim_zr * zoom)
            self.canvas.draw_idle()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Earth view (Sky 3D mode only)
    # ------------------------------------------------------------------
    def _snap_earth_view(self):
        if self._plot_cache is None or self.ax is None:
            return
        elev = self._plot_cache.get("earth_elev")
        azim = self._plot_cache.get("earth_azim")
        if elev is None or azim is None:
            self.log("Earth View: only available in 'Sky 3D' space mode.")
            return
        self.ax.view_init(elev=float(elev), azim=float(azim))
        self.canvas.draw_idle()
        self.log(f"Earth View snapped: elev={elev:.1f}°, azim={azim:.1f}°")

    # ------------------------------------------------------------------
    # Star selection (click → red dot)
    # ------------------------------------------------------------------
    def _on_canvas_click(self, event):
        if event is None or event.inaxes != self.ax:
            return
        # Select stars only on double-click to avoid accidental picks while rotating.
        if not bool(getattr(event, "dblclick", False)):
            return
        if self._plot_cache is None or event.x is None or event.y is None:
            return

        x = np.asarray(self._plot_cache.get("x", []), float)
        y = np.asarray(self._plot_cache.get("y", []), float)
        z = np.asarray(self._plot_cache.get("z", []), float)
        if x.size == 0:
            return

        x2, y2, _ = proj3d.proj_transform(x, y, z, self.ax.get_proj())
        xy_disp = self.ax.transData.transform(np.column_stack([x2, y2]))
        click = np.array([float(event.x), float(event.y)])
        d2 = np.sum((xy_disp - click) ** 2, axis=1)
        i = int(np.argmin(d2))
        if d2[i] > 14.0 ** 2:
            return

        # Remove old highlight
        if self._highlight is not None:
            try:
                self._highlight.remove()
            except Exception:
                pass
            self._highlight = None

        # Add new red dot
        self._highlight = self.ax.scatter(
            [float(x[i])], [float(y[i])], [float(z[i])],
            s=130, c="red", zorder=10,
            linewidths=1.5, edgecolors="white",
            depthshade=False,
        )
        self.canvas.draw_idle()

        # Info text
        row_idx = int(self._plot_cache["row_idx"][i])
        row = self.df.iloc[row_idx]
        sid   = self._format_sid(row.get("source_id",      np.nan))
        gsid  = self._format_sid(row.get("gaia_source_id", np.nan))
        iid   = self._format_sid(row.get("ID",             np.nan))
        gmag  = self._plot_cache["gmag"][i]    if i < len(self._plot_cache["gmag"])    else np.nan
        dist  = self._plot_cache["dist_pc"][i] if i < len(self._plot_cache["dist_pc"]) else np.nan
        plx   = self._plot_cache["plx"][i]     if i < len(self._plot_cache["plx"])     else np.nan
        bp_rp = self._plot_cache["bp_rp"][i]   if i < len(self._plot_cache["bp_rp"])   else np.nan
        gr_s  = self._plot_cache["gr_std"][i]  if i < len(self._plot_cache["gr_std"])  else np.nan
        gr_i  = self._plot_cache["gr_inst"][i] if i < len(self._plot_cache["gr_inst"]) else np.nan
        pmem  = self._plot_cache["pmem"][i]    if i < len(self._plot_cache["pmem"])    else np.nan
        pmra_v  = self._plot_cache["pmra_arr"][i]  if i < len(self._plot_cache["pmra_arr"])  else np.nan
        pmdec_v = self._plot_cache["pmdec_arr"][i] if i < len(self._plot_cache["pmdec_arr"]) else np.nan

        parts = [
            f"ID={iid}",
            f"source_id={sid}",
            f"G={gmag:.3f}" if np.isfinite(gmag) else "G=-",
            f"dist={dist:.1f}pc" if np.isfinite(dist) else "dist=-",
            f"plx={plx:.3f}mas"  if np.isfinite(plx)  else "plx=-",
            f"pm=({pmra_v:.2f},{pmdec_v:.2f})mas/yr" if (np.isfinite(pmra_v) and np.isfinite(pmdec_v)) else "pm=-",
            f"BP-RP={bp_rp:.3f}"  if np.isfinite(bp_rp) else "BP-RP=-",
            f"g-r(std)={gr_s:.3f}" if np.isfinite(gr_s) else "g-r(std)=-",
            f"g-r(inst)={gr_i:.3f}" if np.isfinite(gr_i) else "g-r(inst)=-",
            f"Pmem={pmem:.3f}" if np.isfinite(pmem) else "Pmem=-",
        ]
        msg = " | ".join(parts)
        self.pick_label.setText(f"Selected star: {msg}")
        self.log(msg)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save_png(self):
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save 3D Plot",
            str(self.result_dir / "gaia_cluster_3d.png"),
            "PNG Images (*.png)"
        )
        if not out_path:
            return
        self.figure.savefig(out_path, dpi=170, bbox_inches="tight")
        self.log(f"Saved: {out_path}")

    # ------------------------------------------------------------------
    # Animation export
    # ------------------------------------------------------------------
    def _export_animation(self):
        """Export a rotating 3-D fly-around as GIF or MP4."""
        if self._plot_cache is None or self.ax is None:
            QMessageBox.warning(self, "No Plot", "Plot the data first before exporting.")
            return

        # ── settings dialog ───────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle("Export 3D Animation")
        dlg.setMinimumWidth(290)
        dl = QVBoxLayout(dlg)
        form = QFormLayout()

        spin_dur = QSpinBox()
        spin_dur.setRange(5, 30); spin_dur.setValue(12); spin_dur.setSuffix(" s")
        spin_fps = QSpinBox()
        spin_fps.setRange(8, 24); spin_fps.setValue(15); spin_fps.setSuffix(" fps")
        spin_dpi = QSpinBox()
        spin_dpi.setRange(72, 180); spin_dpi.setValue(100); spin_dpi.setSuffix(" dpi")
        fmt_combo = QComboBox()
        fmt_combo.addItems(["GIF  (Pillow, no ffmpeg needed)", "MP4  (requires ffmpeg)"])

        form.addRow("Duration:", spin_dur)
        form.addRow("FPS:", spin_fps)
        form.addRow("DPI:", spin_dpi)
        form.addRow("Format:", fmt_combo)
        dl.addLayout(form)

        note = QLabel(
            "<small>Rotation path: 360° azimuth orbit + ±20° elevation wave + ×0.8–1.2 zoom pulse</small>"
        )
        note.setWordWrap(True)
        dl.addWidget(note)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        dl.addWidget(bb)

        if dlg.exec_() != QDialog.Accepted:
            return

        duration = spin_dur.value()
        fps      = spin_fps.value()
        dpi      = spin_dpi.value()
        use_mp4  = fmt_combo.currentIndex() == 1
        ext      = "mp4" if use_mp4 else "gif"

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Animation",
            str(self.result_dir / f"cluster_3d.{ext}"),
            f"{'MP4 Video' if use_mp4 else 'GIF Animation'} (*.{ext})"
        )
        if not save_path:
            return

        n_frames = duration * fps

        # Use home state stored by last redraw_plot (same as auto-rotate)
        azim0 = self._anim_azim0
        xc, xr = self._anim_xc, self._anim_xr
        yc, yr = self._anim_yc, self._anim_yr
        zc, zr = self._anim_zc, self._anim_zr
        # Also save current ax limits to restore after export
        elev0   = float(self.ax.elev)
        x0, x1  = self.ax.get_xlim3d()
        y0, y1  = self.ax.get_ylim3d()
        z0, z1  = self.ax.get_zlim3d()

        def _set_frame(frame: int):
            """3-D fly-around path for frame i / n_frames."""
            t = frame / n_frames
            # Azimuth: one full 360° orbit
            azim = azim0 + 360.0 * t
            # Elevation: sinusoidal 0°–40°, 1.5 oscillations per orbit → proper 3-D look
            elev = 20.0 + 20.0 * np.sin(2.0 * np.pi * 1.5 * t)
            # Zoom: 2 cycles, amplitude from UI spinbox
            z_in  = float(self.zoom_in_spin.value())
            z_out = float(self.zoom_out_spin.value())
            zoom  = z_in + (z_out - z_in) * 0.5 * (1.0 + np.cos(2.0 * np.pi * 2.0 * t))
            self.ax.view_init(elev=float(elev), azim=float(azim))
            self.ax.set_xlim3d(xc - xr * zoom, xc + xr * zoom)
            self.ax.set_ylim3d(yc - yr * zoom, yc + yr * zoom)
            self.ax.set_zlim3d(zc - zr * zoom, zc + zr * zoom)

        prog = QProgressDialog("Rendering frames…", "Cancel", 0, n_frames, self)
        prog.setWindowTitle("Exporting Animation")
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.show()

        try:
            if use_mp4:
                self._export_mp4(save_path, n_frames, fps, dpi, _set_frame, prog)
            else:
                self._export_gif(save_path, n_frames, fps, dpi, _set_frame, prog)
        finally:
            prog.close()
            # Restore original view
            self.ax.view_init(elev=elev0, azim=azim0)
            self.ax.set_xlim3d(x0, x1)
            self.ax.set_ylim3d(y0, y1)
            self.ax.set_zlim3d(z0, z1)
            self.canvas.draw_idle()
            self.info_label.setText("Ready")

    def _export_gif(self, path: str, n_frames: int, fps: int, dpi: int,
                    set_frame_fn, prog: QProgressDialog):
        import io
        try:
            from PIL import Image as PilImage
        except ImportError:
            QMessageBox.critical(
                self, "Missing Library",
                "Pillow is required for GIF export.\n\npip install Pillow"
            )
            return

        frames = []
        for i in range(n_frames):
            if prog.wasCanceled():
                self.log("GIF export cancelled.")
                return
            set_frame_fn(i)
            buf = io.BytesIO()
            self.figure.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            buf.seek(0)
            frames.append(PilImage.open(buf).copy().convert("P", dither=PilImage.FLOYDSTEINBERG))
            buf.close()
            prog.setValue(i + 1)
            QApplication.instance().processEvents()

        if not frames:
            return

        self.info_label.setText(f"Saving GIF ({len(frames)} frames)…")
        QApplication.instance().processEvents()
        frames[0].save(
            path, format="GIF",
            save_all=True, append_images=frames[1:],
            duration=int(1000 / fps), loop=0,
        )
        self.log(f"GIF saved: {path}  ({len(frames)} frames @ {fps} fps)")
        QMessageBox.information(self, "Done", f"GIF saved:\n{path}")

    def _export_mp4(self, path: str, n_frames: int, fps: int, dpi: int,
                    set_frame_fn, prog: QProgressDialog):
        from matplotlib.animation import FuncAnimation, FFMpegWriter

        def animate(frame):
            if prog.wasCanceled():
                return []
            set_frame_fn(frame)
            prog.setValue(frame + 1)
            QApplication.instance().processEvents()
            return []

        anim = FuncAnimation(self.figure, animate, frames=n_frames,
                             interval=1000 // fps, blit=False)
        try:
            writer = FFMpegWriter(
                fps=fps, codec="h264", bitrate=2000,
                extra_args=["-pix_fmt", "yuv420p"],
            )
            self.info_label.setText(f"Encoding MP4 ({n_frames} frames)…")
            QApplication.instance().processEvents()
            anim.save(path, writer=writer, dpi=dpi)
            self.log(f"MP4 saved: {path}  ({n_frames} frames @ {fps} fps)")
            QMessageBox.information(self, "Done", f"MP4 saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(
                self, "MP4 Export Failed",
                f"ffmpeg not found or encoding error:\n{e}\n\nUse GIF instead."
            )
            self.log(f"MP4 export failed: {e}")

    def closeEvent(self, event):
        try:
            self._auto_rotate_timer.stop()
        except Exception:
            pass
        super().closeEvent(event)
