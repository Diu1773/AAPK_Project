"""
Extinction & Zeropoint Fitting Tool
Standalone tool for atmospheric extinction fitting and calibration.

Supports two extinction models:
- Model A: diff_mag = k1*X + k2*X^2 + k_c*(B-V)*X + ZP
- Model B: diff_mag = k1*X + k_c1*(B-V)*X + k_c2*(B-V)^2*X + ZP
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib as mpl

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QMessageBox,
    QTextEdit, QComboBox, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QCheckBox
)

from ...utils.step_paths import (
    step1_dir,
    step2_cropped_dir,
    step8_dir,
    step9_dir,
    tool_extinction_dir,
)
from ...utils.astro_utils import compute_airmass_from_header


class ExtinctionFitWorker(QThread):
    """Worker thread for extinction fitting."""

    progress = pyqtSignal(int, int, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, params, result_dir: Path, model: str = "A"):
        super().__init__()
        self.params = params
        self.result_dir = Path(result_dir)
        self.model = model  # "A" or "B"
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def _log(self, msg: str):
        self.log.emit(msg)

    @staticmethod
    def _normalize_filter_key(value: str | None) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()

    @staticmethod
    def _parse_date_from_datetime(value: str | None) -> str:
        """Parse YYYY-MM-DD from datetime-like strings."""
        import re

        if value is None:
            return ""
        s = str(value).strip()
        if not s:
            return ""
        if "T" in s:
            s = s.split("T", 1)[0]
        elif " " in s:
            s = s.split(" ", 1)[0]
        m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
        if m:
            return m.group(1)
        m = re.match(r"(\d{8})", s)
        if m:
            d = m.group(1)
            return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        return ""

    @staticmethod
    def _extract_date_from_header(hdr) -> str:
        """Extract YYYY-MM-DD from FITS header date keys."""
        for key in ("DATE-OBS", "DATEOBS", "DATE"):
            if key in hdr:
                parsed = ExtinctionFitWorker._parse_date_from_datetime(hdr.get(key))
                if parsed:
                    return parsed
        return ""

    @staticmethod
    def _extract_date_from_path(path: Path | str | None = None, fname: str = "") -> str:
        """Extract date from folder path or filename (YYYY-MM-DD or YYYYMMDD)"""
        import re

        if path is not None:
            path_str = str(path)
            for part in reversed(Path(path_str).parts):
                m = re.search(r"(\d{4}-\d{2}-\d{2})", part)
                if m:
                    return m.group(1)
                m = re.search(r"(20\d{6})", part)
                if m:
                    d = m.group(1)
                    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"

        if fname and "__" in fname:
            folder_part = fname.split("__")[0]
            m = re.match(r"(\d{4}-\d{2}-\d{2})", folder_part)
            if m:
                return m.group(1)
            m = re.match(r"(\d{8})", folder_part)
            if m:
                d = m.group(1)
                return f"{d[:4]}-{d[4:6]}-{d[6:8]}"

        if fname:
            m = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
            if m:
                return m.group(1)
            m = re.search(r"(20\d{6})", fname)
            if m:
                d = m.group(1)
                return f"{d[:4]}-{d[4:6]}-{d[6:8]}"

        return "unknown"

    @staticmethod
    def _load_selection_ids(result_dir: Path) -> tuple[int | None, list[int], int | None, list[int]]:
        """Load target/comparison IDs from selection file."""
        selection_path = step8_dir(result_dir) / "target_selection.json"
        if not selection_path.exists():
            selection_path = result_dir / "target_selection.json"
        if not selection_path.exists():
            return None, [], None, []
        try:
            data = json.loads(selection_path.read_text(encoding="utf-8"))
            target_id = data.get("target_id")
            comp_ids = data.get("comparison_ids", [])
            target_sid = data.get("target_source_id")
            comp_sids = data.get("comparison_source_ids", [])
            if target_id is not None:
                target_id = int(target_id)
            if target_sid is not None:
                target_sid = int(target_sid)
            comp_ids = [int(x) for x in comp_ids if x is not None]
            comp_sids = [int(x) for x in comp_sids if x is not None]
            return target_id, comp_ids, target_sid, comp_sids
        except Exception:
            return None, [], None, []

    @staticmethod
    def _load_selection_ids_by_filter(result_dir: Path) -> dict:
        """Load filter-specific selections."""
        step8_out = step8_dir(result_dir)
        filter_selections = {}

        if not step8_out.exists():
            return filter_selections

        for sel_path in step8_out.glob("selection_*.json"):
            try:
                flt_raw = sel_path.stem.replace("selection_", "")
                data = json.loads(sel_path.read_text(encoding="utf-8"))
                flt = ExtinctionFitWorker._normalize_filter_key(flt_raw) or ExtinctionFitWorker._normalize_filter_key(
                    data.get("filter")
                )

                target_id = data.get("target_id")
                comp_ids = data.get("comparison_ids", [])
                target_sid = data.get("target_source_id")
                comp_sids = data.get("comparison_source_ids", [])

                if target_id is not None:
                    target_id = int(target_id)
                if target_sid is not None:
                    target_sid = int(target_sid)
                comp_ids = [int(x) for x in comp_ids if x is not None]
                comp_sids = [int(x) for x in comp_sids if x is not None]

                if (target_id is not None or target_sid is not None) and flt:
                    filter_selections[flt] = {
                        "target_id": target_id,
                        "comp_ids": comp_ids,
                        "target_source_id": target_sid,
                        "comparison_source_ids": comp_sids,
                    }
            except Exception:
                continue

        return filter_selections

    @staticmethod
    def _fit_model_a(X, C, y):
        """
        Model A: diff_mag = k1*X + k2*X^2 + k_c*C*X + ZP
        Where C is the color index (e.g., B-V or g-r)
        """
        X = np.asarray(X, float)
        C = np.asarray(C, float) if C is not None else np.zeros_like(X)
        y = np.asarray(y, float)

        mask = np.isfinite(X) & np.isfinite(y)
        has_color = np.isfinite(C).any() and np.nanstd(C) > 1e-8

        if mask.sum() < 4:
            return {"zp": np.nan, "k1": np.nan, "k2": np.nan, "k_c": np.nan}, mask

        X_m = X[mask]
        y_m = y[mask]
        C_m = C[mask] if has_color else np.zeros_like(X_m)

        # Design matrix: [1, X, X^2, C*X]
        if has_color:
            A = np.column_stack([np.ones_like(X_m), X_m, X_m**2, C_m * X_m])
        else:
            A = np.column_stack([np.ones_like(X_m), X_m, X_m**2])

        try:
            coef, _, _, _ = np.linalg.lstsq(A, y_m, rcond=None)
            if has_color:
                return {"zp": coef[0], "k1": coef[1], "k2": coef[2], "k_c": coef[3]}, mask
            else:
                return {"zp": coef[0], "k1": coef[1], "k2": coef[2], "k_c": np.nan}, mask
        except Exception:
            return {"zp": np.nan, "k1": np.nan, "k2": np.nan, "k_c": np.nan}, mask

    @staticmethod
    def _fit_model_b(X, C, y):
        """
        Model B: diff_mag = k1*X + k_c1*C*X + k_c2*C^2*X + ZP
        Polynomial in color index
        """
        X = np.asarray(X, float)
        C = np.asarray(C, float) if C is not None else np.zeros_like(X)
        y = np.asarray(y, float)

        mask = np.isfinite(X) & np.isfinite(y)
        has_color = np.isfinite(C).any() and np.nanstd(C) > 1e-8

        if mask.sum() < 4:
            return {"zp": np.nan, "k1": np.nan, "k_c1": np.nan, "k_c2": np.nan}, mask

        X_m = X[mask]
        y_m = y[mask]
        C_m = C[mask] if has_color else np.zeros_like(X_m)

        # Design matrix: [1, X, C*X, C^2*X]
        if has_color:
            A = np.column_stack([np.ones_like(X_m), X_m, C_m * X_m, (C_m**2) * X_m])
        else:
            A = np.column_stack([np.ones_like(X_m), X_m])

        try:
            coef, _, _, _ = np.linalg.lstsq(A, y_m, rcond=None)
            if has_color:
                return {"zp": coef[0], "k1": coef[1], "k_c1": coef[2], "k_c2": coef[3]}, mask
            else:
                return {"zp": coef[0], "k1": coef[1], "k_c1": np.nan, "k_c2": np.nan}, mask
        except Exception:
            return {"zp": np.nan, "k1": np.nan, "k_c1": np.nan, "k_c2": np.nan}, mask

    @staticmethod
    def _robust_fit(X, C, y, model="A", clip_sigma=3.0, iters=5, min_points=5):
        """Robust fitting with sigma clipping."""
        X = np.asarray(X, float)
        C = np.asarray(C, float) if C is not None else np.zeros_like(X)
        y = np.asarray(y, float)

        mask = np.isfinite(X) & np.isfinite(y)

        for _ in range(int(iters)):
            if mask.sum() < min_points:
                break

            if model == "A":
                result, _ = ExtinctionFitWorker._fit_model_a(X[mask], C[mask], y[mask])
                zp, k1, k2, k_c = result["zp"], result["k1"], result.get("k2", 0), result.get("k_c", 0)
                if np.isnan(k_c):
                    k_c = 0
                if np.isnan(k2):
                    k2 = 0
                yhat = zp + k1 * X[mask] + k2 * (X[mask]**2) + k_c * C[mask] * X[mask]
            else:
                result, _ = ExtinctionFitWorker._fit_model_b(X[mask], C[mask], y[mask])
                zp, k1, k_c1, k_c2 = result["zp"], result["k1"], result.get("k_c1", 0), result.get("k_c2", 0)
                if np.isnan(k_c1):
                    k_c1 = 0
                if np.isnan(k_c2):
                    k_c2 = 0
                yhat = zp + k1 * X[mask] + k_c1 * C[mask] * X[mask] + k_c2 * (C[mask]**2) * X[mask]

            resid = y[mask] - yhat
            med = np.nanmedian(resid)
            mad = np.nanmedian(np.abs(resid - med))
            sig = 1.4826 * mad if mad > 0 else np.nanstd(resid)

            if not np.isfinite(sig) or sig <= 0:
                break

            keep = np.abs(resid - med) <= float(clip_sigma) * sig
            if keep.all():
                break

            new_mask = mask.copy()
            mask_idx = np.where(mask)[0]
            new_mask[mask_idx] = keep
            mask = new_mask

        # Final fit
        if mask.sum() < min_points:
            if model == "A":
                return {"zp": np.nan, "k1": np.nan, "k2": np.nan, "k_c": np.nan}, mask, 0
            else:
                return {"zp": np.nan, "k1": np.nan, "k_c1": np.nan, "k_c2": np.nan}, mask, 0

        if model == "A":
            result, _ = ExtinctionFitWorker._fit_model_a(X[mask], C[mask], y[mask])
        else:
            result, _ = ExtinctionFitWorker._fit_model_b(X[mask], C[mask], y[mask])

        return result, mask, int(mask.sum())

    def run(self):
        try:
            P = self.params.P
            result_dir = self.result_dir

            # Load selections
            filter_selections = self._load_selection_ids_by_filter(result_dir)
            legacy_target_id, legacy_comp_ids, legacy_target_sid, legacy_comp_sids = self._load_selection_ids(result_dir)

            if not filter_selections and legacy_target_id is None:
                raise RuntimeError("Target/comparison selection missing (step 8)")

            self._log(f"Extinction fit started with Model {self.model}")

            # Load photometry index
            idx_path = step9_dir(result_dir) / "photometry_index.csv"
            if not idx_path.exists():
                idx_path = result_dir / "photometry_index.csv"
            if not idx_path.exists():
                raise FileNotFoundError("photometry_index.csv not found")
            idx = pd.read_csv(idx_path)
            self._log(f"Index frames: {len(idx)}")

            # Load airmass data
            frame_airmass_path = result_dir / "frame_airmass.csv"
            if not frame_airmass_path.exists():
                # Check legacy path
                from ...utils.step_paths import step11_zeropoint_dir
                frame_airmass_path = step11_zeropoint_dir(result_dir) / "frame_airmass.csv"
            if not frame_airmass_path.exists():
                from ...utils.step_paths import step11_dir
                frame_airmass_path = step11_dir(result_dir) / "frame_airmass.csv"

            airmass_map = {}
            date_map = {}
            if frame_airmass_path.exists():
                try:
                    df_air = pd.read_csv(frame_airmass_path)
                    if "file" in df_air.columns and "airmass" in df_air.columns:
                        def _store_airmass(key, value):
                            if not key or not np.isfinite(value):
                                return
                            airmass_map.setdefault(key, float(value))
                            base = Path(key).name
                            if base:
                                airmass_map.setdefault(base, float(value))

                        def _store_date(key, value):
                            if not key:
                                return
                            parsed = ExtinctionFitWorker._parse_date_from_datetime(value)
                            if not parsed:
                                return
                            date_map.setdefault(key, parsed)
                            base = Path(key).name
                            if base:
                                date_map.setdefault(base, parsed)

                        for _, row in df_air.iterrows():
                            fname = str(row.get("file", "")).strip()
                            am = pd.to_numeric(row.get("airmass", np.nan), errors="coerce")
                            _store_airmass(fname, am)
                            for key in ("date", "datetime_utc", "datetime_local"):
                                if key in row:
                                    _store_date(fname, row.get(key))
                                    if fname in date_map:
                                        break
                except Exception:
                    pass

            # Build data for fitting
            rows = []
            files = idx["file"].astype(str).tolist() if "file" in idx.columns else []
            total = len(files)

            for i, fname in enumerate(files, start=1):
                if self._stop_requested:
                    self.finished.emit({"stopped": True})
                    return
                self.progress.emit(i, total, fname)

                # Find photometry file
                phot_path = step9_dir(result_dir) / f"{fname}_photometry.tsv"
                if not phot_path.exists():
                    phot_path = result_dir / f"{fname}_photometry.tsv"
                if not phot_path.exists():
                    continue

                try:
                    dfp = pd.read_csv(phot_path, sep="\t")
                except Exception:
                    dfp = pd.read_csv(phot_path)

                if "mag" not in dfp.columns:
                    continue

                # Get filter
                filt_key = "unknown"
                if "filter" in idx.columns:
                    frow = idx[idx["file"].astype(str) == fname]
                    if not frow.empty:
                        filt_key = self._normalize_filter_key(str(frow["filter"].iloc[0]))

                # Get target/comp IDs
                if filt_key in filter_selections:
                    sel = filter_selections[filt_key]
                    target_id = sel.get("target_id")
                    comp_ids = sel.get("comp_ids", [])
                    target_sid = sel.get("target_source_id")
                    comp_sids = sel.get("comparison_source_ids", [])
                else:
                    target_id = legacy_target_id
                    comp_ids = legacy_comp_ids
                    target_sid = legacy_target_sid
                    comp_sids = legacy_comp_sids

                use_source_ids = target_sid is not None and "source_id" in dfp.columns
                if use_source_ids:
                    id_col = "source_id"
                    target_value = target_sid
                    comp_values = comp_sids
                else:
                    id_col = "ID"
                    target_value = target_id
                    comp_values = comp_ids

                if target_value is None or id_col not in dfp.columns:
                    continue
                if not comp_values:
                    continue

                # Get target magnitude
                try:
                    target_value = int(target_value)
                except Exception:
                    continue
                id_series = pd.to_numeric(dfp[id_col], errors="coerce")
                row_t = dfp[id_series == target_value]
                if row_t.empty:
                    continue
                tmag = pd.to_numeric(row_t["mag"].iloc[0], errors="coerce")
                if not np.isfinite(tmag):
                    continue

                # Get comparison magnitudes
                cmags = []
                for cid in comp_values:
                    try:
                        cid_val = int(cid)
                    except Exception:
                        continue
                    row_c = dfp[id_series == cid_val]
                    if not row_c.empty:
                        cval = pd.to_numeric(row_c["mag"].iloc[0], errors="coerce")
                        if np.isfinite(cval):
                            cmags.append(cval)
                if not cmags:
                    continue

                comp_avg = float(np.nanmean(cmags))
                diff_mag = tmag - comp_avg

                # Resolve FITS path
                fpath = step2_cropped_dir(result_dir) / fname
                if not fpath.exists():
                    fpath = self.params.get_file_path(fname) if hasattr(self.params, 'get_file_path') else None
                if fpath is not None and not fpath.exists():
                    fpath = None

                # Get airmass and date from cached map or header
                airmass_val = airmass_map.get(fname, np.nan)
                obs_date = date_map.get(fname, "")

                if not obs_date:
                    obs_date = self._extract_date_from_path(fpath, fname) if fpath else self._extract_date_from_path(None, fname)

                if (not np.isfinite(airmass_val)) or (not obs_date):
                    if fpath is not None and fpath.exists():
                        try:
                            hdr = fits.getheader(fpath)
                            info = compute_airmass_from_header(
                                hdr,
                                float(getattr(P, "site_lat_deg", 0.0)),
                                float(getattr(P, "site_lon_deg", 0.0)),
                                float(getattr(P, "site_alt_m", 0.0)),
                                float(getattr(P, "site_tz_offset_hours", 0.0)),
                            )
                            if not np.isfinite(airmass_val):
                                airmass_val = info.get("airmass", np.nan)
                            if not obs_date:
                                obs_date = self._parse_date_from_datetime(info.get("datetime_utc")) or \
                                           self._parse_date_from_datetime(info.get("datetime_local")) or \
                                           self._extract_date_from_header(hdr) or \
                                           obs_date
                        except Exception:
                            pass

                if not obs_date:
                    obs_date = "unknown"

                rows.append({
                    "file": fname,
                    "filter": filt_key,
                    "date": obs_date,
                    "airmass": float(airmass_val) if np.isfinite(airmass_val) else np.nan,
                    "target_mag": tmag,
                    "comp_avg": comp_avg,
                    "diff_mag": diff_mag,
                    "color_delta": np.nan,  # TODO: add color term support
                })

            if not rows:
                raise RuntimeError("No valid extinction-fit points")

            df = pd.DataFrame(rows)
            self._log(f"Collected {len(df)} data points")

            # Perform fitting by date and filter
            clip_sigma = float(getattr(P, "extfit_clip_sigma", 3.0))
            fit_iters = int(getattr(P, "extfit_fit_iters", 5))
            min_points = int(getattr(P, "extfit_min_points", 5))

            fit_rows = []
            for (date_val, filt), sub in df.groupby(["date", "filter"]):
                x = sub["airmass"].to_numpy(float)
                y = sub["diff_mag"].to_numpy(float)
                c = sub["color_delta"].to_numpy(float)

                result, mask, n_used = self._robust_fit(
                    x, c, y, model=self.model,
                    clip_sigma=clip_sigma, iters=fit_iters, min_points=min_points
                )

                fit_row = {
                    "date": date_val,
                    "filter": filt,
                    "model": self.model,
                    "n_total": int(np.isfinite(x).sum()),
                    "n_used": n_used,
                }
                fit_row.update(result)
                fit_rows.append(fit_row)

                self._log(f"[{date_val}][{filt}] Model {self.model}: n={n_used}, k1={result.get('k1', np.nan):.5f}")

            # Save results
            out_dir = tool_extinction_dir(result_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            fit_df = pd.DataFrame(fit_rows)
            fit_path = out_dir / "extinction_fit_by_filter.csv"
            points_path = out_dir / "extinction_fit_points.csv"

            fit_df.to_csv(fit_path, index=False)
            df.to_csv(points_path, index=False)

            self._log(f"Saved {fit_path.name} | rows={len(fit_df)}")
            self._log(f"Saved {points_path.name} | rows={len(df)}")

            self.finished.emit({
                "ok": True,
                "fit_path": str(fit_path),
                "points_path": str(points_path),
                "fit_df": fit_df.to_dict("records"),
                "points_df": df.to_dict("records"),
            })

        except Exception as e:
            import traceback
            self._log(f"Error: {e}")
            self._log(traceback.format_exc())
            self.error.emit(str(e))


class ExtinctionZeropointToolWindow(QWidget):
    """Extinction & Zeropoint Fitting Tool Window."""

    def __init__(self, params, project_state, parent=None):
        super().__init__(parent)
        self.params = params
        self.project_state = project_state
        self.worker = None
        self.fit_results = None
        self.points_df = None

        self.setWindowTitle("Extinction & Zeropoint Tool")
        self.setWindowFlag(Qt.Window, True)
        self.resize(1200, 800)
        self.setMinimumSize(900, 600)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Info banner
        info = QLabel(
            "Atmospheric extinction fitting tool for differential photometry.\n"
            "Model A: diff = k1*X + k2*X^2 + k_c*C*X + ZP\n"
            "Model B: diff = k1*X + k_c1*C*X + k_c2*C^2*X + ZP"
        )
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        layout.addWidget(info)

        # Controls
        controls_group = QGroupBox("Fit Settings")
        controls_layout = QHBoxLayout(controls_group)

        controls_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["A: k1*X + k2*X^2 + k_c*C*X", "B: k1*X + k_c1*C*X + k_c2*C^2*X"])
        controls_layout.addWidget(self.model_combo)

        controls_layout.addWidget(QLabel("Date Filter:"))
        self.date_combo = QComboBox()
        self.date_combo.addItem("All Dates")
        controls_layout.addWidget(self.date_combo)

        controls_layout.addWidget(QLabel("Band Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All Filters")
        controls_layout.addWidget(self.filter_combo)

        controls_layout.addStretch()

        self.btn_run = QPushButton("Run Fit")
        self.btn_run.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 20px; }")
        self.btn_run.clicked.connect(self._run_fit)
        controls_layout.addWidget(self.btn_run)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_fit)
        controls_layout.addWidget(self.btn_stop)

        self.btn_save = QPushButton("Save Results")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_results)
        controls_layout.addWidget(self.btn_save)

        layout.addWidget(controls_group)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Main content area with plot and results
        content_layout = QHBoxLayout()

        # Plot area
        plot_group = QGroupBox("Airmass vs Differential Magnitude")
        plot_layout = QVBoxLayout(plot_group)

        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        content_layout.addWidget(plot_group, stretch=2)

        # Results table
        results_group = QGroupBox("Fit Results")
        results_layout = QVBoxLayout(results_group)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels(["Date", "Filter", "k1", "k2/k_c1", "k_c/k_c2", "ZP", "N", "Scatter"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        results_layout.addWidget(self.results_table)

        content_layout.addWidget(results_group, stretch=1)

        layout.addLayout(content_layout)

        # Log area
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_group)

    def _log(self, msg: str):
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {msg}")

    def _run_fit(self):
        result_dir = self.params.P.result_dir

        # Check if required files exist
        idx_path = step9_dir(result_dir) / "photometry_index.csv"
        if not idx_path.exists():
            idx_path = result_dir / "photometry_index.csv"
        if not idx_path.exists():
            QMessageBox.warning(self, "Missing Data", "photometry_index.csv not found. Run Step 9 first.")
            return

        # Get model selection
        model = "A" if self.model_combo.currentIndex() == 0 else "B"

        self._log(f"Starting extinction fit with Model {model}")
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker = ExtinctionFitWorker(self.params, result_dir, model=model)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._log)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _stop_fit(self):
        if self.worker:
            self.worker.stop()
            self._log("Stopping fit...")

    def _on_progress(self, current: int, total: int, msg: str):
        pct = int(100 * current / max(total, 1))
        self.progress_bar.setValue(pct)

    def _on_finished(self, result: dict):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)

        if result.get("stopped"):
            self._log("Fit stopped by user")
            return

        if result.get("ok"):
            self._log("Fit completed successfully")
            self.btn_save.setEnabled(True)

            self.fit_results = result.get("fit_df", [])
            self.points_df = pd.DataFrame(result.get("points_df", []))

            self._update_combos()
            self._update_table()
            self._update_plot()
        else:
            self._log("Fit failed")

    def _on_error(self, error_msg: str):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)
        self._log(f"Error: {error_msg}")
        QMessageBox.critical(self, "Error", f"Fit failed:\n{error_msg}")

    def _update_combos(self):
        if self.points_df is None or self.points_df.empty:
            return

        # Update date combo
        dates = sorted(self.points_df["date"].dropna().unique())
        self.date_combo.clear()
        self.date_combo.addItem("All Dates")
        self.date_combo.addItems(dates)
        self.date_combo.currentTextChanged.connect(self._update_plot)

        # Update filter combo
        filters = sorted(self.points_df["filter"].dropna().unique())
        self.filter_combo.clear()
        self.filter_combo.addItem("All Filters")
        self.filter_combo.addItems(filters)
        self.filter_combo.currentTextChanged.connect(self._update_plot)

    def _update_table(self):
        if not self.fit_results:
            return

        self.results_table.setRowCount(len(self.fit_results))
        for i, row in enumerate(self.fit_results):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(row.get("date", ""))))
            self.results_table.setItem(i, 1, QTableWidgetItem(str(row.get("filter", ""))))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{row.get('k1', np.nan):.5f}"))

            # Model A uses k2, Model B uses k_c1
            k2_or_kc1 = row.get("k2", row.get("k_c1", np.nan))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{k2_or_kc1:.5f}"))

            # Model A uses k_c, Model B uses k_c2
            kc_or_kc2 = row.get("k_c", row.get("k_c2", np.nan))
            self.results_table.setItem(i, 4, QTableWidgetItem(f"{kc_or_kc2:.5f}"))

            self.results_table.setItem(i, 5, QTableWidgetItem(f"{row.get('zp', np.nan):.4f}"))
            self.results_table.setItem(i, 6, QTableWidgetItem(str(row.get("n_used", 0))))
            self.results_table.setItem(i, 7, QTableWidgetItem(""))

    def _update_plot(self):
        if self.points_df is None or self.points_df.empty:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        df = self.points_df.copy()

        # Apply filters
        date_filter = self.date_combo.currentText()
        filter_filter = self.filter_combo.currentText()

        if date_filter != "All Dates":
            df = df[df["date"] == date_filter]
        if filter_filter != "All Filters":
            df = df[df["filter"] == filter_filter]

        if df.empty:
            ax.set_title("No data to plot")
            self.canvas.draw()
            return

        # Color by date
        dates = df["date"].unique()
        colors = mpl.cm.tab10(np.linspace(0, 1, len(dates)))
        date_colors = dict(zip(dates, colors))

        # Plot data points
        for date_val in dates:
            sub = df[df["date"] == date_val]
            color = date_colors[date_val]
            ax.scatter(sub["airmass"], sub["diff_mag"], c=[color], s=20, alpha=0.7, label=date_val)

        # Plot fit lines if available
        if self.fit_results:
            for row in self.fit_results:
                if date_filter != "All Dates" and row.get("date") != date_filter:
                    continue
                if filter_filter != "All Filters" and row.get("filter") != filter_filter:
                    continue

                k1 = row.get("k1", 0)
                k2 = row.get("k2", row.get("k_c1", 0))
                zp = row.get("zp", 0)

                if np.isfinite(k1) and np.isfinite(zp):
                    x_line = np.linspace(1.0, 2.5, 100)
                    y_line = zp + k1 * x_line
                    if np.isfinite(k2):
                        y_line += k2 * x_line**2

                    date_val = row.get("date", "")
                    color = date_colors.get(date_val, "red")
                    ax.plot(x_line, y_line, c=color, linestyle="--", alpha=0.8)

        ax.set_xlabel("Airmass (X)")
        ax.set_ylabel("Differential Magnitude (Target - Comparison)")
        ax.set_title("Airmass vs Differential Magnitude")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def _save_results(self):
        if not self.fit_results:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results",
            str(self.params.P.result_dir / "extinction_fit_results.csv"),
            "CSV Files (*.csv)"
        )

        if file_path:
            df = pd.DataFrame(self.fit_results)
            df.to_csv(file_path, index=False)
            self._log(f"Results saved to {file_path}")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        event.accept()
