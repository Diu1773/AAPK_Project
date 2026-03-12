"""
Step 7: Aperture/Annulus Decision + Aperture Correction
Ported from AAPKI_GUI.ipynb Cell 9.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import SigmaClip
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats, aperture_photometry

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QMessageBox,
    QTextEdit, QDialog, QFormLayout, QDialogButtonBox, QProgressBar,
    QCheckBox, QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from .step_window_base import StepWindowBase
from ...utils.step_paths import (
    step2_cropped_dir,
    step4_dir,
    step5_dir,
    step9_dir,
    legacy_step7_wcs_dir,
)


class ApertureWorker(QThread):
    """Worker for aperture/annulus calculation"""
    progress = pyqtSignal(int, int, str)
    file_done = pyqtSignal(str, dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str, str)

    def __init__(
        self,
        file_list,
        params,
        data_dir,
        result_dir,
        cache_dir,
        use_cropped=False,
        output_dir: Path | None = None,
    ):
        super().__init__()
        self.file_list = list(file_list)
        self.params = params
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.cache_dir = Path(cache_dir)
        self.use_cropped = use_cropped
        self.output_dir = Path(output_dir) if output_dir is not None else self.result_dir
        self._stop_requested = False
        self.last_error = None

    def stop(self):
        self._stop_requested = True

    def _load_fwhm_from_meta(self, fname):
        meta_path = self.cache_dir / f"detect_{fname}.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                for k in ("fwhm_med_rad_px", "fwhm_med_px", "fwhm_px", "fwhm_med"):
                    v = meta.get(k, None)
                    if v is not None:
                        v = float(v)
                        if np.isfinite(v) and v > 0:
                            return v
            except Exception:
                pass
        return float(getattr(self.params.P, "fwhm_pix_guess", 6.0))

    @staticmethod
    def _to_float(val, default):
        try:
            if val is None:
                return float(default)
            out = float(val)
            return out if np.isfinite(out) else float(default)
        except Exception:
            return float(default)

    @staticmethod
    def _to_int(val, default):
        try:
            if val is None:
                return int(default)
            return int(float(val))
        except Exception:
            return int(default)

    @staticmethod
    def _build_scale_grid(start, stop, step):
        start = float(start)
        stop = float(stop)
        step = float(step)
        if step <= 0:
            step = 0.25
        if stop < start:
            start, stop = stop, start
        n = int(np.floor((stop - start) / step + 1e-9)) + 1
        n = int(np.clip(n, 1, 200))
        vals = [start + k * step for k in range(n)]
        if (not vals) or (vals[-1] < stop - 1e-9):
            vals.append(stop)
        return [float(round(v, 6)) for v in vals]

    def run(self):
        try:
            P = self.params.P
            ps = self._to_float(getattr(P, "pixel_scale_arcsec", np.nan), np.nan)
            output_dir = Path(self.output_dir) if self.output_dir is not None else step9_dir(self.result_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            ap_scale = self._to_float(getattr(P, "phot_aperture_scale", 1.0), 1.0)
            ann_in_scale = self._to_float(getattr(P, "fitsky_annulus_scale", 4.0), 4.0)
            ann_out_scale = self._to_float(getattr(P, "fitsky_dannulus_scale", 2.0), 2.0)
            cbox_scale = self._to_float(getattr(P, "center_cbox_scale", 1.5), 1.5)

            fwhm_px_min = self._to_float(getattr(P, "fwhm_px_min", 3.5), 3.5)
            fwhm_px_max = self._to_float(getattr(P, "fwhm_px_max", 8.0), 8.0)

            min_r_ap_px = self._to_float(getattr(P, "min_r_ap_px", 4.0), 4.0)
            min_r_in_px = self._to_float(getattr(P, "min_r_in_px", 12.0), 12.0)
            min_r_out_px = self._to_float(getattr(P, "min_r_out_px", 20.0), 20.0)
            ann_gap = self._to_float(getattr(P, "annulus_min_gap_px", 6.0), 6.0)
            ann_minw = self._to_float(getattr(P, "annulus_min_width_px", 12.0), 12.0)

            apcorr_apply = bool(getattr(P, "apcorr_apply", True))
            apcorr_use_min_n = self._to_int(getattr(P, "apcorr_use_min_n", 20), 20)
            apcorr_scatter_max = self._to_float(getattr(P, "apcorr_scatter_max", 0.05), 0.05)
            apcorr_small_scale = self._to_float(getattr(P, "apcorr_small_scale", 1.0), 1.0)
            apcorr_large_scale = self._to_float(getattr(P, "apcorr_large_scale", 3.0), 3.0)
            ann_sigma = self._to_float(getattr(P, "annulus_sigma_clip", 3.0), 3.0)
            ann_maxiter = self._to_int(getattr(P, "fitsky_max_iter", 5), 5)

            apcorr_optimize_scales = bool(getattr(P, "apcorr_optimize_scales", True))
            apcorr_small_scale_min = self._to_float(getattr(P, "apcorr_small_scale_min", apcorr_small_scale), apcorr_small_scale)
            apcorr_small_scale_max = self._to_float(getattr(P, "apcorr_small_scale_max", apcorr_small_scale), apcorr_small_scale)
            apcorr_large_scale_min = self._to_float(getattr(P, "apcorr_large_scale_min", apcorr_large_scale), apcorr_large_scale)
            apcorr_large_scale_max = self._to_float(getattr(P, "apcorr_large_scale_max", apcorr_large_scale), apcorr_large_scale)
            apcorr_scale_step = max(0.05, self._to_float(getattr(P, "apcorr_scale_step", 0.2), 0.2))
            apcorr_min_gap_fwhm = max(0.2, self._to_float(getattr(P, "apcorr_min_gap_fwhm", 1.0), 1.0))
            apcorr_max_pairs = max(1, self._to_int(getattr(P, "apcorr_max_pairs", 24), 24))
            apcorr_max_sources = max(30, self._to_int(getattr(P, "apcorr_max_sources", 250), 250))

            phot_use_qc_pass_only = bool(getattr(P, "phot_use_qc_pass_only", False))

            fixed_pair = (float(apcorr_small_scale), float(apcorr_large_scale))
            if apcorr_optimize_scales:
                small_grid = self._build_scale_grid(apcorr_small_scale_min, apcorr_small_scale_max, apcorr_scale_step)
                large_grid = self._build_scale_grid(apcorr_large_scale_min, apcorr_large_scale_max, apcorr_scale_step)
                apcorr_pairs = []
                for s_sc in small_grid:
                    for l_sc in large_grid:
                        if l_sc + 1e-9 < (s_sc + apcorr_min_gap_fwhm):
                            continue
                        apcorr_pairs.append((float(s_sc), float(l_sc)))
                if fixed_pair not in apcorr_pairs:
                    apcorr_pairs.append(fixed_pair)
                if len(apcorr_pairs) > apcorr_max_pairs:
                    idx = np.linspace(0, len(apcorr_pairs) - 1, apcorr_max_pairs).astype(int)
                    apcorr_pairs = [apcorr_pairs[i] for i in np.unique(idx)]
            else:
                apcorr_pairs = [fixed_pair]

            files = list(self.file_list)
            if phot_use_qc_pass_only:
                qpath = step5_dir(self.result_dir) / "frame_quality.csv"
                if not qpath.exists():
                    qpath = legacy_step7_wcs_dir(self.result_dir) / "frame_quality.csv"
                if not qpath.exists():
                    qpath = self.result_dir / "frame_quality.csv"
                if qpath.exists():
                    try:
                        dfq = pd.read_csv(qpath)
                        good = set(dfq.loc[dfq["passed"] == True, "file"].astype(str).tolist())
                        files = [f for f in files if f in good]
                    except Exception:
                        pass

            rows_ap = []
            rows_apcorr = []
            rows_apcorr_candidates = []
            total = len(files)

            for i, fname in enumerate(files, 1):
                if self._stop_requested:
                    break

                fwhm_med = float(self._load_fwhm_from_meta(fname))
                fwhm_used = float(np.clip(fwhm_med, fwhm_px_min, fwhm_px_max))

                r_ap = max(ap_scale * fwhm_used, min_r_ap_px)
                r_in = max(ann_in_scale * fwhm_used, max(min_r_in_px, r_ap + ann_gap))
                r_out = max(r_in + ann_out_scale * fwhm_used, r_in + ann_minw, min_r_out_px)
                cbox_px = max(cbox_scale * fwhm_used, 5.0)

                row = dict(
                    file=fname,
                    fwhm_med=fwhm_med,
                    fwhm_used=fwhm_used,
                    r_ap=r_ap,
                    r_in=r_in,
                    r_out=r_out,
                    cbox_px=cbox_px,
                )

                if np.isfinite(ps) and ps > 0:
                    row.update(dict(
                        fwhm_med_arcsec=fwhm_med * ps,
                        fwhm_used_arcsec=fwhm_used * ps,
                        r_ap_arcsec=r_ap * ps,
                        r_in_arcsec=r_in * ps,
                        r_out_arcsec=r_out * ps,
                        cbox_arcsec=cbox_px * ps,
                    ))

                rows_ap.append(row)
                (self.cache_dir / f"ap_{fname}.json").write_text(
                    json.dumps(row, indent=2), encoding="utf-8"
                )

                apc_row = dict(
                    file=fname,
                    apcorr=np.nan,
                    rel_scatter=np.nan,
                    n_used=0,
                    apply=False,
                    optimize_scales=bool(apcorr_optimize_scales),
                    n_candidates=int(len(apcorr_pairs)),
                )
                if apcorr_apply:
                    det_csv = self.cache_dir / f"detect_{fname}.csv"
                    if not det_csv.exists():
                        det_alt = step4_dir(self.result_dir) / f"detect_{fname}.csv"
                        if det_alt.exists():
                            det_csv = det_alt
                    if not det_csv.exists():
                        det_alt = self.result_dir / f"detect_{fname}.csv"
                        if det_alt.exists():
                            det_csv = det_alt
                    if det_csv.exists():
                        try:
                            if self.use_cropped:
                                img_path = step2_cropped_dir(self.result_dir) / fname
                                if not img_path.exists():
                                    img_path = (self.result_dir / "cropped" / fname)
                            else:
                                try:
                                    img_path = self.params.get_file_path(fname)
                                except Exception:
                                    img_path = self.data_dir / fname
                                if not Path(img_path).exists():
                                    img_path = self.data_dir / fname
                            img = fits.getdata(img_path).astype(float)
                            det_df = pd.read_csv(det_csv)
                            x_col = "x" if "x" in det_df.columns else ("xcenter" if "xcenter" in det_df.columns else None)
                            y_col = "y" if "y" in det_df.columns else ("ycenter" if "ycenter" in det_df.columns else None)
                            if x_col is None or y_col is None:
                                raise ValueError(f"detect columns not found in {det_csv.name}")
                            xy_all = det_df[[x_col, y_col]].to_numpy(float)
                            if len(xy_all):
                                finite_xy = np.isfinite(xy_all[:, 0]) & np.isfinite(xy_all[:, 1])
                                xy_all = xy_all[finite_xy]
                            h, w = img.shape
                            if len(xy_all):
                                vals = img[xy_all[:, 1].astype(int).clip(0, h - 1),
                                           xy_all[:, 0].astype(int).clip(0, w - 1)]
                                order = np.argsort(vals)[::-1]
                                xy_all = xy_all[order][:apcorr_max_sources]

                            if len(xy_all):
                                max_large_scale = max((pair[1] for pair in apcorr_pairs), default=apcorr_large_scale)
                                max_small_scale = max((pair[0] for pair in apcorr_pairs), default=apcorr_small_scale)
                                max_large_r = max(
                                    max_large_scale * fwhm_used,
                                    (max_small_scale + apcorr_min_gap_fwhm) * fwhm_used,
                                    r_ap + apcorr_min_gap_fwhm * fwhm_used,
                                )
                                edge_pad = int(np.ceil(max(max_large_r, r_out) + 2.0))
                                edge_mask = (
                                    (xy_all[:, 0] >= edge_pad)
                                    & (xy_all[:, 0] <= (w - edge_pad - 1))
                                    & (xy_all[:, 1] >= edge_pad)
                                    & (xy_all[:, 1] <= (h - edge_pad - 1))
                                )
                                xy_all = xy_all[edge_mask]

                            sc = SigmaClip(ann_sigma, maxiters=ann_maxiter)
                            bkg_med = np.array([], dtype=float)
                            if len(xy_all):
                                try:
                                    an_all = CircularAnnulus(xy_all, r_in=r_in, r_out=r_out)
                                    st_an = ApertureStats(img, an_all, sigma_clip=sc)
                                    bkg_med = np.asarray(st_an.median, dtype=float)
                                    if bkg_med.ndim == 0:
                                        bkg_med = np.full(len(xy_all), float(bkg_med))
                                except Exception:
                                    bkg_med = np.array([], dtype=float)
                            if len(bkg_med) != len(xy_all):
                                bkg_med = np.array([], dtype=float)

                            if len(bkg_med):
                                valid_bkg = np.isfinite(bkg_med)
                                xy_use = xy_all[valid_bkg]
                                bkg_use = bkg_med[valid_bkg]
                            else:
                                xy_use = np.array([], dtype=float).reshape(0, 2)
                                bkg_use = np.array([], dtype=float)

                            cand_rows = []
                            for pair_idx, (small_scale, large_scale) in enumerate(apcorr_pairs):
                                r_small = max(small_scale * fwhm_used, r_ap)
                                r_large = max(
                                    large_scale * fwhm_used,
                                    r_small + apcorr_min_gap_fwhm * fwhm_used,
                                )

                                medc = np.nan
                                rel_sc = np.nan
                                n_used = 0
                                if len(xy_use):
                                    ap_s = CircularAperture(xy_use, r=r_small)
                                    ap_l = CircularAperture(xy_use, r=r_large)
                                    sum_s = np.asarray(aperture_photometry(img, ap_s)["aperture_sum"], dtype=float)
                                    sum_l = np.asarray(aperture_photometry(img, ap_l)["aperture_sum"], dtype=float)
                                    flux_s = sum_s - (bkg_use * ap_s.area)
                                    flux_l = sum_l - (bkg_use * ap_l.area)
                                    valid = np.isfinite(flux_s) & np.isfinite(flux_l) & (flux_s > 0) & (flux_l > 0)
                                    if np.any(valid):
                                        corr = (flux_l[valid] / flux_s[valid]).astype(float)
                                        corr = corr[np.isfinite(corr) & (corr > 0.6) & (corr < 3.0)]
                                        if len(corr) >= 8:
                                            c_med0 = float(np.nanmedian(corr))
                                            c_mad0 = float(np.nanmedian(np.abs(corr - c_med0)))
                                            c_sig0 = 1.4826 * c_mad0 if np.isfinite(c_mad0) else np.nan
                                            if np.isfinite(c_sig0) and c_sig0 > 0:
                                                keep = np.abs(corr - c_med0) <= (3.0 * c_sig0)
                                                if int(np.sum(keep)) >= 5:
                                                    corr = corr[keep]
                                        n_used = int(len(corr))
                                        medc = float(np.nanmedian(corr)) if n_used else np.nan
                                        if n_used >= 5 and np.isfinite(medc) and medc > 0:
                                            mad = float(np.nanmedian(np.abs(corr - medc)))
                                            rel_sc = 1.4826 * mad / medc if np.isfinite(mad) else np.nan

                                apply_flag = bool(
                                    n_used >= apcorr_use_min_n
                                    and np.isfinite(medc)
                                    and (medc >= 0.8)
                                    and (medc <= 2.0)
                                    and np.isfinite(rel_sc)
                                    and (rel_sc <= apcorr_scatter_max)
                                )
                                cand = dict(
                                    file=fname,
                                    pair_index=int(pair_idx),
                                    small_scale=float(small_scale),
                                    large_scale=float(large_scale),
                                    fwhm_med=fwhm_med,
                                    fwhm_used=fwhm_used,
                                    r_small=float(r_small),
                                    r_large=float(r_large),
                                    n_used=n_used,
                                    apcorr=medc,
                                    rel_scatter=rel_sc,
                                    apply=bool(apply_flag),
                                )
                                if np.isfinite(ps) and ps > 0:
                                    cand.update(dict(
                                        r_small_arcsec=float(r_small) * ps,
                                        r_large_arcsec=float(r_large) * ps,
                                    ))
                                cand_rows.append(cand)

                            best = None
                            good = [c for c in cand_rows if np.isfinite(c.get("rel_scatter", np.nan))]
                            enough_n = [c for c in good if int(c.get("n_used", 0)) >= apcorr_use_min_n]
                            if enough_n:
                                best = min(
                                    enough_n,
                                    key=lambda c: (
                                        float(c["rel_scatter"]),
                                        -int(c["n_used"]),
                                        float(c["small_scale"]),
                                    ),
                                )
                            elif good:
                                best = min(
                                    good,
                                    key=lambda c: (
                                        max(0, apcorr_use_min_n - int(c["n_used"])),
                                        float(c["rel_scatter"]),
                                        -int(c["n_used"]),
                                    ),
                                )
                            elif cand_rows:
                                best = max(cand_rows, key=lambda c: int(c.get("n_used", 0)))

                            if best is not None:
                                for cand in cand_rows:
                                    row_c = dict(cand)
                                    row_c["selected"] = bool(cand is best)
                                    rows_apcorr_candidates.append(row_c)

                                apc_row = dict(
                                    file=fname,
                                    fwhm_med=fwhm_med,
                                    fwhm_used=fwhm_used,
                                    small_scale_sel=float(best["small_scale"]),
                                    large_scale_sel=float(best["large_scale"]),
                                    r_small=float(best["r_small"]),
                                    r_large=float(best["r_large"]),
                                    n_used=int(best["n_used"]),
                                    apcorr=float(best["apcorr"]) if np.isfinite(best["apcorr"]) else np.nan,
                                    rel_scatter=float(best["rel_scatter"]) if np.isfinite(best["rel_scatter"]) else np.nan,
                                    apply=bool(best["apply"]),
                                    optimize_scales=bool(apcorr_optimize_scales),
                                    n_candidates=int(len(cand_rows)),
                                )
                                if np.isfinite(ps) and ps > 0:
                                    apc_row.update(dict(
                                        r_small_arcsec=float(best["r_small"]) * ps,
                                        r_large_arcsec=float(best["r_large"]) * ps,
                                    ))
                        except Exception:
                            apc_row = dict(
                                file=fname,
                                apcorr=np.nan,
                                rel_scatter=np.nan,
                                n_used=0,
                                apply=False,
                                optimize_scales=bool(apcorr_optimize_scales),
                                n_candidates=int(len(apcorr_pairs)),
                            )

                rows_apcorr.append(apc_row)
                (self.cache_dir / f"apcorr_{fname}.json").write_text(
                    json.dumps(apc_row, indent=2), encoding="utf-8"
                )

                self.file_done.emit(fname, row)
                self.progress.emit(i, total, fname)

            pd.DataFrame(rows_ap).to_csv(output_dir / "aperture_by_frame.csv", index=False)
            pd.DataFrame(rows_apcorr).to_csv(output_dir / "apcorr_summary.csv", index=False)
            if rows_apcorr_candidates:
                pd.DataFrame(rows_apcorr_candidates).to_csv(output_dir / "apcorr_candidates.csv", index=False)

            self.finished.emit({
                "total": len(rows_ap),
            })
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self.error.emit("WORKER", error_msg)
            self.finished.emit({})


class AperturePhotometryWindow(StepWindowBase):
    """Step 10: Aperture photometry prep"""

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        self.worker = None
        self.results = {}
        self.log_window = None
        self.file_list = []
        self.use_cropped = False

        super().__init__(
            step_index=99,
            step_name="Aperture Photometry (Legacy)",
            params=params,
            project_state=project_state,
            main_window=main_window
        )

        self.setup_step_ui()
        self.restore_state()

    def setup_step_ui(self):
        info = QLabel("Compute per-frame aperture/annulus sizes and aperture correction.")
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        self.content_layout.addWidget(info)

        control_layout = QHBoxLayout()
        btn_params = QPushButton("Aperture Parameters")
        btn_params.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 8px 15px; }")
        btn_params.clicked.connect(self.open_parameters_dialog)
        control_layout.addWidget(btn_params)

        control_layout.addStretch()

        self.btn_run = QPushButton("Run Aperture")
        self.btn_run.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 20px; }")
        self.btn_run.clicked.connect(self.run_aperture)
        control_layout.addWidget(self.btn_run)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px 15px; }")
        self.btn_stop.clicked.connect(self.stop_aperture)
        self.btn_stop.setEnabled(False)
        control_layout.addWidget(self.btn_stop)

        btn_log = QPushButton("Log")
        btn_log.setStyleSheet("QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 8px 15px; }")
        btn_log.clicked.connect(self.show_log_window)
        control_layout.addWidget(btn_log)

        self.content_layout.addLayout(control_layout)

        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("Ready")
        self.progress_label.setMinimumWidth(350)
        progress_layout.addWidget(self.progress_label)
        self.content_layout.addLayout(progress_layout)

        results_group = QGroupBox("Aperture Summary")
        results_layout = QVBoxLayout(results_group)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(["File", "FWHM", "r_ap", "r_in", "r_out"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        results_layout.addWidget(self.results_table)
        self.content_layout.addWidget(results_group)

        self.setup_log_window()
        self.populate_file_list()

    def setup_log_window(self):
        if self.log_window is not None:
            return
        self.log_window = QWidget(self, Qt.Window)
        self.log_window.setWindowTitle("Aperture Log")
        self.log_window.resize(800, 400)
        layout = QVBoxLayout(self.log_window)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        layout.addWidget(self.log_text)

    def show_log_window(self):
        if self.log_window is None:
            self.setup_log_window()
        self.log_window.show()
        self.log_window.raise_()
        self.log_window.activateWindow()

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def populate_file_list(self):
        cropped_dir = step2_cropped_dir(self.params.P.result_dir)
        if cropped_dir.exists() and list(cropped_dir.glob("*.fit*")):
            files = sorted([f.name for f in cropped_dir.glob("*.fit*")])
            self.use_cropped = True
        else:
            if not self.file_manager.filenames:
                try:
                    self.file_manager.scan_files()
                except Exception:
                    pass
            files = self.file_manager.filenames
            self.use_cropped = False
        self.file_list = list(files)

    def open_parameters_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Aperture Parameters")
        dialog.resize(480, 520)

        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        self.param_ap_scale = QDoubleSpinBox()
        self.param_ap_scale.setRange(0.5, 5.0)
        self.param_ap_scale.setSingleStep(0.1)
        self.param_ap_scale.setValue(float(getattr(self.params.P, "phot_aperture_scale", 1.0)))
        form.addRow("Aperture Scale:", self.param_ap_scale)

        self.param_ann_in = QDoubleSpinBox()
        self.param_ann_in.setRange(1.0, 10.0)
        self.param_ann_in.setSingleStep(0.5)
        self.param_ann_in.setValue(float(getattr(self.params.P, "fitsky_annulus_scale", 4.0)))
        form.addRow("Annulus Inner Scale:", self.param_ann_in)

        self.param_ann_out = QDoubleSpinBox()
        self.param_ann_out.setRange(0.5, 10.0)
        self.param_ann_out.setSingleStep(0.5)
        self.param_ann_out.setValue(float(getattr(self.params.P, "fitsky_dannulus_scale", 2.0)))
        form.addRow("Annulus Width Scale:", self.param_ann_out)

        self.param_cbox = QDoubleSpinBox()
        self.param_cbox.setRange(0.5, 5.0)
        self.param_cbox.setSingleStep(0.1)
        self.param_cbox.setValue(float(getattr(self.params.P, "center_cbox_scale", 1.5)))
        form.addRow("Center CBox Scale:", self.param_cbox)

        self.param_fwhm_min = QDoubleSpinBox()
        self.param_fwhm_min.setRange(0.5, 20.0)
        self.param_fwhm_min.setSingleStep(0.5)
        self.param_fwhm_min.setValue(float(getattr(self.params.P, "fwhm_px_min", 3.5)))
        form.addRow("FWHM Min (px):", self.param_fwhm_min)

        self.param_fwhm_max = QDoubleSpinBox()
        self.param_fwhm_max.setRange(1.0, 30.0)
        self.param_fwhm_max.setSingleStep(0.5)
        self.param_fwhm_max.setValue(float(getattr(self.params.P, "fwhm_px_max", 8.0)))
        form.addRow("FWHM Max (px):", self.param_fwhm_max)

        self.param_min_r_ap = QDoubleSpinBox()
        self.param_min_r_ap.setRange(1.0, 50.0)
        self.param_min_r_ap.setValue(float(getattr(self.params.P, "min_r_ap_px", 4.0)))
        form.addRow("Min r_ap (px):", self.param_min_r_ap)

        self.param_min_r_in = QDoubleSpinBox()
        self.param_min_r_in.setRange(1.0, 100.0)
        self.param_min_r_in.setValue(float(getattr(self.params.P, "min_r_in_px", 12.0)))
        form.addRow("Min r_in (px):", self.param_min_r_in)

        self.param_min_r_out = QDoubleSpinBox()
        self.param_min_r_out.setRange(1.0, 200.0)
        self.param_min_r_out.setValue(float(getattr(self.params.P, "min_r_out_px", 20.0)))
        form.addRow("Min r_out (px):", self.param_min_r_out)

        self.param_ann_gap = QDoubleSpinBox()
        self.param_ann_gap.setRange(0.0, 50.0)
        self.param_ann_gap.setValue(float(getattr(self.params.P, "annulus_min_gap_px", 6.0)))
        form.addRow("Annulus Min Gap (px):", self.param_ann_gap)

        self.param_ann_minw = QDoubleSpinBox()
        self.param_ann_minw.setRange(0.0, 100.0)
        self.param_ann_minw.setValue(float(getattr(self.params.P, "annulus_min_width_px", 12.0)))
        form.addRow("Annulus Min Width (px):", self.param_ann_minw)

        self.param_apcorr = QCheckBox("Enable")
        self.param_apcorr.setChecked(bool(getattr(self.params.P, "apcorr_apply", True)))
        form.addRow("Aperture Correction:", self.param_apcorr)

        self.param_apcorr_small = QDoubleSpinBox()
        self.param_apcorr_small.setRange(0.5, 5.0)
        self.param_apcorr_small.setValue(float(getattr(self.params.P, "apcorr_small_scale", 1.0)))
        form.addRow("Apcorr Small Scale:", self.param_apcorr_small)

        self.param_apcorr_large = QDoubleSpinBox()
        self.param_apcorr_large.setRange(1.0, 10.0)
        self.param_apcorr_large.setValue(float(getattr(self.params.P, "apcorr_large_scale", 3.0)))
        form.addRow("Apcorr Large Scale:", self.param_apcorr_large)

        self.param_apcorr_min_n = QSpinBox()
        self.param_apcorr_min_n.setRange(1, 500)
        self.param_apcorr_min_n.setValue(int(getattr(self.params.P, "apcorr_use_min_n", 20)))
        form.addRow("Apcorr Min N:", self.param_apcorr_min_n)

        self.param_apcorr_scatter = QDoubleSpinBox()
        self.param_apcorr_scatter.setRange(0.0, 0.5)
        self.param_apcorr_scatter.setSingleStep(0.01)
        self.param_apcorr_scatter.setValue(float(getattr(self.params.P, "apcorr_scatter_max", 0.05)))
        form.addRow("Apcorr Scatter Max:", self.param_apcorr_scatter)

        self.param_sigma_clip = QDoubleSpinBox()
        self.param_sigma_clip.setRange(1.0, 10.0)
        self.param_sigma_clip.setValue(float(getattr(self.params.P, "annulus_sigma_clip", 3.0)))
        form.addRow("Annulus Sigma Clip:", self.param_sigma_clip)

        self.param_max_iter = QSpinBox()
        self.param_max_iter.setRange(1, 20)
        self.param_max_iter.setValue(int(getattr(self.params.P, "fitsky_max_iter", 5)))
        form.addRow("Annulus Max Iter:", self.param_max_iter)

        self.param_qc_only = QCheckBox("Use QC Pass Only")
        self.param_qc_only.setChecked(bool(getattr(self.params.P, "phot_use_qc_pass_only", False)))
        form.addRow("QC Pass Only:", self.param_qc_only)

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: self.save_parameters(dialog))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec_()

    def save_parameters(self, dialog):
        self.params.P.phot_aperture_scale = self.param_ap_scale.value()
        self.params.P.fitsky_annulus_scale = self.param_ann_in.value()
        self.params.P.fitsky_dannulus_scale = self.param_ann_out.value()
        self.params.P.center_cbox_scale = self.param_cbox.value()
        self.params.P.fwhm_px_min = self.param_fwhm_min.value()
        self.params.P.fwhm_px_max = self.param_fwhm_max.value()
        self.params.P.min_r_ap_px = self.param_min_r_ap.value()
        self.params.P.min_r_in_px = self.param_min_r_in.value()
        self.params.P.min_r_out_px = self.param_min_r_out.value()
        self.params.P.annulus_min_gap_px = self.param_ann_gap.value()
        self.params.P.annulus_min_width_px = self.param_ann_minw.value()
        self.params.P.apcorr_apply = self.param_apcorr.isChecked()
        self.params.P.apcorr_small_scale = self.param_apcorr_small.value()
        self.params.P.apcorr_large_scale = self.param_apcorr_large.value()
        self.params.P.apcorr_use_min_n = self.param_apcorr_min_n.value()
        self.params.P.apcorr_scatter_max = self.param_apcorr_scatter.value()
        self.params.P.annulus_sigma_clip = self.param_sigma_clip.value()
        self.params.P.fitsky_max_iter = self.param_max_iter.value()
        self.params.P.phot_use_qc_pass_only = self.param_qc_only.isChecked()
        self.persist_params()
        self.save_state()
        QMessageBox.information(dialog, "Success", "Parameters saved!")
        dialog.accept()

    def run_aperture(self):
        if not self.file_list:
            QMessageBox.warning(self, "Warning", "No files to process")
            return
        if self.worker and self.worker.isRunning():
            return

        self.results = {}
        self.results_table.setRowCount(0)
        self.log_text.clear()

        self.worker = ApertureWorker(
            self.file_list,
            self.params,
            self.params.P.data_dir,
            self.params.P.result_dir,
            self.params.P.cache_dir,
            self.use_cropped
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.file_done.connect(self.on_file_done)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(self.file_list))
        self.progress_label.setText(f"0/{len(self.file_list)} | Starting...")
        self.worker.start()
        self.show_log_window()

    def stop_aperture(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()

    def on_progress(self, current, total, filename):
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"{current}/{total} | {filename}")

    def on_file_done(self, filename, result):
        self.results[filename] = result
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        self.results_table.setItem(row, 0, QTableWidgetItem(filename))
        fwhm_px = float(result.get("fwhm_med", 0.0))
        pixscale = float(getattr(self.params.P, "pixel_scale_arcsec", np.nan))
        if np.isfinite(pixscale) and pixscale > 0 and np.isfinite(fwhm_px):
            fwhm_arcsec = fwhm_px * pixscale
            fwhm_str = f'{fwhm_arcsec:.2f}" ({fwhm_px:.2f} px)'
        else:
            fwhm_str = f"{fwhm_px:.2f} px" if np.isfinite(fwhm_px) else "N/A"
        self.results_table.setItem(row, 1, QTableWidgetItem(fwhm_str))
        self.results_table.setItem(row, 2, QTableWidgetItem(f"{result.get('r_ap', 0):.2f}"))
        self.results_table.setItem(row, 3, QTableWidgetItem(f"{result.get('r_in', 0):.2f}"))
        self.results_table.setItem(row, 4, QTableWidgetItem(f"{result.get('r_out', 0):.2f}"))
        self.log(f"{filename}: r_ap={result.get('r_ap', 0):.2f} r_in={result.get('r_in', 0):.2f}")

    def on_error(self, filename, error):
        self.log(f"ERROR {filename}: {error}")

    def on_finished(self, summary):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_label.setText("Done")
        if summary:
            self.log(f"Aperture done: {summary.get('total', 0)} files")
        self.save_state()
        self.update_navigation_buttons()

    def validate_step(self) -> bool:
        ap_path = step9_dir(self.params.P.result_dir) / "aperture_by_frame.csv"
        if not ap_path.exists():
            ap_path = self.params.P.result_dir / "aperture_by_frame.csv"
        return ap_path.exists()

    def save_state(self):
        state_data = {
            "aperture_complete": (step9_dir(self.params.P.result_dir) / "aperture_by_frame.csv").exists()
            or (self.params.P.result_dir / "aperture_by_frame.csv").exists(),
            "phot_aperture_scale": getattr(self.params.P, "phot_aperture_scale", 1.0),
            "fitsky_annulus_scale": getattr(self.params.P, "fitsky_annulus_scale", 4.0),
            "fitsky_dannulus_scale": getattr(self.params.P, "fitsky_dannulus_scale", 2.0),
            "center_cbox_scale": getattr(self.params.P, "center_cbox_scale", 1.5),
            "fwhm_px_min": getattr(self.params.P, "fwhm_px_min", 3.5),
            "fwhm_px_max": getattr(self.params.P, "fwhm_px_max", 8.0),
            "min_r_ap_px": getattr(self.params.P, "min_r_ap_px", 4.0),
            "min_r_in_px": getattr(self.params.P, "min_r_in_px", 12.0),
            "min_r_out_px": getattr(self.params.P, "min_r_out_px", 20.0),
            "annulus_min_gap_px": getattr(self.params.P, "annulus_min_gap_px", 6.0),
            "annulus_min_width_px": getattr(self.params.P, "annulus_min_width_px", 12.0),
            "apcorr_apply": getattr(self.params.P, "apcorr_apply", True),
            "apcorr_small_scale": getattr(self.params.P, "apcorr_small_scale", 1.0),
            "apcorr_large_scale": getattr(self.params.P, "apcorr_large_scale", 3.0),
            "apcorr_use_min_n": getattr(self.params.P, "apcorr_use_min_n", 20),
            "apcorr_scatter_max": getattr(self.params.P, "apcorr_scatter_max", 0.05),
            "apcorr_optimize_scales": getattr(self.params.P, "apcorr_optimize_scales", True),
            "apcorr_small_scale_min": getattr(self.params.P, "apcorr_small_scale_min", 0.8),
            "apcorr_small_scale_max": getattr(self.params.P, "apcorr_small_scale_max", 1.4),
            "apcorr_large_scale_min": getattr(self.params.P, "apcorr_large_scale_min", 2.4),
            "apcorr_large_scale_max": getattr(self.params.P, "apcorr_large_scale_max", 4.0),
            "apcorr_scale_step": getattr(self.params.P, "apcorr_scale_step", 0.2),
            "apcorr_min_gap_fwhm": getattr(self.params.P, "apcorr_min_gap_fwhm", 1.0),
            "apcorr_max_pairs": getattr(self.params.P, "apcorr_max_pairs", 24),
            "apcorr_max_sources": getattr(self.params.P, "apcorr_max_sources", 250),
            "annulus_sigma_clip": getattr(self.params.P, "annulus_sigma_clip", 3.0),
            "fitsky_max_iter": getattr(self.params.P, "fitsky_max_iter", 5),
            "phot_use_qc_pass_only": getattr(self.params.P, "phot_use_qc_pass_only", False),
        }
        self.project_state.store_step_data("aperture_photometry", state_data)

    def restore_state(self):
        state_data = self.project_state.get_step_data("aperture_photometry")
        if state_data:
            for key, val in state_data.items():
                if hasattr(self.params.P, key):
                    setattr(self.params.P, key, val)
