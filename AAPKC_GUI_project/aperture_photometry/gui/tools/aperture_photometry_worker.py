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
from scipy.spatial import cKDTree
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats, aperture_photometry

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QMessageBox,
    QTextEdit, QDialog, QFormLayout, QDialogButtonBox, QProgressBar,
    QCheckBox, QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from ..workflow.step_window_base import StepWindowBase
from ...utils.step_paths import step2_cropped_dir, step4_dir, step5_dir, step9_dir, crop_is_active


class ApertureWorker(QThread):
    """Worker for aperture/annulus calculation"""
    progress = pyqtSignal(int, int, str)
    file_done = pyqtSignal(str, dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str, str)

    def __init__(self, file_list, params, data_dir, result_dir, cache_dir, use_cropped=False):
        super().__init__()
        self.file_list = list(file_list)
        self.params = params
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.cache_dir = Path(cache_dir)
        self.use_cropped = use_cropped
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    @staticmethod
    def _norm_path_key(path_value) -> str:
        if path_value is None:
            return ""
        try:
            s = str(path_value).strip().replace("\\", "/")
        except Exception:
            s = str(path_value).replace("\\", "/")
        if len(s) >= 3 and s[1] == ":" and s[2] == "/" and s[0].isalpha():
            s = f"/mnt/{s[0].lower()}/{s[3:]}"
        while "//" in s:
            s = s.replace("//", "/")
        if len(s) > 1 and s.endswith("/"):
            s = s[:-1]
        return s.lower()

    def _resolve_fits_path(self, fname: str) -> Path | None:
        if self.use_cropped and crop_is_active(self.result_dir):
            cdir = step2_cropped_dir(self.result_dir)
            cpath = cdir / fname
            if cpath.exists():
                return cpath
            legacy = self.result_dir / "cropped" / fname
            if legacy.exists():
                return legacy
        try:
            path = Path(self.params.get_file_path(fname))
            if path.exists():
                return path
        except Exception:
            pass
        return None

    def _detect_meta_compatible(self, fname: str, payload: dict, meta_path: Path) -> bool:
        if not isinstance(payload, dict):
            return False
        try:
            schema = int(payload.get("cache_schema", 0) or 0)
        except Exception:
            schema = 0
        if schema < 2:
            return False
        src = self._resolve_fits_path(fname)
        if src is None or not src.exists():
            return False
        try:
            st = src.stat()
            src_size = int(st.st_size)
            src_mtime_ns = int(st.st_mtime_ns)
        except Exception:
            return False
        if bool(payload.get("source_use_cropped", None)) != bool(self.use_cropped):
            return False
        if self._norm_path_key(payload.get("source_path")) != self._norm_path_key(src):
            return False
        try:
            if int(payload.get("source_size")) != src_size:
                return False
            # mtime can drift after in-place FITS header updates (Step5/Step6).
            # Keep cache valid when path/crop/size are unchanged.
            _ = int(payload.get("source_mtime_ns"))
        except Exception:
            return False
        return True

    def _load_fwhm_from_meta(self, fname):
        candidates = [
            self.cache_dir / f"detect_{fname}.json",
            step4_dir(self.result_dir) / f"detect_{fname}.json",
        ]
        candidates = [p for p in candidates if p.exists()]
        candidates.sort(key=lambda p: p.stat().st_mtime_ns if p.exists() else 0, reverse=True)
        for meta_path in candidates:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not self._detect_meta_compatible(fname, meta, meta_path):
                continue
            for k in ("fwhm_med_rad_px", "fwhm_med_px", "fwhm_px", "fwhm_med"):
                v = meta.get(k, None)
                if v is not None:
                    try:
                        v = float(v)
                    except Exception:
                        continue
                    if np.isfinite(v) and v > 0:
                        return v
        return self._to_float(getattr(self.params.P, "fwhm_pix_guess", 6.0), 6.0)

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
            output_dir = step9_dir(self.result_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

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
            ann_sigma = self._to_float(getattr(P, "annulus_sigma_clip", 3.0), 3.0)
            ann_maxiter = self._to_int(getattr(P, "fitsky_max_iter", 5), 5)

            # Growth curve parameters
            gc_scale_min = max(0.3, self._to_float(getattr(P, "apcorr_scale_min", 0.5), 0.5))
            gc_scale_max = max(gc_scale_min + 0.5, self._to_float(getattr(P, "apcorr_scale_max", 5.0), 5.0))
            gc_scale_step = max(0.1, self._to_float(getattr(P, "apcorr_scale_step", 0.25), 0.25))
            gc_large_ref_scale = self._to_float(getattr(P, "apcorr_large_ref_scale", 5.0), 5.0)
            gc_isolation_factor = max(1.0, self._to_float(getattr(P, "apcorr_isolation_factor", 2.0), 2.0))
            apcorr_max_sources = max(30, self._to_int(getattr(P, "apcorr_max_sources", 250), 250))

            gain = self._to_float(getattr(P, "gain_e_per_adu", 1.0), 1.0)
            rn_e = self._to_float(getattr(P, "rdnoise_e", 7.5), 7.5)

            phot_use_qc_pass_only = bool(getattr(P, "phot_use_qc_pass_only", False))

            # Build growth curve scale grid
            gc_scales = self._build_scale_grid(gc_scale_min, gc_scale_max, gc_scale_step)

            files_all = list(self.file_list)
            files = list(files_all)
            if phot_use_qc_pass_only:
                qpath = step5_dir(self.result_dir) / "frame_quality.csv"
                if not qpath.exists():
                    legacy_qpath = self.result_dir / "frame_quality.csv"
                    if legacy_qpath.exists():
                        qpath = legacy_qpath
                if qpath.exists():
                    try:
                        dfq = pd.read_csv(qpath)
                        good = set(dfq.loc[dfq["passed"] == True, "file"].astype(str).tolist())
                        filtered = [f for f in files_all if f in good]
                        files = filtered if filtered else list(files_all)
                    except Exception:
                        pass

            rows_ap = []
            rows_apcorr = []
            rows_growth_curve = []
            total = len(files)

            for i, fname in enumerate(files, 1):
                if self._stop_requested:
                    break

                fwhm_med = float(self._load_fwhm_from_meta(fname))
                fwhm_used = float(np.clip(fwhm_med, fwhm_px_min, fwhm_px_max))

                # Build radius grid for this frame
                radii_px = np.array([max(s * fwhm_used, min_r_ap_px) for s in gc_scales])
                # Remove duplicate radii (can happen when min_r_ap_px clamps small scales)
                _, unique_idx = np.unique(np.round(radii_px, 4), return_index=True)
                unique_idx = np.sort(unique_idx)
                radii_px = radii_px[unique_idx]
                scales_used = np.array(gc_scales)[unique_idx]

                r_large_ref = max(gc_large_ref_scale * fwhm_used, radii_px[-1] if len(radii_px) else min_r_ap_px)

                # Annulus for sky background (same as before)
                r_in = max(ann_in_scale * fwhm_used, max(min_r_in_px, r_large_ref + ann_gap))
                r_out = max(r_in + ann_out_scale * fwhm_used, r_in + ann_minw, min_r_out_px)
                cbox_px = max(cbox_scale * fwhm_used, 5.0)

                # Default aperture row (will be updated if growth curve succeeds)
                r_ap_default = max(ap_scale * fwhm_used, min_r_ap_px)

                apc_row = dict(
                    file=fname,
                    fwhm_med=fwhm_med,
                    fwhm_used=fwhm_used,
                    optimal_scale=ap_scale,
                    r_optimal=r_ap_default,
                    r_large_ref=r_large_ref,
                    n_used=0,
                    apcorr=np.nan,
                    mag_err_optimal=np.nan,
                    snr_optimal=np.nan,
                    apply=False,
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
                            img_path = self._resolve_fits_path(fname)
                            if img_path is None:
                                raise FileNotFoundError(f"FITS not found: {fname}")
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

                            # Sort by brightness and take top sources
                            if len(xy_all):
                                vals = img[xy_all[:, 1].astype(int).clip(0, h - 1),
                                           xy_all[:, 0].astype(int).clip(0, w - 1)]
                                order = np.argsort(vals)[::-1]
                                xy_all = xy_all[order][:apcorr_max_sources]

                            # Edge padding for largest aperture + annulus
                            if len(xy_all):
                                edge_pad = int(np.ceil(max(r_large_ref, r_out) + 2.0))
                                edge_mask = (
                                    (xy_all[:, 0] >= edge_pad)
                                    & (xy_all[:, 0] <= (w - edge_pad - 1))
                                    & (xy_all[:, 1] >= edge_pad)
                                    & (xy_all[:, 1] <= (h - edge_pad - 1))
                                )
                                xy_all = xy_all[edge_mask]

                            # Isolation filter: keep only stars whose nearest neighbor
                            # is at least isolation_factor * r_large_ref away.
                            if len(xy_all) >= 2:
                                tree = cKDTree(xy_all)
                                dists, _ = tree.query(xy_all, k=2)
                                nn_dist = dists[:, 1]  # distance to nearest neighbor
                                min_sep = gc_isolation_factor * r_large_ref
                                iso_mask = nn_dist >= min_sep
                                # Keep at least 10 stars even if few are isolated
                                if np.sum(iso_mask) >= 10:
                                    xy_all = xy_all[iso_mask]

                            # Measure sky background (once, using the outer annulus)
                            sc = SigmaClip(ann_sigma, maxiters=ann_maxiter)
                            bkg_med = np.array([], dtype=float)
                            bkg_std_arr = np.array([], dtype=float)
                            n_sky_arr = np.array([], dtype=float)
                            if len(xy_all):
                                try:
                                    an_all = CircularAnnulus(xy_all, r_in=r_in, r_out=r_out)
                                    st_an = ApertureStats(img, an_all, sigma_clip=sc)
                                    bkg_med = np.asarray(st_an.median, dtype=float)
                                    bkg_std_arr = np.asarray(st_an.std, dtype=float)
                                    # n_sky per source: approximate from annulus area
                                    n_sky_arr = np.full(len(xy_all), float(an_all.area))
                                    if bkg_med.ndim == 0:
                                        bkg_med = np.full(len(xy_all), float(bkg_med))
                                    if bkg_std_arr.ndim == 0:
                                        bkg_std_arr = np.full(len(xy_all), float(bkg_std_arr))
                                except Exception:
                                    bkg_med = np.array([], dtype=float)
                                    bkg_std_arr = np.array([], dtype=float)
                                    n_sky_arr = np.array([], dtype=float)

                            if len(bkg_med) != len(xy_all):
                                bkg_med = np.array([], dtype=float)
                                bkg_std_arr = np.array([], dtype=float)
                                n_sky_arr = np.array([], dtype=float)

                            if len(bkg_med):
                                valid_bkg = np.isfinite(bkg_med) & np.isfinite(bkg_std_arr)
                                xy_use = xy_all[valid_bkg]
                                bkg_use = bkg_med[valid_bkg]
                                bkg_std_use = bkg_std_arr[valid_bkg]
                                n_sky_use = n_sky_arr[valid_bkg]
                            else:
                                xy_use = np.array([], dtype=float).reshape(0, 2)
                                bkg_use = np.array([], dtype=float)
                                bkg_std_use = np.array([], dtype=float)
                                n_sky_use = np.array([], dtype=float)

                            n_stars = len(xy_use)

                            # --- Growth curve: photometry at each radius ---
                            # Per-star arrays: flux_e[n_radii, n_stars], mag_err[n_radii, n_stars]
                            n_radii = len(radii_px)
                            flux_e_all = np.full((n_radii, n_stars), np.nan)
                            mag_err_all = np.full((n_radii, n_stars), np.nan)
                            snr_all = np.full((n_radii, n_stars), np.nan)

                            if n_stars > 0:
                                # Background variance per pixel (electrons^2)
                                # bkg_std already includes sky Poisson + read noise
                                sigma_pix_e2 = np.maximum((bkg_std_use * gain) ** 2 - rn_e ** 2, 0.0)

                                for ri, r in enumerate(radii_px):
                                    ap = CircularAperture(xy_use, r=r)
                                    raw_sum = np.asarray(
                                        aperture_photometry(img, ap)["aperture_sum"], dtype=float
                                    )
                                    flux_adu = raw_sum - bkg_use * ap.area
                                    flux_e = flux_adu * gain

                                    # CCD equation per star
                                    var_source = np.maximum(flux_e, 0.0)
                                    var_bkg = ap.area * sigma_pix_e2
                                    var_bkg_est = (ap.area ** 2 / np.maximum(n_sky_use, 1.0)) * sigma_pix_e2
                                    var_rn = ap.area * rn_e ** 2
                                    var_total = var_source + var_bkg + var_bkg_est + var_rn
                                    sigma_e = np.sqrt(np.maximum(var_total, 0.0))

                                    snr = np.where(sigma_e > 0, flux_e / sigma_e, np.nan)
                                    m_err = np.where(snr > 0, 1.0857 / snr, np.nan)

                                    flux_e_all[ri] = flux_e
                                    mag_err_all[ri] = m_err
                                    snr_all[ri] = snr

                            # Compute median quantities across stars at each radius
                            med_mag_err = np.full(n_radii, np.nan)
                            med_snr = np.full(n_radii, np.nan)
                            med_flux_e = np.full(n_radii, np.nan)
                            med_mag = np.full(n_radii, np.nan)
                            n_valid = np.zeros(n_radii, dtype=int)

                            for ri in range(n_radii):
                                fe = flux_e_all[ri]
                                me = mag_err_all[ri]
                                sn = snr_all[ri]
                                valid = np.isfinite(fe) & (fe > 0) & np.isfinite(me)
                                nv = int(np.sum(valid))
                                n_valid[ri] = nv
                                if nv >= 3:
                                    med_flux_e[ri] = float(np.nanmedian(fe[valid]))
                                    med_mag_err[ri] = float(np.nanmedian(me[valid]))
                                    med_snr[ri] = float(np.nanmedian(sn[valid]))
                                    with np.errstate(divide="ignore", invalid="ignore"):
                                        med_mag[ri] = float(-2.5 * np.log10(max(med_flux_e[ri], 1e-30)))

                            # Find optimal aperture = radius with minimum median mag_err
                            finite_err = np.isfinite(med_mag_err)
                            if np.any(finite_err):
                                opt_idx = int(np.nanargmin(med_mag_err))
                                r_optimal = float(radii_px[opt_idx])
                                opt_scale = float(scales_used[opt_idx])
                                opt_mag_err = float(med_mag_err[opt_idx])
                                opt_snr = float(med_snr[opt_idx])
                                n_used = int(n_valid[opt_idx])
                            else:
                                r_optimal = r_ap_default
                                opt_scale = ap_scale
                                opt_mag_err = np.nan
                                opt_snr = np.nan
                                n_used = 0

                            # Compute apcorr = flux(large_ref) / flux(optimal)
                            apcorr_val = np.nan
                            rel_sc = np.nan
                            if n_stars > 0 and np.any(finite_err):
                                # Photometry at large reference aperture
                                ap_ref = CircularAperture(xy_use, r=r_large_ref)
                                raw_ref = np.asarray(
                                    aperture_photometry(img, ap_ref)["aperture_sum"], dtype=float
                                )
                                flux_ref = (raw_ref - bkg_use * ap_ref.area) * gain
                                flux_opt = flux_e_all[opt_idx]

                                valid_ratio = (
                                    np.isfinite(flux_ref) & np.isfinite(flux_opt)
                                    & (flux_ref > 0) & (flux_opt > 0)
                                )
                                if np.sum(valid_ratio) >= 5:
                                    ratios = flux_ref[valid_ratio] / flux_opt[valid_ratio]
                                    ratios = ratios[np.isfinite(ratios) & (ratios > 0.5) & (ratios < 5.0)]
                                    if len(ratios) >= 5:
                                        # 3-sigma MAD clip
                                        r_med = float(np.nanmedian(ratios))
                                        r_mad = float(np.nanmedian(np.abs(ratios - r_med)))
                                        r_sig = 1.4826 * r_mad
                                        if np.isfinite(r_sig) and r_sig > 0:
                                            keep = np.abs(ratios - r_med) <= 3.0 * r_sig
                                            if np.sum(keep) >= 5:
                                                ratios = ratios[keep]
                                        apcorr_val = float(np.nanmedian(ratios))
                                        mad_final = float(np.nanmedian(np.abs(ratios - apcorr_val)))
                                        rel_sc = 1.4826 * mad_final / apcorr_val if apcorr_val > 0 else np.nan

                            apply_flag = bool(
                                n_used >= apcorr_use_min_n
                                and np.isfinite(apcorr_val)
                                and (apcorr_val >= 0.8) and (apcorr_val <= 5.0)
                                and np.isfinite(rel_sc)
                                and (rel_sc <= apcorr_scatter_max)
                            )

                            apc_row = dict(
                                file=fname,
                                fwhm_med=fwhm_med,
                                fwhm_used=fwhm_used,
                                optimal_scale=opt_scale,
                                r_optimal=r_optimal,
                                r_large_ref=r_large_ref,
                                n_used=n_used,
                                apcorr=float(apcorr_val) if np.isfinite(apcorr_val) else np.nan,
                                rel_scatter=float(rel_sc) if np.isfinite(rel_sc) else np.nan,
                                mag_err_optimal=opt_mag_err,
                                snr_optimal=opt_snr,
                                apply=bool(apply_flag),
                            )

                            # Growth curve rows for this frame
                            for ri in range(n_radii):
                                rows_growth_curve.append(dict(
                                    file=fname,
                                    scale=float(scales_used[ri]),
                                    r_px=float(radii_px[ri]),
                                    median_flux_e=float(med_flux_e[ri]) if np.isfinite(med_flux_e[ri]) else np.nan,
                                    median_mag=float(med_mag[ri]) if np.isfinite(med_mag[ri]) else np.nan,
                                    median_mag_err=float(med_mag_err[ri]) if np.isfinite(med_mag_err[ri]) else np.nan,
                                    median_snr=float(med_snr[ri]) if np.isfinite(med_snr[ri]) else np.nan,
                                    n_stars=int(n_valid[ri]),
                                    selected=bool(np.isfinite(med_mag_err[ri]) and ri == opt_idx) if np.any(finite_err) else False,
                                ))

                        except Exception:
                            apc_row = dict(
                                file=fname,
                                fwhm_med=fwhm_med,
                                fwhm_used=fwhm_used,
                                optimal_scale=ap_scale,
                                r_optimal=r_ap_default,
                                r_large_ref=r_large_ref,
                                n_used=0,
                                apcorr=np.nan,
                                mag_err_optimal=np.nan,
                                snr_optimal=np.nan,
                                apply=False,
                            )

                # Build aperture_by_frame row using optimal radius from growth curve
                r_ap = apc_row.get("r_optimal", r_ap_default)
                r_in_ap = max(ann_in_scale * fwhm_used, max(min_r_in_px, r_ap + ann_gap))
                r_out_ap = max(r_in_ap + ann_out_scale * fwhm_used, r_in_ap + ann_minw, min_r_out_px)

                row = dict(
                    file=fname,
                    fwhm_med=fwhm_med,
                    fwhm_used=fwhm_used,
                    r_ap=r_ap,
                    r_in=r_in_ap,
                    r_out=r_out_ap,
                    cbox_px=cbox_px,
                )
                if np.isfinite(ps) and ps > 0:
                    row.update(dict(
                        fwhm_med_arcsec=fwhm_med * ps,
                        fwhm_used_arcsec=fwhm_used * ps,
                        r_ap_arcsec=r_ap * ps,
                        r_in_arcsec=r_in_ap * ps,
                        r_out_arcsec=r_out_ap * ps,
                        cbox_arcsec=cbox_px * ps,
                    ))

                rows_ap.append(row)
                try:
                    (self.cache_dir / f"ap_{fname}.json").write_text(
                        json.dumps(row, indent=2), encoding="utf-8"
                    )
                except Exception:
                    pass

                rows_apcorr.append(apc_row)
                try:
                    (self.cache_dir / f"apcorr_{fname}.json").write_text(
                        json.dumps(apc_row, indent=2), encoding="utf-8"
                    )
                except Exception:
                    pass

                self.file_done.emit(fname, row)
                self.progress.emit(i, total, fname)

            ap_cols = [
                "file", "fwhm_med", "fwhm_used", "r_ap", "r_in", "r_out", "cbox_px",
                "fwhm_med_arcsec", "fwhm_used_arcsec", "r_ap_arcsec", "r_in_arcsec", "r_out_arcsec", "cbox_arcsec",
            ]
            apcorr_cols = [
                "file", "fwhm_med", "fwhm_used",
                "optimal_scale", "r_optimal", "r_large_ref",
                "n_used", "apcorr", "rel_scatter", "mag_err_optimal", "snr_optimal", "apply",
            ]
            gc_cols = [
                "file", "scale", "r_px",
                "median_flux_e", "median_mag", "median_mag_err", "median_snr",
                "n_stars", "selected",
            ]

            pd.DataFrame(rows_ap, columns=ap_cols).to_csv(output_dir / "aperture_by_frame.csv", index=False)
            pd.DataFrame(rows_apcorr, columns=apcorr_cols).to_csv(output_dir / "apcorr_summary.csv", index=False)
            pd.DataFrame(rows_growth_curve, columns=gc_cols).to_csv(output_dir / "growth_curve.csv", index=False)

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
            step_index=9,
            step_name="Aperture Photometry",
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
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
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

    def open_parameters_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Aperture Parameters")
        dialog.resize(520, 700)

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

        # --- Growth Curve Apcorr ---
        self.param_apcorr = QCheckBox("Enable")
        self.param_apcorr.setChecked(bool(getattr(self.params.P, "apcorr_apply", True)))
        form.addRow("Aperture Correction:", self.param_apcorr)

        self.param_apcorr_min_n = QSpinBox()
        self.param_apcorr_min_n.setRange(1, 500)
        self.param_apcorr_min_n.setValue(int(getattr(self.params.P, "apcorr_use_min_n", 20)))
        form.addRow("Apcorr Min N:", self.param_apcorr_min_n)

        self.param_apcorr_scatter = QDoubleSpinBox()
        self.param_apcorr_scatter.setRange(0.0, 0.5)
        self.param_apcorr_scatter.setSingleStep(0.01)
        self.param_apcorr_scatter.setValue(float(getattr(self.params.P, "apcorr_scatter_max", 0.05)))
        form.addRow("Apcorr Scatter Max:", self.param_apcorr_scatter)

        self.param_apcorr_sources = QSpinBox()
        self.param_apcorr_sources.setRange(30, 1000)
        self.param_apcorr_sources.setValue(int(getattr(self.params.P, "apcorr_max_sources", 250)))
        form.addRow("Apcorr Max Sources:", self.param_apcorr_sources)

        self.param_gc_scale_min = QDoubleSpinBox()
        self.param_gc_scale_min.setRange(0.3, 3.0)
        self.param_gc_scale_min.setSingleStep(0.1)
        self.param_gc_scale_min.setValue(float(getattr(self.params.P, "apcorr_scale_min", 0.5)))
        form.addRow("Growth Curve Scale Min (×FWHM):", self.param_gc_scale_min)

        self.param_gc_scale_max = QDoubleSpinBox()
        self.param_gc_scale_max.setRange(1.0, 10.0)
        self.param_gc_scale_max.setSingleStep(0.25)
        self.param_gc_scale_max.setValue(float(getattr(self.params.P, "apcorr_scale_max", 5.0)))
        form.addRow("Growth Curve Scale Max (×FWHM):", self.param_gc_scale_max)

        self.param_apcorr_step = QDoubleSpinBox()
        self.param_apcorr_step.setRange(0.05, 1.0)
        self.param_apcorr_step.setSingleStep(0.05)
        self.param_apcorr_step.setValue(float(getattr(self.params.P, "apcorr_scale_step", 0.25)))
        form.addRow("Growth Curve Scale Step (×FWHM):", self.param_apcorr_step)

        self.param_gc_large_ref = QDoubleSpinBox()
        self.param_gc_large_ref.setRange(2.0, 12.0)
        self.param_gc_large_ref.setSingleStep(0.5)
        self.param_gc_large_ref.setValue(float(getattr(self.params.P, "apcorr_large_ref_scale", 5.0)))
        form.addRow("Large Ref Scale (×FWHM):", self.param_gc_large_ref)

        self.param_gc_isolation = QDoubleSpinBox()
        self.param_gc_isolation.setRange(1.0, 6.0)
        self.param_gc_isolation.setSingleStep(0.1)
        self.param_gc_isolation.setValue(float(getattr(self.params.P, "apcorr_isolation_factor", 2.0)))
        form.addRow("Isolation Factor:", self.param_gc_isolation)

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
        self.params.P.apcorr_use_min_n = self.param_apcorr_min_n.value()
        self.params.P.apcorr_scatter_max = self.param_apcorr_scatter.value()
        self.params.P.apcorr_max_sources = self.param_apcorr_sources.value()
        self.params.P.apcorr_scale_min = self.param_gc_scale_min.value()
        self.params.P.apcorr_scale_max = self.param_gc_scale_max.value()
        self.params.P.apcorr_scale_step = self.param_apcorr_step.value()
        self.params.P.apcorr_large_ref_scale = self.param_gc_large_ref.value()
        self.params.P.apcorr_isolation_factor = self.param_gc_isolation.value()
        self.params.P.annulus_sigma_clip = self.param_sigma_clip.value()
        self.params.P.fitsky_max_iter = self.param_max_iter.value()
        self.params.P.phot_use_qc_pass_only = self.param_qc_only.isChecked()
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
            "apcorr_use_min_n": getattr(self.params.P, "apcorr_use_min_n", 20),
            "apcorr_scatter_max": getattr(self.params.P, "apcorr_scatter_max", 0.05),
            "apcorr_max_sources": getattr(self.params.P, "apcorr_max_sources", 250),
            "apcorr_scale_min": getattr(self.params.P, "apcorr_scale_min", 0.5),
            "apcorr_scale_max": getattr(self.params.P, "apcorr_scale_max", 5.0),
            "apcorr_scale_step": getattr(self.params.P, "apcorr_scale_step", 0.25),
            "apcorr_large_ref_scale": getattr(self.params.P, "apcorr_large_ref_scale", 5.0),
            "apcorr_isolation_factor": getattr(self.params.P, "apcorr_isolation_factor", 2.0),
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
