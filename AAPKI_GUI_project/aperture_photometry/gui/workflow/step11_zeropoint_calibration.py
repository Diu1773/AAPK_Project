"""
Step 11: Zeropoint & Standardization
Ported from AAPKI_GUI.ipynb Cell 14/15 (calibration prep).
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
import astropy.units as u

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib as mpl

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QMessageBox,
    QTextEdit, QDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox,
    QSpinBox, QCheckBox, QComboBox, QWidget
)

from .step_window_base import StepWindowBase
from ...utils.astro_utils import compute_airmass_from_header
from ...utils.step_paths import (
    step2_cropped_dir,
    crop_is_active,
    step5_dir,
    step7_dir,
    step9_dir,
    step11_dir,
    step11_extinction_dir,
)


class ZeropointCalibrationWorker(QThread):
    progress = pyqtSignal(int, int, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, params, data_dir: Path, result_dir: Path, cache_dir: Path):
        super().__init__()
        self.params = params
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.cache_dir = Path(cache_dir)
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def _log(self, msg: str):
        self.log.emit(msg)

    def _pick_col(self, cols, cands):
        for c in cands:
            if c in cols:
                return c
        return None

    def _resolve_path(self, p):
        p = str(p) if p is not None else ""
        if p.strip() == "":
            return None
        p0 = Path(p)
        if p0.is_absolute() and p0.exists():
            return p0
        for base in (
            step9_dir(self.result_dir),
            self.result_dir,
            self.result_dir / "phot",
            self.result_dir / "photometry",
            self.result_dir / "result",
        ):
            p1 = base / p
            if p1.exists():
                return p1
        return None

    @staticmethod
    def _robust_median_and_err(arr):
        x = np.asarray(arr, float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return (np.nan, np.nan, 0)
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        err = float(1.4826 * mad / np.sqrt(max(len(x), 1)))
        return (med, err, int(len(x)))

    @staticmethod
    def _weighted_mean_mag(mag, mag_err, clip_sigma=3.0, iters=4):
        mags = np.asarray(mag, float)
        errs = np.asarray(mag_err, float)
        mask = np.isfinite(mags) & np.isfinite(errs) & (errs > 0)
        if mask.sum() == 0:
            return (np.nan, np.nan, 0)
        x = mags[mask]
        e = errs[mask]
        for _ in range(int(iters)):
            med = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - med))
            sig = 1.4826 * mad if mad > 0 else np.nanstd(x)
            if not np.isfinite(sig) or sig <= 0:
                break
            keep = np.abs(x - med) <= float(clip_sigma) * sig
            if keep.sum() == len(x):
                break
            x = x[keep]
            e = e[keep]
            if len(x) == 0:
                break
        n = int(len(x))
        if n == 0:
            return (np.nan, np.nan, 0)

        flux = 10.0 ** (-0.4 * x)
        sigma_flux = flux * (np.log(10.0) / 2.5) * e
        w = np.where(np.isfinite(sigma_flux) & (sigma_flux > 0), 1.0 / (sigma_flux ** 2), 0.0)
        wsum = float(np.nansum(w))
        if not np.isfinite(wsum) or wsum <= 0:
            return (np.nan, np.nan, n)
        flux_w = float(np.nansum(w * flux) / wsum)
        if not np.isfinite(flux_w) or flux_w <= 0:
            return (np.nan, np.nan, n)
        sigma_flux_w = float(np.sqrt(1.0 / wsum))
        mag_w = float(-2.5 * np.log10(flux_w))
        mag_w_err = float(1.0857 * (sigma_flux_w / flux_w))
        return (mag_w, mag_w_err, n)

    @staticmethod
    def _robust_clip(x, clip_sigma=3.0, iters=5):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return x
        for _ in range(int(iters)):
            med = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - med))
            sig = 1.4826 * mad if mad > 0 else np.nanstd(x)
            if not np.isfinite(sig) or sig <= 0:
                break
            keep = np.abs(x - med) <= float(clip_sigma) * sig
            if keep.sum() == len(x):
                break
            x = x[keep]
            if len(x) == 0:
                break
        return x

    def _robust_location(self, arr, clip_sigma=3.0, iters=5):
        x = self._robust_clip(arr, clip_sigma=clip_sigma, iters=iters)
        if len(x) == 0:
            return (np.nan, np.nan, 0, np.nan)
        med = float(np.nanmedian(x))
        std = float(np.nanstd(x))
        n = int(len(x))
        outlier_frac = np.nan
        try:
            outlier_frac = float(1.0 - (len(x) / max(len(arr), 1)))
        except Exception:
            outlier_frac = np.nan
        return (med, std, n, outlier_frac)

    def _has_wcs(self, header):
        try:
            w0 = WCS(header)
            return bool(w0.has_celestial)
        except Exception:
            return False

    def _list_frames(self):
        crop_active = crop_is_active(self.result_dir)
        cropped_dir = step2_cropped_dir(self.result_dir)
        if crop_active and cropped_dir.exists() and list(cropped_dir.glob("*.fit*")):
            return sorted(cropped_dir.glob("*.fit*"))
        if crop_active:
            legacy = self.result_dir / "cropped"
            if legacy.exists() and list(legacy.glob("*.fit*")):
                return sorted(legacy.glob("*.fit*"))
        return sorted(self.data_dir.glob("*.fit*"))

    def _load_ref_wcs(self):
        P = self.params.P
        files = self._list_frames()

        ref_val = getattr(P, "ref_frame", None)
        if ref_val is not None and str(ref_val).strip() != "":
            ref_txt = str(ref_val).strip()
            if ref_txt.isdigit():
                idx = int(ref_txt)
                if files and 0 <= idx < len(files):
                    fp = files[idx]
                    hdr = fits.getheader(fp)
                    if self._has_wcs(hdr):
                        self._log(f"WCS from ref_frame index: {fp.name}")
                        return WCS(hdr), fp
            else:
                fp = Path(ref_txt)
                if not fp.is_absolute():
                    cropped_dir = step2_cropped_dir(self.result_dir)
                    search_dirs = [self.result_dir, self.data_dir]
                    if crop_is_active(self.result_dir):
                        if not cropped_dir.exists():
                            cropped_dir = self.result_dir / "cropped"
                        search_dirs.append(cropped_dir)
                    for base in search_dirs:
                        cand = base / fp
                        if cand.exists():
                            fp = cand
                            break
                if fp.exists():
                    hdr = fits.getheader(fp)
                    if self._has_wcs(hdr):
                        self._log(f"WCS from ref_frame: {fp.name}")
                        return WCS(hdr), fp

        # Check cropped directory FIRST (coordinates are from cropped images)
        patterns = ["ref*.fit*", "rc_*.fit*", "Crop_*.fit*", "crop_*.fit*", "*.fit*"]
        cropped_dir = step2_cropped_dir(self.result_dir)
        search_dirs = [self.result_dir, self.data_dir]
        if crop_is_active(self.result_dir):
            if not cropped_dir.exists():
                cropped_dir = self.result_dir / "cropped"
            search_dirs.insert(0, cropped_dir)
        for base in search_dirs:
            if not base.exists():
                continue
            for pat in patterns:
                for fp in sorted(base.glob(pat)):
                    try:
                        hdr = fits.getheader(fp)
                        if self._has_wcs(hdr):
                            self._log(f"WCS auto-detected: {fp.name} (from {base.name})")
                            return WCS(hdr), fp
                    except Exception:
                        continue
        return None, None

    def _build_frame_airmass(self, idx: pd.DataFrame) -> pd.DataFrame:
        P = self.params.P
        lat = float(getattr(P, "site_lat_deg", 0.0))
        lon = float(getattr(P, "site_lon_deg", 0.0))
        alt = float(getattr(P, "site_alt_m", 0.0))
        tz = float(getattr(P, "site_tz_offset_hours", 0.0))

        rows = []
        for _, r in idx.iterrows():
            fname = str(r.get("file", "")).strip()
            if fname == "":
                continue
            fpath = self.data_dir / fname
            if not fpath.exists():
                if crop_is_active(self.result_dir):
                    cand = step2_cropped_dir(self.result_dir) / fname
                    if not cand.exists():
                        cand = (self.result_dir / "cropped") / fname
                    if cand.exists():
                        fpath = cand
            if not fpath.exists():
                continue
            try:
                hdr = fits.getheader(fpath)
                info = compute_airmass_from_header(hdr, lat, lon, alt, tz)
                filt = str(r.get("filter", hdr.get("FILTER", ""))).strip().lower()
                rows.append({
                    "file": fname,
                    "filter": filt,
                    **info,
                })
            except Exception:
                continue
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["file", "filter", "airmass", "airmass_source", "alt_deg", "zenith_deg", "datetime_utc", "datetime_local", "ra_deg", "dec_deg"])
        if len(df):
            output_dir = step11_dir(self.result_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "frame_airmass.csv"
            df.to_csv(out_path, index=False)
            self._log(f"Saved {out_path.name} | rows={len(df)}")
        return df

    def _poly_eval(self, x, coeffs):
        x = np.asarray(x, float)
        y = np.zeros_like(x, dtype=float)
        p = np.ones_like(x, dtype=float)
        for a in coeffs:
            y += a * p
            p *= x
        return y

    def _robust_linfit(self, x, y, clip_sigma=3.0, iters=5, slope_absmax=1.0, min_n=10):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m0 = np.isfinite(x) & np.isfinite(y)
        x = x[m0]
        y = y[m0]
        if len(x) < min_n:
            return (float(np.nanmedian(y)) if len(y) else np.nan, 0.0, int(len(x)))

        ct, zp = np.polyfit(x, y, 1)

        for _ in range(int(iters)):
            yhat = zp + ct * x
            r = y - yhat
            med = np.nanmedian(r)
            mad = np.nanmedian(np.abs(r - med)) + 1e-12
            sig = 1.4826 * mad
            keep = np.abs(r - med) <= float(clip_sigma) * sig
            if keep.sum() < min_n:
                break
            ct, zp = np.polyfit(x[keep], y[keep], 1)
            x, y = x[keep], y[keep]

        if abs(ct) > float(slope_absmax):
            return (float(np.nanmedian(y)), 0.0, int(len(x)))

        return float(zp), float(ct), int(len(x))

    def run(self):
        try:
            P = self.params.P
            result_dir = self.result_dir
            output_dir = step11_dir(result_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            phot_dir = step9_dir(result_dir)

            idx_candidates = [
                phot_dir / "photometry_index.csv",
                phot_dir / "phot_index.csv",
                result_dir / "photometry_index.csv",
                result_dir / "phot_index.csv",
                result_dir / "phot" / "phot_index.csv",
                result_dir / "phot" / "photometry_index.csv",
            ]
            idx_candidates += sorted(phot_dir.glob("*phot*index*.csv"))
            idx_candidates += sorted(result_dir.glob("*phot*index*.csv"))
            if (result_dir / "phot").exists():
                idx_candidates += sorted((result_dir / "phot").glob("*phot*index*.csv"))

            idx_path = next((p for p in idx_candidates if p.exists()), None)
            if idx_path is None:
                raise FileNotFoundError("photometry index csv not found")

            idx = pd.read_csv(idx_path)
            self._log(f"Index = {idx_path.name} | rows={len(idx)}")

            if "path" not in idx.columns:
                for cand in ("phot_tsv", "tsv", "out", "output"):
                    if cand in idx.columns:
                        idx = idx.rename(columns={cand: "path"})
                        break

            if "file" not in idx.columns:
                c_file = self._pick_col(idx.columns, ["fname", "frame", "image", "fits", "name"])
                if c_file:
                    idx = idx.rename(columns={c_file: "file"})

            if "filter" in idx.columns:
                idx["filter"] = idx["filter"].astype(str).str.strip().str.lower()
            elif "FILTER" in idx.columns:
                idx["filter"] = idx["FILTER"].astype(str).str.strip().str.lower()
            else:
                idx["filter"] = "unknown"

            fq_path = step5_dir(result_dir) / "frame_quality.csv"
            if not fq_path.exists():
                fq_path = result_dir / "frame_quality.csv"
            if fq_path.exists() and ("file" in idx.columns):
                fq = pd.read_csv(fq_path)
                if ("file" in fq.columns) and ("passed" in fq.columns):
                    idx = idx.merge(fq[["file", "passed"]], on="file", how="left")
                    before = len(idx)
                    idx = idx[idx["passed"] != False].copy()
                    self._log(f"QC passed only: {before} -> {len(idx)}")

            min_snr_for_mag = float(getattr(P, "min_snr_for_mag", 0.0))

            rows = []
            n_missing = 0
            total = len(idx)
            for i, (_, r) in enumerate(idx.iterrows(), start=1):
                if self._stop_requested:
                    self.finished.emit({"stopped": True})
                    return
                p = self._resolve_path(r.get("path", ""))
                if p is None or (not p.exists()):
                    n_missing += 1
                    continue

                try:
                    dfp = pd.read_csv(p, sep="\t")
                except Exception:
                    dfp = pd.read_csv(p)

                if "is_saturated" in dfp.columns:
                    dfp = dfp[~dfp["is_saturated"].fillna(False).astype(bool)]
                if "is_nonlinear" in dfp.columns:
                    dfp = dfp[~dfp["is_nonlinear"].fillna(False).astype(bool)]
                if "centroid_outlier" in dfp.columns:
                    dfp = dfp[~dfp["centroid_outlier"].fillna(False).astype(bool)]
                if "recenter_capped" in dfp.columns:
                    dfp = dfp[~dfp["recenter_capped"].fillna(False).astype(bool)]

                if "ID" not in dfp.columns:
                    raise RuntimeError(f"{p.name}: ID column missing")

                if "FILTER" in dfp.columns:
                    dfp["FILTER"] = dfp["FILTER"].astype(str).str.strip().str.lower()
                else:
                    dfp["FILTER"] = str(r.get("filter", "unknown")).strip().lower()

                mag_col = None
                for cand in ("mag_inst", "mag", "mag_ap", "mag_apcorr"):
                    if cand in dfp.columns:
                        mag_col = cand
                        break
                if mag_col is None:
                    raise RuntimeError(f"{p.name}: mag column missing")

                err_col = None
                for cand in ("mag_err", "emag", "emag_inst", "magerr"):
                    if cand in dfp.columns:
                        err_col = cand
                        break
                if err_col is None:
                    dfp["mag_err"] = np.nan
                    err_col = "mag_err"

                snr_col = "snr" if "snr" in dfp.columns else None

                tmp = dfp[["ID", "FILTER", mag_col, err_col] + ([snr_col] if snr_col else [])].copy()
                tmp = tmp.rename(columns={mag_col: "mag_inst", err_col: "mag_err"})
                if snr_col is None:
                    tmp["snr"] = np.nan
                else:
                    tmp = tmp.rename(columns={snr_col: "snr"})
                if np.isnan(tmp["mag_err"].to_numpy(float)).all():
                    snr_vals = tmp["snr"].to_numpy(float)
                    if np.isfinite(snr_vals).any():
                        tmp.loc[np.isfinite(snr_vals) & (snr_vals > 0), "mag_err"] = 1.0857 / snr_vals

                tmp["file"] = str(r.get("file", "")) if ("file" in idx.columns) else ""

                if min_snr_for_mag > 0:
                    m = np.isfinite(tmp["snr"].to_numpy(float))
                    tmp = tmp[(~m) | (tmp["snr"].to_numpy(float) >= min_snr_for_mag)].copy()

                rows.append(tmp)
                self.progress.emit(i, total, str(r.get("file", "")))

            self._log(f"Read frames: {len(rows)} | missing paths: {n_missing}")

            all_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["ID", "FILTER", "mag_inst", "mag_err", "snr", "file"])
            all_df["FILTER"] = all_df["FILTER"].astype(str).str.strip().str.lower()

            def _combine_group_raw(g):
                med, med_err, n_med = self._robust_median_and_err(g["mag_inst"])
                wmean, werr, _ = self._weighted_mean_mag(g["mag_inst"], g["mag_err"])
                snr_vals = np.asarray(g.get("snr", np.nan), float)
                snr_med = float(np.nanmedian(snr_vals)) if np.isfinite(snr_vals).any() else np.nan
                return pd.Series({
                    "mag_inst_med": med,
                    "mag_inst_med_err": med_err,
                    "mag_inst_wmean": wmean,
                    "mag_inst_werr": werr,
                    "n_frames": n_med,
                    "snr_med": snr_med,
                })

            grp_raw = all_df.groupby(["ID", "FILTER"]).apply(_combine_group_raw).reset_index()

            grp_raw_path = output_dir / "median_by_ID_filter_raw.csv"
            grp_raw.to_csv(grp_raw_path, index=False, na_rep="NaN")
            self._log(f"Saved {grp_raw_path.name} | rows={len(grp_raw)}")

            wide_raw_mag_w = grp_raw.pivot_table(index="ID", columns="FILTER", values="mag_inst_wmean", aggfunc="median")
            wide_raw_err_w = grp_raw.pivot_table(index="ID", columns="FILTER", values="mag_inst_werr", aggfunc="median")
            wide_raw_mag_med = grp_raw.pivot_table(index="ID", columns="FILTER", values="mag_inst_med", aggfunc="median")
            wide_raw_err_med = grp_raw.pivot_table(index="ID", columns="FILTER", values="mag_inst_med_err", aggfunc="median")
            wide_raw_mag = wide_raw_mag_w.combine_first(wide_raw_mag_med)
            wide_raw_err = wide_raw_err_w.combine_first(wide_raw_err_med)
            wide_raw_snr = grp_raw.pivot_table(index="ID", columns="FILTER", values="snr_med", aggfunc="median")

            wide_raw_mag.columns = [f"mag_inst_{c}" for c in wide_raw_mag.columns]
            wide_raw_err.columns = [f"mag_inst_err_{c}" for c in wide_raw_err.columns]
            wide_raw_snr.columns = [f"snr_{c}" for c in wide_raw_snr.columns]
            wide_raw_mag_w.columns = [f"mag_inst_wmean_{c}" for c in wide_raw_mag_w.columns]
            wide_raw_err_w.columns = [f"mag_inst_werr_{c}" for c in wide_raw_err_w.columns]
            wide_raw_mag_med.columns = [f"mag_inst_med_{c}" for c in wide_raw_mag_med.columns]
            wide_raw_err_med.columns = [f"mag_inst_med_err_{c}" for c in wide_raw_err_med.columns]

            wide_raw = pd.concat(
                [
                    wide_raw_mag,
                    wide_raw_err,
                    wide_raw_mag_w,
                    wide_raw_err_w,
                    wide_raw_mag_med,
                    wide_raw_err_med,
                    wide_raw_snr,
                ],
                axis=1,
            ).reset_index()

            wide_raw_path = output_dir / "median_by_ID_filter_wide_raw.csv"
            wide_raw.to_csv(wide_raw_path, index=False, na_rep="NaN")
            self._log(f"Saved {wide_raw_path.name} | rows={len(wide_raw)}")

            master_path = step7_dir(result_dir) / "master_catalog.tsv"
            if not master_path.exists():
                master_path = result_dir / "master_catalog.tsv"
            if not master_path.exists():
                raise FileNotFoundError("master_catalog.tsv missing")
            master = pd.read_csv(master_path, sep="\t")
            if "ID" not in master.columns:
                raise RuntimeError("master_catalog.tsv missing ID column")

            # Merge wide with master to get Gaia mags from master_catalog
            merge_cols = ["ID"]
            if "source_id" in master.columns:
                merge_cols.append("source_id")
            col_xm = self._pick_col(master.columns, ["x_ref", "x", "x_pix", "x_center", "x_med"])
            col_ym = self._pick_col(master.columns, ["y_ref", "y", "y_pix", "y_center", "y_med"])
            if col_xm:
                merge_cols.append(col_xm)
            if col_ym:
                merge_cols.append(col_ym)
            for col in ("gaia_G", "gaia_BP", "gaia_RP", "gmag", "bpmag", "rpmag", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"):
                if col in master.columns and col not in merge_cols:
                    merge_cols.append(col)

            df = wide_raw.merge(master[merge_cols], on="ID", how="left")
            if col_xm:
                df = df.rename(columns={col_xm: "x_pix"})
            if col_ym:
                df = df.rename(columns={col_ym: "y_pix"})
            g_col = self._pick_col(df.columns, ["gaia_G", "gmag", "phot_g_mean_mag"])
            bp_col = self._pick_col(df.columns, ["gaia_BP", "bpmag", "phot_bp_mean_mag"])
            rp_col = self._pick_col(df.columns, ["gaia_RP", "rpmag", "phot_rp_mean_mag"])

            gaia_from_master = True
            if g_col is None or bp_col is None or rp_col is None:
                gaia_from_master = False
                if "source_id" not in df.columns:
                    raise RuntimeError("master_catalog missing Gaia mags and source_id for Gaia join")
                gaia_path = step5_dir(result_dir) / "gaia_fov.ecsv"
                if not gaia_path.exists():
                    gaia_path = result_dir / "gaia_fov.ecsv"
                if not gaia_path.exists():
                    raise RuntimeError("master_catalog missing Gaia mags and gaia_fov.ecsv not found")
                t_gaia = Table.read(gaia_path, format="ascii.ecsv")
                gaia_df = t_gaia.to_pandas()
                if "source_id" in gaia_df.columns:
                    gaia_df["source_id"] = gaia_df["source_id"].astype("int64")
                gaia_cols = ["source_id", "phot_g_mean_mag"]
                if "phot_bp_mean_mag" in gaia_df.columns:
                    gaia_cols.append("phot_bp_mean_mag")
                if "phot_rp_mean_mag" in gaia_df.columns:
                    gaia_cols.append("phot_rp_mean_mag")
                df = df.merge(gaia_df[gaia_cols], on="source_id", how="left")
                g_col = self._pick_col(df.columns, ["phot_g_mean_mag"])
                bp_col = self._pick_col(df.columns, ["phot_bp_mean_mag"])
                rp_col = self._pick_col(df.columns, ["phot_rp_mean_mag"])

            if g_col is None or bp_col is None or rp_col is None:
                raise RuntimeError("Gaia magnitude columns not available (gaia_G/gaia_BP/gaia_RP or Gaia ECSV)")

            df["gaia_G"] = pd.to_numeric(df[g_col], errors="coerce")
            df["gaia_BP"] = pd.to_numeric(df[bp_col], errors="coerce")
            df["gaia_RP"] = pd.to_numeric(df[rp_col], errors="coerce")

            dfm = df[np.isfinite(df["gaia_G"]) & np.isfinite(df["gaia_BP"]) & np.isfinite(df["gaia_RP"])].copy()
            dfm["gaia_BP_RP"] = dfm["gaia_BP"] - dfm["gaia_RP"]
            src_note = "master_catalog" if gaia_from_master else "gaia_fov.ecsv"
            self._log(f"Gaia mags from {src_note}: {len(dfm)} / {len(df)}")

            min_match = int(getattr(P, "min_master_gaia_matches", 10))
            if len(dfm) < min_match:
                dfm.to_csv(output_dir / "gaia_sdss_calibrator_by_ID.csv", index=False)
                raise RuntimeError("Not enough Gaia matches for calibration")

            out_cal = dfm.copy()

            xcol = out_cal["gaia_BP_RP"].to_numpy(float)
            G = out_cal["gaia_G"].to_numpy(float)

            m_g = np.isfinite(xcol) & (xcol >= 0.3) & (xcol <= 3.0)
            m_r = np.isfinite(xcol) & (xcol >= 0.0) & (xcol <= 3.0)
            m_i = np.isfinite(xcol) & (xcol >= 0.5) & (xcol <= 2.0)

            G_minus_g = np.full_like(G, np.nan)
            G_minus_r = np.full_like(G, np.nan)
            G_minus_i = np.full_like(G, np.nan)

            # Jordi+ (2010) transformations: Gaia G -> SDSS gri
            self._log("=== Jordi+ (2010) Gaia->SDSS Transformations ===")
            self._log("g_sdss = G - (0.2199 - 0.6365*(BP-RP) - 0.1548*(BP-RP)^2 + 0.0064*(BP-RP)^3)")
            self._log("r_sdss = G - (-0.0984 + 0.0859*(BP-RP) + 0.1907*(BP-RP)^2 - 0.1701*(BP-RP)^3 + 0.0226*(BP-RP)^4)")
            self._log("i_sdss = G - (-0.293 + 0.6404*(BP-RP) - 0.0961*(BP-RP)^2 - 0.0021*(BP-RP)^3)")

            G_minus_g[m_g] = self._poly_eval(xcol[m_g], [0.2199, -0.6365, -0.1548, 0.0064])
            G_minus_r[m_r] = self._poly_eval(xcol[m_r], [-0.09837, 0.08592, 0.1907, -0.1701, 0.02263])
            G_minus_i[m_i] = self._poly_eval(xcol[m_i], [-0.293, 0.6404, -0.09609, -0.002104])

            out_cal["sdss_g_ref"] = G - G_minus_g
            out_cal["sdss_r_ref"] = G - G_minus_r
            out_cal["sdss_i_ref"] = G - G_minus_i
            self._log(f"Jordi applied: g_ref valid={np.isfinite(out_cal['sdss_g_ref']).sum()}, r_ref valid={np.isfinite(out_cal['sdss_r_ref']).sum()}, i_ref valid={np.isfinite(out_cal['sdss_i_ref']).sum()}")

            snr_cut_sdss = float(getattr(P, "cmd_snr_calib_min", 20.0))
            m_snr_sdss = np.ones(len(out_cal), dtype=bool)
            for band in ("g", "r", "i"):
                sc = f"snr_{band}"
                if sc in out_cal.columns:
                    svals = out_cal[sc].to_numpy(float)
                    m_snr_sdss &= np.isfinite(svals) & (svals >= snr_cut_sdss)

            def _arr(col):
                return out_cal[col].to_numpy(float) if col in out_cal.columns else np.full(len(out_cal), np.nan)

            g_inst = _arr("mag_inst_g")
            r_inst = _arr("mag_inst_r")
            i_inst = _arr("mag_inst_i")

            color_gr = g_inst - r_inst
            color_ri = r_inst - i_inst

            clip_sigma = float(getattr(P, "zp_clip_sigma", 3.0))
            fit_iters = int(getattr(P, "zp_fit_iters", 5))
            slope_absmax = float(getattr(P, "zp_slope_absmax", 1.0))

            delta_g = out_cal["sdss_g_ref"].to_numpy(float) - g_inst
            mg = np.isfinite(delta_g) & np.isfinite(color_gr) & np.isfinite(g_inst) & m_snr_sdss
            zp_g, ct_g, Ng = self._robust_linfit(color_gr[mg], delta_g[mg], clip_sigma=clip_sigma, iters=fit_iters, slope_absmax=slope_absmax, min_n=min_match)

            delta_r = out_cal["sdss_r_ref"].to_numpy(float) - r_inst
            mr = np.isfinite(delta_r) & np.isfinite(color_gr) & np.isfinite(r_inst) & m_snr_sdss
            zp_r, ct_r, Nr = self._robust_linfit(color_gr[mr], delta_r[mr], clip_sigma=clip_sigma, iters=fit_iters, slope_absmax=slope_absmax, min_n=min_match)

            delta_i = out_cal["sdss_i_ref"].to_numpy(float) - i_inst
            mi = np.isfinite(delta_i) & np.isfinite(color_ri) & np.isfinite(i_inst) & m_snr_sdss
            zp_i, ct_i, Ni = self._robust_linfit(color_ri[mi], delta_i[mi], clip_sigma=clip_sigma, iters=fit_iters, slope_absmax=slope_absmax, min_n=min_match)

            self._log(f"g_std = g_inst + {zp_g:+.4f} + {ct_g:+.4f}*(g-r)_inst (N={Ng})")
            self._log(f"r_std = r_inst + {zp_r:+.4f} + {ct_r:+.4f}*(g-r)_inst (N={Nr})")
            self._log(f"i_std = i_inst + {zp_i:+.4f} + {ct_i:+.4f}*(r-i)_inst (N={Ni})")

            out_cal_path = output_dir / "gaia_sdss_calibrator_by_ID.csv"
            out_cal.to_csv(out_cal_path, index=False)
            self._log(f"Saved {out_cal_path.name} | rows={len(out_cal)}")
            apply_ext = bool(getattr(P, "cmd_apply_extinction", False))
            ext_mode = str(getattr(P, "cmd_extinction_mode", "absorb")).strip().lower()
            if ext_mode not in ("absorb", "two_step"):
                ext_mode = "absorb"
            min_frame_refs = int(getattr(P, "frame_zp_min_n", 5))

            frame_airmass = self._build_frame_airmass(idx)
            if frame_airmass is None or frame_airmass.empty:
                frame_airmass = pd.DataFrame(columns=["file", "filter", "airmass"])

            ext_df = None
            ext_map = {}
            if apply_ext:
                ext_dir = step11_extinction_dir(result_dir)
                ext_path = ext_dir / "extinction_fit_by_filter.csv"
                if not ext_path.exists():
                    ext_path = result_dir / "extinction_fit_by_filter.csv"
                if ext_path.exists():
                    ext_df = pd.read_csv(ext_path)
                    if {"filter", "k"}.issubset(ext_df.columns):
                        for _, er in ext_df.iterrows():
                            ext_map[str(er["filter"]).strip().lower()] = float(er["k"])

            out_cal["color_gr"] = color_gr
            out_cal["color_ri"] = color_ri

            color_df = wide_raw[["ID"]].copy()
            if "mag_inst_g" in wide_raw.columns and "mag_inst_r" in wide_raw.columns:
                color_df["color_gr"] = wide_raw["mag_inst_g"].to_numpy(float) - wide_raw["mag_inst_r"].to_numpy(float)
            else:
                color_df["color_gr"] = np.nan
            if "mag_inst_r" in wide_raw.columns and "mag_inst_i" in wide_raw.columns:
                color_df["color_ri"] = wide_raw["mag_inst_r"].to_numpy(float) - wide_raw["mag_inst_i"].to_numpy(float)
            else:
                color_df["color_ri"] = np.nan

            cal_cols = ["ID", "sdss_g_ref", "sdss_r_ref", "sdss_i_ref", "color_gr", "color_ri", "gaia_BP_RP"]
            obs = all_df.merge(out_cal[cal_cols], on="ID", how="left")
            obs = obs.merge(frame_airmass[["file", "filter", "airmass"]], left_on=["file", "FILTER"], right_on=["file", "filter"], how="left")

            obs["ref_mag"] = np.nan
            obs.loc[obs["FILTER"] == "g", "ref_mag"] = obs.loc[obs["FILTER"] == "g", "sdss_g_ref"]
            obs.loc[obs["FILTER"] == "r", "ref_mag"] = obs.loc[obs["FILTER"] == "r", "sdss_r_ref"]
            obs.loc[obs["FILTER"] == "i", "ref_mag"] = obs.loc[obs["FILTER"] == "i", "sdss_i_ref"]

            obs["color_term"] = np.nan
            obs.loc[obs["FILTER"] == "g", "color_term"] = ct_g * obs.loc[obs["FILTER"] == "g", "color_gr"]
            obs.loc[obs["FILTER"] == "r", "color_term"] = ct_r * obs.loc[obs["FILTER"] == "r", "color_gr"]
            obs.loc[obs["FILTER"] == "i", "color_term"] = ct_i * obs.loc[obs["FILTER"] == "i", "color_ri"]

            obs["k_term"] = 0.0
            if apply_ext and ext_map:
                for f, k in ext_map.items():
                    m = obs["FILTER"] == f
                    obs.loc[m, "k_term"] = k * obs.loc[m, "airmass"].to_numpy(float)

            obs["delta"] = obs["ref_mag"] - (obs["mag_inst"] + obs["color_term"])
            if apply_ext and ext_mode == "two_step":
                obs["delta"] = obs["ref_mag"] - (obs["mag_inst"] + obs["color_term"] + obs["k_term"])

            snr_cut_sdss = float(getattr(P, "cmd_snr_calib_min", 20.0))
            obs["snr_ok"] = True
            if "snr" in obs.columns:
                svals = obs["snr"].to_numpy(float)
                obs["snr_ok"] = np.isfinite(svals) & (svals >= snr_cut_sdss)

            obs["cal_ok"] = False
            bp = obs["gaia_BP_RP"].to_numpy(float)
            obs.loc[(obs["FILTER"] == "g") & np.isfinite(bp) & (bp >= 0.3) & (bp <= 3.0), "cal_ok"] = True
            obs.loc[(obs["FILTER"] == "r") & np.isfinite(bp) & (bp >= 0.0) & (bp <= 3.0), "cal_ok"] = True
            obs.loc[(obs["FILTER"] == "i") & np.isfinite(bp) & (bp >= 0.5) & (bp <= 2.0), "cal_ok"] = True

            obs_all = obs.copy()
            obs = obs_all[np.isfinite(obs_all["delta"]) & obs_all["snr_ok"] & obs_all["cal_ok"]].copy()

            if len(obs_all):
                summary_rows = []
                for filt, sub in obs_all.groupby("FILTER"):
                    ref_ok = np.isfinite(sub["ref_mag"].to_numpy(float))
                    delta_ok = np.isfinite(sub["delta"].to_numpy(float))
                    snr_ok = sub["snr_ok"].to_numpy(bool)
                    cal_ok = sub["cal_ok"].to_numpy(bool)
                    kept = delta_ok & snr_ok & cal_ok
                    summary_rows.append({
                        "filter": filt,
                        "n_total": int(len(sub)),
                        "n_ref_ok": int(np.sum(ref_ok)),
                        "n_delta_ok": int(np.sum(delta_ok)),
                        "n_snr_ok": int(np.sum(snr_ok)),
                        "n_cal_ok": int(np.sum(cal_ok)),
                        "n_kept": int(np.sum(kept)),
                    })
                cut_df = pd.DataFrame(summary_rows)
                cut_path = output_dir / "frame_zeropoint_cut_summary.csv"
                cut_df.to_csv(cut_path, index=False)
                self._log(f"Saved {cut_path.name} | rows={len(cut_df)}")

            frame_rows = []
            reject_rows = []
            for (fname, filt), sub in obs.groupby(["file", "FILTER"]):
                med, std, n, out_frac = self._robust_location(sub["delta"].to_numpy(float), clip_sigma=clip_sigma, iters=fit_iters)
                if n < min_frame_refs:
                    reject_rows.append({
                        "file": fname,
                        "filter": filt,
                        "n_ref": int(n),
                        "min_required": int(min_frame_refs),
                        "reason": "n_ref_below_min",
                    })
                    continue
                frame_rows.append({
                    "file": fname,
                    "filter": filt,
                    "zp_frame": med,
                    "zp_scatter": std,
                    "n_ref": n,
                    "outlier_fraction": out_frac,
                    "snr_med": float(np.nanmedian(sub["snr"].to_numpy(float))) if "snr" in sub.columns else np.nan,
                })

            frame_df = pd.DataFrame(frame_rows)
            if frame_df.empty:
                frame_df = pd.DataFrame(columns=["file", "filter", "zp_frame", "zp_scatter", "n_ref", "outlier_fraction", "snr_med"])
                self._log("No per-frame ZP points; falling back to global ZP by filter.")
            if len(frame_df):
                frame_df = frame_df.merge(frame_airmass, on=["file", "filter"], how="left")
                frame_zp_path = output_dir / "frame_zeropoint.csv"
                frame_df.to_csv(frame_zp_path, index=False)
                self._log(f"Saved {frame_zp_path.name} | rows={len(frame_df)}")
            if reject_rows:
                reject_df = pd.DataFrame(reject_rows)
                reject_path = output_dir / "frame_zeropoint_rejects.csv"
                reject_df.to_csv(reject_path, index=False)
                self._log(f"Saved {reject_path.name} | rows={len(reject_df)}")

            obs = all_df.merge(frame_df[["file", "filter", "zp_frame"]], left_on=["file", "FILTER"], right_on=["file", "filter"], how="left")
            zp_map = {"g": zp_g, "r": zp_r, "i": zp_i}
            obs["zp_frame"] = obs["zp_frame"].fillna(obs["FILTER"].map(zp_map))
            obs = obs.merge(color_df[["ID", "color_gr", "color_ri"]], on="ID", how="left")
            obs = obs.merge(frame_airmass[["file", "filter", "airmass"]], left_on=["file", "FILTER"], right_on=["file", "filter"], how="left")

            obs["color_term"] = np.nan
            obs.loc[obs["FILTER"] == "g", "color_term"] = ct_g * obs.loc[obs["FILTER"] == "g", "color_gr"]
            obs.loc[obs["FILTER"] == "r", "color_term"] = ct_r * obs.loc[obs["FILTER"] == "r", "color_gr"]
            obs.loc[obs["FILTER"] == "i", "color_term"] = ct_i * obs.loc[obs["FILTER"] == "i", "color_ri"]

            obs["k_term"] = 0.0
            if apply_ext and ext_map:
                for f, k in ext_map.items():
                    m = obs["FILTER"] == f
                    obs.loc[m, "k_term"] = k * obs.loc[m, "airmass"].to_numpy(float)

            obs["mag_cal"] = obs["mag_inst"] + obs["zp_frame"] + obs["color_term"]
            if apply_ext and ext_mode == "two_step":
                obs["mag_cal"] = obs["mag_cal"] + obs["k_term"]

            def _combine_group_cal(g):
                med, med_err, n_med = self._robust_median_and_err(g["mag_cal"])
                wmean, werr, _ = self._weighted_mean_mag(g["mag_cal"], g["mag_err"])
                snr_vals = np.asarray(g.get("snr", np.nan), float)
                snr_med = float(np.nanmedian(snr_vals)) if np.isfinite(snr_vals).any() else np.nan
                return pd.Series({
                    "mag_inst_med": med,
                    "mag_inst_med_err": med_err,
                    "mag_inst_wmean": wmean,
                    "mag_inst_werr": werr,
                    "n_frames": n_med,
                    "snr_med": snr_med,
                })

            grp_cal = obs.groupby(["ID", "FILTER"]).apply(_combine_group_cal).reset_index()

            grp_path = output_dir / "median_by_ID_filter.csv"
            grp_cal.to_csv(grp_path, index=False, na_rep="NaN")
            self._log(f"Saved {grp_path.name} | rows={len(grp_cal)}")

            wide_mag_w = grp_cal.pivot_table(index="ID", columns="FILTER", values="mag_inst_wmean", aggfunc="median")
            wide_err_w = grp_cal.pivot_table(index="ID", columns="FILTER", values="mag_inst_werr", aggfunc="median")
            wide_mag_med = grp_cal.pivot_table(index="ID", columns="FILTER", values="mag_inst_med", aggfunc="median")
            wide_err_med = grp_cal.pivot_table(index="ID", columns="FILTER", values="mag_inst_med_err", aggfunc="median")
            wide_mag = wide_mag_w.combine_first(wide_mag_med)
            wide_err = wide_err_w.combine_first(wide_err_med)
            wide_snr = grp_cal.pivot_table(index="ID", columns="FILTER", values="snr_med", aggfunc="median")

            wide_mag.columns = [f"mag_inst_{c}" for c in wide_mag.columns]
            wide_err.columns = [f"mag_inst_err_{c}" for c in wide_err.columns]
            wide_snr.columns = [f"snr_{c}" for c in wide_snr.columns]
            wide_mag_w.columns = [f"mag_inst_wmean_{c}" for c in wide_mag_w.columns]
            wide_err_w.columns = [f"mag_inst_werr_{c}" for c in wide_err_w.columns]
            wide_mag_med.columns = [f"mag_inst_med_{c}" for c in wide_mag_med.columns]
            wide_err_med.columns = [f"mag_inst_med_err_{c}" for c in wide_err_med.columns]

            wide = pd.concat(
                [
                    wide_mag,
                    wide_err,
                    wide_mag_w,
                    wide_err_w,
                    wide_mag_med,
                    wide_err_med,
                    wide_snr,
                ],
                axis=1,
            ).reset_index()

            wide_path = output_dir / "median_by_ID_filter_wide.csv"
            wide.to_csv(wide_path, index=False, na_rep="NaN")
            self._log(f"Saved {wide_path.name} | rows={len(wide)}")

            df_out = wide_raw.merge(master[merge_cols], on="ID", how="left")
            if col_xm:
                df_out = df_out.rename(columns={col_xm: "x_pix"})
            if col_ym:
                df_out = df_out.rename(columns={col_ym: "y_pix"})

            for col in ("gaia_G", "gaia_BP", "gaia_RP"):
                if col not in df_out.columns and col in df.columns:
                    df_out[col] = df[col]

            for band in ("g", "r", "i"):
                c_inst = f"mag_inst_{band}"
                c_std = f"mag_std_{band}"
                if c_inst in wide.columns:
                    df_out[c_std] = wide[c_inst].to_numpy(float)
                else:
                    df_out[c_std] = np.nan

            gi_std = df_out["mag_std_g"].to_numpy(float) - df_out["mag_std_i"].to_numpy(float)
            m_gi_std = np.isfinite(gi_std) & (gi_std >= 1.0) & (gi_std <= 9.0)

            G_minus_g_from_gi = np.full(len(df_out), np.nan)
            BPmRP_from_gi = np.full(len(df_out), np.nan)

            G_minus_g_from_gi[m_gi_std] = self._poly_eval(gi_std[m_gi_std], [-0.1064, -0.4964, -0.09339, 0.004444])
            BPmRP_from_gi[m_gi_std] = self._poly_eval(gi_std[m_gi_std], [0.3971, 0.777, -0.04164, 0.008237])

            df_out["gaia_G_syn"] = df_out["mag_std_g"].to_numpy(float) + G_minus_g_from_gi
            df_out["gaia_BP_RP_syn"] = BPmRP_from_gi

            out_cmd_path = output_dir / "median_by_ID_filter_wide_cmd.csv"
            df_out.to_csv(out_cmd_path, index=False, na_rep="NaN")
            self._log(f"Saved {out_cmd_path.name} | rows={len(df_out)}")

            summary = {
                "ok": True,
                "wide": str(wide_path),
                "cmd": str(out_cmd_path),
                "frame_airmass": str((output_dir / "frame_airmass.csv")) if (output_dir / "frame_airmass.csv").exists() else "",
                "frame_zeropoint": str((output_dir / "frame_zeropoint.csv")) if (output_dir / "frame_zeropoint.csv").exists() else "",
            }
            self.finished.emit(summary)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class CmdViewerWindow(QWidget):
    """Interactive CMD viewer (Qt)."""

    def __init__(self, df: pd.DataFrame, result_dir: Path, parent=None, embedded: bool = False):
        super().__init__(parent)
        self.df = df
        self.result_dir = Path(result_dir)

        self.setWindowTitle("CMD Viewer")
        if embedded:
            self.setWindowFlags(Qt.Widget)
            self.setMinimumSize(900, 600)
        else:
            self.setWindowFlag(Qt.Window, True)
            self.resize(1200, 900)
            self.setMinimumSize(1000, 720)

        # View mode: 0=inst, 1=std, 2=gaia, 3=all
        self.view_mode = 0

        self.inst_bands = sorted({c.replace("mag_inst_", "") for c in df.columns if c.startswith("mag_inst_")})
        self.std_bands = sorted({c.replace("mag_std_", "") for c in df.columns if c.startswith("mag_std_")})
        self.has_std = (len(self.std_bands) > 0) and np.isfinite(df[[c for c in df.columns if c.startswith("mag_std_")]].to_numpy(float)).any()

        self.all_bands = sorted(set(self.inst_bands) | set(self.std_bands))

        self.x_allowed = [("g", "r"), ("r", "i"), ("g", "i")]
        self.y_allowed_scalars = ["g", "r", "i"]
        self.y_allowed_colors = [("g", "r"), ("r", "i"), ("g", "i")]

        self.x_pairs = [(a, b) for (a, b) in self.x_allowed if (a in self.inst_bands and b in self.inst_bands)]
        self.y_scalar_opts = [b for b in self.y_allowed_scalars if b in self.inst_bands]
        self.y_color_pairs = [(a, b) for (a, b) in self.y_allowed_colors if (a in self.inst_bands and b in self.inst_bands)]

        self.snr_cols = [c for c in df.columns if c.startswith("snr_")]
        self.has_snr = len(self.snr_cols) > 0

        self.has_gaia_inst = (
            {"gaia_G_inst", "gaia_BP_RP_inst"}.issubset(df.columns)
            and np.isfinite(df["gaia_G_inst"].to_numpy(float)).any()
            and np.isfinite(df["gaia_BP_RP_inst"].to_numpy(float)).any()
        )
        self.has_gaia_syn = (
            {"gaia_G_syn", "gaia_BP_RP_syn"}.issubset(df.columns)
            and np.isfinite(df["gaia_G_syn"].to_numpy(float)).any()
            and np.isfinite(df["gaia_BP_RP_syn"].to_numpy(float)).any()
        )
        if self.has_gaia_inst:
            self.gaia_mode = "inst"
        elif self.has_gaia_syn:
            self.gaia_mode = "syn"
        else:
            self.gaia_mode = None

        self.teff_vmin = 2400.0
        self.teff_vmax = 40000.0
        self.ob_norm = Normalize(vmin=self.teff_vmin, vmax=self.teff_vmax, clip=True)

        anchors = [
            (2400, "#E53935"),
            (3200, "#FF6A3D"),
            (4500, "#FFB84D"),
            (5800, "#FFE36A"),
            (6500, "#FFF6C7"),
            (8000, "#FFFFFF"),
            (10000, "#FFFFFF"),
            (20000, "#2D5BFF"),
            (40000, "#7A3CFF"),
        ]
        anchors = sorted(anchors, key=lambda x: x[0])
        pos = [(t - self.teff_vmin) / (self.teff_vmax - self.teff_vmin) for t, _ in anchors]
        pos[0] = 0.0
        pos[-1] = 1.0

        self.ob_cmap = LinearSegmentedColormap.from_list(
            "obafgkm_like",
            list(zip(pos, [c for _, c in anchors])),
            N=256
        )
        self.ob_cmap.set_bad("#777777")

        self.color_anchors = {
            "g-r": {
                "x": np.array([-0.40, -0.20, 0.00, 0.30, 0.45, 0.80, 1.40, 1.80], float),
                "t": np.array([35000, 20000, 10000, 7500, 6000, 4500, 3200, 2400], float),
            },
            "r-i": {
                "x": np.array([-0.30, -0.20, -0.05, 0.10, 0.20, 0.40, 0.80, 1.10], float),
                "t": np.array([35000, 20000, 10000, 7500, 6000, 4500, 3200, 2400], float),
            },
            "g-i": {
                "x": np.array([-0.70, -0.45, 0.00, 0.50, 0.80, 1.50, 2.60, 3.20], float),
                "t": np.array([35000, 20000, 10000, 7500, 6000, 4500, 3200, 2400], float),
            },
            "BP-RP": {
                "x": np.array([-0.40, -0.20, 0.00, 0.40, 0.80, 1.30, 2.30, 3.50], float),
                "t": np.array([35000, 20000, 10000, 7500, 6000, 4500, 3200, 2400], float),
            },
        }

        # Determine available views
        self.available_views = ["inst"]
        if self.has_std:
            self.available_views.append("std")
        if self.gaia_mode is not None:
            self.available_views.append("gaia")
        self.available_views.append("all")

        self._plot_cache = {}
        self.last_pick_info = None
        self.pick_log = []
        self._build_ui()
        self._build_figure()
        self._redraw()
        self.setFocusPolicy(Qt.StrongFocus)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("X(color):"))
        self.x_combo = QComboBox()
        self.x_combo.addItems([f"{a}-{b}" for (a, b) in self.x_pairs] or ["(none)"])
        controls.addWidget(self.x_combo)

        controls.addWidget(QLabel("Y:"))
        y_opts = self.y_scalar_opts + [f"{a}-{b}" for (a, b) in self.y_color_pairs]
        self.y_combo = QComboBox()
        self.y_combo.addItems(y_opts or ["(none)"])
        controls.addWidget(self.y_combo)

        controls.addWidget(QLabel("SNR >="))
        self.snr_spin = QSpinBox()
        self.snr_spin.setRange(0, 100)
        self.snr_spin.setValue(20)
        controls.addWidget(self.snr_spin)

        self.invert_y = QCheckBox("Invert Y")
        self.invert_y.setChecked(True)
        controls.addWidget(self.invert_y)

        self.save_btn = QPushButton("Save PNG")
        controls.addWidget(self.save_btn)

        controls.addStretch()
        self.view_label = QLabel("View: Instrumental")
        self.view_label.setStyleSheet("QLabel { color: #2196F3; font-weight: bold; }")
        controls.addWidget(self.view_label)
        layout.addLayout(controls)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFixedHeight(90)
        self.info_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        layout.addWidget(self.info_text)

        self.figure = Figure(figsize=(12, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(800, 400)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        view_controls = QHBoxLayout()
        self.btn_prev_view = QPushButton("Prev View")
        self.btn_next_view = QPushButton("Next View")
        view_controls.addStretch()
        view_controls.addWidget(self.btn_prev_view)
        view_controls.addWidget(self.btn_next_view)
        view_controls.addStretch()
        layout.addLayout(view_controls)

        self.x_combo.currentTextChanged.connect(self._redraw)
        self.y_combo.currentTextChanged.connect(self._redraw)
        self.snr_spin.valueChanged.connect(self._redraw)
        self.invert_y.stateChanged.connect(self._redraw)
        self.save_btn.clicked.connect(self._save_png)
        self.btn_prev_view.clicked.connect(lambda: self._switch_view(-1))
        self.btn_next_view.clicked.connect(lambda: self._switch_view(1))
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)

    def _build_figure(self):
        self.figure.clear()
        view_name = self.available_views[self.view_mode]

        # Single view modes
        if view_name == "inst":
            self.ax_inst = self.figure.add_subplot(1, 1, 1)
            self.ax_std = None
            self.ax_gaia = None
            self.figure.subplots_adjust(left=0.10, right=0.88, bottom=0.10, top=0.92)
            self.cax = self.figure.add_axes([0.90, 0.12, 0.02, 0.78])
        elif view_name == "std":
            self.ax_inst = None
            self.ax_std = self.figure.add_subplot(1, 1, 1)
            self.ax_gaia = None
            self.figure.subplots_adjust(left=0.10, right=0.88, bottom=0.10, top=0.92)
            self.cax = self.figure.add_axes([0.90, 0.12, 0.02, 0.78])
        elif view_name == "gaia":
            self.ax_inst = None
            self.ax_std = None
            self.ax_gaia = self.figure.add_subplot(1, 1, 1)
            self.figure.subplots_adjust(left=0.10, right=0.88, bottom=0.10, top=0.92)
            self.cax = self.figure.add_axes([0.90, 0.12, 0.02, 0.78])
        elif view_name == "all":
            # Show all available CMDs
            if self.has_std and self.gaia_mode is not None:
                self.ax_inst = self.figure.add_subplot(1, 3, 1)
                self.ax_std = self.figure.add_subplot(1, 3, 2)
                self.ax_gaia = self.figure.add_subplot(1, 3, 3)
                self.figure.subplots_adjust(left=0.055, right=0.90, bottom=0.12, top=0.88, wspace=0.25)
                self.cax = self.figure.add_axes([0.92, 0.16, 0.015, 0.70])
            elif self.has_std:
                self.ax_inst = self.figure.add_subplot(1, 2, 1)
                self.ax_std = self.figure.add_subplot(1, 2, 2)
                self.ax_gaia = None
                self.figure.subplots_adjust(left=0.07, right=0.90, bottom=0.12, top=0.88, wspace=0.25)
                self.cax = self.figure.add_axes([0.92, 0.16, 0.015, 0.70])
            elif self.gaia_mode is not None:
                self.ax_inst = self.figure.add_subplot(1, 2, 1)
                self.ax_std = None
                self.ax_gaia = self.figure.add_subplot(1, 2, 2)
                self.figure.subplots_adjust(left=0.07, right=0.90, bottom=0.12, top=0.88, wspace=0.25)
                self.cax = self.figure.add_axes([0.92, 0.16, 0.015, 0.70])
            else:
                self.ax_inst = self.figure.add_subplot(1, 1, 1)
                self.ax_std = None
                self.ax_gaia = None
                self.figure.subplots_adjust(left=0.13, right=0.88, bottom=0.12, top=0.88)
                self.cax = self.figure.add_axes([0.90, 0.16, 0.02, 0.70])
        else:
            self.ax_inst = self.figure.add_subplot(1, 1, 1)
            self.ax_std = None
            self.ax_gaia = None
            self.figure.subplots_adjust(left=0.13, right=0.88, bottom=0.12, top=0.88)
            self.cax = self.figure.add_axes([0.90, 0.16, 0.02, 0.70])

        self.figure.patch.set_facecolor("black")
        for ax in [self.ax_inst, self.ax_std, self.ax_gaia]:
            if ax is None:
                continue
            ax.set_facecolor("black")
            for sp in ax.spines.values():
                sp.set_color("white")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")

        self.cax.set_facecolor("black")
        sm = mpl.cm.ScalarMappable(norm=self.ob_norm, cmap=self.ob_cmap)
        sm.set_array([])
        cbar = self.figure.colorbar(sm, cax=self.cax)
        cbar.set_label("Teff (K) + OBAFGKM-like color", fontsize=9, color="white")
        ticks = [35000, 20000, 10000, 7500, 6000, 4500, 3200]
        labels = ["35000 K (O)", "20000 K (B)", "10000 K (A)", " 7500 K (F)", " 6000 K (G)", " 4500 K (K)", " 3200 K (M)"]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels)
        cbar.ax.tick_params(colors="white")
        for sp in cbar.ax.spines.values():
            sp.set_color("white")

    def _safe_float(self, series):
        return pd.to_numeric(series, errors="coerce").to_numpy(float)

    def _teff_from_color_index(self, color_x: np.ndarray, mode: str):
        a = self.color_anchors.get(mode, None)
        if a is None or color_x.size == 0:
            return np.full_like(color_x, np.nan, dtype=float)
        x = np.asarray(color_x, float)
        teff = np.interp(x, a["x"], a["t"])
        return np.clip(teff, self.teff_vmin, self.teff_vmax)

    def _get_y_mode(self, yval):
        if yval in self.y_scalar_opts:
            return ("scalar", yval)
        if isinstance(yval, str) and "-" in yval:
            a, b = yval.split("-", 1)
            if (a, b) in self.y_color_pairs:
                return ("color", (a, b))
        return (None, None)

    def _compute_arrays_and_mask(self, system: str, x_pair, y_choice, snr_cut: float):
        a, b = x_pair
        col_ax = f"mag_{system}_{a}"
        col_bx = f"mag_{system}_{b}"
        if (col_ax not in self.df.columns) or (col_bx not in self.df.columns):
            return np.array([]), np.array([]), np.zeros(len(self.df), bool), np.array([])

        Ax = self._safe_float(self.df[col_ax])
        Bx = self._safe_float(self.df[col_bx])
        xcolor = Ax - Bx
        x = xcolor

        y_mode, y_param = self._get_y_mode(y_choice)
        involved = set([a, b])

        if y_mode == "scalar":
            by = y_param
            col_y = f"mag_{system}_{by}"
            if col_y not in self.df.columns:
                return np.array([]), np.array([]), np.zeros(len(self.df), bool), np.array([])
            y = self._safe_float(self.df[col_y])
            involved.add(by)
        elif y_mode == "color":
            ya, yb = y_param
            col_ya = f"mag_{system}_{ya}"
            col_yb = f"mag_{system}_{yb}"
            if (col_ya not in self.df.columns) or (col_yb not in self.df.columns):
                return np.array([]), np.array([]), np.zeros(len(self.df), bool), np.array([])
            Ya = self._safe_float(self.df[col_ya])
            Yb = self._safe_float(self.df[col_yb])
            y = Ya - Yb
            involved.update([ya, yb])
        else:
            return np.array([]), np.array([]), np.zeros(len(self.df), bool), np.array([])

        mask = np.isfinite(x) & np.isfinite(y)

        if snr_cut > 0 and self.has_snr:
            for band in involved:
                sc = f"snr_{band}"
                if sc in self.df.columns:
                    sv = self._safe_float(self.df[sc])
                    mask &= np.isfinite(sv) & (sv >= snr_cut)

        return x[mask], y[mask], mask, xcolor[mask]

    def _compute_gaia_arrays_and_mask(self, snr_cut: float):
        if self.gaia_mode is None:
            return np.array([]), np.array([]), np.zeros(len(self.df), bool), np.array([])

        if self.gaia_mode == "inst":
            G = self._safe_float(self.df["gaia_G_inst"])
            C = self._safe_float(self.df["gaia_BP_RP_inst"])
        else:
            G = self._safe_float(self.df["gaia_G_syn"])
            C = self._safe_float(self.df["gaia_BP_RP_syn"])

        mask = np.isfinite(G) & np.isfinite(C)

        if snr_cut > 0 and self.has_snr:
            for band in ("g", "r", "i"):
                sc = f"snr_{band}"
                if sc in self.df.columns:
                    sv = self._safe_float(self.df[sc])
                    mask &= np.isfinite(sv) & (sv >= snr_cut)

        return C[mask], G[mask], mask, C[mask]

    def _style_axis(self, ax):
        ax.set_facecolor("black")
        for sp in ax.spines.values():
            sp.set_color("white")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    def _apply_y_orientation(self):
        axes = [self.ax_inst, self.ax_std, self.ax_gaia]
        for ax in axes:
            if ax is None:
                continue
            ymin, ymax = ax.get_ylim()
            if self.invert_y.isChecked():
                if ymin < ymax:
                    ax.set_ylim(ymax, ymin)
            else:
                if ymin > ymax:
                    ax.set_ylim(ymax, ymin)

    def _redraw(self):
        ax_primary = self.ax_inst or self.ax_std or self.ax_gaia
        self._plot_cache = {}
        if not self.x_pairs:
            if ax_primary is not None:
                ax_primary.clear()
                ax_primary.set_title("No available X color index", fontsize=10, color="white")
            self.canvas.draw_idle()
            return

        x_text = self.x_combo.currentText()
        if x_text not in [f"{a}-{b}" for (a, b) in self.x_pairs]:
            self.canvas.draw_idle()
            return
        a, b = x_text.split("-", 1)
        x_pair = (a, b)

        yval = self.y_combo.currentText()
        snr_cut = float(self.snr_spin.value())

        if self.ax_inst is not None:
            self.ax_inst.clear()
        if self.ax_std is not None:
            self.ax_std.clear()
        if self.ax_gaia is not None:
            self.ax_gaia.clear()

        if self.ax_inst is not None:
            x_i, y_i, mask_i, xcol_i = self._compute_arrays_and_mask("inst", x_pair, yval, snr_cut)
            teff_i = self._teff_from_color_index(xcol_i, f"{a}-{b}")
        else:
            x_i, y_i, mask_i, teff_i = np.array([]), np.array([]), np.zeros(len(self.df), bool), np.array([])

        if self.has_std and self.ax_std is not None:
            x_s, y_s, mask_s, xcol_s = self._compute_arrays_and_mask("std", x_pair, yval, snr_cut)
            teff_s = self._teff_from_color_index(xcol_s, f"{a}-{b}")
        else:
            x_s, y_s, mask_s, teff_s = np.array([]), np.array([]), np.zeros(len(self.df), bool), np.array([])

        if self.gaia_mode is not None and self.ax_gaia is not None:
            x_g, y_g, mask_g, xcol_g = self._compute_gaia_arrays_and_mask(snr_cut)
            teff_g = self._teff_from_color_index(xcol_g, "BP-RP")
        else:
            x_g, y_g, mask_g, teff_g = np.array([]), np.array([]), np.zeros(len(self.df), bool), np.array([])

        if self.ax_inst is not None:
            self._style_axis(self.ax_inst)
            if len(x_i) > 0:
                self.ax_inst.scatter(x_i, y_i, s=12, alpha=0.92, linewidths=0, rasterized=True, c=teff_i, cmap=self.ob_cmap, norm=self.ob_norm)
                self.ax_inst.set_title(f"Instrumental CMD (N={len(x_i)})", fontsize=11, color="white")
                self._plot_cache[self.ax_inst] = {
                    "system": "inst",
                    "x": x_i,
                    "y": y_i,
                    "df_index": np.where(mask_i)[0],
                }
            else:
                self.ax_inst.set_title("Instrumental CMD (N=0)", fontsize=11, color="white")

        if self.ax_std is not None:
            self._style_axis(self.ax_std)
            if len(x_s) > 0:
                self.ax_std.scatter(x_s, y_s, s=12, alpha=0.92, linewidths=0, rasterized=True, c=teff_s, cmap=self.ob_cmap, norm=self.ob_norm)
                self.ax_std.set_title(f"SDSS CMD (N={len(x_s)})", fontsize=11, color="white")
                self._plot_cache[self.ax_std] = {
                    "system": "std",
                    "x": x_s,
                    "y": y_s,
                    "df_index": np.where(mask_s)[0],
                }
            else:
                self.ax_std.set_title("SDSS CMD (N=0)", fontsize=11, color="white")

        if self.ax_gaia is not None:
            self._style_axis(self.ax_gaia)
            if len(x_g) > 0:
                self.ax_gaia.scatter(x_g, y_g, s=12, alpha=0.92, linewidths=0, rasterized=True, c=teff_g, cmap=self.ob_cmap, norm=self.ob_norm)
                title = "Gaia CMD (inst)" if self.gaia_mode == "inst" else "Gaia CMD (syn)"
                self.ax_gaia.set_title(f"{title} (N={len(x_g)})", fontsize=11, color="white")
                self._plot_cache[self.ax_gaia] = {
                    "system": "gaia",
                    "x": x_g,
                    "y": y_g,
                    "df_index": np.where(mask_g)[0],
                }
            else:
                self.ax_gaia.set_title("Gaia CMD (N=0)", fontsize=11, color="white")

        x_label = f"{a}-{b} (mag)"
        y_label = f"{yval} (mag)" if yval else ""
        if self.ax_inst is not None:
            self.ax_inst.set_xlabel(x_label)
            self.ax_inst.set_ylabel(y_label)
        if self.ax_std is not None:
            self.ax_std.set_xlabel(x_label)
            self.ax_std.set_ylabel(y_label)
        if self.ax_gaia is not None:
            self.ax_gaia.set_xlabel("BP-RP (mag)")
            self.ax_gaia.set_ylabel("G (mag)")

        self._apply_y_orientation()

        def _rng(arr):
            arr = np.asarray(arr, float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return "n/a"
            return f"{arr.min():.0f}-{arr.max():.0f} K"

        lines = [
            f"X={a}-{b}, Y={yval}, SNR>={snr_cut:.0f}",
            f"[Inst] N={len(x_i)} | Teff range: {_rng(teff_i)}",
        ]
        if self.has_std:
            lines.append(f"[Std]  N={len(x_s)} | Teff range: {_rng(teff_s)}")
        if self.gaia_mode is not None:
            lines.append(f"[Gaia:{self.gaia_mode}] N={len(x_g)} | Teff range: {_rng(teff_g)}")
        if not self.has_snr:
            lines.append("(snr_* columns missing: SNR cut disabled)")
        if self.last_pick_info:
            lines.append(self.last_pick_info)
        if self.pick_log:
            lines.append("Pick log (latest 5):")
            lines.extend(self.pick_log[-5:])
        self.info_text.setPlainText("\n".join(lines))

        self.canvas.draw_idle()

    def _fmt_val(self, v, nd=3):
        try:
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return "NaN"
            return f"{float(v):.{nd}f}"
        except Exception:
            return str(v)

    def _on_plot_click(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if not getattr(event, "dblclick", False):
            return
        if event.inaxes not in self._plot_cache:
            return
        cache = self._plot_cache[event.inaxes]
        x = cache["x"]
        y = cache["y"]
        if x.size == 0:
            return
        xy_disp = event.inaxes.transData.transform(np.column_stack([x, y]))
        click = np.array([event.x, event.y])
        d2 = np.sum((xy_disp - click) ** 2, axis=1)
        idx = int(np.argmin(d2))
        if d2[idx] > (12.0 ** 2):
            return
        df_idx = int(cache["df_index"][idx])
        row = self.df.iloc[df_idx]

        parts = [f"Pick[{cache['system']}] ID={row.get('ID', 'n/a')}", f"source_id={row.get('source_id', 'n/a')}"]
        for band in ("g", "r", "i"):
            c_inst = f"mag_inst_{band}"
            if c_inst in row:
                parts.append(f"{band}_inst={self._fmt_val(row.get(c_inst))}")
        for band in ("g", "r", "i"):
            c_std = f"mag_std_{band}"
            if c_std in row:
                parts.append(f"{band}_std={self._fmt_val(row.get(c_std))}")
        if "gaia_G" in row:
            parts.append(f"gaia_G={self._fmt_val(row.get('gaia_G'))}")
        if "gaia_BP" in row:
            parts.append(f"gaia_BP={self._fmt_val(row.get('gaia_BP'))}")
        if "gaia_RP" in row:
            parts.append(f"gaia_RP={self._fmt_val(row.get('gaia_RP'))}")
        msg = " | ".join(parts)
        self.last_pick_info = msg
        self.pick_log.append(msg)
        self._redraw()

    def _save_png(self):
        if not self.x_pairs:
            self.info_text.setPlainText("No available X color index")
            return
        a, b = self.x_combo.currentText().split("-", 1)
        yv = self.y_combo.currentText().replace(" ", "")

        if self.has_std and self.gaia_mode is not None:
            mode = f"inst_std_gaia{self.gaia_mode}"
        elif self.has_std:
            mode = "inst_std"
        elif self.gaia_mode is not None:
            mode = f"inst_gaia{self.gaia_mode}"
        else:
            mode = "inst_only"

        out = self.result_dir / f"cmd_{mode}_{a}-{b}_vs_{yv}_snr{int(self.snr_spin.value())}_OBcolor_dark.png"
        self.figure.savefig(out, dpi=170, bbox_inches="tight", facecolor=self.figure.get_facecolor(), edgecolor="none")
        self.info_text.setPlainText(f"Saved: {out}")

    def keyPressEvent(self, event):
        super().keyPressEvent(event)

    def _switch_view(self, delta: int):
        """Switch between views: inst, std, gaia, all"""
        self.view_mode = (self.view_mode + delta) % len(self.available_views)
        view_name = self.available_views[self.view_mode]
        view_labels = {"inst": "Instrumental", "std": "SDSS Standard", "gaia": "Gaia", "all": "All CMDs"}
        self.view_label.setText(f"View: {view_labels.get(view_name, view_name)}")
        self._build_figure()
        self._redraw()


class ZeropointCalibrationWindow(StepWindowBase):
    """Step 11: Zeropoint & Standardization"""

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        self.worker = None
        self.viewer = None

        super().__init__(
            step_index=10,
            step_name="Zeropoint Calibration",
            params=params,
            project_state=project_state,
            main_window=main_window
        )

        self.setup_step_ui()
        self.restore_state()

    def setup_step_ui(self):
        info = QLabel(
            "Build per-frame ZP calibration and standardized catalogs."
        )
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        self.content_layout.addWidget(info)

        control_layout = QHBoxLayout()
        btn_params = QPushButton("Calibration Parameters")
        btn_params.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 8px 15px; }")
        btn_params.clicked.connect(self.open_parameters_dialog)
        control_layout.addWidget(btn_params)

        self.btn_run = QPushButton("Run ZP Calibration")
        self.btn_run.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 15px; }")
        self.btn_run.clicked.connect(self.run_analysis)
        control_layout.addWidget(self.btn_run)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px 15px; }")
        self.btn_stop.clicked.connect(self.stop_analysis)
        self.btn_stop.setEnabled(False)
        control_layout.addWidget(self.btn_stop)

        btn_log = QPushButton("Open Log")
        btn_log.setStyleSheet("QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 8px 15px; }")
        btn_log.clicked.connect(self.show_log_window)
        control_layout.addWidget(btn_log)

        control_layout.addStretch()
        self.content_layout.addLayout(control_layout)

        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_label = QLabel("Idle")
        progress_layout.addWidget(self.progress_label)
        self.content_layout.addWidget(progress_group)

        self.log_window = QWidget(self, Qt.Window)
        self.log_window.setWindowTitle("Calibration Log")
        self.log_window.resize(800, 400)
        log_layout = QVBoxLayout(self.log_window)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        log_layout.addWidget(self.log_text)

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def show_log_window(self):
        self.log_window.show()
        self.log_window.raise_()
        self.log_window.activateWindow()

    def open_parameters_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Calibration Parameters")
        dialog.resize(480, 480)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        self.param_pix = QDoubleSpinBox()
        self.param_pix.setRange(0.0, 50.0)
        self.param_pix.setValue(float(getattr(self.params.P, "pixel_scale_arcsec", 0.0) or 0.0))
        self.param_pix.setEnabled(False)
        form.addRow("Pixel scale (arcsec):", self.param_pix)

        self.param_match = QDoubleSpinBox()
        self.param_match.setRange(0.1, 20.0)
        self.param_match.setValue(float(getattr(self.params.P, "match_tol_px", 5.0)))
        form.addRow("Match tol (px):", self.param_match)

        self.param_min_match = QSpinBox()
        self.param_min_match.setRange(3, 1000)
        self.param_min_match.setValue(int(getattr(self.params.P, "min_master_gaia_matches", 10)))
        form.addRow("Min Gaia matches:", self.param_min_match)

        self.param_cmd_snr = QDoubleSpinBox()
        self.param_cmd_snr.setRange(0.0, 200.0)
        self.param_cmd_snr.setValue(float(getattr(self.params.P, "cmd_snr_calib_min", 20.0)))
        form.addRow("CMD calib SNR min:", self.param_cmd_snr)

        self.param_frame_min = QSpinBox()
        self.param_frame_min.setRange(1, 1000)
        self.param_frame_min.setValue(int(getattr(self.params.P, "frame_zp_min_n", 5)))
        form.addRow("Frame ZP min refs:", self.param_frame_min)

        self.param_apply_ext = QCheckBox("Enable")
        self.param_apply_ext.setChecked(bool(getattr(self.params.P, "cmd_apply_extinction", False)))
        form.addRow("Apply extinction (k·X):", self.param_apply_ext)

        self.param_ext_mode = QComboBox()
        self.param_ext_mode.addItems(["absorb", "two_step"])
        self.param_ext_mode.setCurrentText(str(getattr(self.params.P, "cmd_extinction_mode", "absorb")))
        form.addRow("Extinction mode:", self.param_ext_mode)

        self.param_clip = QDoubleSpinBox()
        self.param_clip.setRange(0.5, 10.0)
        self.param_clip.setValue(float(getattr(self.params.P, "zp_clip_sigma", 3.0)))
        form.addRow("ZP clip sigma:", self.param_clip)

        self.param_iters = QSpinBox()
        self.param_iters.setRange(1, 20)
        self.param_iters.setValue(int(getattr(self.params.P, "zp_fit_iters", 5)))
        form.addRow("ZP fit iters:", self.param_iters)

        self.param_slope = QDoubleSpinBox()
        self.param_slope.setRange(0.1, 5.0)
        self.param_slope.setValue(float(getattr(self.params.P, "zp_slope_absmax", 1.0)))
        form.addRow("ZP slope abs max:", self.param_slope)

        self.param_gaia_snr = QDoubleSpinBox()
        self.param_gaia_snr.setRange(0.0, 200.0)
        self.param_gaia_snr.setValue(float(getattr(self.params.P, "gaia_snr_calib_min", 20.0)))
        form.addRow("Gaia calib SNR min:", self.param_gaia_snr)

        self.param_gi_min = QDoubleSpinBox()
        self.param_gi_min.setRange(-2.0, 10.0)
        self.param_gi_min.setValue(float(getattr(self.params.P, "gaia_gi_min", -0.5)))
        form.addRow("Gaia g-i min:", self.param_gi_min)

        self.param_gi_max = QDoubleSpinBox()
        self.param_gi_max.setRange(-2.0, 10.0)
        self.param_gi_max.setValue(float(getattr(self.params.P, "gaia_gi_max", 4.5)))
        form.addRow("Gaia g-i max:", self.param_gi_max)

        self.param_gz_slope = QDoubleSpinBox()
        self.param_gz_slope.setRange(0.1, 5.0)
        self.param_gz_slope.setValue(float(getattr(self.params.P, "gaia_zp_slope_absmax", 1.0)))
        form.addRow("Gaia G slope abs max:", self.param_gz_slope)

        self.param_gc_slope = QDoubleSpinBox()
        self.param_gc_slope.setRange(0.1, 10.0)
        self.param_gc_slope.setValue(float(getattr(self.params.P, "gaia_color_slope_absmax", 2.0)))
        form.addRow("Gaia color slope abs max:", self.param_gc_slope)

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: self.save_parameters(dialog))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec_()

    def save_parameters(self, dialog):
        self.params.P.match_tol_px = self.param_match.value()
        self.params.P.min_master_gaia_matches = self.param_min_match.value()
        self.params.P.cmd_snr_calib_min = self.param_cmd_snr.value()
        self.params.P.frame_zp_min_n = self.param_frame_min.value()
        self.params.P.cmd_apply_extinction = self.param_apply_ext.isChecked()
        self.params.P.cmd_extinction_mode = self.param_ext_mode.currentText().strip()
        self.params.P.zp_clip_sigma = self.param_clip.value()
        self.params.P.zp_fit_iters = self.param_iters.value()
        self.params.P.zp_slope_absmax = self.param_slope.value()
        self.params.P.gaia_snr_calib_min = self.param_gaia_snr.value()
        self.params.P.gaia_gi_min = self.param_gi_min.value()
        self.params.P.gaia_gi_max = self.param_gi_max.value()
        self.params.P.gaia_zp_slope_absmax = self.param_gz_slope.value()
        self.params.P.gaia_color_slope_absmax = self.param_gc_slope.value()
        self.save_state()
        QMessageBox.information(dialog, "Success", "Parameters saved!")
        dialog.accept()

    def run_analysis(self):
        if self.worker and self.worker.isRunning():
            return
        self.log_text.clear()
        self.progress_label.setText("Starting...")

        self.worker = ZeropointCalibrationWorker(
            self.params,
            self.params.P.data_dir,
            self.params.P.result_dir,
            self.params.P.cache_dir,
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.log)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.worker.start()
        self.show_log_window()

    def stop_analysis(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log("Stop requested")

    def on_progress(self, current, total, filename):
        self.progress_label.setText(f"{current}/{total} | {filename}")

    def on_finished(self, summary):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if summary.get("stopped"):
            self.progress_label.setText("Stopped")
            self.log("Analysis stopped")
        else:
            self.progress_label.setText("Done")
            self.log("ZP calibration complete")
            self.save_state()
            self.update_navigation_buttons()

    def on_error(self, message):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_label.setText("Error")
        self.log(f"ERROR: {message}")

    def validate_step(self) -> bool:
        result_dir = self.params.P.result_dir
        wide_cmd = step11_dir(result_dir) / "median_by_ID_filter_wide_cmd.csv"
        wide = step11_dir(result_dir) / "median_by_ID_filter_wide.csv"
        if not wide_cmd.exists() and not wide.exists():
            wide_cmd = result_dir / "median_by_ID_filter_wide_cmd.csv"
            wide = result_dir / "median_by_ID_filter_wide.csv"
        return wide_cmd.exists() or wide.exists()

    def save_state(self):
        state_data = {
            "match_tol_px": getattr(self.params.P, "match_tol_px", 5.0),
            "min_master_gaia_matches": getattr(self.params.P, "min_master_gaia_matches", 10),
            "cmd_snr_calib_min": getattr(self.params.P, "cmd_snr_calib_min", 20.0),
            "frame_zp_min_n": getattr(self.params.P, "frame_zp_min_n", 5),
            "cmd_apply_extinction": getattr(self.params.P, "cmd_apply_extinction", False),
            "cmd_extinction_mode": getattr(self.params.P, "cmd_extinction_mode", "absorb"),
            "zp_clip_sigma": getattr(self.params.P, "zp_clip_sigma", 3.0),
            "zp_fit_iters": getattr(self.params.P, "zp_fit_iters", 5),
            "zp_slope_absmax": getattr(self.params.P, "zp_slope_absmax", 1.0),
            "gaia_snr_calib_min": getattr(self.params.P, "gaia_snr_calib_min", 20.0),
            "gaia_gi_min": getattr(self.params.P, "gaia_gi_min", -0.5),
            "gaia_gi_max": getattr(self.params.P, "gaia_gi_max", 4.5),
            "gaia_zp_slope_absmax": getattr(self.params.P, "gaia_zp_slope_absmax", 1.0),
            "gaia_color_slope_absmax": getattr(self.params.P, "gaia_color_slope_absmax", 2.0),
        }
        self.project_state.store_step_data("zeropoint_calibration", state_data)

    def restore_state(self):
        state_data = self.project_state.get_step_data("zeropoint_calibration")
        if not state_data:
            state_data = self.project_state.get_step_data("cmd_analysis")
        if state_data:
            for key, val in state_data.items():
                if key == "pixel_scale_arcsec":
                    continue
                if hasattr(self.params.P, key):
                    setattr(self.params.P, key, val)
