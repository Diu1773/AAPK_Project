"""
Extinction (Airmass Fit) Tool Window
Fits per-filter extinction coefficients using instrumental magnitudes.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit,
    QGroupBox, QMessageBox, QComboBox, QCheckBox, QSplitter, QTableWidget,
    QTableWidgetItem, QHeaderView
)

from ...utils.astro_utils import compute_airmass_from_header
from ...utils.step_paths import (
    step2_cropped_dir,
    step5_dir,
    step6_dir,
    step9_dir,
    step11_dir,
    step11_extinction_dir,
    legacy_step5_refbuild_dir,
    legacy_step7_wcs_dir,
    legacy_step7_refbuild_dir,
)


class ExtinctionFitWorker(QThread):
    progress = pyqtSignal(int, int, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, params, data_dir: Path, result_dir: Path):
        super().__init__()
        self.params = params
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def _log(self, msg: str):
        self.log.emit(msg)

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

    def _poly_eval(self, x, coeffs):
        x = np.asarray(x, float)
        y = np.zeros_like(x, dtype=float)
        p = np.ones_like(x, dtype=float)
        for a in coeffs:
            y += a * p
            p *= x
        return y

    def _robust_linfit(self, x, y, clip_sigma=3.0, iters=5, min_n=10):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m0 = np.isfinite(x) & np.isfinite(y)
        x = x[m0]
        y = y[m0]
        if len(x) < min_n:
            return (np.nan, np.nan, np.nan, 0, np.nan)
        k, zp = np.polyfit(x, y, 1)
        base_n = len(x)
        for _ in range(int(iters)):
            yhat = zp + k * x
            r = y - yhat
            med = np.nanmedian(r)
            mad = np.nanmedian(np.abs(r - med)) + 1e-12
            sig = 1.4826 * mad
            keep = np.abs(r - med) <= float(clip_sigma) * sig
            if keep.sum() < min_n:
                break
            if keep.sum() == len(x):
                break
            x, y = x[keep], y[keep]
            k, zp = np.polyfit(x, y, 1)
        yhat = zp + k * x
        scatter = float(np.nanstd(y - yhat)) if len(x) else np.nan
        outlier_frac = float(1.0 - (len(x) / max(base_n, 1)))
        return (float(k), float(zp), scatter, int(len(x)), outlier_frac)

    def _robust_quadfit(self, x, y, clip_sigma=3.0, iters=5, min_n=15):
        """2차 다항식 피팅: y = k2*x^2 + k1*x + zp"""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m0 = np.isfinite(x) & np.isfinite(y)
        x = x[m0]
        y = y[m0]
        if len(x) < min_n:
            return (np.nan, np.nan, np.nan, np.nan, 0, np.nan, False)
        try:
            coeffs = np.polyfit(x, y, 2)  # [k2, k1, zp]
            k2, k1, zp = coeffs
        except Exception:
            return (np.nan, np.nan, np.nan, np.nan, 0, np.nan, False)
        base_n = len(x)
        for _ in range(int(iters)):
            yhat = zp + k1 * x + k2 * x * x
            r = y - yhat
            med = np.nanmedian(r)
            mad = np.nanmedian(np.abs(r - med)) + 1e-12
            sig = 1.4826 * mad
            keep = np.abs(r - med) <= float(clip_sigma) * sig
            if keep.sum() < min_n:
                break
            if keep.sum() == len(x):
                break
            x, y = x[keep], y[keep]
            try:
                coeffs = np.polyfit(x, y, 2)
                k2, k1, zp = coeffs
            except Exception:
                break
        yhat = zp + k1 * x + k2 * x * x
        scatter = float(np.nanstd(y - yhat)) if len(x) else np.nan
        outlier_frac = float(1.0 - (len(x) / max(base_n, 1)))
        # k2가 유의미한지 체크 (|k2| > 0.01 이고 scatter 개선이 있으면)
        k2_significant = abs(k2) > 0.005 if np.isfinite(k2) else False
        return (float(k1), float(k2), float(zp), scatter, int(len(x)), outlier_frac, k2_significant)

    def _robust_color_extinction_fit(self, X, C, y, clip_sigma=3.0, iters=5, min_n=15):
        """
        색 의존 소광 피팅: delta = k' * X + k'' * C * X + zp

        Parameters:
        - X: airmass 배열
        - C: 색지수 배열 (예: g-r, BP-RP)
        - y: delta (m_ref - m_inst) 배열

        Returns:
        - k_prime: 평균 소광 계수
        - k_double_prime: 색 의존 소광 계수
        - zp: 제로포인트
        - scatter, n_used, outlier_frac, significant
        """
        X = np.asarray(X, float)
        C = np.asarray(C, float)
        y = np.asarray(y, float)

        m0 = np.isfinite(X) & np.isfinite(C) & np.isfinite(y)
        X = X[m0]
        C = C[m0]
        y = y[m0]

        if len(X) < min_n:
            return (np.nan, np.nan, np.nan, np.nan, 0, np.nan, False)

        # 다중 선형 회귀: y = a0 + a1*X + a2*(C*X)
        # Design matrix: [1, X, C*X]
        try:
            A = np.column_stack([np.ones_like(X), X, C * X])
            coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
            zp, k_prime, k_double_prime = coeffs
        except Exception:
            return (np.nan, np.nan, np.nan, np.nan, 0, np.nan, False)

        base_n = len(X)
        for _ in range(int(iters)):
            yhat = zp + k_prime * X + k_double_prime * C * X
            r = y - yhat
            med = np.nanmedian(r)
            mad = np.nanmedian(np.abs(r - med)) + 1e-12
            sig = 1.4826 * mad
            keep = np.abs(r - med) <= float(clip_sigma) * sig
            if keep.sum() < min_n:
                break
            if keep.sum() == len(X):
                break
            X, C, y = X[keep], C[keep], y[keep]
            try:
                A = np.column_stack([np.ones_like(X), X, C * X])
                coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                zp, k_prime, k_double_prime = coeffs
            except Exception:
                break

        yhat = zp + k_prime * X + k_double_prime * C * X
        scatter = float(np.nanstd(y - yhat)) if len(X) else np.nan
        outlier_frac = float(1.0 - (len(X) / max(base_n, 1)))

        # k''가 유의미한지 체크
        k_double_significant = abs(k_double_prime) > 0.01 if np.isfinite(k_double_prime) else False

        return (float(k_prime), float(k_double_prime), float(zp), scatter, int(len(X)), outlier_frac, k_double_significant)

    def _extract_date_from_file(self, fname: str, hdr=None) -> str:
        """파일명/폴더명 또는 헤더에서 날짜 추출 (YYYY-MM-DD or YYYYMMDD)"""
        import re
        # 1. 파일명에 날짜 폴더가 포함된 경우 (예: "2024-01-15__pp_image.fits")
        if "__" in fname:
            folder_part = fname.split("__")[0]
            # YYYY-MM-DD 패턴
            m = re.match(r"(\d{4}-\d{2}-\d{2})", folder_part)
            if m:
                return m.group(1)
            # YYYYMMDD 패턴
            m = re.match(r"(\d{8})", folder_part)
            if m:
                d = m.group(1)
                return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        # 2. DATE-OBS 헤더에서 추출
        if hdr is not None:
            date_obs = str(hdr.get("DATE-OBS", ""))
            if date_obs:
                m = re.match(r"(\d{4}-\d{2}-\d{2})", date_obs)
                if m:
                    return m.group(1)
        return "unknown"

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
            fpath = self.params.get_file_path(fname)
            if not fpath.exists():
                cand = step2_cropped_dir(self.result_dir) / fname
                if cand.exists():
                    fpath = cand
            if not fpath.exists():
                continue
            try:
                hdr = fits.getheader(fpath)
                info = compute_airmass_from_header(hdr, lat, lon, alt, tz)
                filt = str(r.get("filter", hdr.get("FILTER", ""))).strip().lower()
                obs_date = self._extract_date_from_file(fname, hdr)
                rows.append({
                    "file": fname,
                    "filter": filt,
                    "date": obs_date,
                    **info,
                })
            except Exception:
                continue
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["file", "filter", "date", "airmass", "airmass_source", "alt_deg", "zenith_deg", "datetime_utc", "datetime_local", "ra_deg", "dec_deg"])
        if len(df):
            out_path = step11_dir(self.result_dir) / "frame_airmass.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            self._log(f"Saved {out_path.name} | rows={len(df)}")
        return df

    def run(self):
        try:
            P = self.params.P
            result_dir = self.result_dir

            idx_candidates = [
                step9_dir(result_dir) / "photometry_index.csv",
                result_dir / "photometry_index.csv",
                result_dir / "phot_index.csv",
                result_dir / "phot" / "photometry_index.csv",
                result_dir / "phot" / "phot_index.csv",
            ]
            idx_path = next((p for p in idx_candidates if p.exists()), None)
            if idx_path is None:
                raise FileNotFoundError("photometry index csv not found")
            idx = pd.read_csv(idx_path)
            if "path" not in idx.columns:
                for cand in ("phot_tsv", "tsv", "out", "output"):
                    if cand in idx.columns:
                        idx = idx.rename(columns={cand: "path"})
                        break
            if "file" not in idx.columns:
                for cand in ("fname", "frame", "image", "fits", "name"):
                    if cand in idx.columns:
                        idx = idx.rename(columns={cand: "file"})
                        break
            if "filter" in idx.columns:
                idx["filter"] = idx["filter"].astype(str).str.strip().str.lower()
            elif "FILTER" in idx.columns:
                idx["filter"] = idx["FILTER"].astype(str).str.strip().str.lower()
            else:
                idx["filter"] = ""

            rows = []
            total = len(idx)
            for i, r in idx.iterrows():
                if self._stop_requested:
                    self.finished.emit({"stopped": True})
                    return
                p = r.get("path", "")
                p = str(p) if p is not None else ""
                if p.strip() == "":
                    continue
                tsv = step9_dir(self.result_dir) / p
                if not tsv.exists():
                    tsv = self.result_dir / p
                if not tsv.exists():
                    continue
                dfp = pd.read_csv(tsv, sep="\t")
                mag_col = None
                for cand in ("mag_inst", "mag", "mag_ap", "mag_apcorr"):
                    if cand in dfp.columns:
                        mag_col = cand
                        break
                if mag_col is None:
                    continue
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
                tmp["file"] = str(r.get("file", ""))
                rows.append(tmp)
                self.progress.emit(i + 1, total, str(r.get("file", "")))

            if not rows:
                raise RuntimeError("No photometry data found")
            all_df = pd.concat(rows, ignore_index=True)
            all_df["FILTER"] = all_df["FILTER"].astype(str).str.strip().str.lower()

            grp = (
                all_df.groupby(["ID", "FILTER"])
                .agg(
                    mag_inst_med=("mag_inst", lambda s: self._robust_median_and_err(s)[0]),
                    n_frames=("mag_inst", lambda s: self._robust_median_and_err(s)[2]),
                    snr_med=("snr", lambda s: float(np.nanmedian(np.asarray(s, float))) if np.isfinite(np.asarray(s, float)).any() else np.nan),
                )
                .reset_index()
            )

            wide_mag = grp.pivot_table(index="ID", columns="FILTER", values="mag_inst_med", aggfunc="median")
            wide_snr = grp.pivot_table(index="ID", columns="FILTER", values="snr_med", aggfunc="median")
            wide_mag.columns = [f"mag_inst_{c}" for c in wide_mag.columns]
            wide_snr.columns = [f"snr_{c}" for c in wide_snr.columns]
            wide = pd.concat([wide_mag, wide_snr], axis=1).reset_index()

            master_path = step6_dir(result_dir) / "ref_catalog.tsv"
            if not master_path.exists():
                master_path = legacy_step5_refbuild_dir(result_dir) / "ref_catalog.tsv"
            if not master_path.exists():
                master_path = legacy_step7_refbuild_dir(result_dir) / "master_catalog.tsv"
            if not master_path.exists():
                master_path = result_dir / "master_catalog.tsv"
            if not master_path.exists():
                raise FileNotFoundError("master_catalog.tsv missing")
            master = pd.read_csv(master_path, sep="\t")
            if "ID" not in master.columns:
                raise RuntimeError("master_catalog.tsv missing ID column")

            merge_cols = ["ID"]
            if "source_id" in master.columns:
                merge_cols.append("source_id")
            for col in ("gaia_G", "gaia_BP", "gaia_RP", "gmag", "bpmag", "rpmag", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"):
                if col in master.columns and col not in merge_cols:
                    merge_cols.append(col)
            df = wide.merge(master[merge_cols], on="ID", how="left")

            g_col = None
            bp_col = None
            rp_col = None
            for cand in ("gaia_G", "gmag", "phot_g_mean_mag"):
                if cand in df.columns:
                    g_col = cand
                    break
            for cand in ("gaia_BP", "bpmag", "phot_bp_mean_mag"):
                if cand in df.columns:
                    bp_col = cand
                    break
            for cand in ("gaia_RP", "rpmag", "phot_rp_mean_mag"):
                if cand in df.columns:
                    rp_col = cand
                    break

            if g_col is None or bp_col is None or rp_col is None:
                if "source_id" not in df.columns:
                    raise RuntimeError("Gaia mags not available and source_id missing")
                gaia_path = step5_dir(result_dir) / "gaia_fov.ecsv"
                if not gaia_path.exists():
                    gaia_path = legacy_step7_wcs_dir(result_dir) / "gaia_fov.ecsv"
                if not gaia_path.exists():
                    gaia_path = result_dir / "gaia_fov.ecsv"
                if not gaia_path.exists():
                    raise RuntimeError("gaia_fov.ecsv not found")
                t_gaia = Table.read(gaia_path, format="ascii.ecsv")
                gaia_df = t_gaia.to_pandas()
                gaia_df["source_id"] = gaia_df["source_id"].astype("int64")
                gaia_cols = ["source_id", "phot_g_mean_mag"]
                if "phot_bp_mean_mag" in gaia_df.columns:
                    gaia_cols.append("phot_bp_mean_mag")
                if "phot_rp_mean_mag" in gaia_df.columns:
                    gaia_cols.append("phot_rp_mean_mag")
                df = df.merge(gaia_df[gaia_cols], on="source_id", how="left")
                g_col = "phot_g_mean_mag"
                bp_col = "phot_bp_mean_mag"
                rp_col = "phot_rp_mean_mag"

            df["gaia_G"] = pd.to_numeric(df[g_col], errors="coerce")
            df["gaia_BP"] = pd.to_numeric(df[bp_col], errors="coerce")
            df["gaia_RP"] = pd.to_numeric(df[rp_col], errors="coerce")
            dfm = df[np.isfinite(df["gaia_G"]) & np.isfinite(df["gaia_BP"]) & np.isfinite(df["gaia_RP"])].copy()
            dfm["gaia_BP_RP"] = dfm["gaia_BP"] - dfm["gaia_RP"]

            out_cal = dfm.copy()
            xcol = out_cal["gaia_BP_RP"].to_numpy(float)
            G = out_cal["gaia_G"].to_numpy(float)

            m_g = np.isfinite(xcol) & (xcol >= 0.3) & (xcol <= 3.0)
            m_r = np.isfinite(xcol) & (xcol >= 0.0) & (xcol <= 3.0)
            m_i = np.isfinite(xcol) & (xcol >= 0.5) & (xcol <= 2.0)

            G_minus_g = np.full_like(G, np.nan)
            G_minus_r = np.full_like(G, np.nan)
            G_minus_i = np.full_like(G, np.nan)

            G_minus_g[m_g] = self._poly_eval(xcol[m_g], [0.2199, -0.6365, -0.1548, 0.0064])
            G_minus_r[m_r] = self._poly_eval(xcol[m_r], [-0.09837, 0.08592, 0.1907, -0.1701, 0.02263])
            G_minus_i[m_i] = self._poly_eval(xcol[m_i], [-0.293, 0.6404, -0.09609, -0.002104])

            out_cal["sdss_g_ref"] = G - G_minus_g
            out_cal["sdss_r_ref"] = G - G_minus_r
            out_cal["sdss_i_ref"] = G - G_minus_i

            g_inst = out_cal.get("mag_inst_g", pd.Series(np.full(len(out_cal), np.nan))).to_numpy(float)
            r_inst = out_cal.get("mag_inst_r", pd.Series(np.full(len(out_cal), np.nan))).to_numpy(float)
            i_inst = out_cal.get("mag_inst_i", pd.Series(np.full(len(out_cal), np.nan))).to_numpy(float)

            color_gr = g_inst - r_inst
            color_ri = r_inst - i_inst

            clip_sigma = float(getattr(P, "zp_clip_sigma", 3.0))
            fit_iters = int(getattr(P, "zp_fit_iters", 5))
            min_match = int(getattr(P, "min_master_gaia_matches", 10))

            delta_g = out_cal["sdss_g_ref"].to_numpy(float) - g_inst
            mg = np.isfinite(delta_g) & np.isfinite(color_gr) & np.isfinite(g_inst)
            ct_g = self._robust_linfit(color_gr[mg], delta_g[mg], clip_sigma=clip_sigma, iters=fit_iters, min_n=min_match)[0]
            if not np.isfinite(ct_g):
                ct_g = 0.0

            delta_r = out_cal["sdss_r_ref"].to_numpy(float) - r_inst
            mr = np.isfinite(delta_r) & np.isfinite(color_gr) & np.isfinite(r_inst)
            ct_r = self._robust_linfit(color_gr[mr], delta_r[mr], clip_sigma=clip_sigma, iters=fit_iters, min_n=min_match)[0]
            if not np.isfinite(ct_r):
                ct_r = 0.0

            delta_i = out_cal["sdss_i_ref"].to_numpy(float) - i_inst
            mi = np.isfinite(delta_i) & np.isfinite(color_ri) & np.isfinite(i_inst)
            ct_i = self._robust_linfit(color_ri[mi], delta_i[mi], clip_sigma=clip_sigma, iters=fit_iters, min_n=min_match)[0]
            if not np.isfinite(ct_i):
                ct_i = 0.0

            out_cal["color_gr"] = color_gr
            out_cal["color_ri"] = color_ri

            frame_airmass = self._build_frame_airmass(idx)
            if frame_airmass is None or frame_airmass.empty:
                raise RuntimeError("frame_airmass.csv missing and airmass computation failed")

            cal_cols = ["ID", "sdss_g_ref", "sdss_r_ref", "sdss_i_ref", "color_gr", "color_ri", "gaia_BP_RP"]
            obs = all_df.merge(out_cal[cal_cols], on="ID", how="left")
            # date 컬럼도 함께 merge
            merge_cols = ["file", "filter", "airmass"]
            if "date" in frame_airmass.columns:
                merge_cols.append("date")
            obs = obs.merge(frame_airmass[merge_cols], left_on=["file", "FILTER"], right_on=["file", "filter"], how="left")
            if "date" not in obs.columns:
                obs["date"] = "unknown"

            obs["ref_mag"] = np.nan
            obs.loc[obs["FILTER"] == "g", "ref_mag"] = obs.loc[obs["FILTER"] == "g", "sdss_g_ref"]
            obs.loc[obs["FILTER"] == "r", "ref_mag"] = obs.loc[obs["FILTER"] == "r", "sdss_r_ref"]
            obs.loc[obs["FILTER"] == "i", "ref_mag"] = obs.loc[obs["FILTER"] == "i", "sdss_i_ref"]

            obs["color_term"] = np.nan
            obs.loc[obs["FILTER"] == "g", "color_term"] = ct_g * obs.loc[obs["FILTER"] == "g", "color_gr"]
            obs.loc[obs["FILTER"] == "r", "color_term"] = ct_r * obs.loc[obs["FILTER"] == "r", "color_gr"]
            obs.loc[obs["FILTER"] == "i", "color_term"] = ct_i * obs.loc[obs["FILTER"] == "i", "color_ri"]

            obs["delta"] = obs["ref_mag"] - (obs["mag_inst"] + obs["color_term"])
            obs["cal_ok"] = False
            bp = obs["gaia_BP_RP"].to_numpy(float)
            obs.loc[(obs["FILTER"] == "g") & np.isfinite(bp) & (bp >= 0.3) & (bp <= 3.0), "cal_ok"] = True
            obs.loc[(obs["FILTER"] == "r") & np.isfinite(bp) & (bp >= 0.0) & (bp <= 3.0), "cal_ok"] = True
            obs.loc[(obs["FILTER"] == "i") & np.isfinite(bp) & (bp >= 0.5) & (bp <= 2.0), "cal_ok"] = True

            snr_cut = float(getattr(P, "cmd_snr_calib_min", 20.0))
            obs["snr_ok"] = True
            if "snr" in obs.columns:
                svals = obs["snr"].to_numpy(float)
                obs["snr_ok"] = np.isfinite(svals) & (svals >= snr_cut)

            filters_seen = sorted(set(all_df["FILTER"].dropna().astype(str)))
            out_dir = result_dir / "extinction"
            out_dir.mkdir(parents=True, exist_ok=True)
            step11_out = step11_extinction_dir(result_dir)
            step11_out.mkdir(parents=True, exist_ok=True)
            stats_rows = []
            for filt in filters_seen:
                fmask = obs["FILTER"] == filt
                n_total = int(fmask.sum())
                n_ref = int(np.isfinite(obs.loc[fmask, "ref_mag"]).sum())
                n_air = int(np.isfinite(obs.loc[fmask, "airmass"]).sum())
                n_color = int(np.isfinite(obs.loc[fmask, "color_term"]).sum())
                n_snr = int(obs.loc[fmask, "snr_ok"].sum()) if "snr_ok" in obs.columns else n_total
                n_cal = int(obs.loc[fmask, "cal_ok"].sum())
                n_delta = int(np.isfinite(obs.loc[fmask, "delta"]).sum())
                self._log(
                    f"Filter[{filt}] total={n_total}, ref={n_ref}, color={n_color}, "
                    f"airmass={n_air}, snr_ok={n_snr}, cal_ok={n_cal}, delta_ok={n_delta}"
                )
                stats_rows.append({
                    "filter": filt,
                    "n_total": n_total,
                    "n_ref": n_ref,
                    "n_color": n_color,
                    "n_airmass": n_air,
                    "n_snr_ok": n_snr,
                    "n_cal_ok": n_cal,
                    "n_delta_ok": n_delta,
                })

            if stats_rows:
                stats_df = pd.DataFrame(stats_rows)
                stats_path = out_dir / "extinction_fit_filter_stats.csv"
                stats_df.to_csv(stats_path, index=False)
                self._log(f"Saved {stats_path.name} | rows={len(stats_df)}")

            obs = obs[np.isfinite(obs["delta"]) & np.isfinite(obs["airmass"]) & obs["cal_ok"] & obs["snr_ok"]].copy()

            # 날짜별 분류 통계
            dates_seen = sorted(set(obs["date"].dropna().astype(str)))
            self._log(f"Dates found: {dates_seen}")
            if len(obs):
                for (date_val, filt), sub in obs.groupby(["date", "FILTER"]):
                    self._log(f"Fit candidates [{date_val}][{filt}]: {len(sub)} points")

            fit_rows = []
            point_rows = []
            use_quadratic = bool(getattr(P, "extinction_use_quadratic", True))
            use_color_extinction = bool(getattr(P, "extinction_use_color_dependent", True))
            min_quad = int(getattr(P, "extinction_min_points_quadratic", 20))
            min_color = int(getattr(P, "extinction_min_points_color", 30))

            # 날짜별 + 필터별 피팅
            for (date_val, filt), sub in obs.groupby(["date", "FILTER"]):
                x = sub["airmass"].to_numpy(float)
                y = sub["delta"].to_numpy(float)

                # 색지수 선택: g/r 필터면 g-r, i 필터면 r-i
                if filt in ("g", "r"):
                    color_col = "color_gr"
                elif filt == "i":
                    color_col = "color_ri"
                else:
                    color_col = "color_gr"  # 기본값

                color = sub[color_col].to_numpy(float) if color_col in sub.columns else np.full(len(x), np.nan)

                # 1차 피팅 (기본)
                k1_lin, zp_lin, scatter_lin, n_ref_lin, out_frac_lin = self._robust_linfit(
                    x, y, clip_sigma=clip_sigma, iters=fit_iters, min_n=min_match
                )

                # 색 의존 소광 피팅: delta = k' * X + k'' * C * X
                k_prime, k_color, zp_col, scatter_col, n_ref_col, out_frac_col, k_col_sig = (
                    np.nan, np.nan, np.nan, np.nan, 0, np.nan, False
                )
                n_valid_color = np.sum(np.isfinite(x) & np.isfinite(color) & np.isfinite(y))
                if use_color_extinction and n_valid_color >= min_color:
                    k_prime, k_color, zp_col, scatter_col, n_ref_col, out_frac_col, k_col_sig = self._robust_color_extinction_fit(
                        x, color, y, clip_sigma=clip_sigma, iters=fit_iters, min_n=min_color
                    )
                    if k_col_sig and np.isfinite(k_color):
                        self._log(f"[{date_val}][{filt}] color extinction fit: k'={k_prime:.4f}, k''={k_color:.4f} (n={n_ref_col})")

                # 2차 피팅 시도 (에어매스 비선형)
                k1_q, k2_q, zp_q, scatter_q, n_ref_q, out_frac_q, k2_sig = (
                    np.nan, np.nan, np.nan, np.nan, 0, np.nan, False
                )
                if use_quadratic and len(x) >= min_quad:
                    k1_q, k2_q, zp_q, scatter_q, n_ref_q, out_frac_q, k2_sig = self._robust_quadfit(
                        x, y, clip_sigma=clip_sigma, iters=fit_iters, min_n=min_quad
                    )

                # 최선의 피팅 선택
                # 우선순위: 색 의존 피팅 > 2차 피팅 > 1차 피팅
                k1, k2, k_color_final, zp = k1_lin, 0.0, 0.0, zp_lin
                scatter, n_ref, out_frac = scatter_lin, n_ref_lin, out_frac_lin
                fit_order = 1

                # 색 의존 피팅이 유의미하면 사용
                if k_col_sig and np.isfinite(k_color) and scatter_col < scatter_lin * 0.90:
                    k1, k2, k_color_final, zp = k_prime, 0.0, k_color, zp_col
                    scatter, n_ref, out_frac = scatter_col, n_ref_col, out_frac_col
                    fit_order = 3  # color-dependent
                    yhat = zp + k1 * x + k_color_final * color * x
                    self._log(f"[{date_val}][{filt}] using COLOR-DEPENDENT fit: k'={k1:.4f}, k''={k_color_final:.4f}")
                # 2차 피팅이 유의미하면 사용
                elif k2_sig and np.isfinite(k2_q) and scatter_q < scatter_lin * 0.95:
                    k1, k2, zp = k1_q, k2_q, zp_q
                    scatter, n_ref, out_frac = scatter_q, n_ref_q, out_frac_q
                    fit_order = 2
                    yhat = zp + k1 * x + k2 * x * x
                    self._log(f"[{date_val}][{filt}] using quadratic fit (k2={k2:.4f})")
                else:
                    yhat = zp + k1 * x

                if not np.isfinite(k1):
                    self._log(f"[{date_val}][{filt}] FAILED: insufficient points (n={len(x)})")
                    continue

                resid = y - yhat
                fit_rows.append({
                    "date": date_val,
                    "filter": filt,
                    "k1": k1,
                    "k2": k2,
                    "k_color": k_color_final,  # 색 의존 소광 계수 k''
                    "zp": zp,
                    "scatter": scatter,
                    "n_ref": n_ref,
                    "outlier_fraction": out_frac,
                    "fit_order": fit_order,
                })
                for xi, yi, ri, ci in zip(x, y, resid, color):
                    point_rows.append({"date": date_val, "filter": filt, "airmass": xi, "delta": yi, "resid": ri, "color": ci})

            fit_df = pd.DataFrame(fit_rows)
            if fit_df.empty:
                raise RuntimeError("No valid extinction fits produced")

            # Step 11 전용 폴더에 저장
            fit_path = step11_out / "step11_extinction_fit_by_filter.csv"
            fit_df.to_csv(fit_path, index=False)
            self._log(f"Saved {fit_path.name} | rows={len(fit_df)}")

            # 기존 경로에도 저장 (호환성)
            fit_path_old = out_dir / "extinction_fit_by_filter.csv"
            fit_df.to_csv(fit_path_old, index=False)

            for _, row in fit_df.iterrows():
                k2_str = f", k2={row['k2']:.4f}" if row.get('k2', 0) != 0 else ""
                k_color_str = f", k''={row['k_color']:.4f}" if row.get('k_color', 0) != 0 else ""
                fit_type = {1: "linear", 2: "quadratic", 3: "color-dep"}.get(row.get('fit_order', 1), "linear")
                date_str = row.get('date', 'unknown')
                self._log(
                    f"[{date_str}][{row['filter']}] k'={row['k1']:.4f}{k2_str}{k_color_str}, zp={row['zp']:.4f}, "
                    f"scatter={row['scatter']:.4f}, n={int(row['n_ref'])}, type={fit_type}"
                )

            points_df = pd.DataFrame(point_rows)
            points_path = step11_out / "step11_extinction_fit_points.csv"
            points_df.to_csv(points_path, index=False)
            self._log(f"Saved {points_path.name} | rows={len(points_df)}")

            self.finished.emit({"fit": fit_df, "points": points_df})
        except Exception as e:
            self.error.emit(str(e))


class ExtinctionFitWindow(QWidget):
    """Extinction (Airmass Fit) tool window."""

    def __init__(self, params, data_dir: Path, result_dir: Path, parent=None):
        super().__init__(parent)
        self.params = params
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.worker = None

        self.setWindowTitle("Extinction (Airmass Fit)")
        self.resize(900, 700)

        layout = QVBoxLayout(self)

        info = QLabel("Fit per-filter extinction coefficient k using inst mags and frame airmass.")
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 8px; border-radius: 5px; }")
        layout.addWidget(info)

        control_layout = QHBoxLayout()
        self.btn_run = QPushButton("Run Extinction Fit")
        self.btn_run.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 6px 12px; }")
        self.btn_run.clicked.connect(self.run_fit)
        control_layout.addWidget(self.btn_run)

        self.btn_save = QPushButton("Save Plots")
        self.btn_save.setStyleSheet("QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 6px 12px; }")
        self.btn_save.clicked.connect(self.save_plots)
        self.btn_save.setEnabled(False)
        control_layout.addWidget(self.btn_save)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 6px 12px; }")
        self.btn_stop.clicked.connect(self.stop_fit)
        self.btn_stop.setEnabled(False)
        control_layout.addWidget(self.btn_stop)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        # 날짜/필터 선택 UI
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Date:"))
        self.date_combo = QComboBox()
        self.date_combo.addItem("All Dates")
        self.date_combo.currentIndexChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.date_combo)

        filter_layout.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All Filters")
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_combo)

        self.chk_show_fit = QCheckBox("Show Fit Lines")
        self.chk_show_fit.setChecked(True)
        self.chk_show_fit.stateChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.chk_show_fit)

        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        plot_group = QGroupBox("Fit Diagnostics")
        plot_layout = QVBoxLayout(plot_group)
        self.figure = Figure(figsize=(7, 5))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(NavigationToolbar(self.canvas, self))
        plot_layout.addWidget(self.canvas)
        layout.addWidget(plot_group)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

        self.points_df = None
        self.fit_df = None

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def run_fit(self):
        if self.worker and self.worker.isRunning():
            return
        self.log_text.clear()
        self.figure.clear()
        self.canvas.draw()
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False)

        self.worker = ExtinctionFitWorker(self.params, self.data_dir, self.result_dir)
        self.worker.progress.connect(lambda c, t, f: self.log(f"{c}/{t} | {f}"))
        self.worker.log.connect(self.log)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def stop_fit(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log("Stop requested")

    def on_finished(self, results):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.fit_df = results.get("fit")
        self.points_df = results.get("points")
        if isinstance(self.fit_df, pd.DataFrame) and isinstance(self.points_df, pd.DataFrame):
            # 콤보박스 채우기
            self._populate_combos()
            self._plot_results()
            self.btn_save.setEnabled(True)
        self.log("Extinction fit complete")

    def _populate_combos(self):
        """날짜/필터 콤보박스 채우기"""
        # 날짜 콤보
        self.date_combo.blockSignals(True)
        self.date_combo.clear()
        self.date_combo.addItem("All Dates")
        if self.fit_df is not None and "date" in self.fit_df.columns:
            dates = sorted(set(self.fit_df["date"].dropna().astype(str)))
            for d in dates:
                self.date_combo.addItem(d)
        self.date_combo.blockSignals(False)

        # 필터 콤보
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        self.filter_combo.addItem("All Filters")
        if self.fit_df is not None and "filter" in self.fit_df.columns:
            filters = sorted(set(self.fit_df["filter"].dropna().astype(str)))
            for f in filters:
                self.filter_combo.addItem(f)
        self.filter_combo.blockSignals(False)

    def _on_filter_changed(self):
        """필터/날짜 변경 시 플롯 갱신"""
        self._plot_results()

    def on_error(self, message: str):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        QMessageBox.critical(self, "Error", f"Extinction fit failed:\n{message}")

    def _plot_results(self):
        self.figure.clear()
        ax1, ax2 = self.figure.subplots(2, 1, figsize=(8, 6))

        if self.fit_df is None or self.fit_df.empty:
            self.canvas.draw()
            return

        # 선택된 날짜/필터 가져오기
        sel_date = self.date_combo.currentText()
        sel_filter = self.filter_combo.currentText()
        show_fit = self.chk_show_fit.isChecked()

        # 데이터 필터링
        fit_df = self.fit_df.copy()
        points_df = self.points_df.copy() if self.points_df is not None else pd.DataFrame()

        if sel_date != "All Dates" and "date" in fit_df.columns:
            fit_df = fit_df[fit_df["date"] == sel_date]
            if not points_df.empty and "date" in points_df.columns:
                points_df = points_df[points_df["date"] == sel_date]

        if sel_filter != "All Filters" and "filter" in fit_df.columns:
            fit_df = fit_df[fit_df["filter"] == sel_filter]
            if not points_df.empty and "filter" in points_df.columns:
                points_df = points_df[points_df["filter"] == sel_filter]

        x_min = None
        x_max = None
        if not points_df.empty:
            x_min = float(points_df["airmass"].min())
            x_max = float(points_df["airmass"].max())

        # 색상: 날짜별로 구분 (같은 필터라도 날짜가 다르면 다른 색)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        markers = ["o", "s", "^", "v", "D", "<", ">", "p", "h", "*"]

        for i, (_, row) in enumerate(fit_df.iterrows()):
            filt = str(row["filter"])
            date_val = str(row.get("date", "unknown"))
            k1 = float(row.get("k1", row.get("k", 0)))
            k2 = float(row.get("k2", 0)) if "k2" in row else 0.0
            zp = float(row["zp"])
            fit_order = int(row.get("fit_order", 1)) if "fit_order" in row else (2 if k2 != 0 else 1)
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            # 해당 날짜+필터의 포인트 데이터
            if not points_df.empty:
                mask = points_df["filter"] == filt
                if "date" in points_df.columns:
                    mask &= points_df["date"] == date_val
                sub = points_df[mask]
            else:
                sub = pd.DataFrame(columns=["airmass", "delta"])

            label_prefix = f"{date_val}/{filt}" if sel_date == "All Dates" else filt
            if len(sub):
                ax1.scatter(sub["airmass"], sub["delta"], s=15, alpha=0.6, color=color, marker=marker, label=f"{label_prefix} data")
                x = np.linspace(sub["airmass"].min(), sub["airmass"].max(), 100)
            elif x_min is not None and x_max is not None:
                x = np.linspace(x_min, x_max, 100)
            else:
                x = np.linspace(1.0, 2.5, 100)

            if show_fit:
                if fit_order == 2 and k2 != 0:
                    y_fit = zp + k1 * x + k2 * x * x
                    label = f"{label_prefix}: k={k1:.3f}, k2={k2:.4f}"
                else:
                    y_fit = zp + k1 * x
                    label = f"{label_prefix}: k={k1:.3f}"
                ax1.plot(x, y_fit, lw=2, color=color, linestyle="-", label=label)

        ax1.set_xlabel("Airmass (X)", fontsize=10)
        ax1.set_ylabel("Δ = m_ref - m_inst (mag)", fontsize=10)
        title = "Extinction Fit"
        if sel_date != "All Dates":
            title += f" [{sel_date}]"
        if sel_filter != "All Filters":
            title += f" [{sel_filter}]"
        ax1.set_title(title, fontsize=11, fontweight="bold")
        if len(fit_df) > 0:
            ax1.legend(loc="best", fontsize=7, ncol=min(3, len(fit_df)))
        ax1.grid(True, alpha=0.3)

        # 히스토그램도 필터링된 데이터 사용
        if not points_df.empty:
            if sel_date == "All Dates":
                # 날짜+필터별로 그룹화
                grp_cols = ["date", "filter"] if "date" in points_df.columns else ["filter"]
            else:
                grp_cols = ["filter"]
            for i, (key, sub) in enumerate(points_df.groupby(grp_cols)):
                color = colors[i % len(colors)]
                resid = sub["resid"].dropna()
                if len(resid) > 0:
                    if isinstance(key, tuple):
                        lbl = "/".join(str(k) for k in key)
                    else:
                        lbl = str(key)
                    ax2.hist(resid, bins=30, alpha=0.5, color=color, label=f"{lbl} (σ={resid.std():.3f})")
        ax2.set_xlabel("Residual (mag)", fontsize=10)
        ax2.set_ylabel("Count", fontsize=10)
        ax2.set_title("Residual Histogram", fontsize=11, fontweight="bold")
        ax2.legend(loc="best", fontsize=7)
        ax2.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def save_plots(self):
        try:
            # Step 11 전용 폴더에 저장
            step11_out = step11_extinction_dir(self.result_dir)
            step11_out.mkdir(parents=True, exist_ok=True)
            out = step11_out / "step11_extinction_fit_plot.png"
            self.figure.savefig(out, dpi=150, bbox_inches="tight")
            self.log(f"Saved {out}")

            # 기존 경로에도 저장 (호환성)
            out_dir = self.result_dir / "extinction"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_old = out_dir / "extinction_fit_plot.png"
            self.figure.savefig(out_old, dpi=150, bbox_inches="tight")
            self.log(f"Saved {out_old}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save plot:\n{e}")
