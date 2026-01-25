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
    QGroupBox, QMessageBox
)

from ...utils.astro_utils import compute_airmass_from_header
from ...utils.step_paths import (
    step2_cropped_dir,
    step5_dir,
    step7_dir,
    step9_dir,
    step11_dir,
    step11_extinction_dir,
    crop_is_active,
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

    def run(self):
        try:
            P = self.params.P
            result_dir = self.result_dir
            phot_dir = step9_dir(result_dir)
            output_dir = step11_extinction_dir(result_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            idx_candidates = [
                phot_dir / "photometry_index.csv",
                phot_dir / "phot_index.csv",
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
                tsv = phot_dir / p
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

            master_path = step7_dir(result_dir) / "master_catalog.tsv"
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
            obs = obs.merge(frame_airmass[["file", "filter", "airmass"]], left_on=["file", "FILTER"], right_on=["file", "filter"], how="left")

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
            out_dir = output_dir
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

            if len(obs):
                for filt, sub in obs.groupby("FILTER"):
                    self._log(f"Fit candidates [{filt}]: {len(sub)} points")

            fit_rows = []
            point_rows = []
            for filt, sub in obs.groupby("FILTER"):
                x = sub["airmass"].to_numpy(float)
                y = sub["delta"].to_numpy(float)
                k, zp, scatter, n_ref, out_frac = self._robust_linfit(x, y, clip_sigma=clip_sigma, iters=fit_iters, min_n=min_match)
                if not np.isfinite(k):
                    continue
                yhat = zp + k * x
                resid = y - yhat
                fit_rows.append({
                    "filter": filt,
                    "k": k,
                    "zp": zp,
                    "scatter": scatter,
                    "n_ref": n_ref,
                    "outlier_fraction": out_frac,
                })
                for xi, yi, ri in zip(x, y, resid):
                    point_rows.append({"filter": filt, "airmass": xi, "delta": yi, "resid": ri})

            fit_df = pd.DataFrame(fit_rows)
            if fit_df.empty:
                raise RuntimeError("No valid extinction fits produced")

            fit_path = out_dir / "extinction_fit_by_filter.csv"
            fit_df.to_csv(fit_path, index=False)
            self._log(f"Saved {fit_path.name} | rows={len(fit_df)}")
            for _, row in fit_df.iterrows():
                self._log(
                    f"k[{row['filter']}]={row['k']:.4f}, zp={row['zp']:.4f}, "
                    f"scatter={row['scatter']:.4f}, n_ref={int(row['n_ref'])}, outlier={row['outlier_fraction']:.2f}"
                )
            for filt in filters_seen:
                if filt not in set(fit_df["filter"].astype(str)):
                    self._log(f"No fit for filter [{filt}]: insufficient valid points after cuts")

            points_df = pd.DataFrame(point_rows)
            points_path = out_dir / "extinction_fit_points.csv"
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
            self._plot_results()
            self.btn_save.setEnabled(True)
        self.log("Extinction fit complete")

    def on_error(self, message: str):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        QMessageBox.critical(self, "Error", f"Extinction fit failed:\n{message}")

    def _plot_results(self):
        self.figure.clear()
        ax1, ax2 = self.figure.subplots(2, 1)

        if self.fit_df is None or self.fit_df.empty:
            return

        x_min = None
        x_max = None
        if self.points_df is not None and not self.points_df.empty:
            x_min = float(self.points_df["airmass"].min())
            x_max = float(self.points_df["airmass"].max())

        for _, row in self.fit_df.iterrows():
            filt = str(row["filter"])
            k = float(row["k"])
            zp = float(row["zp"])
            if self.points_df is not None and not self.points_df.empty:
                sub = self.points_df[self.points_df["filter"] == filt]
            else:
                sub = pd.DataFrame(columns=["airmass", "delta"])
            if len(sub):
                ax1.scatter(sub["airmass"], sub["delta"], s=12, alpha=0.5, label=filt)
                x = np.linspace(sub["airmass"].min(), sub["airmass"].max(), 50)
            elif x_min is not None and x_max is not None:
                x = np.linspace(x_min, x_max, 50)
            else:
                x = np.linspace(1.0, 2.5, 50)
            ax1.plot(x, zp + k * x, lw=1.5, label=f"{filt} fit")

        ax1.set_xlabel("Airmass (X)")
        ax1.set_ylabel("Δ = m_ref - m_inst")
        ax1.set_title("Extinction Fit (per filter)")
        ax1.legend()

        if self.points_df is not None and not self.points_df.empty:
            for filt, sub in self.points_df.groupby("filter"):
                ax2.hist(sub["resid"].dropna(), bins=30, alpha=0.5, label=filt)
        ax2.set_xlabel("Residual (mag)")
        ax2.set_ylabel("Count")
        ax2.set_title("Residual Histogram")
        ax2.legend()

        self.figure.tight_layout()
        self.canvas.draw()

    def save_plots(self):
        try:
            out_dir = step11_extinction_dir(self.result_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / "extinction_fit_plot.png"
            self.figure.savefig(out, dpi=150, bbox_inches="tight")
            self.log(f"Saved {out}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save plot:\n{e}")
