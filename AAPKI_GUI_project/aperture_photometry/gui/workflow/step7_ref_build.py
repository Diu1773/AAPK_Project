"""
Step 7: REF Build (master catalog + ID mapping)
Ported from AAPKI_GUI.ipynb Cell 11.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QMessageBox,
    QTextEdit, QDialog, QFormLayout, QDialogButtonBox, QProgressBar,
    QCheckBox, QSpinBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QWidget, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from .step_window_base import StepWindowBase
from ...utils.step_paths import step2_cropped_dir, step5_dir, step6_dir, step7_dir, step8_dir, crop_is_active


def _is_up_to_date(out_path: Path, deps):
    try:
        t_out = out_path.stat().st_mtime
    except Exception:
        return False
    for d in deps:
        try:
            if Path(d).stat().st_mtime > t_out:
                return False
        except Exception:
            return False
    return True


class RefBuildWorker(QThread):
    """Worker thread for REF build"""
    progress = pyqtSignal(int, int, str)
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

    def run(self):
        try:
            P = self.params.P
            result_dir = self.result_dir
            output_dir = step7_dir(result_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            cache_dir = self.cache_dir
            idmatch_dir = cache_dir / "idmatch"
            idmatch_dir.mkdir(parents=True, exist_ok=True)

            frames = list(self.file_list)
            if not frames:
                raise RuntimeError("[REF] frames empty")

            resume = bool(getattr(P, "resume_mode", True))
            force = bool(getattr(P, "force_master_build", False))
            min_frames_xy = int(getattr(P, "master_min_frames_xy", 1))
            preserve_ids = bool(getattr(P, "master_preserve_ids", True))

            master_path = output_dir / "master_catalog.tsv"
            map_path = output_dir / "sourceid_to_ID.csv"
            frame_map_tsv = output_dir / "frame_sourceid_to_ID.tsv"
            debug_json = output_dir / "master_ref_build_debug.json"
            missing_csv = output_dir / "master_missing_in_ref.csv"

            master_ids_path = step8_dir(result_dir) / "master_star_ids.csv"
            master_ids_back = step8_dir(result_dir) / "master_star_ids.orig.csv"
            if not master_ids_path.exists():
                master_ids_path = step6_dir(result_dir) / "master_star_ids.csv"
                master_ids_back = step6_dir(result_dir) / "master_star_ids.orig.csv"
            if not master_ids_path.exists():
                master_ids_path = result_dir / "master_star_ids.csv"
                master_ids_back = result_dir / "master_star_ids.orig.csv"

            deps = []
            for fn in frames:
                p = idmatch_dir / f"idmatch_{fn}.csv"
                if p.exists():
                    deps.append(p)
            if master_ids_path.exists():
                deps.append(master_ids_path)
            gaia_fov_path = step5_dir(result_dir) / "gaia_fov.ecsv"
            if not gaia_fov_path.exists():
                gaia_fov_path = result_dir / "gaia_fov.ecsv"
            if gaia_fov_path.exists():
                deps.append(gaia_fov_path)

            if resume and (not force) and master_path.exists() and _is_up_to_date(master_path, deps):
                self.finished.emit({"skipped": True, "master_path": str(master_path)})
                return

            rows = []
            per_frame_stat = []
            total = len(frames)
            for k, fn in enumerate(frames, 1):
                if self._stop_requested:
                    break
                p = idmatch_dir / f"idmatch_{fn}.csv"
                if not p.exists() or p.stat().st_size == 0:
                    per_frame_stat.append(dict(file=fn, n_matched=0, ok=False, reason="missing_idmatch_csv"))
                    self.progress.emit(k, total, fn)
                    continue
                try:
                    df = pd.read_csv(p)
                except Exception:
                    per_frame_stat.append(dict(file=fn, n_matched=0, ok=False, reason="read_fail"))
                    self.progress.emit(k, total, fn)
                    continue

                need = {"source_id", "x", "y"}
                if (not need.issubset(df.columns)) or (len(df) == 0):
                    per_frame_stat.append(dict(file=fn, n_matched=0, ok=False, reason="no_rows_or_missing_cols"))
                    self.progress.emit(k, total, fn)
                    continue

                df = df.copy()
                df["file"] = fn
                df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce").astype("Int64")
                df["x"] = pd.to_numeric(df["x"], errors="coerce")
                df["y"] = pd.to_numeric(df["y"], errors="coerce")
                df["sep_arcsec"] = pd.to_numeric(df["sep_arcsec"], errors="coerce") if "sep_arcsec" in df.columns else np.nan
                if "gaia_G" not in df.columns and "gmag" in df.columns:
                    df["gaia_G"] = df["gmag"]
                if "gaia_BP" not in df.columns and "bpmag" in df.columns:
                    df["gaia_BP"] = df["bpmag"]
                if "gaia_RP" not in df.columns and "rpmag" in df.columns:
                    df["gaia_RP"] = df["rpmag"]
                for col in ("gaia_G", "gaia_BP", "gaia_RP"):
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                if "ra_deg" in df.columns and "dec_deg" in df.columns:
                    df["ra_deg"] = pd.to_numeric(df["ra_deg"], errors="coerce")
                    df["dec_deg"] = pd.to_numeric(df["dec_deg"], errors="coerce")
                else:
                    df["ra_deg"] = pd.to_numeric(df.get("ra", np.nan), errors="coerce")
                    df["dec_deg"] = pd.to_numeric(df.get("dec", np.nan), errors="coerce")

                df = df[np.isfinite(df["x"]) & np.isfinite(df["y"]) & df["source_id"].notna()].copy()
                df["source_id"] = df["source_id"].astype("int64")
                df = df.sort_values("sep_arcsec").drop_duplicates(subset=["file", "source_id"], keep="first")
                keep_cols = ["file", "source_id", "x", "y", "sep_arcsec", "ra_deg", "dec_deg"]
                for col in ("gaia_G", "gaia_BP", "gaia_RP"):
                    if col in df.columns:
                        keep_cols.append(col)
                rows.append(df[keep_cols])
                per_frame_stat.append(dict(
                    file=fn,
                    n_matched=int(df["source_id"].nunique()),
                    ok=True,
                    sep_med_arcsec=float(np.nanmedian(df["sep_arcsec"].to_numpy(float))) if np.isfinite(df["sep_arcsec"]).any() else np.nan
                ))
                self.progress.emit(k, total, fn)

            all_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
                columns=["file", "source_id", "x", "y", "sep_arcsec", "ra_deg", "dec_deg"]
            )
            per_frame_stat = pd.DataFrame(per_frame_stat)
            if len(all_df) == 0:
                raise RuntimeError("[REF] idmatch 결과가 비었습니다.")

            # choose ref frame
            ref_frame = None
            ref_req = getattr(P, "ref_frame", None)
            if ref_req is not None:
                if isinstance(ref_req, (int, np.integer)):
                    i = int(ref_req)
                    i = max(0, min(i, len(frames) - 1))
                    ref_frame = frames[i]
                else:
                    s = str(ref_req).strip()
                    if s in frames:
                        ref_frame = s
                    else:
                        hit = [f for f in frames if f.endswith(s)]
                        ref_frame = hit[0] if hit else None
            if ref_frame is None:
                pf = per_frame_stat.copy()
                pf["n_matched"] = pd.to_numeric(pf.get("n_matched", 0), errors="coerce").fillna(0).astype(int)
                pf = pf.sort_values("n_matched", ascending=False)
                ref_frame = str(pf.iloc[0]["file"])
            if ref_frame not in frames:
                raise RuntimeError(f"[REF] ref_frame 결정 실패: {ref_frame}")

            # master ids
            if master_ids_path.exists():
                ms = pd.read_csv(master_ids_path)
                if "source_id" in ms.columns and len(ms):
                    master_ids = pd.to_numeric(ms["source_id"], errors="coerce").dropna().astype("int64").to_numpy()
                else:
                    master_ids = np.array(sorted(all_df["source_id"].unique()), dtype=np.int64)
            else:
                master_ids = np.array(sorted(all_df["source_id"].unique()), dtype=np.int64)
            master_ids = np.array(sorted(pd.unique(master_ids)), dtype=np.int64)

            # manual added set
            manual_added_set = set()
            if master_ids_path.exists() and master_ids_back.exists():
                try:
                    cur = pd.read_csv(master_ids_path)
                    org = pd.read_csv(master_ids_back)
                    if "source_id" in cur.columns and "source_id" in org.columns:
                        cur_set = set(pd.to_numeric(cur["source_id"], errors="coerce").dropna().astype("int64").tolist())
                        org_set = set(pd.to_numeric(org["source_id"], errors="coerce").dropna().astype("int64").tolist())
                        manual_added_set = cur_set - org_set
                except Exception:
                    manual_added_set = set()

            # radec from best idmatch
            best = all_df.sort_values(["source_id", "sep_arcsec"]).drop_duplicates("source_id", keep="first").copy()
            sid2radec = best.set_index("source_id")[["ra_deg", "dec_deg"]]

            # radec fill from gaia_fov
            gaia_sid2radec = None
            if gaia_fov_path.exists():
                try:
                    tg = Table.read(str(gaia_fov_path), format="ascii.ecsv")
                    cols = list(tg.colnames)
                    low = [c.lower() for c in cols]
                    if low != cols:
                        if len(set(low)) != len(low):
                            raise RuntimeError("gaia_fov colname lower() collision")
                        tg.rename_columns(cols, low)
                    sid_col = "source_id" if "source_id" in tg.colnames else None
                    if sid_col is None:
                        for c in tg.colnames:
                            if c.endswith("source_id"):
                                sid_col = c
                                break
                    if sid_col and ("ra" in tg.colnames) and ("dec" in tg.colnames):
                        gdf = tg[[sid_col, "ra", "dec"]].to_pandas()
                        gdf[sid_col] = pd.to_numeric(gdf[sid_col], errors="coerce").astype("Int64")
                        gdf["ra"] = pd.to_numeric(gdf["ra"], errors="coerce")
                        gdf["dec"] = pd.to_numeric(gdf["dec"], errors="coerce")
                        gdf = gdf.dropna(subset=[sid_col, "ra", "dec"]).copy()
                        gdf[sid_col] = gdf[sid_col].astype("int64")
                        gaia_sid2radec = gdf.drop_duplicates(subset=[sid_col]).set_index(sid_col)[["ra", "dec"]]
                except Exception:
                    gaia_sid2radec = None

            # ref xy
            ref_df = all_df[all_df["file"] == ref_frame].copy()
            ref_df = ref_df.sort_values("sep_arcsec").drop_duplicates("source_id", keep="first")
            ref_xy = ref_df.set_index("source_id")[["x", "y"]]

            # ref wcs
            if self.use_cropped:
                cropped_dir = step2_cropped_dir(self.result_dir)
                if not cropped_dir.exists():
                    cropped_dir = self.result_dir / "cropped"
                ref_path = cropped_dir / ref_frame
            else:
                ref_path = self.data_dir / ref_frame
            hdr = fits.getheader(ref_path)
            w = WCS(hdr, relax=True)
            has_wcs = bool(getattr(w, "has_celestial", False))

            def _mad(a):
                a = np.asarray(a, float)
                a = a[np.isfinite(a)]
                if len(a) == 0:
                    return np.nan
                med = np.median(a)
                return float(np.median(np.abs(a - med)))

            grp = (all_df.groupby("source_id").agg(
                n_frames_xy=("file", "nunique"),
                x_mad=("x", _mad),
                y_mad=("y", _mad),
            ).reset_index())
            grp["source_id"] = grp["source_id"].astype("int64")

            mtab = pd.DataFrame({"source_id": master_ids})
            mtab = mtab.merge(grp, on="source_id", how="left")
            if min_frames_xy > 1:
                mtab = mtab[mtab["n_frames_xy"].fillna(0).astype(int) >= min_frames_xy].copy()

            mtab["ra_deg"] = mtab["source_id"].map(sid2radec["ra_deg"])
            mtab["dec_deg"] = mtab["source_id"].map(sid2radec["dec_deg"])
            mtab["radec_origin"] = "idmatch_best"

            if gaia_sid2radec is not None:
                miss = ~(np.isfinite(mtab["ra_deg"].to_numpy(float)) & np.isfinite(mtab["dec_deg"].to_numpy(float)))
                if miss.any():
                    sid_miss = mtab.loc[miss, "source_id"].astype("int64")
                    ra_fill = sid_miss.map(gaia_sid2radec["ra"])
                    dec_fill = sid_miss.map(gaia_sid2radec["dec"])
                    ok = np.isfinite(ra_fill.to_numpy(float)) & np.isfinite(dec_fill.to_numpy(float))
                    idx = mtab.index[miss].to_numpy()
                    idx_ok = idx[ok]
                    mtab.loc[idx_ok, "ra_deg"] = ra_fill.to_numpy(float)[ok]
                    mtab.loc[idx_ok, "dec_deg"] = dec_fill.to_numpy(float)[ok]
                    mtab.loc[idx_ok, "radec_origin"] = "manual_gaia_fov"

            mtab["x_ref"] = np.nan
            mtab["y_ref"] = np.nan
            mtab["xy_origin"] = "none"
            mtab["in_ref"] = False

            inref = mtab["source_id"].isin(ref_xy.index.to_numpy())
            mtab.loc[inref, "x_ref"] = mtab.loc[inref, "source_id"].map(ref_xy["x"])
            mtab.loc[inref, "y_ref"] = mtab.loc[inref, "source_id"].map(ref_xy["y"])
            mtab.loc[inref, "xy_origin"] = "ref_measured"
            mtab.loc[inref, "in_ref"] = True

            miss = ~inref
            if has_wcs and miss.any():
                ra = mtab.loc[miss, "ra_deg"].to_numpy(float)
                dec = mtab.loc[miss, "dec_deg"].to_numpy(float)
                ok = np.isfinite(ra) & np.isfinite(dec)
                if ok.any():
                    sc = SkyCoord(ra[ok] * u.deg, dec[ok] * u.deg, frame="icrs")
                    try:
                        xw, yw = w.celestial.world_to_pixel(sc)
                    except Exception:
                        xw, yw = w.all_world2pix(ra[ok], dec[ok], 0)
                    idx_miss = mtab.index[miss].to_numpy()
                    mtab.loc[idx_miss[ok], "x_ref"] = np.asarray(xw, float)
                    mtab.loc[idx_miss[ok], "y_ref"] = np.asarray(yw, float)
                    mtab.loc[idx_miss[ok], "xy_origin"] = "ref_wcs_fill"

            m_ok = np.isfinite(mtab["x_ref"].to_numpy(float)) & np.isfinite(mtab["y_ref"].to_numpy(float))
            dropped = int((~m_ok).sum())
            mtab = mtab[m_ok].copy()

            mtab["is_manual_added"] = mtab["source_id"].map(lambda s: int(s) in manual_added_set)

            old_map = None
            if preserve_ids and map_path.exists():
                try:
                    old_map = pd.read_csv(map_path)
                    if {"source_id", "ID"} <= set(old_map.columns):
                        old_map["source_id"] = pd.to_numeric(old_map["source_id"], errors="coerce").astype("Int64")
                        old_map["ID"] = pd.to_numeric(old_map["ID"], errors="coerce").astype("Int64")
                        old_map = old_map.dropna(subset=["source_id", "ID"]).copy()
                        old_map["source_id"] = old_map["source_id"].astype("int64")
                        old_map["ID"] = old_map["ID"].astype("int64")
                    else:
                        old_map = None
                except Exception:
                    old_map = None

            mtab["_old_ID"] = np.nan
            if old_map is not None and len(old_map):
                om = old_map.drop_duplicates("source_id").set_index("source_id")["ID"]
                mtab["_old_ID"] = mtab["source_id"].map(om).astype("float")

            used = set(mtab.loc[np.isfinite(mtab["_old_ID"]), "_old_ID"].astype(int).tolist())
            next_id = (max(used) + 1) if used else 1

            mtab = mtab.sort_values(["y_ref", "x_ref", "source_id"], ascending=[True, True, True]).reset_index(drop=True)
            new_ids = []
            for i in range(len(mtab)):
                v = mtab.loc[i, "_old_ID"]
                if np.isfinite(v):
                    new_ids.append(int(v))
                else:
                    while next_id in used:
                        next_id += 1
                    new_ids.append(int(next_id))
                    used.add(int(next_id))
                    next_id += 1

            mtab["ID"] = np.array(new_ids, dtype=int)
            mtab = mtab.sort_values("ID").reset_index(drop=True)

            if {"gaia_G", "gaia_BP", "gaia_RP"} & set(best.columns):
                for col in ("gaia_G", "gaia_BP", "gaia_RP"):
                    if col in best.columns:
                        mtab[col] = mtab["source_id"].map(best.set_index("source_id")[col])

            master_cols = [
                "ID", "source_id", "ra_deg", "dec_deg",
                "x_ref", "y_ref", "xy_origin", "in_ref", "n_frames_xy", "x_mad", "y_mad",
                "radec_origin", "is_manual_added"
            ]
            for col in ("gaia_G", "gaia_BP", "gaia_RP"):
                if col in mtab.columns:
                    master_cols.append(col)

            master = mtab[master_cols].copy()

            master.to_csv(master_path, sep="\t", index=False, na_rep="NaN", encoding="utf-8-sig")
            map_df = master[["source_id", "ID"]].copy()
            map_df.to_csv(map_path, index=False, encoding="utf-8-sig")

            all_dbg = all_df.merge(map_df, on="source_id", how="left")
            all_dbg = all_dbg[["file", "source_id", "ID", "x", "y", "sep_arcsec"]].copy()
            all_dbg.to_csv(frame_map_tsv, sep="\t", index=False, na_rep="NaN", encoding="utf-8-sig")

            miss_list = master[master["in_ref"] == False].copy()
            miss_list.to_csv(missing_csv, index=False, encoding="utf-8-sig")

            dbg = dict(
                ref_frame=ref_frame,
                master_min_frames_xy=min_frames_xy,
                preserve_ids=preserve_ids,
                n_master=int(len(master)),
                n_in_ref=int(master["in_ref"].sum()),
                n_missing_in_ref=int((master["in_ref"] == False).sum()),
                n_xy_ref_measured=int((master["xy_origin"] == "ref_measured").sum()),
                n_xy_ref_wcs_fill=int((master["xy_origin"] == "ref_wcs_fill").sum()),
                n_radec_from_idmatch=int((master["radec_origin"] == "idmatch_best").sum()),
                n_radec_from_gaia_fov=int((master["radec_origin"] == "manual_gaia_fov").sum()),
                n_manual_added=int(master["is_manual_added"].sum()),
                dropped_no_xy=dropped,
            )
            debug_json.write_text(json.dumps(dbg, indent=2, ensure_ascii=False), encoding="utf-8")

            # per_frame_stat is a DataFrame, convert to list of dicts for iteration
            pfs_records = per_frame_stat.to_dict('records')
            n_ok = sum(1 for r in pfs_records if r.get("ok"))
            n_missing = sum(1 for r in pfs_records if r.get("reason") == "missing_idmatch_csv")
            n_read_fail = sum(1 for r in pfs_records if r.get("reason") == "read_fail")
            n_no_match = sum(1 for r in pfs_records if r.get("reason") == "no_rows_or_missing_cols")
            self.finished.emit({
                "master_rows": int(len(master)),
                "ref_frame": ref_frame,
                "dropped": int(dropped),
                "total_frames": int(len(frames)),
                "ok_frames": int(n_ok),
                "missing_idmatch": int(n_missing),
                "read_fail": int(n_read_fail),
                "no_match": int(n_no_match),
                "n_in_ref": int(master["in_ref"].sum()),
                "n_missing_in_ref": int((master["in_ref"] == False).sum()),
            })
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self.error.emit("WORKER", error_msg)
            self.finished.emit({})


class RefBuildWindow(StepWindowBase):
    """Step 7: REF Build"""

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        self.worker = None
        self.file_list = []
        self.use_cropped = False
        self.log_window = None

        super().__init__(
            step_index=6,
            step_name="REF Build",
            params=params,
            project_state=project_state,
            main_window=main_window
        )

        self.setup_step_ui()
        self.restore_state()

    def setup_step_ui(self):
        info = QLabel("Build master_catalog.tsv and sourceid_to_ID.csv from idmatch results.")
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        self.content_layout.addWidget(info)

        ref_group = QGroupBox("Ref Frame")
        ref_layout = QHBoxLayout(ref_group)
        ref_layout.addWidget(QLabel("Ref:"))
        self.ref_combo = QComboBox()
        self.ref_combo.currentIndexChanged.connect(self.on_ref_changed)
        ref_layout.addWidget(self.ref_combo)
        self.content_layout.addWidget(ref_group)

        control_layout = QHBoxLayout()
        btn_params = QPushButton("REF Parameters")
        btn_params.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 8px 15px; }")
        btn_params.clicked.connect(self.open_parameters_dialog)
        control_layout.addWidget(btn_params)

        control_layout.addStretch()

        self.btn_run = QPushButton("Run REF Build")
        self.btn_run.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 20px; }")
        self.btn_run.clicked.connect(self.run_ref_build)
        control_layout.addWidget(self.btn_run)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px 15px; }")
        self.btn_stop.clicked.connect(self.stop_ref_build)
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

        results_group = QGroupBox("REF Summary")
        results_layout = QVBoxLayout(results_group)
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Ref Frame", "Master Rows", "Dropped"])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        results_layout.addWidget(self.results_table)
        self.content_layout.addWidget(results_group)

        self.log_window = QWidget(self, Qt.Window)
        self.log_window.setWindowTitle("REF Build Log")
        self.log_window.resize(800, 400)
        log_layout = QVBoxLayout(self.log_window)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        log_layout.addWidget(self.log_text)

        self.populate_file_list()

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def populate_file_list(self):
        cropped_dir = step2_cropped_dir(self.params.P.result_dir)
        legacy_cropped = self.params.P.result_dir / "cropped"
        crop_active = crop_is_active(self.params.P.result_dir)
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
        if hasattr(self, "ref_combo"):
            self.ref_combo.blockSignals(True)
            self.ref_combo.clear()
            self.ref_combo.addItem("Auto (best matched)")
            self.ref_combo.addItems(self.file_list)
            ref_frame = getattr(self.params.P, "ref_frame", None)
            if ref_frame in self.file_list:
                self.ref_combo.setCurrentIndex(self.file_list.index(ref_frame) + 1)
            else:
                self.ref_combo.setCurrentIndex(0)
            self.ref_combo.blockSignals(False)

    def on_ref_changed(self, index):
        if index <= 0:
            self.params.P.ref_frame = None
        else:
            self.params.P.ref_frame = self.ref_combo.currentText()
        self.save_state()

    def open_parameters_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("REF Parameters")
        dialog.resize(460, 320)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        self.param_force = QCheckBox("Force rebuild")
        self.param_force.setChecked(bool(getattr(self.params.P, "force_master_build", False)))
        form.addRow("Force:", self.param_force)

        self.param_min_frames = QSpinBox()
        self.param_min_frames.setRange(1, 100)
        self.param_min_frames.setValue(int(getattr(self.params.P, "master_min_frames_xy", 1)))
        form.addRow("Min Frames XY:", self.param_min_frames)

        self.param_preserve = QCheckBox("Preserve IDs")
        self.param_preserve.setChecked(bool(getattr(self.params.P, "master_preserve_ids", True)))
        form.addRow("Preserve IDs:", self.param_preserve)

        self.param_ref = QLineEdit()
        self.param_ref.setText(str(getattr(self.params.P, "ref_frame", "")))
        self.param_ref.setPlaceholderText("index or filename (optional)")
        form.addRow("Ref Frame:", self.param_ref)

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: self.save_parameters(dialog))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec_()

    def save_parameters(self, dialog):
        self.params.P.force_master_build = self.param_force.isChecked()
        self.params.P.master_min_frames_xy = self.param_min_frames.value()
        self.params.P.master_preserve_ids = self.param_preserve.isChecked()
        ref_val = self.param_ref.text().strip()
        self.params.P.ref_frame = ref_val if ref_val else None
        if hasattr(self, "ref_combo"):
            self.populate_file_list()
        self.save_state()
        QMessageBox.information(dialog, "Success", "Parameters saved!")
        dialog.accept()

    def run_ref_build(self):
        if not self.file_list:
            QMessageBox.warning(self, "Warning", "No files to process")
            return
        if self.worker and self.worker.isRunning():
            return

        self.results_table.setRowCount(0)
        self.log_text.clear()
        self.log(
            "Params | "
            f"files={len(self.file_list)} | use_cropped={self.use_cropped} | "
            f"resume={getattr(self.params.P, 'resume_mode', True)} | "
            f"force={getattr(self.params.P, 'force_master_build', False)} | "
            f"min_frames_xy={getattr(self.params.P, 'master_min_frames_xy', 1)} | "
            f"preserve_ids={getattr(self.params.P, 'master_preserve_ids', True)} | "
            f"ref_frame={getattr(self.params.P, 'ref_frame', None)}"
        )

        self.worker = RefBuildWorker(
            self.file_list,
            self.params,
            self.params.P.data_dir,
            self.params.P.result_dir,
            self.params.P.cache_dir,
            self.use_cropped
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(self.file_list))
        self.progress_label.setText(f"0/{len(self.file_list)} | Starting...")
        self.worker.start()
        self.show_log_window()

    def stop_ref_build(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()

    def on_progress(self, current, total, filename):
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"{current}/{total} | {filename}")

    def on_finished(self, summary):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_label.setText("Done")
        if summary:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            self.results_table.setItem(row, 0, QTableWidgetItem(str(summary.get("ref_frame", ""))))
            self.results_table.setItem(row, 1, QTableWidgetItem(str(summary.get("master_rows", ""))))
            self.results_table.setItem(row, 2, QTableWidgetItem(str(summary.get("dropped", ""))))
            self.log(
                "REF build done | "
                f"master_rows={summary.get('master_rows')} | "
                f"dropped={summary.get('dropped')} | "
                f"frames_ok={summary.get('ok_frames')}/{summary.get('total_frames')} | "
                f"missing_idmatch={summary.get('missing_idmatch')} | "
                f"read_fail={summary.get('read_fail')} | "
                f"no_match={summary.get('no_match')} | "
                f"in_ref={summary.get('n_in_ref')} | "
                f"missing_in_ref={summary.get('n_missing_in_ref')}"
            )
        self.save_state()
        self.update_navigation_buttons()

    def on_error(self, filename, error):
        self.log(f"ERROR {filename}: {error}")

    def show_log_window(self):
        self.log_window.show()
        self.log_window.raise_()
        self.log_window.activateWindow()

    def validate_step(self) -> bool:
        master_path = step7_dir(self.params.P.result_dir) / "master_catalog.tsv"
        if not master_path.exists():
            master_path = self.params.P.result_dir / "master_catalog.tsv"
        return master_path.exists()

    def save_state(self):
        state_data = {
            "force_master_build": getattr(self.params.P, "force_master_build", False),
            "master_min_frames_xy": getattr(self.params.P, "master_min_frames_xy", 1),
            "master_preserve_ids": getattr(self.params.P, "master_preserve_ids", True),
            "ref_frame": getattr(self.params.P, "ref_frame", None),
        }
        self.project_state.store_step_data("ref_build", state_data)

    def restore_state(self):
        state_data = self.project_state.get_step_data("ref_build")
        if state_data:
            for key, val in state_data.items():
                if hasattr(self.params.P, key):
                    setattr(self.params.P, key, val)
        if hasattr(self, "ref_combo"):
            self.populate_file_list()
