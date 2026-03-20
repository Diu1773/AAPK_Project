"""
Exoplanet Transit Analysis Tool

Features:
  1. Load light curve + auto-fetch prior parameters (NASA Exoplanet Archive / ExoClock)
  2. Manual transit trimming (T_start, T_end)
  3. Batman transit model fitting (scipy.optimize leastsq)
  4. MCMC fit (emcee) — optional, requires emcee
  5. O-C timing residuals across multiple transits
  6. Parameter summary: Rp/Rs, a/Rs, i, T₀, P, transit depth/duration

Required (core): scipy, batman-package
Optional: emcee (MCMC), astroquery (NASA archive fetch)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

_CORR_MODE_RE = re.compile(r"lightcurve_.*?_(global|color|offset|raw)\b", re.IGNORECASE)
_CORR_MODE_LABELS = {
    "global": "Global ensemble",
    "color": "Color-dependent",
    "offset": "Nightly offset",
    "raw": "Raw",
}


def _detect_corr_mode_from_df(df: pd.DataFrame, filename: str) -> str:
    if "correction_mode" in df.columns:
        vals = df["correction_mode"].dropna().astype(str).str.strip().str.lower()
        if not vals.empty:
            key = vals.iloc[0]
            if key:
                return _CORR_MODE_LABELS.get(key, key)
    m = _CORR_MODE_RE.search(filename)
    if m:
        return _CORR_MODE_LABELS.get(m.group(1).lower(), m.group(1))
    return ""


def _resolve_check_filter(filters) -> str | None:
    if filters is None:
        return None
    unique_filters = sorted({str(f) for f in filters if str(f).strip() and str(f).lower() != "nan"})
    return unique_filters[0] if len(unique_filters) == 1 else None

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QTabWidget, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QSplitter, QMessageBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from ..workflow.step12_period_analysis import PeriodAnalysisWorker


def _load_check_star_for_plot(result_dir: Path, filt: str | None = None):
    """Load check star CSV from step10 output for plotting. Returns (check_id, df_or_None)."""
    try:
        from ..workflow.step10_light_curve_builder import _load_check_star_csv
        check_id, df = _load_check_star_csv(result_dir, filt=filt)
        return check_id, (df if not df.empty else None)
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class FetchParamsWorker(QThread):
    """Fetch transit parameters from NASA Exoplanet Archive."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, target_name: str):
        super().__init__()
        self.target_name = target_name.strip()

    def run(self):
        try:
            self.progress.emit(f"Querying NASA Exoplanet Archive for '{self.target_name}'…")
            from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
            result = NasaExoplanetArchive.query_object(
                self.target_name,
                select="pl_name,pl_orbper,pl_tranmid,pl_ratror,pl_ratdor,pl_orbincl,pl_orbeccen"
            )
            if len(result) == 0:
                self.error.emit(f"No results for '{self.target_name}'")
                return

            row = result[0]

            def safe(col, default=np.nan):
                try:
                    v = float(row[col])
                    return v if np.isfinite(v) else default
                except Exception:
                    return default

            params = {
                "name": str(row["pl_name"]),
                "period": safe("pl_orbper"),
                "t0_bjd": safe("pl_tranmid"),
                "rp_rs": safe("pl_ratror"),
                "a_rs": safe("pl_ratdor"),
                "inc_deg": safe("pl_orbincl"),
                "ecc": safe("pl_orbeccen", 0.0),
                "source": "NASA Exoplanet Archive",
            }
            self.progress.emit("Done")
            self.finished.emit(params)

        except ImportError:
            # Fall back to ExoClock REST API
            self._fetch_exoclock()
        except Exception as e:
            # Also try ExoClock as fallback
            self.progress.emit(f"Archive failed ({e}), trying ExoClock…")
            self._fetch_exoclock()

    def _fetch_exoclock(self):
        try:
            import requests
            name_enc = self.target_name.replace(" ", "-").lower()
            url = f"https://www.exoclock.space/api/planets/{name_enc}"
            self.progress.emit(f"ExoClock API: {url}")
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                self.error.emit(f"ExoClock returned {resp.status_code}")
                return
            data = resp.json()

            def safe(key, default=np.nan):
                try:
                    v = data.get(key)
                    return float(v) if v is not None else default
                except Exception:
                    return default

            params = {
                "name": data.get("name", self.target_name),
                "period": safe("period"),
                "t0_bjd": safe("t0"),
                "rp_rs": safe("rp"),
                "a_rs": safe("ar"),
                "inc_deg": safe("inc"),
                "ecc": safe("ecc", 0.0),
                "source": "ExoClock",
            }
            self.progress.emit("Done (ExoClock)")
            self.finished.emit(params)
        except Exception as e:
            self.error.emit(f"ExoClock failed: {e}")


class BatmanFitWorker(QThread):
    """Fit batman transit model using scipy least-squares."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, time, flux, flux_err, init_params: dict, fixed: dict):
        super().__init__()
        self.time = np.asarray(time, dtype=float)
        self.flux = np.asarray(flux, dtype=float)
        self.flux_err = np.asarray(flux_err, dtype=float) if flux_err is not None else None
        self.init = init_params
        self.fixed = fixed  # keys to hold fixed

    def run(self):
        try:
            import batman
            from scipy.optimize import least_squares

            self.progress.emit("Building batman model…")

            def make_model(p_arr):
                params = batman.TransitParams()
                param_names = ["t0", "per", "rp", "a", "inc", "ecc", "w", "u1", "u2"]
                vals = dict(zip(param_names, p_arr))
                params.t0 = vals["t0"]
                params.per = vals["per"]
                params.rp = vals["rp"]
                params.a = vals["a"]
                params.inc = vals["inc"]
                params.ecc = vals["ecc"]
                params.w = vals["w"]
                params.u = [vals["u1"], vals["u2"]]
                params.limb_dark = "quadratic"
                m = batman.TransitModel(params, self.time)
                return m.light_curve(params)

            # Build free parameter list
            all_names = ["t0", "per", "rp", "a", "inc", "ecc", "w", "u1", "u2"]
            free_names = [n for n in all_names if n not in self.fixed]
            fixed_vals = {n: self.fixed.get(n, self.init.get(n, 0.0)) for n in all_names}
            x0 = np.array([self.init.get(n, fixed_vals[n]) for n in free_names])

            def residuals(x):
                p_full = np.array([
                    x[free_names.index(n)] if n in free_names else fixed_vals[n]
                    for n in all_names
                ])
                model = make_model(p_full)
                r = self.flux - model
                if self.flux_err is not None:
                    r /= self.flux_err
                return r

            self.progress.emit("Fitting (least-squares)…")
            result = least_squares(residuals, x0, method="lm")

            fitted = {
                n: result.x[free_names.index(n)] if n in free_names else fixed_vals[n]
                for n in all_names
            }

            # Compute model light curve for plotting
            p_arr = np.array([fitted[n] for n in all_names])
            model_lc = make_model(p_arr)

            # Transit depth, duration
            rp = abs(fitted["rp"])
            depth = rp**2
            duration_hr = np.nan
            try:
                # Approximate duration: T14 = P/π × arcsin(Rs/a × √((1+k)²−b²) / sin(i))
                per = fitted["per"]
                a = fitted["a"]
                inc = np.radians(fitted["inc"])
                b = a * np.cos(inc)
                k = rp
                duration_days = (per / np.pi) * np.arcsin(
                    (1.0 / a) * np.sqrt((1 + k)**2 - b**2) / np.sin(inc)
                )
                duration_hr = duration_days * 24.0
            except Exception:
                pass

            self.finished.emit({
                "fitted_params": fitted,
                "free_names": free_names,
                "residuals": result.fun,
                "cost": float(result.cost),
                "model_lc": model_lc,
                "depth": depth,
                "duration_hr": duration_hr,
            })

        except ImportError:
            self.error.emit(
                "batman-package가 설치되지 않았습니다.\n"
                "pip install batman-package 로 설치하세요."
            )
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class MCMCWorker(QThread):
    """MCMC fitting using emcee."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, time, flux, flux_err, init_params: dict,
                 free_names: list, n_walkers: int = 32, n_steps: int = 1000):
        super().__init__()
        self.time = np.asarray(time, dtype=float)
        self.flux = np.asarray(flux, dtype=float)
        self.flux_err = np.asarray(flux_err, dtype=float) if flux_err is not None else None
        self.init = init_params
        self.free_names = free_names
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            import emcee
            import batman

            all_names = ["t0", "per", "rp", "a", "inc", "ecc", "w", "u1", "u2"]
            fixed = {n: self.init[n] for n in all_names if n not in self.free_names}

            def make_model_lc(x):
                params = batman.TransitParams()
                vals = dict(zip(self.free_names, x))
                vals.update(fixed)
                params.t0 = vals.get("t0", self.init["t0"])
                params.per = vals.get("per", self.init["per"])
                params.rp = vals.get("rp", self.init["rp"])
                params.a = vals.get("a", self.init["a"])
                params.inc = vals.get("inc", self.init["inc"])
                params.ecc = vals.get("ecc", 0.0)
                params.w = vals.get("w", 90.0)
                params.u = [vals.get("u1", 0.3), vals.get("u2", 0.1)]
                params.limb_dark = "quadratic"
                m = batman.TransitModel(params, self.time)
                return m.light_curve(params)

            sigma = self.flux_err if self.flux_err is not None else np.ones_like(self.flux) * 0.001

            def log_likelihood(x):
                try:
                    model = make_model_lc(x)
                    return -0.5 * np.sum(((self.flux - model) / sigma)**2)
                except Exception:
                    return -np.inf

            def log_prior(x):
                # Loose uninformative priors
                vals = dict(zip(self.free_names, x))
                if "rp" in vals and not (0 < vals["rp"] < 0.5):
                    return -np.inf
                if "inc" in vals and not (50 < vals["inc"] < 90.5):
                    return -np.inf
                if "a" in vals and not (1 < vals["a"] < 100):
                    return -np.inf
                return 0.0

            def log_prob(x):
                lp = log_prior(x)
                if not np.isfinite(lp):
                    return -np.inf
                return lp + log_likelihood(x)

            ndim = len(self.free_names)
            p0_center = np.array([self.init.get(n, 0.0) for n in self.free_names])
            p0 = p0_center + 1e-4 * np.random.randn(self.n_walkers, ndim)

            sampler = emcee.EnsembleSampler(self.n_walkers, ndim, log_prob)
            burn_in = min(200, self.n_steps // 5)

            self.progress.emit(f"Burn-in ({burn_in} steps)…")
            sampler.run_mcmc(p0, burn_in, progress=False)
            if self._stop:
                return
            sampler.reset()

            self.progress.emit(f"Production ({self.n_steps} steps)…")
            sampler.run_mcmc(sampler.get_last_sample(), self.n_steps, progress=False)

            flat_samples = sampler.get_chain(flat=True)
            medians = np.median(flat_samples, axis=0)
            q16 = np.percentile(flat_samples, 16, axis=0)
            q84 = np.percentile(flat_samples, 84, axis=0)
            lo = medians - q16
            hi = q84 - medians

            fitted = {
                n: float(medians[i]) for i, n in enumerate(self.free_names)
            }
            fitted.update(fixed)

            self.finished.emit({
                "fitted_params": fitted,
                "medians": medians,
                "lo": lo,
                "hi": hi,
                "flat_samples": flat_samples,
                "free_names": self.free_names,
            })

        except ImportError:
            self.error.emit(
                "emcee가 설치되지 않았습니다.\n"
                "pip install emcee 로 설치하세요."
            )
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class TransitToolWindow(QWidget):
    """Exoplanet Transit Analysis Tool."""

    def __init__(self, params, project_state, parent=None):
        super().__init__(parent)
        self.params = params
        self.project_state = project_state
        self.lc_data: Optional[dict] = None
        self.prior_params: dict = {
            "t0": 2458000.0, "per": 1.0, "rp": 0.1, "a": 10.0,
            "inc": 88.0, "ecc": 0.0, "w": 90.0, "u1": 0.3, "u2": 0.1,
        }
        self.scan_result: Optional[dict] = None
        self.fit_result: Optional[dict] = None
        self._scan_worker: Optional[PeriodAnalysisWorker] = None
        self._fetch_worker: Optional[FetchParamsWorker] = None
        self._fit_worker: Optional[BatmanFitWorker] = None
        self._mcmc_worker: Optional[MCMCWorker] = None
        self.workspace_dir = Path(self.params.P.result_dir)

        self.setWindowTitle("Exoplanet Transit Analysis")
        self.resize(1300, 850)
        self._build_ui()
        self._load_lc_from_workspace()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)

        header = QLabel(
            "<b>Exoplanet Transit Analysis</b> — load transit light curve → fetch prior "
            "parameters (NASA/ExoClock) → batman model fit → optional MCMC → O-C timing"
        )
        header.setStyleSheet("QLabel { background: #E8F5E9; padding: 8px; border-radius: 4px; }")
        header.setWordWrap(True)
        root.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        # ---- Left panel ----
        left = QWidget()
        left.setMaximumWidth(340)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        splitter.addWidget(left)

        # Target + parameter fetch
        tgt_group = QGroupBox("Target & Prior Parameters")
        tgt_form = QFormLayout(tgt_group)
        name_row = QHBoxLayout()
        self.target_edit = QLineEdit()
        self.target_edit.setPlaceholderText("e.g. HAT-P-7 b")
        name_row.addWidget(self.target_edit)
        btn_fetch = QPushButton("Fetch")
        btn_fetch.setStyleSheet(
            "QPushButton { background: #1976D2; color: white; font-weight: bold; padding: 4px 10px; }"
        )
        btn_fetch.clicked.connect(self._fetch_params)
        name_row.addWidget(btn_fetch)
        tgt_form.addRow("Target:", name_row)
        self.fetch_status = QLabel("")
        self.fetch_status.setStyleSheet("font-size: 8pt; color: #555;")
        tgt_form.addRow(self.fetch_status)

        param_defs = [
            ("T₀ (BJD)", "t0", 2400000.0, 2600000.0, 6, ""),
            ("Period (d)", "per", 0.001, 1000.0, 8, " d"),
            ("Rp/Rs", "rp", 0.001, 0.5, 6, ""),
            ("a/Rs", "a", 1.0, 200.0, 4, ""),
            ("inc (deg)", "inc", 50.0, 90.5, 4, "°"),
            ("ecc", "ecc", 0.0, 1.0, 4, ""),
            ("u₁ (LD)", "u1", 0.0, 1.0, 4, ""),
            ("u₂ (LD)", "u2", 0.0, 1.0, 4, ""),
        ]
        self._param_spins: dict[str, QDoubleSpinBox] = {}
        self._param_fixed: dict[str, QCheckBox] = {}
        for label, key, lo, hi, dec, suf in param_defs:
            row = QHBoxLayout()
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setDecimals(dec)
            spin.setValue(self.prior_params.get(key, 0.0))
            if suf:
                spin.setSuffix(suf)
            spin.valueChanged.connect(self._sync_prior)
            self._param_spins[key] = spin
            row.addWidget(spin)
            chk = QCheckBox("fix")
            chk.setChecked(key in ("per", "ecc", "w"))
            self._param_fixed[key] = chk
            row.addWidget(chk)
            tgt_form.addRow(label + ":", row)
        ll.addWidget(tgt_group)

        # Light curve
        lc_group = QGroupBox("Light Curve")
        lc_form = QFormLayout(lc_group)
        self.lc_status = QLabel("Not loaded")
        self.lc_status.setWordWrap(True)
        lc_form.addRow("Status:", self.lc_status)
        ws_row = QWidget()
        ws_layout = QHBoxLayout(ws_row)
        ws_layout.setContentsMargins(0, 0, 0, 0)
        self.workspace_edit = QLineEdit(str(self.workspace_dir))
        btn_workspace = QPushButton("Browse…")
        btn_workspace.clicked.connect(self._browse_workspace)
        btn_reload = QPushButton("Load")
        btn_reload.clicked.connect(self._load_lc_from_workspace)
        ws_layout.addWidget(self.workspace_edit, 1)
        ws_layout.addWidget(btn_workspace)
        ws_layout.addWidget(btn_reload)
        lc_form.addRow("Workspace:", ws_row)
        # Trim
        trim_row = QHBoxLayout()
        self.trim_start = QDoubleSpinBox()
        self.trim_start.setRange(2400000, 2600000); self.trim_start.setDecimals(6)
        self.trim_start.setValue(2458000.0)
        trim_row.addWidget(QLabel("from:"))
        trim_row.addWidget(self.trim_start)
        lc_form.addRow("Trim (BJD):", trim_row)
        trim_row2 = QHBoxLayout()
        self.trim_end = QDoubleSpinBox()
        self.trim_end.setRange(2400000, 2600000); self.trim_end.setDecimals(6)
        self.trim_end.setValue(2459000.0)
        trim_row2.addWidget(QLabel("to:"))
        trim_row2.addWidget(self.trim_end)
        lc_form.addRow("", trim_row2)
        btn_trim = QPushButton("Apply Trim & Show")
        btn_trim.clicked.connect(self._apply_trim)
        lc_form.addRow(btn_trim)
        ll.addWidget(lc_group)

        # Period scan (BLS 권장 — transit용)
        scan_group = QGroupBox("Period Scan")
        scan_form = QFormLayout(scan_group)
        self.scan_min_p = QDoubleSpinBox()
        self.scan_min_p.setRange(0.001, 500); self.scan_min_p.setDecimals(4)
        self.scan_min_p.setValue(0.5); self.scan_min_p.setSuffix(" d")
        scan_form.addRow("P min:", self.scan_min_p)
        self.scan_max_p = QDoubleSpinBox()
        self.scan_max_p.setRange(0.01, 2000); self.scan_max_p.setDecimals(4)
        self.scan_max_p.setValue(30.0); self.scan_max_p.setSuffix(" d")
        scan_form.addRow("P max:", self.scan_max_p)
        self.scan_spp = QSpinBox()
        self.scan_spp.setRange(5, 50); self.scan_spp.setValue(10)
        scan_form.addRow("Samples/peak:", self.scan_spp)
        scan_method_row = QHBoxLayout()
        self.scan_chk_ls = QCheckBox("LS"); self.scan_chk_ls.setChecked(True)
        self.scan_chk_bls = QCheckBox("BLS"); self.scan_chk_bls.setChecked(True)
        scan_method_row.addWidget(self.scan_chk_ls)
        scan_method_row.addWidget(self.scan_chk_bls)
        scan_form.addRow("Methods:", scan_method_row)
        btn_scan = QPushButton("Scan Period")
        btn_scan.setStyleSheet(
            "QPushButton { background: #00796B; color: white; font-weight: bold; padding: 5px; }"
        )
        btn_scan.clicked.connect(self._run_scan)
        scan_form.addRow(btn_scan)
        self.scan_status = QLabel("")
        self.scan_status.setStyleSheet("font-size: 8pt; color: #555;")
        scan_form.addRow(self.scan_status)
        ll.addWidget(scan_group)

        # Fit buttons
        fit_group = QGroupBox("Fit")
        fit_form = QFormLayout(fit_group)
        btn_fit = QPushButton("Batman Fit (least-squares)")
        btn_fit.setStyleSheet(
            "QPushButton { background: #388E3C; color: white; font-weight: bold; padding: 6px; }"
        )
        btn_fit.clicked.connect(self._run_batman_fit)
        fit_form.addRow(btn_fit)
        mcmc_row = QHBoxLayout()
        self.n_walkers_spin = QSpinBox()
        self.n_walkers_spin.setRange(16, 128); self.n_walkers_spin.setValue(32)
        self.n_steps_spin = QSpinBox()
        self.n_steps_spin.setRange(100, 5000); self.n_steps_spin.setValue(1000)
        mcmc_row.addWidget(QLabel("walkers:"))
        mcmc_row.addWidget(self.n_walkers_spin)
        mcmc_row.addWidget(QLabel("steps:"))
        mcmc_row.addWidget(self.n_steps_spin)
        fit_form.addRow(mcmc_row)
        btn_mcmc = QPushButton("MCMC (emcee)")
        btn_mcmc.setStyleSheet(
            "QPushButton { background: #7B1FA2; color: white; font-weight: bold; padding: 6px; }"
        )
        btn_mcmc.clicked.connect(self._run_mcmc)
        fit_form.addRow(btn_mcmc)
        self.fit_status = QLabel("")
        self.fit_status.setStyleSheet("font-size: 8pt; color: #555;")
        fit_form.addRow(self.fit_status)
        ll.addWidget(fit_group)

        ll.addStretch()

        self.btn_log_toggle = QPushButton("Log ▼")
        self.btn_log_toggle.setCheckable(True)
        self.btn_log_toggle.setChecked(False)
        self.btn_log_toggle.setStyleSheet(
            "QPushButton { text-align: left; font-size: 8pt; padding: 2px 6px; }"
        )
        self.btn_log_toggle.toggled.connect(self._toggle_log)
        ll.addWidget(self.btn_log_toggle)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(150)
        self.log_box.setStyleSheet("font-family: monospace; font-size: 8pt;")
        self.log_box.hide()
        ll.addWidget(self.log_box)

        # ---- Right panel ----
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)
        splitter.addWidget(right)
        splitter.setSizes([330, 970])

        self.tabs = QTabWidget()
        rl.addWidget(self.tabs)

        # Periodogram tab
        pg_tab = QWidget()
        pgl = QVBoxLayout(pg_tab)
        self.pg_canvas = FigureCanvas(Figure(figsize=(9, 4)))
        pgl.addWidget(NavigationToolbar(self.pg_canvas, pg_tab))
        pgl.addWidget(self.pg_canvas)
        self.tabs.addTab(pg_tab, "Periodogram")

        # Light curve tab
        lc_tab = QWidget()
        lt = QVBoxLayout(lc_tab)
        self.lc_canvas = FigureCanvas(Figure(figsize=(9, 4)))
        lt.addWidget(NavigationToolbar(self.lc_canvas, lc_tab))
        lt.addWidget(self.lc_canvas)
        self.tabs.addTab(lc_tab, "Light Curve")

        # Fit result tab
        fit_tab = QWidget()
        ft = QVBoxLayout(fit_tab)
        self.fit_label = QLabel("Batman 피팅 결과가 여기에 표시됩니다.")
        self.fit_label.setStyleSheet(
            "QLabel { background: #E8F5E9; padding: 8px; border-radius: 4px; "
            "font-family: monospace; font-size: 9pt; }"
        )
        self.fit_label.setWordWrap(True)
        ft.addWidget(self.fit_label)
        self.fit_canvas = FigureCanvas(Figure(figsize=(9, 5)))
        ft.addWidget(NavigationToolbar(self.fit_canvas, fit_tab))
        ft.addWidget(self.fit_canvas, 1)
        self.tabs.addTab(fit_tab, "Fit Result")

        # MCMC corner tab
        mcmc_tab = QWidget()
        mt = QVBoxLayout(mcmc_tab)
        self.mcmc_label = QLabel("MCMC 결과가 여기에 표시됩니다.")
        self.mcmc_label.setStyleSheet(
            "QLabel { background: #F3E5F5; padding: 8px; border-radius: 4px; "
            "font-family: monospace; font-size: 9pt; }"
        )
        self.mcmc_label.setWordWrap(True)
        mt.addWidget(self.mcmc_label)
        self.mcmc_canvas = FigureCanvas(Figure(figsize=(9, 7)))
        mt.addWidget(NavigationToolbar(self.mcmc_canvas, mcmc_tab))
        mt.addWidget(self.mcmc_canvas, 1)
        self.tabs.addTab(mcmc_tab, "MCMC")

        # O-C tab
        oc_tab = self._build_oc_tab()
        self.tabs.addTab(oc_tab, "O-C")

    def _build_oc_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("T₀ (BJD):"))
        self.oc_t0 = QDoubleSpinBox()
        self.oc_t0.setRange(2400000, 2600000); self.oc_t0.setDecimals(6)
        self.oc_t0.setValue(2458000.0); self.oc_t0.setMinimumWidth(130)
        self.oc_t0.valueChanged.connect(self._recompute_oc)
        hdr.addWidget(self.oc_t0)
        hdr.addWidget(QLabel("P (d):"))
        self.oc_p = QDoubleSpinBox()
        self.oc_p.setRange(0.00001, 10000); self.oc_p.setDecimals(8)
        self.oc_p.setValue(1.0); self.oc_p.setMinimumWidth(120)
        self.oc_p.valueChanged.connect(self._recompute_oc)
        hdr.addWidget(self.oc_p)
        btn_from_fit = QPushButton("← From Fit")
        btn_from_fit.clicked.connect(self._oc_from_fit)
        hdr.addWidget(btn_from_fit)
        hdr.addStretch()
        layout.addLayout(hdr)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.oc_table = QTableWidget()
        self.oc_table.setColumnCount(4)
        self.oc_table.setHorizontalHeaderLabels(["n", "BJD_mid", "O-C (d)", "err (d)"])
        self.oc_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.oc_table.horizontalHeader().setStretchLastSection(True)
        self.oc_table.setMinimumWidth(300)
        ll.addWidget(self.oc_table, 1)
        btn_row = QHBoxLayout()
        for label, slot in [("Add", self._oc_add), ("Del", self._oc_del),
                             ("Import CSV", self._oc_import), ("Export CSV", self._oc_export)]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        btn_row.addStretch()
        ll.addLayout(btn_row)
        splitter.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.oc_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        rl.addWidget(NavigationToolbar(self.oc_canvas, right))
        rl.addWidget(self.oc_canvas, 1)
        self.oc_fit_label = QLabel("")
        self.oc_fit_label.setWordWrap(True)
        self.oc_fit_label.setStyleSheet(
            "QLabel { background: #E8F5E9; padding: 6px; border-radius: 4px; "
            "font-family: monospace; font-size: 9pt; }"
        )
        rl.addWidget(self.oc_fit_label)
        splitter.addWidget(right)
        splitter.setSizes([330, 670])
        layout.addWidget(splitter, 1)

        return tab

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sync_prior(self):
        for key, spin in self._param_spins.items():
            self.prior_params[key] = spin.value()

    def _current_workspace_dir(self) -> Path:
        text = self.workspace_edit.text().strip() if hasattr(self, "workspace_edit") else ""
        path = Path(text) if text else Path(self.workspace_dir)
        self.workspace_dir = path
        return path

    def _browse_workspace(self):
        start_dir = self._current_workspace_dir()
        start = str(start_dir.parent if start_dir.exists() else Path(self.params.P.result_dir).parent)
        path = QFileDialog.getExistingDirectory(self, "Workspace 선택", start)
        if path:
            self.workspace_edit.setText(path)
            self._load_lc_from_workspace()

    def _load_lc_from_workspace(self):
        try:
            from ...utils.step_paths import find_best_lightcurve_csv
            rd = self._current_workspace_dir()
            path = find_best_lightcurve_csv(rd)
            if path is None:
                self.lc_status.setText(f"No lightcurve_*.csv found in\n{rd}")
                return
            self._load_csv(path)
        except Exception as e:
            self.lc_status.setText(f"Workspace load failed: {e}")

    def _load_csv(self, path: Path):
        try:
            df = pd.read_csv(path)
            # Apply step10 frame exclusions
            try:
                from ...utils.qc_utils import load_frame_excludes as _lfe
                rd = self._current_workspace_dir()
                excl = set(_lfe(rd).keys())
                if excl and "file" in df.columns:
                    before = len(df)
                    df = df[~df["file"].astype(str).isin(excl)].reset_index(drop=True)
                    if len(df) < before:
                        self.lc_status.setText(f"Frame QC: {before - len(df)} frame(s) excluded")
            except Exception:
                pass
            time_col = next(
                (c for c in ["BJD", "bjd", "HJD", "hjd", "JD", "jd", "time"] if c in df.columns),
                None
            )
            if time_col is None:
                self.lc_status.setText("No time column found")
                return
            mag_col = next(
                (c for c in ["diff_mag_corr", "diff_mag", "mag_corr", "diff_mag_raw", "mag_raw", "mag"]
                 if c in df.columns), None
            )
            if mag_col is None:
                self.lc_status.setText("No magnitude column found")
                return
            err_col = next(
                (c for c in ["diff_err", "diff_err_corr", "mag_err", "err", "sigma"] if c in df.columns),
                None
            )
            t_raw = df[time_col].to_numpy(float)
            m_raw = df[mag_col].to_numpy(float)
            e_raw = df[err_col].to_numpy(float) if err_col else None

            # Convert mag to relative flux (normalise to out-of-transit baseline)
            mask = np.isfinite(t_raw) & np.isfinite(m_raw)
            t_v = t_raw[mask]
            m_v = m_raw[mask]
            e_v = e_raw[mask] if e_raw is not None else None

            # Normalize: flux = 10^(-(mag - mag_median)/2.5)
            med_mag = np.nanmedian(m_v)
            flux = 10.0 ** (-(m_v - med_mag) / 2.5)
            flux_err = (np.log(10) / 2.5) * flux * e_v if e_v is not None else None

            self.lc_data = {
                "time": t_v, "flux": flux, "flux_err": flux_err,
                "mag": m_v, "mag_err": e_v, "source": path.name,
            }
            # Set trim defaults
            self.trim_start.setValue(float(t_v.min()))
            self.trim_end.setValue(float(t_v.max()))

            n = len(t_v)
            raw_corr = "corr" if any(x in mag_col for x in ["corr", "cal"]) else "raw"
            corr_tag = _detect_corr_mode_from_df(df, path.name)
            corr_line = f"  detrend: {corr_tag}" if corr_tag else ""
            workspace_dir = self._current_workspace_dir()
            workspace_name = workspace_dir.name
            workspace_type = ""
            try:
                from ...utils.run_workspace import load_run_manifest
                run_meta = load_run_manifest(workspace_dir)
                run_type = str(run_meta.get("run_type") or "").strip().lower()
                if run_type:
                    workspace_type = f" [{run_type}]"
            except Exception:
                pass
            self.lc_status.setText(
                f"{workspace_name}{workspace_type}\n{path.name}\n{n} pts, flux{corr_line}\nmag src: {mag_col} [{raw_corr}]"
            )
            self.lc_status.setStyleSheet("color: green;")
            self.log(f"Loaded: {path.name} ({n} pts, mag→flux, detrend={corr_tag or 'N/A'})")
            self._draw_lc()
        except Exception as e:
            self.lc_status.setText(f"Error: {e}")
            self.log(f"[ERROR] {e}")

    def _apply_trim(self):
        if self.lc_data is None:
            return
        t = self.lc_data["time"]
        t_start = self.trim_start.value()
        t_end = self.trim_end.value()
        mask = (t >= t_start) & (t <= t_end)
        if mask.sum() < 5:
            QMessageBox.warning(self, "Trim", "트림 후 데이터가 너무 적습니다 (< 5).")
            return
        self.lc_data["trim_mask"] = mask
        self.log(f"Trim applied: {mask.sum()} pts between {t_start:.4f} – {t_end:.4f}")
        self._draw_lc()

    def _get_trimmed(self):
        if self.lc_data is None:
            return None, None, None
        mask = self.lc_data.get("trim_mask", np.ones(len(self.lc_data["time"]), dtype=bool))
        t = self.lc_data["time"][mask]
        f = self.lc_data["flux"][mask]
        fe = self.lc_data["flux_err"][mask] if self.lc_data.get("flux_err") is not None else None
        return t, f, fe

    def _draw_lc(self):
        if self.lc_data is None:
            return
        fig = self.lc_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        t = self.lc_data["time"]
        f = self.lc_data["flux"]
        fe = self.lc_data.get("flux_err")
        if fe is not None:
            ax.errorbar(t, f, yerr=fe, fmt="o", color="#1565C0", markersize=3,
                        elinewidth=0.5, capsize=0, alpha=0.7, label="Target")
        else:
            ax.scatter(t, f, color="#1565C0", s=8, alpha=0.7, label="Target")

        # Check star overlay (flux-normalised)
        try:
            _rd = self._current_workspace_dir()
            _check_filter = _resolve_check_filter(self.lc_data.get("filters") if self.lc_data else None)
            _ck_id, _ck_df = _load_check_star_for_plot(_rd, filt=_check_filter)
            if _ck_df is not None and not _ck_df.empty:
                _t_col = next((c for c in ["JD", "bjd", "hjd", "time"] if c in _ck_df.columns), None)
                _y_col = next((c for c in ["diff_mag_raw", "diff_mag", "mag"] if c in _ck_df.columns), None)
                if _t_col and _y_col:
                    _ct = pd.to_numeric(_ck_df[_t_col], errors="coerce").to_numpy(float)
                    _cm = pd.to_numeric(_ck_df[_y_col], errors="coerce").to_numpy(float)
                    _mask = np.isfinite(_ct) & np.isfinite(_cm)
                    if _mask.any():
                        _ck_label = f"Check ID {_ck_id}" if _ck_id is not None else "Check"
                        _med = np.nanmedian(_cm[_mask])
                        _cf = 10.0 ** (-(_cm[_mask] - _med) / 2.5)
                        ax.scatter(_ct[_mask], _cf, s=8, color="#FFD700", alpha=0.4,
                                   zorder=2, label=_ck_label, marker="^")
                        ax.legend(fontsize=8)
        except Exception:
            pass

        ax.set_xlabel("BJD")
        ax.set_ylabel("Relative Flux")
        ax.set_title(f"Transit Light Curve — {self.lc_data.get('source', '')}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.lc_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def _run_scan(self):
        if self.lc_data is None:
            QMessageBox.warning(self, "No Data", "광도곡선을 먼저 로드하세요.")
            return
        if self._scan_worker and self._scan_worker.isRunning():
            return
        methods = []
        if self.scan_chk_ls.isChecked(): methods.append("ls")
        if self.scan_chk_bls.isChecked(): methods.append("bls")
        if not methods:
            QMessageBox.warning(self, "No Method", "메서드를 선택하세요.")
            return
        self.scan_status.setText("Running…")
        self.tabs.setCurrentIndex(0)  # Periodogram tab
        flux = self.lc_data["flux"]
        # Convert flux back to mag-like (invert for periodogram — flux dip = period signal)
        mag_like = -2.5 * np.log10(np.clip(flux, 1e-6, None))
        self._scan_worker = PeriodAnalysisWorker(
            time=self.lc_data["time"],
            mag_raw=mag_like,
            mag_corr=None,
            mag_err=None,
            min_period=self.scan_min_p.value(),
            max_period=self.scan_max_p.value(),
            samples_per_peak=self.scan_spp.value(),
            methods=methods,
        )
        self._scan_worker.progress.connect(self.scan_status.setText)
        self._scan_worker.finished.connect(self._on_scan_done)
        self._scan_worker.error.connect(lambda e: (self.scan_status.setText("Error"), self.log(e)))
        self._scan_worker.start()

    def _on_scan_done(self, results: dict):
        self.scan_result = results
        # Prefer BLS for transit
        for key in ("raw_bls", "raw_ls"):
            d = results.get(key)
            if d and "error" not in d and np.isfinite(d.get("best_period", np.nan)):
                best = float(d["best_period"])
                self.scan_status.setText(f"Best P = {best:.6f} d  [{key.split('_')[1].upper()}]")
                self._param_spins["per"].setValue(best)
                self.prior_params["per"] = best
                self.oc_p.setValue(best)
                break
        self._draw_periodogram(results)
        parts = [f"{k}\u2192{v['best_period']:.4f}d" for k, v in results.items() if "error" not in v]
        self.log(f"Scan done: {', '.join(parts)}")

    def _draw_periodogram(self, results: dict):
        method_labels = {"ls": "Lomb-Scargle", "bls": "BLS (transit)"}
        method_colors = {"ls": "#1E88E5", "bls": "#FF9800"}
        y_labels = {"ls": "LS Power", "bls": "BLS Power"}

        n = len(results)
        fig = self.pg_canvas.figure
        fig.clear()
        if n == 0:
            self.pg_canvas.draw_idle()
            return
        axes = fig.subplots(1, n, squeeze=False)[0]
        for ax, (key, data) in zip(axes, results.items()):
            method = key.split("_", 1)[1]
            if "error" in data:
                ax.text(0.5, 0.5, data["error"], ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                ax.set_title(method_labels.get(method, method))
                continue
            periods = (1.0 / data["frequency"]) if "frequency" in data else data["trial_periods"]
            power = data["power"]
            best = data["best_period"]
            color = method_colors.get(method, "#666")
            ax.plot(periods, power, color=color, lw=0.8, alpha=0.9)
            ax.axvline(best, color="red", ls="--", lw=1.5, label=f"P={best:.6f} d")
            ax.scatter([best], [data["best_power"]], color="red", s=50, zorder=5)
            ax.set_xscale("log")
            ax.set_xlabel("Period (days)")
            ax.set_ylabel(y_labels.get(method, "Power"))
            ax.set_title(f"{method_labels.get(method, method)}\nP={best:.6f} d")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.pg_canvas.draw_idle()

    def _fetch_params(self):
        name = self.target_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Target", "행성 이름을 입력하세요.")
            return
        if self._fetch_worker and self._fetch_worker.isRunning():
            return
        self.fetch_status.setText("Fetching…")
        self._fetch_worker = FetchParamsWorker(name)
        self._fetch_worker.progress.connect(self.fetch_status.setText)
        self._fetch_worker.finished.connect(self._on_fetch_done)
        self._fetch_worker.error.connect(lambda e: (self.fetch_status.setText(f"Error: {e}"), self.log(e)))
        self._fetch_worker.start()

    def _on_fetch_done(self, params: dict):
        src = params.get("source", "")
        self.fetch_status.setText(f"Done ({src})")
        mapping = {
            "t0": "t0_bjd", "per": "period", "rp": "rp_rs",
            "a": "a_rs", "inc": "inc_deg", "ecc": "ecc",
        }
        for spin_key, fetch_key in mapping.items():
            v = params.get(fetch_key, np.nan)
            if np.isfinite(v):
                self._param_spins[spin_key].setValue(v)
                self.prior_params[spin_key] = v
        self.log(
            f"Fetched from {src}: {params.get('name','')}  "
            f"P={params.get('period', np.nan):.6f} d  "
            f"T₀={params.get('t0_bjd', np.nan):.4f}  "
            f"Rp/Rs={params.get('rp_rs', np.nan):.4f}"
        )
        self.oc_t0.setValue(self.prior_params.get("t0", 2458000.0))
        self.oc_p.setValue(self.prior_params.get("per", 1.0))

    # ------------------------------------------------------------------
    # Batman fit
    # ------------------------------------------------------------------

    def _run_batman_fit(self):
        t, f, fe = self._get_trimmed()
        if t is None or len(t) < 10:
            QMessageBox.warning(self, "No Data", "트림된 광도곡선 데이터가 없습니다.")
            return
        if self._fit_worker and self._fit_worker.isRunning():
            return
        self._sync_prior()
        fixed = {k: v for k, v in self.prior_params.items()
                 if self._param_fixed.get(k, QCheckBox()).isChecked()}
        self.fit_status.setText("Fitting…")
        self.tabs.setCurrentIndex(1)
        self._fit_worker = BatmanFitWorker(t, f, fe, dict(self.prior_params), fixed)
        self._fit_worker.progress.connect(self.fit_status.setText)
        self._fit_worker.finished.connect(self._on_fit_done)
        self._fit_worker.error.connect(self._on_fit_error)
        self._fit_worker.start()

    def _on_fit_done(self, result: dict):
        self.fit_result = result
        fitted = result["fitted_params"]
        depth = result["depth"]
        dur = result["duration_hr"]

        lines = ["=== Batman Fit Result ==="]
        for n, v in fitted.items():
            lines.append(f"  {n:6s} = {v:.6f}")
        lines.append(f"\n  Transit depth  = {depth:.6f}  ({depth*100:.4f} %)")
        if np.isfinite(dur):
            lines.append(f"  Duration T₁₄   = {dur:.3f} hr")
        lines.append(f"\n  χ² cost = {result['cost']:.4f}")
        self.fit_label.setText("\n".join(lines))

        self._draw_fit(result)

        # Update O-C T₀ and P from fit
        self.oc_t0.setValue(fitted.get("t0", self.oc_t0.value()))
        self.oc_p.setValue(fitted.get("per", self.oc_p.value()))

        self.fit_status.setText(
            f"Done — depth={depth*100:.3f}%  dur={dur:.2f} hr" if np.isfinite(dur)
            else f"Done — depth={depth*100:.3f}%"
        )
        self.log(
            f"Batman fit: Rp/Rs={fitted.get('rp',.0):.4f}  a/Rs={fitted.get('a',.0):.2f}  "
            f"inc={fitted.get('inc',.0):.2f}°  T₀={fitted.get('t0',.0):.6f}"
        )

    def _on_fit_error(self, msg: str):
        self.fit_status.setText("Error")
        QMessageBox.warning(self, "Fit Error", msg)
        self.log(f"[FIT ERROR] {msg}")

    def _draw_fit(self, result: dict):
        t, f, fe = self._get_trimmed()
        if t is None:
            return
        fig = self.fit_canvas.figure
        fig.clear()
        ax1, ax2 = fig.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

        model = result["model_lc"]
        if fe is not None:
            ax1.errorbar(t, f, yerr=fe, fmt="o", color="#1565C0", markersize=3,
                         elinewidth=0.5, capsize=0, alpha=0.7, label="Data")
        else:
            ax1.scatter(t, f, color="#1565C0", s=8, alpha=0.7, label="Data")
        sort = np.argsort(t)
        ax1.plot(t[sort], model[sort], color="red", lw=1.5, label="Batman model")
        ax1.set_ylabel("Rel. Flux")
        ax1.set_title("Transit Fit")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        resid = f - model
        if fe is not None:
            ax2.errorbar(t, resid, yerr=fe, fmt="o", color="#E53935", markersize=3,
                         elinewidth=0.5, capsize=0, alpha=0.7, label="Residual")
        else:
            ax2.scatter(t, resid, color="#E53935", s=8, alpha=0.7, label="Residual")
        ax2.axhline(0, color="gray", ls="--", lw=1)

        # Check star overlay on residual plot (if check star shows a dip → systematic)
        try:
            _rd = self._current_workspace_dir()
            _check_filter = _resolve_check_filter(self.lc_data.get("filters") if self.lc_data else None)
            _ck_id, _ck_df = _load_check_star_for_plot(_rd, filt=_check_filter)
            if _ck_df is not None and not _ck_df.empty:
                _t_col = next((c for c in ["JD", "bjd", "hjd", "time"] if c in _ck_df.columns), None)
                _y_col = next((c for c in ["diff_mag_raw", "diff_mag", "mag"] if c in _ck_df.columns), None)
                if _t_col and _y_col:
                    _ct = pd.to_numeric(_ck_df[_t_col], errors="coerce").to_numpy(float)
                    _cm = pd.to_numeric(_ck_df[_y_col], errors="coerce").to_numpy(float)
                    _mask = np.isfinite(_ct) & np.isfinite(_cm)
                    if _mask.any():
                        _ck_label = f"Check ID {_ck_id}" if _ck_id is not None else "Check"
                        _med = np.nanmedian(_cm[_mask])
                        _cf = 10.0 ** (-(_cm[_mask] - _med) / 2.5) - 1.0
                        ax2.scatter(_ct[_mask], _cf, s=8, color="#FFD700", alpha=0.4,
                                    zorder=2, label=f"{_ck_label} (flux-1)", marker="^")
                        ax2.legend(fontsize=7)
        except Exception:
            pass

        ax2.set_xlabel("BJD")
        ax2.set_ylabel("Residual")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        self.fit_canvas.draw_idle()

    # ------------------------------------------------------------------
    # MCMC
    # ------------------------------------------------------------------

    def _run_mcmc(self):
        t, f, fe = self._get_trimmed()
        if t is None or len(t) < 10:
            QMessageBox.warning(self, "No Data", "트림된 광도곡선 데이터가 없습니다.")
            return
        if self._mcmc_worker and self._mcmc_worker.isRunning():
            return
        self._sync_prior()
        fixed = {k: v for k, v in self.prior_params.items()
                 if self._param_fixed.get(k, QCheckBox()).isChecked()}
        free_names = [k for k in self.prior_params if k not in fixed]
        self.fit_status.setText("MCMC running…")
        self.tabs.setCurrentIndex(2)
        self._mcmc_worker = MCMCWorker(
            t, f, fe, dict(self.prior_params), free_names,
            n_walkers=self.n_walkers_spin.value(),
            n_steps=self.n_steps_spin.value(),
        )
        self._mcmc_worker.progress.connect(self.fit_status.setText)
        self._mcmc_worker.finished.connect(self._on_mcmc_done)
        self._mcmc_worker.error.connect(
            lambda e: (self.fit_status.setText("Error"), QMessageBox.warning(self, "MCMC Error", e))
        )
        self._mcmc_worker.start()

    def _on_mcmc_done(self, result: dict):
        self.fit_status.setText("MCMC done")
        medians = result["medians"]
        lo = result["lo"]
        hi = result["hi"]
        free_names = result["free_names"]

        lines = ["=== MCMC Result (median ± 1σ) ==="]
        for i, n in enumerate(free_names):
            lines.append(f"  {n:6s} = {medians[i]:.6f}  +{hi[i]:.6f} / -{lo[i]:.6f}")
        self.mcmc_label.setText("\n".join(lines))

        self._draw_mcmc_corner(result)
        self.log("MCMC done — see MCMC tab for corner plot")

    def _draw_mcmc_corner(self, result: dict):
        """Draw a simple pair-plot (no corner package required)."""
        samples = result["flat_samples"]
        free_names = result["free_names"]
        n = len(free_names)

        fig = self.mcmc_canvas.figure
        fig.clear()
        if n == 0:
            self.mcmc_canvas.draw_idle()
            return
        if n == 1:
            ax = fig.add_subplot(111)
            ax.hist(samples[:, 0], bins=40, color="#7B1FA2", alpha=0.7)
            ax.set_xlabel(free_names[0])
            ax.set_ylabel("Count")
            fig.tight_layout()
            self.mcmc_canvas.draw_idle()
            return

        axes = fig.subplots(n, n)
        for i in range(n):
            for j in range(n):
                ax = axes[i][j]
                if i == j:
                    ax.hist(samples[:, i], bins=30, color="#7B1FA2", alpha=0.7)
                    ax.set_yticklabels([])
                elif i > j:
                    ax.scatter(samples[::5, j], samples[::5, i],
                               s=1, alpha=0.3, color="#1E88E5")
                else:
                    ax.set_visible(False)
                if i == n - 1:
                    ax.set_xlabel(free_names[j], fontsize=7)
                else:
                    ax.set_xticklabels([])
                if j == 0 and i > 0:
                    ax.set_ylabel(free_names[i], fontsize=7)
                else:
                    ax.set_yticklabels([])
        fig.tight_layout()
        self.mcmc_canvas.draw_idle()

    # ------------------------------------------------------------------
    # O-C
    # ------------------------------------------------------------------

    def _oc_from_fit(self):
        if self.fit_result:
            fp = self.fit_result["fitted_params"]
            self.oc_t0.setValue(fp.get("t0", self.oc_t0.value()))
            self.oc_p.setValue(fp.get("per", self.oc_p.value()))

    def _oc_add(self):
        r = self.oc_table.rowCount()
        self.oc_table.insertRow(r)
        for c in range(4):
            self.oc_table.setItem(r, c, QTableWidgetItem(""))

    def _oc_del(self):
        rows = sorted({i.row() for i in self.oc_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.oc_table.removeRow(r)
        self._draw_oc()

    def _oc_import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import O-C CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            df = pd.read_csv(path)
            bjd_col = next(
                (c for c in df.columns if c.lower() in ("bjd_mid", "bjd", "hjd", "jd")), None
            )
            if bjd_col is None:
                QMessageBox.warning(self, "Error", "BJD 컬럼을 찾을 수 없습니다.")
                return
            err_col = next(
                (c for c in df.columns if c.lower() in ("err", "error", "sigma")), None
            )
            t0 = self.oc_t0.value(); p = self.oc_p.value()
            self.oc_table.setRowCount(0)
            for _, row_data in df.iterrows():
                bjd = float(row_data[bjd_col])
                n = int(round((bjd - t0) / p)) if p > 0 else 0
                oc = bjd - (t0 + n * p)
                err = f"{float(row_data[err_col]):.6f}" if err_col else ""
                r = self.oc_table.rowCount()
                self.oc_table.insertRow(r)
                self.oc_table.setItem(r, 0, QTableWidgetItem(str(n)))
                self.oc_table.setItem(r, 1, QTableWidgetItem(f"{bjd:.6f}"))
                self.oc_table.setItem(r, 2, QTableWidgetItem(f"{oc:.6f}"))
                self.oc_table.setItem(r, 3, QTableWidgetItem(err))
            self._draw_oc()
        except Exception as e:
            QMessageBox.warning(self, "Import Error", str(e))

    def _oc_export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export O-C CSV", "", "CSV (*.csv)")
        if not path:
            return
        rows = []
        for r in range(self.oc_table.rowCount()):
            rows.append([
                self.oc_table.item(r, c).text() if self.oc_table.item(r, c) else ""
                for c in range(4)
            ])
        pd.DataFrame(rows, columns=["n", "BJD_mid", "O-C (d)", "err (d)"]).to_csv(path, index=False)
        self.log(f"Exported O-C to {path}")

    def _recompute_oc(self):
        t0 = self.oc_t0.value(); p = self.oc_p.value()
        if p <= 0:
            return
        for r in range(self.oc_table.rowCount()):
            bjd_it = self.oc_table.item(r, 1)
            if not bjd_it or not bjd_it.text():
                continue
            try:
                bjd = float(bjd_it.text())
                n = int(round((bjd - t0) / p))
                oc = bjd - (t0 + n * p)
                self.oc_table.setItem(r, 0, QTableWidgetItem(str(n)))
                self.oc_table.setItem(r, 2, QTableWidgetItem(f"{oc:.6f}"))
            except ValueError:
                pass
        self._draw_oc()

    def _get_oc_arrays(self):
        ns, ocs, errs = [], [], []
        for r in range(self.oc_table.rowCount()):
            try:
                n_it = self.oc_table.item(r, 0)
                oc_it = self.oc_table.item(r, 2)
                er_it = self.oc_table.item(r, 3)
                if not n_it or not oc_it or not n_it.text() or not oc_it.text():
                    continue
                ns.append(int(n_it.text()))
                ocs.append(float(oc_it.text()))
                errs.append(float(er_it.text()) if (er_it and er_it.text()) else np.nan)
            except (ValueError, TypeError):
                pass
        return np.array(ns), np.array(ocs), np.array(errs)

    def _draw_oc(self):
        fig = self.oc_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ns, ocs, errs = self._get_oc_arrays()
        if len(ns) == 0:
            ax.text(0.5, 0.5, "O-C 데이터 없음\n(Import CSV or Add rows)",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            fig.tight_layout()
            self.oc_canvas.draw_idle()
            return
        oc_min = ocs * 1440
        has_err = np.any(np.isfinite(errs))
        if has_err:
            ax.errorbar(ns, oc_min, yerr=np.where(np.isfinite(errs), errs * 1440, 0),
                        fmt="o", color="#1565C0", markersize=5, elinewidth=1, capsize=3)
        else:
            ax.scatter(ns, oc_min, color="#1565C0", s=40, zorder=5)
        ax.axhline(0, color="gray", ls="--", lw=1, alpha=0.6)
        ax.set_xlabel("Transit Epoch (n)")
        ax.set_ylabel("O-C (minutes)")
        ax.set_title("Transit Timing Variations (O-C)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.oc_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------

    def _toggle_log(self, checked: bool):
        self.log_box.setVisible(checked)
        self.btn_log_toggle.setText("Log ▲" if checked else "Log ▼")

    def log(self, msg: str):
        self.log_box.append(msg)
