"""
IRAF Photometry Tool Window
Comprehensive IRAF/DAOPHOT photometry interface with all parameters.
Uses WSL subprocess to run PyRAF on Windows.
"""

from __future__ import annotations

import os
import sys
import subprocess
import tempfile
import json
import tomllib
try:
    import tomli_w  # type: ignore
except Exception:
    tomli_w = None
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QGroupBox, QFormLayout, QLineEdit, QDoubleSpinBox, QSpinBox,
    QFileDialog, QMessageBox, QProgressBar, QTabWidget, QComboBox,
    QCheckBox, QSplitter, QTableWidget, QTableWidgetItem, QScrollArea,
    QFrame, QSlider, QShortcut, QDialog, QHeaderView
)

from ...utils.step_paths import step2_cropped_dir


def windows_to_wsl_path(win_path: str) -> str:
    """Convert Windows path to WSL path."""
    path = str(win_path).replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        path = f"/mnt/{drive}{path[2:]}"
    return path


def wsl_to_windows_path(wsl_path: str) -> str:
    """Convert WSL path to Windows path."""
    if wsl_path.startswith("/mnt/") and len(wsl_path) > 6:
        drive = wsl_path[5].upper()
        return f"{drive}:{wsl_path[6:]}".replace("/", "\\")
    return wsl_path


# ============================================================================
# IRAF Parameter Dataclass-like storage
# ============================================================================
class IRAFParameters:
    """Storage for all IRAF DAOPHOT parameters."""

    def __init__(self):
        # === DATAPARS ===
        self.scale = 1.0           # Image scale in units/pixel
        self.emission = True       # Emission features (True) or absorption (False)
        self.datamax = 60000.0     # Maximum good data value
        self.noise = "poisson"     # Noise model: poisson, constant, file
        self.readnoise = 1.39      # CCD readout noise in electrons
        self.epadu = 0.1           # Gain in electrons per ADU
        self.exposure = "EXPTIME"  # Exposure time header keyword
        self.itime = 1.0           # Integration time (if not in header)

        # === Filter-specific FWHM (seeing in arcsec) ===
        self.seeing_g = 2.5
        self.seeing_r = 2.5
        self.seeing_i = 2.5
        self.seeing_default = 2.5

        # === Filter-specific sigma (sky background std) ===
        self.sigma_g = 50.0
        self.sigma_r = 50.0
        self.sigma_i = 50.0
        self.sigma_default = 50.0

        # === FINDPARS ===
        # Filter-specific thresholds (g, r, i)
        self.threshold_g = 4.0
        self.threshold_r = 4.5
        self.threshold_i = 5.0
        self.threshold_default = 5.0  # For unknown filters
        self.nsigma = 1.5          # Width of convolution kernel in sigma
        self.ratio = 1.0           # Ratio of minor to major axis of Gaussian
        self.theta = 0.0           # Position angle of major axis
        # Filter-specific sharplo
        self.sharplo_g = 0.2
        self.sharplo_r = 0.2
        self.sharplo_i = 0.4
        self.sharplo_default = 0.2
        self.sharphi = 1.0         # Upper bound on sharpness
        self.roundlo = -1.0        # Lower bound on roundness
        self.roundhi = 1.0         # Upper bound on roundness
        # Filter-specific datamin
        self.datamin_g = -100.0
        self.datamin_r = -100.0
        self.datamin_i = 0.0
        self.datamin_default = -100.0

        # === CENTERPARS ===
        self.calgorithm = "centroid"  # Centering algorithm
        self.cbox_mult = 2.0       # Centering box = FWHM * cbox_mult
        self.cthreshold = 0.0      # Centering threshold in sigma
        self.minsnratio = 1.0      # Minimum signal-to-noise ratio
        self.cmaxiter = 10         # Maximum iterations for centering
        self.maxshift = 1.0        # Maximum shift in scale units
        self.clean = False         # Symmetry clean before centering
        self.rclean = 1.0          # Cleaning radius in scale units
        self.rclip = 2.0           # Clipping radius in scale units
        self.kclean = 3.0          # K-sigma rejection criterion

        # === FITSKYPARS (FWHM multipliers) ===
        self.salgorithm = "mode"   # Sky algorithm
        self.annulus_mult = 4.0    # Inner radius = FWHM * annulus_mult
        self.dannulus_mult = 2.0   # Width = FWHM * dannulus_mult
        self.skyvalue = 0.0        # User sky value (for constant algorithm)
        self.smaxiter = 10         # Maximum iterations for sky fitting
        self.sloclip = 0.0         # Lower clipping factor (sigma)
        self.shiclip = 0.0         # Upper clipping factor (sigma)
        self.snreject = 50         # Maximum number of rejection iterations
        self.sloreject = 3.0       # Lower K-sigma rejection limit
        self.shireject = 3.0       # Upper K-sigma rejection limit
        self.khist = 3.0           # Half-width of histogram in sigma
        self.binsize = 0.1         # Binsize of histogram in sigma
        self.smooth = False        # Smooth histogram before fitting
        self.rgrow = 0.0           # Region growing radius in scale units

        # === PHOTPARS (FWHM multiplier) ===
        self.aperture_mult = 1.0   # Aperture = FWHM * aperture_mult
        self.zmag = 25.0           # Zero point of magnitude scale
        self.mkapert = False       # Make aperture plots

        # === Convenience settings ===
        self.pix_scale = 0.392     # Pixel scale in arcsec/pixel
        self.sigma_ref = 50.0      # Reference sigma for threshold scaling

    def to_dict(self):
        """Convert to dictionary for script generation."""
        return {k: v for k, v in self.__dict__.items()}

    def from_dict(self, d: dict):
        """Load parameters from dictionary."""
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)


# ============================================================================
# PyRAF Script Template
# ============================================================================
PYRAF_SCRIPT_TEMPLATE = '''#!/usr/bin/env python3
"""Auto-generated PyRAF photometry script with filter-specific parameters"""

from pyraf import iraf
import os, glob, sys
import numpy as np

iraf.noao(); iraf.digiphot(); iraf.daophot()

# =====================
# Paths
# =====================
DATA_DIR = "{data_dir}"
OUTDIR = "{output_dir}"
FILE_PATTERN = "{file_pattern}"
SKIP_EXISTING = {skip_existing}

# =====================
# Filter config (from TOML)
# =====================
FILTER_PARAMS = {filter_params}
FILTER_ALIASES = {filter_aliases}
PARAM_DEFAULTS = {param_defaults}

# =====================
# Pixel scale
# =====================
PIX_SCALE = {pix_scale}  # arcsec/pixel

# =====================
# Filter-specific SEEING (arcsec)
# =====================
SEEING = {{"g": {seeing_g}, "r": {seeing_r}, "i": {seeing_i}, "default": {seeing_default}}}

# =====================
# Filter-specific SIGMA (sky background std)
# =====================
SIGMA = {{"g": {sigma_g}, "r": {sigma_r}, "i": {sigma_i}, "default": {sigma_default}}}
SIGMA_REF = {sigma_ref}

# =====================
# DATAPARS (base values)
# =====================
DATAPARS = {{
    "scale": {scale},
    "emission": {emission},
    "datamax": {datamax},
    "noise": "{noise}",
    "readnoise": {readnoise},
    "epadu": {epadu},
    "exposure": "{exposure}",
    "itime": {itime},
}}

# =====================
# FINDPARS - Filter-specific
# =====================
THRESHOLD = {{"g": {threshold_g}, "r": {threshold_r}, "i": {threshold_i}, "default": {threshold_default}}}
SHARPLO = {{"g": {sharplo_g}, "r": {sharplo_r}, "i": {sharplo_i}, "default": {sharplo_default}}}
DATAMIN = {{"g": {datamin_g}, "r": {datamin_r}, "i": {datamin_i}, "default": {datamin_default}}}

FINDPARS_BASE = {{
    "nsigma": {nsigma},
    "ratio": {ratio},
    "theta": {theta},
    "sharphi": {sharphi},
    "roundlo": {roundlo},
    "roundhi": {roundhi},
}}

# =====================
# CENTERPARS
# =====================
CENTERPARS = {{
    "calgorithm": "{calgorithm}",
    "cbox_mult": {cbox_mult},
    "cthreshold": {cthreshold},
    "minsnratio": {minsnratio},
    "cmaxiter": {cmaxiter},
    "maxshift": {maxshift},
    "clean": {clean},
    "rclean": {rclean},
    "rclip": {rclip},
    "kclean": {kclean},
}}

# =====================
# FITSKYPARS (FWHM multipliers)
# =====================
FITSKYPARS = {{
    "salgorithm": "{salgorithm}",
    "annulus_mult": {annulus_mult},
    "dannulus_mult": {dannulus_mult},
    "skyvalue": {skyvalue},
    "smaxiter": {smaxiter},
    "sloclip": {sloclip},
    "shiclip": {shiclip},
    "snreject": {snreject},
    "sloreject": {sloreject},
    "shireject": {shireject},
    "khist": {khist},
    "binsize": {binsize},
    "smooth": {smooth},
    "rgrow": {rgrow},
}}

# =====================
# PHOTPARS (FWHM multiplier)
# =====================
PHOTPARS = {{
    "aperture_mult": {aperture_mult},
    "zmag": {zmag},
    "mkapert": {mkapert},
}}

# =====================
# Helper functions
# =====================
def safe_rm(p):
    try: os.remove(p)
    except FileNotFoundError: pass

def get_header(im, key):
    try:
        out = iraf.hselect(im, key, "yes", Stdout=1)
        v = out[0].strip() if out else ""
        if v in ["", "INDEF", "indef"]:
            return None
        return v
    except:
        return None

def normalize_filter(val):
    key = str(val).strip().lower()
    if key in FILTER_ALIASES:
        return str(FILTER_ALIASES[key]).strip().lower()
    return key

def guess_filter(im):
    for k in ["FILTER", "FILTER1", "FILTER2", "FILTNAM"]:
        v = get_header(im, k)
        if v:
            return normalize_filter(v)
    low = os.path.basename(im).lower()
    for alias, canon in FILTER_ALIASES.items():
        if alias and alias in low:
            return str(canon).strip().lower()
    for key in FILTER_PARAMS.keys():
        if key != "default" and key in low:
            return key
    if "-g" in low: return "g"
    if "-r" in low: return "r"
    if "-i" in low: return "i"
    return "unknown"

def get_param(name, band, default=None):
    b = normalize_filter(band)
    if b in FILTER_PARAMS and name in FILTER_PARAMS[b]:
        return FILTER_PARAMS[b][name]
    if "default" in FILTER_PARAMS and name in FILTER_PARAMS["default"]:
        return FILTER_PARAMS["default"][name]
    if name in PARAM_DEFAULTS:
        return PARAM_DEFAULTS[name]
    return default

def estimate_sigma(image):
    try:
        out = iraf.imstat(image, fields="stddev", format="no",
                         nclip=5, lsigma=3., usigma=3., Stdout=1)
        return float(out[0].strip())
    except:
        return None

def fwhm_pix_from_header_or_default(im, band):
    # Try arcsec keys from header
    for k in ["FWHMARC", "SEEING", "FWHM_AS"]:
        v = get_header(im, k)
        if v:
            try:
                val = float(v)
                if val > 0:
                    return val / PIX_SCALE
            except:
                pass
    # Try pixel keys from header
    for k in ["FWHMPSF", "FWHM_PIX"]:
        v = get_header(im, k)
        if v:
            try:
                val = float(v)
                if val > 0:
                    return val
            except:
                pass
    # Use filter-specific seeing
    seeing_arcsec = get_param("seeing", band, SEEING.get(band, SEEING["default"]))
    return seeing_arcsec / PIX_SCALE

# =====================
# Unlearn all tasks
# =====================
for t in ["daofind", "phot", "datapars", "findpars", "centerpars", "fitskypars", "photpars"]:
    try: iraf.unlearn(t)
    except: pass

# =====================
# Apply base DATAPARS
# =====================
iraf.datapars.scale = DATAPARS["scale"]
iraf.datapars.emission = "yes" if DATAPARS["emission"] else "no"
iraf.datapars.datamax = DATAPARS["datamax"]
iraf.datapars.noise = DATAPARS["noise"]
iraf.datapars.readnoise = DATAPARS["readnoise"]
iraf.datapars.epadu = DATAPARS["epadu"]
iraf.datapars.exposure = DATAPARS["exposure"]
iraf.datapars.itime = DATAPARS["itime"]

# =====================
# Apply base FINDPARS
# =====================
iraf.findpars.nsigma = FINDPARS_BASE["nsigma"]
iraf.findpars.ratio = FINDPARS_BASE["ratio"]
iraf.findpars.theta = FINDPARS_BASE["theta"]
iraf.findpars.sharphi = FINDPARS_BASE["sharphi"]
iraf.findpars.roundlo = FINDPARS_BASE["roundlo"]
iraf.findpars.roundhi = FINDPARS_BASE["roundhi"]

# =====================
# Apply CENTERPARS (base)
# =====================
iraf.centerpars.calgorithm = CENTERPARS["calgorithm"]
iraf.centerpars.cthreshold = CENTERPARS["cthreshold"]
iraf.centerpars.minsnratio = CENTERPARS["minsnratio"]
iraf.centerpars.cmaxiter = CENTERPARS["cmaxiter"]
iraf.centerpars.maxshift = CENTERPARS["maxshift"]
iraf.centerpars.clean = "yes" if CENTERPARS["clean"] else "no"
iraf.centerpars.rclean = CENTERPARS["rclean"]
iraf.centerpars.rclip = CENTERPARS["rclip"]
iraf.centerpars.kclean = CENTERPARS["kclean"]

# =====================
# Apply FITSKYPARS (base)
# =====================
iraf.fitskypars.salgorithm = FITSKYPARS["salgorithm"]
iraf.fitskypars.skyvalue = FITSKYPARS["skyvalue"]
iraf.fitskypars.smaxiter = FITSKYPARS["smaxiter"]
iraf.fitskypars.sloclip = FITSKYPARS["sloclip"]
iraf.fitskypars.shiclip = FITSKYPARS["shiclip"]
iraf.fitskypars.snreject = FITSKYPARS["snreject"]
iraf.fitskypars.sloreject = FITSKYPARS["sloreject"]
iraf.fitskypars.shireject = FITSKYPARS["shireject"]
iraf.fitskypars.khist = FITSKYPARS["khist"]
iraf.fitskypars.binsize = FITSKYPARS["binsize"]
iraf.fitskypars.smooth = "yes" if FITSKYPARS["smooth"] else "no"
iraf.fitskypars.rgrow = FITSKYPARS["rgrow"]

# =====================
# Apply PHOTPARS (base)
# =====================
iraf.photpars.zmag = PHOTPARS["zmag"]
iraf.photpars.mkapert = "yes" if PHOTPARS["mkapert"] else "no"

print("[INIT] PyRAF initialized")
print(f"[PARAM] pix_scale={{PIX_SCALE:.3f}} arcsec/pix")
print(f"[PARAM] epadu={{DATAPARS['epadu']}}, rdnoise={{DATAPARS['readnoise']}}")
print(f"[PARAM] FWHM multipliers: ap={{PHOTPARS['aperture_mult']}}x, ann={{FITSKYPARS['annulus_mult']}}x, dann={{FITSKYPARS['dannulus_mult']}}x")
print(f"[PARAM] Filter-specific seeing: g={{SEEING['g']}}, r={{SEEING['r']}}, i={{SEEING['i']}} arcsec")
print(f"[PARAM] Filter-specific sigma: g={{SIGMA['g']}}, r={{SIGMA['r']}}, i={{SIGMA['i']}}")

os.makedirs(OUTDIR, exist_ok=True)
os.chdir(DATA_DIR)
imgs = sorted(glob.glob(FILE_PATTERN))

if not imgs:
    print(f"[ERROR] No images found matching {{FILE_PATTERN}}")
    sys.exit(1)

print(f"[INFO] Found {{len(imgs)}} images")

skipped = 0
processed = 0

for idx, im in enumerate(imgs):
    base = os.path.splitext(os.path.basename(im))[0]
    txt = os.path.join(OUTDIR, f"{{base}}.txt")

    # Skip if already processed
    if SKIP_EXISTING and os.path.exists(txt):
        print(f"[SKIP] {{idx+1}}/{{len(imgs)}} {{im}} (already exists)")
        skipped += 1
        continue

    band = guess_filter(im)

    # Get filter-specific FWHM
    fwhm_pix = fwhm_pix_from_header_or_default(im, band)
    iraf.datapars.fwhmpsf = float(fwhm_pix)

    # Get filter-specific sigma (or auto-estimate)
    if {auto_sigma}:
        sig = estimate_sigma(im)
        if sig is None or sig <= 0:
            sig = get_param("sigma", band, SIGMA.get(band, SIGMA["default"]))
    else:
        sig = get_param("sigma", band, SIGMA.get(band, SIGMA["default"]))
    iraf.datapars.sigma = float(sig)

    # Filter-specific threshold with sigma scaling
    base_thr = get_param("threshold", band, THRESHOLD.get(band, THRESHOLD["default"]))
    sigma_ref = float(get_param("sigma_ref", band, SIGMA_REF))
    thr = base_thr * np.clip(sig / sigma_ref, 0.8, 1.6)
    thr = float(np.clip(thr, 3.5, 15.0))
    iraf.findpars.threshold = thr

    # Filter-specific sharplo
    sharplo = get_param("sharplo", band, SHARPLO.get(band, SHARPLO["default"]))
    iraf.findpars.sharplo = float(sharplo)

    # Filter-specific datamin
    datamin = get_param("datamin", band, DATAMIN.get(band, DATAMIN["default"]))
    iraf.datapars.datamin = float(datamin)

    # Per-filter overrides
    iraf.datapars.scale = float(get_param("scale", band, DATAPARS["scale"]))
    iraf.datapars.emission = "yes" if get_param("emission", band, DATAPARS["emission"]) else "no"
    iraf.datapars.datamax = float(get_param("datamax", band, DATAPARS["datamax"]))
    iraf.datapars.noise = str(get_param("noise", band, DATAPARS["noise"]))
    iraf.datapars.readnoise = float(get_param("readnoise", band, DATAPARS["readnoise"]))
    iraf.datapars.epadu = float(get_param("epadu", band, DATAPARS["epadu"]))
    iraf.datapars.exposure = str(get_param("exposure", band, DATAPARS["exposure"]))
    iraf.datapars.itime = float(get_param("itime", band, DATAPARS["itime"]))

    iraf.findpars.nsigma = float(get_param("nsigma", band, FINDPARS_BASE["nsigma"]))
    iraf.findpars.ratio = float(get_param("ratio", band, FINDPARS_BASE["ratio"]))
    iraf.findpars.theta = float(get_param("theta", band, FINDPARS_BASE["theta"]))
    iraf.findpars.sharphi = float(get_param("sharphi", band, FINDPARS_BASE["sharphi"]))
    iraf.findpars.roundlo = float(get_param("roundlo", band, FINDPARS_BASE["roundlo"]))
    iraf.findpars.roundhi = float(get_param("roundhi", band, FINDPARS_BASE["roundhi"]))

    iraf.centerpars.calgorithm = str(get_param("calgorithm", band, CENTERPARS["calgorithm"]))
    iraf.centerpars.cthreshold = float(get_param("cthreshold", band, CENTERPARS["cthreshold"]))
    iraf.centerpars.minsnratio = float(get_param("minsnratio", band, CENTERPARS["minsnratio"]))
    iraf.centerpars.cmaxiter = int(get_param("cmaxiter", band, CENTERPARS["cmaxiter"]))
    iraf.centerpars.maxshift = float(get_param("maxshift", band, CENTERPARS["maxshift"]))
    iraf.centerpars.clean = "yes" if get_param("clean", band, CENTERPARS["clean"]) else "no"
    iraf.centerpars.rclean = float(get_param("rclean", band, CENTERPARS["rclean"]))
    iraf.centerpars.rclip = float(get_param("rclip", band, CENTERPARS["rclip"]))
    iraf.centerpars.kclean = float(get_param("kclean", band, CENTERPARS["kclean"]))

    iraf.fitskypars.salgorithm = str(get_param("salgorithm", band, FITSKYPARS["salgorithm"]))
    iraf.fitskypars.skyvalue = float(get_param("skyvalue", band, FITSKYPARS["skyvalue"]))
    iraf.fitskypars.smaxiter = int(get_param("smaxiter", band, FITSKYPARS["smaxiter"]))
    iraf.fitskypars.sloclip = float(get_param("sloclip", band, FITSKYPARS["sloclip"]))
    iraf.fitskypars.shiclip = float(get_param("shiclip", band, FITSKYPARS["shiclip"]))
    iraf.fitskypars.snreject = int(get_param("snreject", band, FITSKYPARS["snreject"]))
    iraf.fitskypars.sloreject = float(get_param("sloreject", band, FITSKYPARS["sloreject"]))
    iraf.fitskypars.shireject = float(get_param("shireject", band, FITSKYPARS["shireject"]))
    iraf.fitskypars.khist = float(get_param("khist", band, FITSKYPARS["khist"]))
    iraf.fitskypars.binsize = float(get_param("binsize", band, FITSKYPARS["binsize"]))
    iraf.fitskypars.smooth = "yes" if get_param("smooth", band, FITSKYPARS["smooth"]) else "no"
    iraf.fitskypars.rgrow = float(get_param("rgrow", band, FITSKYPARS["rgrow"]))

    # FWHM-based parameters
    cbox = fwhm_pix * float(get_param("cbox_mult", band, CENTERPARS["cbox_mult"]))
    iraf.centerpars.cbox = float(cbox)

    aperture = fwhm_pix * float(get_param("aperture_mult", band, PHOTPARS["aperture_mult"]))
    iraf.photpars.apertures = f"{{aperture:.2f}}"

    annulus = fwhm_pix * float(get_param("annulus_mult", band, FITSKYPARS["annulus_mult"]))
    dannulus = fwhm_pix * float(get_param("dannulus_mult", band, FITSKYPARS["dannulus_mult"]))
    iraf.fitskypars.annulus = float(annulus)
    iraf.fitskypars.dannulus = float(dannulus)

    iraf.photpars.zmag = float(get_param("zmag", band, PHOTPARS["zmag"]))
    iraf.photpars.mkapert = "yes" if get_param("mkapert", band, PHOTPARS["mkapert"]) else "no"

    coo = os.path.join(OUTDIR, f"{{base}}.coo")
    mag = os.path.join(OUTDIR, f"{{base}}.mag")
    safe_rm(coo); safe_rm(mag); safe_rm(txt)

    print(f"[PROGRESS] {{idx+1}}/{{len(imgs)}} {{im}} band={{band}}")
    print(f"  fwhm={{fwhm_pix:.2f}}px sigma={{sig:.1f}} thr={{thr:.2f}} sharplo={{sharplo}} datamin={{datamin}}")
    print(f"  ap={{aperture:.2f}}px ann={{annulus:.2f}}px dann={{dannulus:.2f}}px cbox={{cbox:.2f}}px")

    try:
        iraf.daofind(im, output=coo, verify="no", interactive="no", verbose="no")
        iraf.phot(im, coords=coo, output=mag, verify="no", interactive="no")
        iraf.txdump(mag, fields="ID,XCENTER,YCENTER,MAG,MERR,MSKY,STDEV,NSKY",
                   expr="yes", headers="no", Stdout=txt)

        n_stars = 0
        try:
            with open(txt, 'r') as f:
                n_stars = sum(1 for _ in f)
        except:
            pass
        print(f"  -> {{n_stars}} stars detected")
        processed += 1
    except Exception as e:
        print(f"  -> ERROR: {{e}}")

print(f"[DONE] Processed: {{processed}}, Skipped: {{skipped}}, Output: {{OUTDIR}}")
'''


# ============================================================================
# Worker Thread
# ============================================================================
class IRAFPhotometryWorker(QThread):
    """Worker thread for running IRAF photometry via WSL PyRAF."""

    progress = pyqtSignal(int, int, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, data_dir: Path, output_dir: Path, file_pattern: str,
                 params: IRAFParameters, auto_sigma: bool = True, skip_existing: bool = True,
                 filter_params: dict | None = None, filter_aliases: dict | None = None,
                 param_defaults: dict | None = None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.file_pattern = file_pattern
        self.params = params
        self.auto_sigma = auto_sigma
        self.skip_existing = skip_existing
        self.filter_params = filter_params or {}
        self.filter_aliases = filter_aliases or {}
        self.param_defaults = param_defaults or {}
        self._stop_requested = False
        self._process = None
        self._script_path = None

    def stop(self):
        self._stop_requested = True
        if self._process:
            self._process.terminate()

    def _log(self, msg: str):
        self.log.emit(msg)

    def run(self):
        try:
            self._script_path = None
            is_windows = sys.platform == "win32"

            if is_windows:
                data_dir_str = windows_to_wsl_path(str(self.data_dir))
                output_dir_str = windows_to_wsl_path(str(self.output_dir))
            else:
                data_dir_str = str(self.data_dir)
                output_dir_str = str(self.output_dir)

            p = self.params
            script_content = PYRAF_SCRIPT_TEMPLATE.format(
                data_dir=data_dir_str,
                output_dir=output_dir_str,
                file_pattern=self.file_pattern,
                skip_existing="True" if self.skip_existing else "False",
                filter_params=repr(self.filter_params),
                filter_aliases=repr(self.filter_aliases),
                param_defaults=repr(self.param_defaults),
                # Pixel scale
                pix_scale=p.pix_scale,
                # Filter-specific seeing
                seeing_g=p.seeing_g,
                seeing_r=p.seeing_r,
                seeing_i=p.seeing_i,
                seeing_default=p.seeing_default,
                # Filter-specific sigma
                sigma_g=p.sigma_g,
                sigma_r=p.sigma_r,
                sigma_i=p.sigma_i,
                sigma_default=p.sigma_default,
                sigma_ref=p.sigma_ref,
                # DATAPARS
                scale=p.scale,
                emission="True" if p.emission else "False",
                datamax=p.datamax,
                noise=p.noise,
                readnoise=p.readnoise,
                epadu=p.epadu,
                exposure=p.exposure,
                itime=p.itime,
                # Filter-specific FINDPARS
                threshold_g=p.threshold_g,
                threshold_r=p.threshold_r,
                threshold_i=p.threshold_i,
                threshold_default=p.threshold_default,
                sharplo_g=p.sharplo_g,
                sharplo_r=p.sharplo_r,
                sharplo_i=p.sharplo_i,
                sharplo_default=p.sharplo_default,
                datamin_g=p.datamin_g,
                datamin_r=p.datamin_r,
                datamin_i=p.datamin_i,
                datamin_default=p.datamin_default,
                # FINDPARS base
                nsigma=p.nsigma,
                ratio=p.ratio,
                theta=p.theta,
                sharphi=p.sharphi,
                roundlo=p.roundlo,
                roundhi=p.roundhi,
                # CENTERPARS
                calgorithm=p.calgorithm,
                cbox_mult=p.cbox_mult,
                cthreshold=p.cthreshold,
                minsnratio=p.minsnratio,
                cmaxiter=p.cmaxiter,
                maxshift=p.maxshift,
                clean="True" if p.clean else "False",
                rclean=p.rclean,
                rclip=p.rclip,
                kclean=p.kclean,
                # FITSKYPARS (FWHM multipliers)
                salgorithm=p.salgorithm,
                annulus_mult=p.annulus_mult,
                dannulus_mult=p.dannulus_mult,
                skyvalue=p.skyvalue,
                smaxiter=p.smaxiter,
                sloclip=p.sloclip,
                shiclip=p.shiclip,
                snreject=p.snreject,
                sloreject=p.sloreject,
                shireject=p.shireject,
                khist=p.khist,
                binsize=p.binsize,
                smooth="True" if p.smooth else "False",
                rgrow=p.rgrow,
                # PHOTPARS (FWHM multiplier)
                aperture_mult=p.aperture_mult,
                zmag=p.zmag,
                mkapert="True" if p.mkapert else "False",
                # Additional
                auto_sigma="True" if self.auto_sigma else "False",
            )

            # Save script
            if is_windows:
                script_dir = Path(self.output_dir)
                script_dir.mkdir(parents=True, exist_ok=True)
                script_path = script_dir / "_pyraf_photometry.py"
                script_path.write_text(script_content, encoding="utf-8")
                self._script_path = script_path
                wsl_script_path = windows_to_wsl_path(str(script_path))
                self._log(f"Script: {script_path}")
                cmd = ["wsl", "python3", wsl_script_path]
            else:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(script_content)
                    script_path = f.name
                self._script_path = Path(script_path)
                self._log(f"Script: {script_path}")
                cmd = ["python3", script_path]

            self._log("Starting PyRAF via WSL..." if is_windows else "Starting PyRAF...")

            self._process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )

            total_images = 0
            results = []

            for line in iter(self._process.stdout.readline, ''):
                if self._stop_requested:
                    self._process.terminate()
                    break

                line = line.strip()
                if not line:
                    continue

                self._log(line)

                if line.startswith("[INFO] Found"):
                    try:
                        total_images = int(line.split()[2])
                    except:
                        pass
                elif line.startswith("[PROGRESS]"):
                    try:
                        parts = line.split()
                        current = int(parts[1].split("/")[0])
                        self.progress.emit(current, total_images, parts[2] if len(parts) > 2 else "")
                    except:
                        pass
                elif "stars detected" in line:
                    try:
                        n_stars = int(line.split("->")[1].split()[0])
                        results.append({"n_stars": n_stars})
                    except:
                        pass

            self._process.wait()

            if self._process.returncode == 0:
                self._log("\n[SUCCESS] PyRAF photometry completed!")
                if self._script_path:
                    try:
                        self._script_path.unlink()
                        self._log(f"[CLEANUP] Removed script: {self._script_path}")
                    except Exception as e:
                        self._log(f"[CLEANUP] Failed to remove script: {e}")
                self.finished.emit({"results": results, "output_dir": str(self.output_dir)})
            else:
                self.error.emit(f"PyRAF exited with code {self._process.returncode}")

        except FileNotFoundError:
            self.error.emit("WSL not found. Install WSL and PyRAF first.")
        except Exception as e:
            self.error.emit(f"Error: {e}")


# ============================================================================
# Comparison Functions (from iraf_comparison_window.py)
# ============================================================================
def _read_iraf_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+",
                     names=["ID", "x", "y", "mag", "merr", "msky", "stdev", "nsky"],
                     engine="python")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "x" in df.columns:
        df["x"] = df["x"] - 1.0
    if "y" in df.columns:
        df["y"] = df["y"] - 1.0
    return df


def _read_iraf_coo(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=["x", "y", "mag", "sharp", "sround", "ground", "ID"],
        engine="python",
    )
    for col in ["ID", "x", "y", "mag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["ID", "x", "y", "mag"]].copy()
    df["x"] = df["x"] - 1.0
    df["y"] = df["y"] - 1.0
    return df


def _read_aapki_tsv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t")
    except:
        return pd.read_csv(path)


def _normalize_frame_key(stem: str) -> str:
    key = stem
    if key.endswith("_photometry"):
        key = key[:-len("_photometry")]
    if key.endswith(".fit") or key.endswith(".fits"):
        key = key.rsplit(".", 1)[0]
    if key.startswith("Crop_"):
        key = key[len("Crop_"):]
    return key


BASE_IRAF_SHIFT = -1.0
AUTO_SHIFT_THRESHOLD = 0.6


def _auto_axis_shift(delta_med: float) -> float:
    if not np.isfinite(delta_med):
        return 0.0
    if abs(delta_med - 1.0) <= AUTO_SHIFT_THRESHOLD:
        return 1.0
    if abs(delta_med + 1.0) <= AUTO_SHIFT_THRESHOLD:
        return -1.0
    return 0.0


def _pick_first(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


# ============================================================================
# Main Window
# ============================================================================
class IRAFPhotometryWindow(QMainWindow):
    """Comprehensive IRAF Photometry tool with parameters and comparison."""

    def __init__(self, params, data_dir: Path, result_dir: Path, project_state=None, parent=None):
        super().__init__(parent)
        self.app_params = params
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.project_state = project_state
        self.iraf_params = IRAFParameters()
        self.worker = None

        # Comparison data
        self.frame_rows = []
        self.frame_matches = {}
        self.matched_all = None

        # Overlay data (comparison)
        self.overlay_file_list = []
        self.overlay_keys = []
        self.overlay_image_map = {}
        self.overlay_key_to_index = {}
        self.overlay_last_image_dir = None
        self.overlay_current_index = 0
        self.overlay_image_data = None
        self.overlay_header = None
        self.overlay_aapki_map = {}
        self.overlay_iraf_map = {}
        self.overlay_filter_cache = {}
        self._overlay_normalized_cache = None
        self.overlay_xlim_original = None
        self.overlay_ylim_original = None
        self.overlay_panning = False
        self.overlay_pan_start = None
        self._overlay_imshow_obj = None

        # Stretch plot window (2D Plot)
        self.overlay_stretch_plot_dialog = None
        self.overlay_stretch_plot_canvas = None
        self.overlay_stretch_plot_ax = None
        self.overlay_stretch_plot_fig = None
        self.overlay_stretch_plot_info_label = None
        self._overlay_stretch_vmin = None
        self._overlay_stretch_vmax = None
        self._overlay_stretch_data_range = None
        self._overlay_stretch_dragging = False
        self._overlay_stretch_drag_target = None
        self._overlay_stretch_marker_min_line = None
        self._overlay_stretch_marker_max_line = None

        self.setWindowTitle("IRAF/DAOPHOT Photometry Tool")
        self.setMinimumSize(1200, 900)
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Main tab widget
        self.tabs = QTabWidget()

        # Tab 1: Run Photometry
        self.tabs.addTab(self._create_run_tab(), "Run Photometry")

        # Tab 2: IRAF Parameters
        self.tabs.addTab(self._create_params_tab(), "IRAF Parameters")

        # Tab 3: Comparison
        self.tabs.addTab(self._create_comparison_tab(), "Comparison")

        layout.addWidget(self.tabs)

    def closeEvent(self, event):
        self._auto_save_params()
        super().closeEvent(event)

    # ========================================================================
    # Tab 1: Run Photometry
    # ========================================================================
    def _create_run_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Info
        info = QLabel(
            "Run IRAF DAOPHOT photometry via WSL/PyRAF.\n"
            "Configure parameters in the 'IRAF Parameters' tab before running."
        )
        info.setStyleSheet("color: #555; font-style: italic; padding: 5px;")
        layout.addWidget(info)

        # Paths
        paths_group = QGroupBox("Paths")
        paths_layout = QFormLayout()

        # Data directory
        data_row = QHBoxLayout()
        default_data = step2_cropped_dir(self.result_dir)
        if not default_data.exists():
            default_data = step2_cropped_dir(self.data_dir / "result")
        self.data_edit = QLineEdit(str(default_data))
        data_btn = QPushButton("Browse")
        data_btn.clicked.connect(lambda: self._browse_dir(self.data_edit))
        data_row.addWidget(self.data_edit)
        data_row.addWidget(data_btn)
        data_w = QWidget()
        data_w.setLayout(data_row)
        paths_layout.addRow("Data Directory:", data_w)

        # Output directory
        out_row = QHBoxLayout()
        self.out_edit = QLineEdit(str(self.result_dir / "iraf_phot"))
        out_btn = QPushButton("Browse")
        out_btn.clicked.connect(lambda: self._browse_dir(self.out_edit))
        out_row.addWidget(self.out_edit)
        out_row.addWidget(out_btn)
        out_w = QWidget()
        out_w.setLayout(out_row)
        paths_layout.addRow("Output Directory:", out_w)

        # File pattern
        self.pattern_edit = QLineEdit("*.fit*")
        paths_layout.addRow("File Pattern:", self.pattern_edit)

        # Auto sigma
        self.auto_sigma_check = QCheckBox("Auto-estimate sigma per image")
        self.auto_sigma_check.setChecked(True)
        paths_layout.addRow("", self.auto_sigma_check)

        # Skip existing
        self.skip_existing_check = QCheckBox("Skip already processed files")
        self.skip_existing_check.setChecked(True)
        paths_layout.addRow("", self.skip_existing_check)

        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)

        # Buttons
        btn_layout = QHBoxLayout()

        self.run_btn = QPushButton("Run IRAF Photometry")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.run_btn.clicked.connect(self.run_photometry)
        btn_layout.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_photometry)
        btn_layout.addWidget(self.stop_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        layout.addWidget(self.progress_label)

        # Log
        log_group = QGroupBox("Execution Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(300)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        return tab

    # ========================================================================
    # Tab 2: IRAF Parameters
    # ========================================================================
    def _create_params_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Sub-tabs for each parameter group
        param_tabs = QTabWidget()

        param_tabs.addTab(self._create_datapars_panel(), "DATAPARS")
        param_tabs.addTab(self._create_findpars_panel(), "FINDPARS")
        param_tabs.addTab(self._create_centerpars_panel(), "CENTERPARS")
        param_tabs.addTab(self._create_fitskypars_panel(), "FITSKYPARS")
        param_tabs.addTab(self._create_photpars_panel(), "PHOTPARS")

        layout.addWidget(param_tabs)

        btn_layout = QHBoxLayout()

        note = QLabel("Parameters auto-save on run/close.")
        note.setStyleSheet("color: #555; font-style: italic;")
        btn_layout.addWidget(note)
        btn_layout.addStretch()

        defaults_btn = QPushButton("Reset to Defaults")
        defaults_btn.clicked.connect(self._load_defaults)
        btn_layout.addWidget(defaults_btn)

        layout.addLayout(btn_layout)

        # Auto-load saved parameters if file exists
        self._auto_load_params()

        return tab

    def _create_datapars_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QWidget()
        layout = QFormLayout(panel)

        # --- Pixel Scale ---
        layout.addRow(QLabel("--- Pixel Scale ---"))
        self.dp_pix_scale = QDoubleSpinBox()
        self.dp_pix_scale.setRange(0.01, 10.0)
        self.dp_pix_scale.setDecimals(3)
        self.dp_pix_scale.setValue(self.iraf_params.pix_scale)
        layout.addRow("pix_scale (arcsec/pix):", self.dp_pix_scale)

        # --- Filter-specific SEEING (FWHM in arcsec) ---
        layout.addRow(QLabel(""))
        layout.addRow(QLabel("--- Filter-specific Seeing (arcsec) ---"))

        self.dp_seeing_g = QDoubleSpinBox()
        self.dp_seeing_g.setRange(0.1, 20.0)
        self.dp_seeing_g.setDecimals(2)
        self.dp_seeing_g.setValue(self.iraf_params.seeing_g)
        layout.addRow("seeing (g band):", self.dp_seeing_g)

        self.dp_seeing_r = QDoubleSpinBox()
        self.dp_seeing_r.setRange(0.1, 20.0)
        self.dp_seeing_r.setDecimals(2)
        self.dp_seeing_r.setValue(self.iraf_params.seeing_r)
        layout.addRow("seeing (r band):", self.dp_seeing_r)

        self.dp_seeing_i = QDoubleSpinBox()
        self.dp_seeing_i.setRange(0.1, 20.0)
        self.dp_seeing_i.setDecimals(2)
        self.dp_seeing_i.setValue(self.iraf_params.seeing_i)
        layout.addRow("seeing (i band):", self.dp_seeing_i)

        self.dp_seeing_default = QDoubleSpinBox()
        self.dp_seeing_default.setRange(0.1, 20.0)
        self.dp_seeing_default.setDecimals(2)
        self.dp_seeing_default.setValue(self.iraf_params.seeing_default)
        layout.addRow("seeing (default):", self.dp_seeing_default)

        # --- Filter-specific SIGMA (sky background std) ---
        layout.addRow(QLabel(""))
        layout.addRow(QLabel("--- Filter-specific Sigma (sky std) ---"))

        self.dp_sigma_g = QDoubleSpinBox()
        self.dp_sigma_g.setRange(0.1, 10000.0)
        self.dp_sigma_g.setDecimals(1)
        self.dp_sigma_g.setValue(self.iraf_params.sigma_g)
        layout.addRow("sigma (g band):", self.dp_sigma_g)

        self.dp_sigma_r = QDoubleSpinBox()
        self.dp_sigma_r.setRange(0.1, 10000.0)
        self.dp_sigma_r.setDecimals(1)
        self.dp_sigma_r.setValue(self.iraf_params.sigma_r)
        layout.addRow("sigma (r band):", self.dp_sigma_r)

        self.dp_sigma_i = QDoubleSpinBox()
        self.dp_sigma_i.setRange(0.1, 10000.0)
        self.dp_sigma_i.setDecimals(1)
        self.dp_sigma_i.setValue(self.iraf_params.sigma_i)
        layout.addRow("sigma (i band):", self.dp_sigma_i)

        self.dp_sigma_default = QDoubleSpinBox()
        self.dp_sigma_default.setRange(0.1, 10000.0)
        self.dp_sigma_default.setDecimals(1)
        self.dp_sigma_default.setValue(self.iraf_params.sigma_default)
        layout.addRow("sigma (default):", self.dp_sigma_default)

        # Sigma reference (for threshold scaling)
        self.dp_sigma_ref = QDoubleSpinBox()
        self.dp_sigma_ref.setRange(1.0, 1000.0)
        self.dp_sigma_ref.setDecimals(1)
        self.dp_sigma_ref.setValue(self.iraf_params.sigma_ref)
        layout.addRow("sigma_ref (thr scaling):", self.dp_sigma_ref)

        # --- Other DATAPARS ---
        layout.addRow(QLabel(""))
        layout.addRow(QLabel("--- Other DATAPARS ---"))

        # Scale
        self.dp_scale = QDoubleSpinBox()
        self.dp_scale.setRange(0.01, 100.0)
        self.dp_scale.setDecimals(3)
        self.dp_scale.setValue(self.iraf_params.scale)
        layout.addRow("scale (units/pixel):", self.dp_scale)

        # Emission
        self.dp_emission = QCheckBox()
        self.dp_emission.setChecked(self.iraf_params.emission)
        layout.addRow("emission:", self.dp_emission)

        # Datamax
        self.dp_datamax = QDoubleSpinBox()
        self.dp_datamax.setRange(0, 1000000)
        self.dp_datamax.setDecimals(0)
        self.dp_datamax.setValue(self.iraf_params.datamax)
        layout.addRow("datamax:", self.dp_datamax)

        # Noise model
        self.dp_noise = QComboBox()
        self.dp_noise.addItems(["poisson", "constant", "file"])
        self.dp_noise.setCurrentText(self.iraf_params.noise)
        layout.addRow("noise model:", self.dp_noise)

        # Readnoise
        self.dp_readnoise = QDoubleSpinBox()
        self.dp_readnoise.setRange(0.0, 1000.0)
        self.dp_readnoise.setDecimals(2)
        self.dp_readnoise.setValue(self.iraf_params.readnoise)
        layout.addRow("readnoise (e-):", self.dp_readnoise)

        # Gain (epadu)
        self.dp_epadu = QDoubleSpinBox()
        self.dp_epadu.setRange(0.001, 1000.0)
        self.dp_epadu.setDecimals(3)
        self.dp_epadu.setValue(self.iraf_params.epadu)
        layout.addRow("epadu (e-/ADU):", self.dp_epadu)

        # Exposure keyword
        self.dp_exposure = QLineEdit(self.iraf_params.exposure)
        layout.addRow("exposure keyword:", self.dp_exposure)

        # Integration time
        self.dp_itime = QDoubleSpinBox()
        self.dp_itime.setRange(0.001, 100000.0)
        self.dp_itime.setDecimals(2)
        self.dp_itime.setValue(self.iraf_params.itime)
        layout.addRow("itime (default):", self.dp_itime)

        scroll.setWidget(panel)
        return scroll

    def _create_findpars_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QWidget()
        layout = QFormLayout(panel)

        # --- Filter-specific Threshold ---
        layout.addRow(QLabel("--- Filter-specific Threshold ---"))
        layout.addRow(QLabel("(threshold = base × clip(sigma/sigma_ref, 0.8, 1.6))"))

        self.fp_threshold_g = QDoubleSpinBox()
        self.fp_threshold_g.setRange(0.1, 100.0)
        self.fp_threshold_g.setDecimals(2)
        self.fp_threshold_g.setValue(self.iraf_params.threshold_g)
        layout.addRow("threshold (g band):", self.fp_threshold_g)

        self.fp_threshold_r = QDoubleSpinBox()
        self.fp_threshold_r.setRange(0.1, 100.0)
        self.fp_threshold_r.setDecimals(2)
        self.fp_threshold_r.setValue(self.iraf_params.threshold_r)
        layout.addRow("threshold (r band):", self.fp_threshold_r)

        self.fp_threshold_i = QDoubleSpinBox()
        self.fp_threshold_i.setRange(0.1, 100.0)
        self.fp_threshold_i.setDecimals(2)
        self.fp_threshold_i.setValue(self.iraf_params.threshold_i)
        layout.addRow("threshold (i band):", self.fp_threshold_i)

        self.fp_threshold_default = QDoubleSpinBox()
        self.fp_threshold_default.setRange(0.1, 100.0)
        self.fp_threshold_default.setDecimals(2)
        self.fp_threshold_default.setValue(self.iraf_params.threshold_default)
        layout.addRow("threshold (default):", self.fp_threshold_default)

        # --- Filter-specific Sharplo ---
        layout.addRow(QLabel(""))
        layout.addRow(QLabel("--- Filter-specific Sharplo ---"))

        self.fp_sharplo_g = QDoubleSpinBox()
        self.fp_sharplo_g.setRange(-10.0, 10.0)
        self.fp_sharplo_g.setDecimals(2)
        self.fp_sharplo_g.setValue(self.iraf_params.sharplo_g)
        layout.addRow("sharplo (g band):", self.fp_sharplo_g)

        self.fp_sharplo_r = QDoubleSpinBox()
        self.fp_sharplo_r.setRange(-10.0, 10.0)
        self.fp_sharplo_r.setDecimals(2)
        self.fp_sharplo_r.setValue(self.iraf_params.sharplo_r)
        layout.addRow("sharplo (r band):", self.fp_sharplo_r)

        self.fp_sharplo_i = QDoubleSpinBox()
        self.fp_sharplo_i.setRange(-10.0, 10.0)
        self.fp_sharplo_i.setDecimals(2)
        self.fp_sharplo_i.setValue(self.iraf_params.sharplo_i)
        layout.addRow("sharplo (i band):", self.fp_sharplo_i)

        self.fp_sharplo_default = QDoubleSpinBox()
        self.fp_sharplo_default.setRange(-10.0, 10.0)
        self.fp_sharplo_default.setDecimals(2)
        self.fp_sharplo_default.setValue(self.iraf_params.sharplo_default)
        layout.addRow("sharplo (default):", self.fp_sharplo_default)

        # --- Filter-specific Datamin ---
        layout.addRow(QLabel(""))
        layout.addRow(QLabel("--- Filter-specific Datamin ---"))

        self.fp_datamin_g = QDoubleSpinBox()
        self.fp_datamin_g.setRange(-100000, 100000)
        self.fp_datamin_g.setDecimals(1)
        self.fp_datamin_g.setValue(self.iraf_params.datamin_g)
        layout.addRow("datamin (g band):", self.fp_datamin_g)

        self.fp_datamin_r = QDoubleSpinBox()
        self.fp_datamin_r.setRange(-100000, 100000)
        self.fp_datamin_r.setDecimals(1)
        self.fp_datamin_r.setValue(self.iraf_params.datamin_r)
        layout.addRow("datamin (r band):", self.fp_datamin_r)

        self.fp_datamin_i = QDoubleSpinBox()
        self.fp_datamin_i.setRange(-100000, 100000)
        self.fp_datamin_i.setDecimals(1)
        self.fp_datamin_i.setValue(self.iraf_params.datamin_i)
        layout.addRow("datamin (i band):", self.fp_datamin_i)

        self.fp_datamin_default = QDoubleSpinBox()
        self.fp_datamin_default.setRange(-100000, 100000)
        self.fp_datamin_default.setDecimals(1)
        self.fp_datamin_default.setValue(self.iraf_params.datamin_default)
        layout.addRow("datamin (default):", self.fp_datamin_default)

        # --- Other FINDPARS ---
        layout.addRow(QLabel(""))
        layout.addRow(QLabel("--- Other FINDPARS ---"))

        # Nsigma
        self.fp_nsigma = QDoubleSpinBox()
        self.fp_nsigma.setRange(0.1, 10.0)
        self.fp_nsigma.setDecimals(2)
        self.fp_nsigma.setValue(self.iraf_params.nsigma)
        layout.addRow("nsigma:", self.fp_nsigma)

        # Ratio
        self.fp_ratio = QDoubleSpinBox()
        self.fp_ratio.setRange(0.0, 1.0)
        self.fp_ratio.setDecimals(2)
        self.fp_ratio.setValue(self.iraf_params.ratio)
        layout.addRow("ratio (minor/major):", self.fp_ratio)

        # Theta
        self.fp_theta = QDoubleSpinBox()
        self.fp_theta.setRange(-180.0, 180.0)
        self.fp_theta.setDecimals(1)
        self.fp_theta.setValue(self.iraf_params.theta)
        layout.addRow("theta (degrees):", self.fp_theta)

        # Sharphi
        self.fp_sharphi = QDoubleSpinBox()
        self.fp_sharphi.setRange(-10.0, 10.0)
        self.fp_sharphi.setDecimals(2)
        self.fp_sharphi.setValue(self.iraf_params.sharphi)
        layout.addRow("sharphi:", self.fp_sharphi)

        # Roundlo
        self.fp_roundlo = QDoubleSpinBox()
        self.fp_roundlo.setRange(-10.0, 10.0)
        self.fp_roundlo.setDecimals(2)
        self.fp_roundlo.setValue(self.iraf_params.roundlo)
        layout.addRow("roundlo:", self.fp_roundlo)

        # Roundhi
        self.fp_roundhi = QDoubleSpinBox()
        self.fp_roundhi.setRange(-10.0, 10.0)
        self.fp_roundhi.setDecimals(2)
        self.fp_roundhi.setValue(self.iraf_params.roundhi)
        layout.addRow("roundhi:", self.fp_roundhi)

        scroll.setWidget(panel)
        return scroll

    def _create_centerpars_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QWidget()
        layout = QFormLayout(panel)

        # Centering algorithm
        self.cp_calgorithm = QComboBox()
        self.cp_calgorithm.addItems(["none", "centroid", "gauss", "ofilter"])
        self.cp_calgorithm.setCurrentText(self.iraf_params.calgorithm)
        layout.addRow("calgorithm:", self.cp_calgorithm)

        # Cbox multiplier (cbox = FWHM * cbox_mult)
        layout.addRow(QLabel("--- FWHM Multiplier ---"))
        self.cp_cbox_mult = QDoubleSpinBox()
        self.cp_cbox_mult.setRange(0.5, 10.0)
        self.cp_cbox_mult.setDecimals(1)
        self.cp_cbox_mult.setValue(self.iraf_params.cbox_mult)
        layout.addRow("cbox_mult (cbox = FWHM ×):", self.cp_cbox_mult)
        layout.addRow(QLabel(""))

        # Cthreshold
        self.cp_cthreshold = QDoubleSpinBox()
        self.cp_cthreshold.setRange(0.0, 100.0)
        self.cp_cthreshold.setDecimals(2)
        self.cp_cthreshold.setValue(self.iraf_params.cthreshold)
        layout.addRow("cthreshold:", self.cp_cthreshold)

        # Min SNR
        self.cp_minsnratio = QDoubleSpinBox()
        self.cp_minsnratio.setRange(0.0, 100.0)
        self.cp_minsnratio.setDecimals(2)
        self.cp_minsnratio.setValue(self.iraf_params.minsnratio)
        layout.addRow("minsnratio:", self.cp_minsnratio)

        # Max iterations
        self.cp_cmaxiter = QSpinBox()
        self.cp_cmaxiter.setRange(1, 100)
        self.cp_cmaxiter.setValue(self.iraf_params.cmaxiter)
        layout.addRow("cmaxiter:", self.cp_cmaxiter)

        # Max shift
        self.cp_maxshift = QDoubleSpinBox()
        self.cp_maxshift.setRange(0.0, 100.0)
        self.cp_maxshift.setDecimals(2)
        self.cp_maxshift.setValue(self.iraf_params.maxshift)
        layout.addRow("maxshift:", self.cp_maxshift)

        # Clean
        self.cp_clean = QCheckBox()
        self.cp_clean.setChecked(self.iraf_params.clean)
        layout.addRow("clean:", self.cp_clean)

        # Rclean
        self.cp_rclean = QDoubleSpinBox()
        self.cp_rclean.setRange(0.0, 100.0)
        self.cp_rclean.setDecimals(2)
        self.cp_rclean.setValue(self.iraf_params.rclean)
        layout.addRow("rclean:", self.cp_rclean)

        # Rclip
        self.cp_rclip = QDoubleSpinBox()
        self.cp_rclip.setRange(0.0, 100.0)
        self.cp_rclip.setDecimals(2)
        self.cp_rclip.setValue(self.iraf_params.rclip)
        layout.addRow("rclip:", self.cp_rclip)

        # Kclean
        self.cp_kclean = QDoubleSpinBox()
        self.cp_kclean.setRange(0.0, 100.0)
        self.cp_kclean.setDecimals(2)
        self.cp_kclean.setValue(self.iraf_params.kclean)
        layout.addRow("kclean:", self.cp_kclean)

        scroll.setWidget(panel)
        return scroll

    def _create_fitskypars_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QWidget()
        layout = QFormLayout(panel)

        # Sky algorithm
        self.sp_salgorithm = QComboBox()
        self.sp_salgorithm.addItems([
            "constant", "file", "mean", "median", "mode",
            "centroid", "gauss", "ofilter", "crosscor", "histplot"
        ])
        self.sp_salgorithm.setCurrentText(self.iraf_params.salgorithm)
        layout.addRow("salgorithm:", self.sp_salgorithm)

        # --- FWHM Multipliers ---
        layout.addRow(QLabel("--- FWHM Multipliers ---"))

        # Annulus multiplier (annulus = FWHM * annulus_mult)
        self.sp_annulus_mult = QDoubleSpinBox()
        self.sp_annulus_mult.setRange(1.0, 20.0)
        self.sp_annulus_mult.setDecimals(1)
        self.sp_annulus_mult.setValue(self.iraf_params.annulus_mult)
        layout.addRow("annulus_mult (ann = FWHM ×):", self.sp_annulus_mult)

        # Dannulus multiplier (dannulus = FWHM * dannulus_mult)
        self.sp_dannulus_mult = QDoubleSpinBox()
        self.sp_dannulus_mult.setRange(0.5, 10.0)
        self.sp_dannulus_mult.setDecimals(1)
        self.sp_dannulus_mult.setValue(self.iraf_params.dannulus_mult)
        layout.addRow("dannulus_mult (width = FWHM ×):", self.sp_dannulus_mult)

        layout.addRow(QLabel(""))

        # Skyvalue
        self.sp_skyvalue = QDoubleSpinBox()
        self.sp_skyvalue.setRange(-100000, 100000)
        self.sp_skyvalue.setDecimals(2)
        self.sp_skyvalue.setValue(self.iraf_params.skyvalue)
        layout.addRow("skyvalue:", self.sp_skyvalue)

        # Smaxiter
        self.sp_smaxiter = QSpinBox()
        self.sp_smaxiter.setRange(1, 100)
        self.sp_smaxiter.setValue(self.iraf_params.smaxiter)
        layout.addRow("smaxiter:", self.sp_smaxiter)

        # Sloclip
        self.sp_sloclip = QDoubleSpinBox()
        self.sp_sloclip.setRange(0.0, 100.0)
        self.sp_sloclip.setDecimals(2)
        self.sp_sloclip.setValue(self.iraf_params.sloclip)
        layout.addRow("sloclip:", self.sp_sloclip)

        # Shiclip
        self.sp_shiclip = QDoubleSpinBox()
        self.sp_shiclip.setRange(0.0, 100.0)
        self.sp_shiclip.setDecimals(2)
        self.sp_shiclip.setValue(self.iraf_params.shiclip)
        layout.addRow("shiclip:", self.sp_shiclip)

        # Snreject
        self.sp_snreject = QSpinBox()
        self.sp_snreject.setRange(0, 1000)
        self.sp_snreject.setValue(self.iraf_params.snreject)
        layout.addRow("snreject:", self.sp_snreject)

        # Sloreject
        self.sp_sloreject = QDoubleSpinBox()
        self.sp_sloreject.setRange(0.0, 100.0)
        self.sp_sloreject.setDecimals(2)
        self.sp_sloreject.setValue(self.iraf_params.sloreject)
        layout.addRow("sloreject (sigma):", self.sp_sloreject)

        # Shireject
        self.sp_shireject = QDoubleSpinBox()
        self.sp_shireject.setRange(0.0, 100.0)
        self.sp_shireject.setDecimals(2)
        self.sp_shireject.setValue(self.iraf_params.shireject)
        layout.addRow("shireject (sigma):", self.sp_shireject)

        # Khist
        self.sp_khist = QDoubleSpinBox()
        self.sp_khist.setRange(0.0, 100.0)
        self.sp_khist.setDecimals(2)
        self.sp_khist.setValue(self.iraf_params.khist)
        layout.addRow("khist:", self.sp_khist)

        # Binsize
        self.sp_binsize = QDoubleSpinBox()
        self.sp_binsize.setRange(0.001, 10.0)
        self.sp_binsize.setDecimals(3)
        self.sp_binsize.setValue(self.iraf_params.binsize)
        layout.addRow("binsize:", self.sp_binsize)

        # Smooth
        self.sp_smooth = QCheckBox()
        self.sp_smooth.setChecked(self.iraf_params.smooth)
        layout.addRow("smooth:", self.sp_smooth)

        # Rgrow
        self.sp_rgrow = QDoubleSpinBox()
        self.sp_rgrow.setRange(0.0, 100.0)
        self.sp_rgrow.setDecimals(2)
        self.sp_rgrow.setValue(self.iraf_params.rgrow)
        layout.addRow("rgrow:", self.sp_rgrow)

        scroll.setWidget(panel)
        return scroll

    def _create_photpars_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QWidget()
        layout = QFormLayout(panel)

        # --- FWHM Multiplier ---
        layout.addRow(QLabel("--- FWHM Multiplier ---"))

        # Aperture multiplier (aperture = FWHM * aperture_mult)
        self.pp_aperture_mult = QDoubleSpinBox()
        self.pp_aperture_mult.setRange(0.5, 10.0)
        self.pp_aperture_mult.setDecimals(1)
        self.pp_aperture_mult.setValue(self.iraf_params.aperture_mult)
        layout.addRow("aperture_mult (ap = FWHM ×):", self.pp_aperture_mult)

        layout.addRow(QLabel(""))
        layout.addRow(QLabel("--- Other PHOTPARS ---"))

        # Zmag
        self.pp_zmag = QDoubleSpinBox()
        self.pp_zmag.setRange(0.0, 50.0)
        self.pp_zmag.setDecimals(2)
        self.pp_zmag.setValue(self.iraf_params.zmag)
        layout.addRow("zmag (zero point):", self.pp_zmag)

        # Mkapert
        self.pp_mkapert = QCheckBox()
        self.pp_mkapert.setChecked(self.iraf_params.mkapert)
        layout.addRow("mkapert:", self.pp_mkapert)

        # Info note
        layout.addRow(QLabel(""))
        layout.addRow(QLabel("--- Summary ---"))
        layout.addRow(QLabel("FWHM is auto-calculated per image from header"))
        layout.addRow(QLabel("(SEEING, FWHMARC) or defaults to seeing/pix_scale"))
        layout.addRow(QLabel(""))
        layout.addRow(QLabel("aperture = FWHM × aperture_mult"))
        layout.addRow(QLabel("annulus = FWHM × annulus_mult (in FITSKYPARS)"))
        layout.addRow(QLabel("dannulus = FWHM × dannulus_mult (in FITSKYPARS)"))
        layout.addRow(QLabel("cbox = FWHM × cbox_mult (in CENTERPARS)"))

        scroll.setWidget(panel)
        return scroll

    def _get_config_path(self) -> Path:
        """Get path for IRAF config file (separate from project state)."""
        return self.result_dir / "iraf_config.json"

    def _load_iraf_params_from_toml(self) -> bool:
        toml_path = Path("parameters.toml")
        if not toml_path.exists():
            return False
        try:
            data = toml_path.read_text(encoding="utf-8")
            cfg = tomllib.loads(data)
        except Exception:
            return False

        tools = cfg.get("tools", {}) if isinstance(cfg, dict) else {}
        iraf_cfg = tools.get("iraf", {}) if isinstance(tools, dict) else {}
        if not iraf_cfg:
            iraf_cfg = cfg.get("iraf", {})
        params = iraf_cfg.get("params", {})
        if isinstance(params, dict) and params:
            try:
                self.iraf_params.from_dict(params)
                self._update_ui_from_params()
                return True
            except Exception:
                return False
        return False

    def _save_iraf_params_to_toml(self, notify: bool = False):
        if tomli_w is None:
            if notify:
                QMessageBox.warning(self, "Error", "tomli_w is required to write parameters.toml")
            return
        toml_path = Path("parameters.toml")
        try:
            if toml_path.exists():
                cfg = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            else:
                cfg = {}
            tools = cfg.get("tools", {})
            if not isinstance(tools, dict):
                tools = {}
            iraf_cfg = tools.get("iraf", {})
            if not isinstance(iraf_cfg, dict):
                iraf_cfg = {}
            iraf_cfg["params"] = self.iraf_params.to_dict()
            tools["iraf"] = iraf_cfg
            cfg["tools"] = tools
            toml_path.write_text(tomli_w.dumps(cfg), encoding="utf-8")
            if notify:
                QMessageBox.information(self, "Saved", f"IRAF parameters saved to:\n{toml_path}")
        except Exception as e:
            if notify:
                QMessageBox.warning(self, "Error", f"Failed to save: {e}")

    def _save_params_to_file(self):
        """Save current parameters to separate config file."""
        self._apply_params_silent()  # Apply UI values to params object first
        self._save_iraf_params_to_toml(notify=True)

    def _auto_save_params(self):
        """Auto-save current parameters without UI prompts."""
        self._apply_params_silent()
        self._save_iraf_params_to_toml(notify=False)

    def _load_params_from_file(self):
        """Load parameters from config file."""
        if self._load_iraf_params_from_toml():
            QMessageBox.information(self, "Loaded", "IRAF parameters loaded from parameters.toml")
            return
        config_path = self._get_config_path()
        if not config_path.exists():
            QMessageBox.information(self, "Not Found", "No IRAF config found")
            return
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.iraf_params.from_dict(data)
            self._update_ui_from_params()
            QMessageBox.information(self, "Loaded", f"IRAF parameters loaded from:\n{config_path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load: {e}")

    def _auto_load_params(self):
        """Auto-load parameters from config file if exists."""
        if self._load_iraf_params_from_toml():
            return
        config_path = self._get_config_path()
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.iraf_params.from_dict(data)
                self._update_ui_from_params()
            except Exception:
                pass  # Silently ignore errors on auto-load

    def _apply_params_silent(self):
        """Apply UI values to params object without showing message."""
        p = self.iraf_params
        # DATAPARS - Pixel scale
        p.pix_scale = self.dp_pix_scale.value()
        # DATAPARS - Filter-specific seeing
        p.seeing_g = self.dp_seeing_g.value()
        p.seeing_r = self.dp_seeing_r.value()
        p.seeing_i = self.dp_seeing_i.value()
        p.seeing_default = self.dp_seeing_default.value()
        # DATAPARS - Filter-specific sigma
        p.sigma_g = self.dp_sigma_g.value()
        p.sigma_r = self.dp_sigma_r.value()
        p.sigma_i = self.dp_sigma_i.value()
        p.sigma_default = self.dp_sigma_default.value()
        p.sigma_ref = self.dp_sigma_ref.value()
        # DATAPARS - Other
        p.scale = self.dp_scale.value()
        p.emission = self.dp_emission.isChecked()
        p.datamax = self.dp_datamax.value()
        p.noise = self.dp_noise.currentText()
        p.readnoise = self.dp_readnoise.value()
        p.epadu = self.dp_epadu.value()
        p.exposure = self.dp_exposure.text()
        p.itime = self.dp_itime.value()
        # FINDPARS - Filter-specific
        p.threshold_g = self.fp_threshold_g.value()
        p.threshold_r = self.fp_threshold_r.value()
        p.threshold_i = self.fp_threshold_i.value()
        p.threshold_default = self.fp_threshold_default.value()
        p.sharplo_g = self.fp_sharplo_g.value()
        p.sharplo_r = self.fp_sharplo_r.value()
        p.sharplo_i = self.fp_sharplo_i.value()
        p.sharplo_default = self.fp_sharplo_default.value()
        p.datamin_g = self.fp_datamin_g.value()
        p.datamin_r = self.fp_datamin_r.value()
        p.datamin_i = self.fp_datamin_i.value()
        p.datamin_default = self.fp_datamin_default.value()
        # FINDPARS - Other
        p.nsigma = self.fp_nsigma.value()
        p.ratio = self.fp_ratio.value()
        p.theta = self.fp_theta.value()
        p.sharphi = self.fp_sharphi.value()
        p.roundlo = self.fp_roundlo.value()
        p.roundhi = self.fp_roundhi.value()
        # CENTERPARS
        p.calgorithm = self.cp_calgorithm.currentText()
        p.cbox_mult = self.cp_cbox_mult.value()
        p.cthreshold = self.cp_cthreshold.value()
        p.minsnratio = self.cp_minsnratio.value()
        p.cmaxiter = self.cp_cmaxiter.value()
        p.maxshift = self.cp_maxshift.value()
        p.clean = self.cp_clean.isChecked()
        p.rclean = self.cp_rclean.value()
        p.rclip = self.cp_rclip.value()
        p.kclean = self.cp_kclean.value()
        # FITSKYPARS
        p.salgorithm = self.sp_salgorithm.currentText()
        p.annulus_mult = self.sp_annulus_mult.value()
        p.dannulus_mult = self.sp_dannulus_mult.value()
        p.skyvalue = self.sp_skyvalue.value()
        p.smaxiter = self.sp_smaxiter.value()
        p.sloclip = self.sp_sloclip.value()
        p.shiclip = self.sp_shiclip.value()
        p.snreject = self.sp_snreject.value()
        p.sloreject = self.sp_sloreject.value()
        p.shireject = self.sp_shireject.value()
        p.khist = self.sp_khist.value()
        p.binsize = self.sp_binsize.value()
        p.smooth = self.sp_smooth.isChecked()
        p.rgrow = self.sp_rgrow.value()
        # PHOTPARS
        p.aperture_mult = self.pp_aperture_mult.value()
        p.zmag = self.pp_zmag.value()
        p.mkapert = self.pp_mkapert.isChecked()

    def _load_defaults(self):
        self.iraf_params = IRAFParameters()
        self._update_ui_from_params()
        QMessageBox.information(self, "Defaults Loaded", "Default IRAF parameters restored.")

    def _update_ui_from_params(self):
        p = self.iraf_params
        # DATAPARS - Pixel scale
        self.dp_pix_scale.setValue(p.pix_scale)
        # DATAPARS - Filter-specific seeing
        self.dp_seeing_g.setValue(p.seeing_g)
        self.dp_seeing_r.setValue(p.seeing_r)
        self.dp_seeing_i.setValue(p.seeing_i)
        self.dp_seeing_default.setValue(p.seeing_default)
        # DATAPARS - Filter-specific sigma
        self.dp_sigma_g.setValue(p.sigma_g)
        self.dp_sigma_r.setValue(p.sigma_r)
        self.dp_sigma_i.setValue(p.sigma_i)
        self.dp_sigma_default.setValue(p.sigma_default)
        self.dp_sigma_ref.setValue(p.sigma_ref)
        # DATAPARS - Other
        self.dp_scale.setValue(p.scale)
        self.dp_emission.setChecked(p.emission)
        self.dp_datamax.setValue(p.datamax)
        self.dp_noise.setCurrentText(p.noise)
        self.dp_readnoise.setValue(p.readnoise)
        self.dp_epadu.setValue(p.epadu)
        self.dp_exposure.setText(p.exposure)
        self.dp_itime.setValue(p.itime)
        # FINDPARS - Filter-specific
        self.fp_threshold_g.setValue(p.threshold_g)
        self.fp_threshold_r.setValue(p.threshold_r)
        self.fp_threshold_i.setValue(p.threshold_i)
        self.fp_threshold_default.setValue(p.threshold_default)
        self.fp_sharplo_g.setValue(p.sharplo_g)
        self.fp_sharplo_r.setValue(p.sharplo_r)
        self.fp_sharplo_i.setValue(p.sharplo_i)
        self.fp_sharplo_default.setValue(p.sharplo_default)
        self.fp_datamin_g.setValue(p.datamin_g)
        self.fp_datamin_r.setValue(p.datamin_r)
        self.fp_datamin_i.setValue(p.datamin_i)
        self.fp_datamin_default.setValue(p.datamin_default)
        # FINDPARS - Other
        self.fp_nsigma.setValue(p.nsigma)
        self.fp_ratio.setValue(p.ratio)
        self.fp_theta.setValue(p.theta)
        self.fp_sharphi.setValue(p.sharphi)
        self.fp_roundlo.setValue(p.roundlo)
        self.fp_roundhi.setValue(p.roundhi)
        # CENTERPARS
        self.cp_calgorithm.setCurrentText(p.calgorithm)
        self.cp_cbox_mult.setValue(p.cbox_mult)
        self.cp_cthreshold.setValue(p.cthreshold)
        self.cp_minsnratio.setValue(p.minsnratio)
        self.cp_cmaxiter.setValue(p.cmaxiter)
        self.cp_maxshift.setValue(p.maxshift)
        self.cp_clean.setChecked(p.clean)
        self.cp_rclean.setValue(p.rclean)
        self.cp_rclip.setValue(p.rclip)
        self.cp_kclean.setValue(p.kclean)
        # FITSKYPARS
        self.sp_salgorithm.setCurrentText(p.salgorithm)
        self.sp_annulus_mult.setValue(p.annulus_mult)
        self.sp_dannulus_mult.setValue(p.dannulus_mult)
        self.sp_skyvalue.setValue(p.skyvalue)
        self.sp_smaxiter.setValue(p.smaxiter)
        self.sp_sloclip.setValue(p.sloclip)
        self.sp_shiclip.setValue(p.shiclip)
        self.sp_snreject.setValue(p.snreject)
        self.sp_sloreject.setValue(p.sloreject)
        self.sp_shireject.setValue(p.shireject)
        self.sp_khist.setValue(p.khist)
        self.sp_binsize.setValue(p.binsize)
        self.sp_smooth.setChecked(p.smooth)
        self.sp_rgrow.setValue(p.rgrow)
        # PHOTPARS
        self.pp_aperture_mult.setValue(p.aperture_mult)
        self.pp_zmag.setValue(p.zmag)
        self.pp_mkapert.setChecked(p.mkapert)

    def _apply_params(self):
        p = self.iraf_params
        # DATAPARS - Pixel scale
        p.pix_scale = self.dp_pix_scale.value()
        # DATAPARS - Filter-specific seeing
        p.seeing_g = self.dp_seeing_g.value()
        p.seeing_r = self.dp_seeing_r.value()
        p.seeing_i = self.dp_seeing_i.value()
        p.seeing_default = self.dp_seeing_default.value()
        # DATAPARS - Filter-specific sigma
        p.sigma_g = self.dp_sigma_g.value()
        p.sigma_r = self.dp_sigma_r.value()
        p.sigma_i = self.dp_sigma_i.value()
        p.sigma_default = self.dp_sigma_default.value()
        p.sigma_ref = self.dp_sigma_ref.value()
        # DATAPARS - Other
        p.scale = self.dp_scale.value()
        p.emission = self.dp_emission.isChecked()
        p.datamax = self.dp_datamax.value()
        p.noise = self.dp_noise.currentText()
        p.readnoise = self.dp_readnoise.value()
        p.epadu = self.dp_epadu.value()
        p.exposure = self.dp_exposure.text()
        p.itime = self.dp_itime.value()
        # FINDPARS - Filter-specific
        p.threshold_g = self.fp_threshold_g.value()
        p.threshold_r = self.fp_threshold_r.value()
        p.threshold_i = self.fp_threshold_i.value()
        p.threshold_default = self.fp_threshold_default.value()
        p.sharplo_g = self.fp_sharplo_g.value()
        p.sharplo_r = self.fp_sharplo_r.value()
        p.sharplo_i = self.fp_sharplo_i.value()
        p.sharplo_default = self.fp_sharplo_default.value()
        p.datamin_g = self.fp_datamin_g.value()
        p.datamin_r = self.fp_datamin_r.value()
        p.datamin_i = self.fp_datamin_i.value()
        p.datamin_default = self.fp_datamin_default.value()
        # FINDPARS - Other
        p.nsigma = self.fp_nsigma.value()
        p.ratio = self.fp_ratio.value()
        p.theta = self.fp_theta.value()
        p.sharphi = self.fp_sharphi.value()
        p.roundlo = self.fp_roundlo.value()
        p.roundhi = self.fp_roundhi.value()
        # CENTERPARS
        p.calgorithm = self.cp_calgorithm.currentText()
        p.cbox_mult = self.cp_cbox_mult.value()
        p.cthreshold = self.cp_cthreshold.value()
        p.minsnratio = self.cp_minsnratio.value()
        p.cmaxiter = self.cp_cmaxiter.value()
        p.maxshift = self.cp_maxshift.value()
        p.clean = self.cp_clean.isChecked()
        p.rclean = self.cp_rclean.value()
        p.rclip = self.cp_rclip.value()
        p.kclean = self.cp_kclean.value()
        # FITSKYPARS
        p.salgorithm = self.sp_salgorithm.currentText()
        p.annulus_mult = self.sp_annulus_mult.value()
        p.dannulus_mult = self.sp_dannulus_mult.value()
        p.skyvalue = self.sp_skyvalue.value()
        p.smaxiter = self.sp_smaxiter.value()
        p.sloclip = self.sp_sloclip.value()
        p.shiclip = self.sp_shiclip.value()
        p.snreject = self.sp_snreject.value()
        p.sloreject = self.sp_sloreject.value()
        p.shireject = self.sp_shireject.value()
        p.khist = self.sp_khist.value()
        p.binsize = self.sp_binsize.value()
        p.smooth = self.sp_smooth.isChecked()
        p.rgrow = self.sp_rgrow.value()
        # PHOTPARS
        p.aperture_mult = self.pp_aperture_mult.value()
        p.zmag = self.pp_zmag.value()
        p.mkapert = self.pp_mkapert.isChecked()

        self._auto_save_params()
        QMessageBox.information(self, "Applied", "Parameters applied to current session.")

    # ========================================================================
    # Tab 3: Comparison
    # ========================================================================
    def _create_comparison_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Settings
        settings_group = QGroupBox("Comparison Settings")
        settings_layout = QFormLayout()

        # AAPKI dir
        aapki_row = QHBoxLayout()
        self.cmp_aapki_edit = QLineEdit(str(self.result_dir))
        aapki_btn = QPushButton("Browse")
        aapki_btn.clicked.connect(lambda: self._browse_dir(self.cmp_aapki_edit))
        aapki_row.addWidget(self.cmp_aapki_edit)
        aapki_row.addWidget(aapki_btn)
        aapki_w = QWidget()
        aapki_w.setLayout(aapki_row)
        settings_layout.addRow("AAPKI Result Dir:", aapki_w)
        self.cmp_aapki_edit.editingFinished.connect(self._overlay_refresh_maps)

        # IRAF dir
        iraf_row = QHBoxLayout()
        self.cmp_iraf_edit = QLineEdit(str(self.result_dir / "iraf_phot"))
        iraf_btn = QPushButton("Browse")
        iraf_btn.clicked.connect(lambda: self._browse_dir(self.cmp_iraf_edit))
        iraf_row.addWidget(self.cmp_iraf_edit)
        iraf_row.addWidget(iraf_btn)
        iraf_w = QWidget()
        iraf_w.setLayout(iraf_row)
        settings_layout.addRow("IRAF Result Dir:", iraf_w)
        self.cmp_iraf_edit.editingFinished.connect(self._overlay_refresh_maps)

        # Image dir (FITS)
        img_row = QHBoxLayout()
        default_img = step2_cropped_dir(self.result_dir)
        if not default_img.exists():
            default_img = step2_cropped_dir(self.data_dir / "result")
        if not default_img.exists():
            default_img = self.data_dir
        self.cmp_image_edit = QLineEdit(str(default_img))
        img_btn = QPushButton("Browse")
        img_btn.clicked.connect(lambda: self._browse_dir(self.cmp_image_edit))
        img_row.addWidget(self.cmp_image_edit)
        img_row.addWidget(img_btn)
        img_w = QWidget()
        img_w.setLayout(img_row)
        settings_layout.addRow("Image Dir:", img_w)
        self.cmp_image_edit.editingFinished.connect(self._overlay_reload)

        # Tolerance
        self.cmp_tol = QDoubleSpinBox()
        self.cmp_tol.setRange(0.1, 20.0)
        self.cmp_tol.setValue(1.5)
        settings_layout.addRow("Match Tolerance (px):", self.cmp_tol)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self.cmp_run_btn = QPushButton("Run Comparison")
        self.cmp_run_btn.clicked.connect(self.run_comparison)
        btn_layout.addWidget(self.cmp_run_btn)

        self.cmp_export_btn = QPushButton("Export CSV")
        self.cmp_export_btn.clicked.connect(self.export_comparison)
        btn_layout.addWidget(self.cmp_export_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Summary
        self.cmp_summary = QLabel("No comparison run yet.")
        self.cmp_summary.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self.cmp_summary)

        # Tabs: results + log
        tabs = QTabWidget()

        result_tab = QWidget()
        result_layout = QVBoxLayout(result_tab)

        # Splitter: table + plot
        splitter = QSplitter(Qt.Horizontal)

        # Table
        self.cmp_table = QTableWidget(0, 13)
        self.cmp_table.setHorizontalHeaderLabels([
            "Frame", "Matched", "dmag_med", "dmag_std", "dx_med", "dy_med",
            "dist_med", "dist_p95", "frac<=tol", "shift_x", "shift_y",
            "N_iraf", "N_aapki"
        ])
        self.cmp_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.cmp_table.horizontalHeader().setStretchLastSection(True)
        self.cmp_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.cmp_table.setSelectionMode(QTableWidget.SingleSelection)
        self.cmp_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cmp_table.itemSelectionChanged.connect(self._plot_comparison)
        splitter.addWidget(self.cmp_table)

        # Plot
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        self.cmp_fig = Figure(figsize=(8, 6), tight_layout=True)
        self.cmp_canvas = FigureCanvas(self.cmp_fig)
        self.cmp_toolbar = NavigationToolbar(self.cmp_canvas, self)
        plot_layout.addWidget(self.cmp_toolbar)
        plot_layout.addWidget(self.cmp_canvas)
        splitter.addWidget(plot_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        result_layout.addWidget(splitter)
        tabs.addTab(result_tab, "Results")

        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.cmp_log_text = QTextEdit()
        self.cmp_log_text.setReadOnly(True)
        self.cmp_log_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        log_layout.addWidget(self.cmp_log_text)
        tabs.addTab(log_tab, "Log")

        overlay_widget = self._create_overlay_widget()
        overlay_widget.setMinimumHeight(420)

        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(tabs)
        main_splitter.addWidget(overlay_widget)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 3)

        layout.addWidget(main_splitter)

        return tab

    def _create_overlay_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        info = QLabel(
            "Overlay AAPKI TSV positions with IRAF daofind positions on FITS frames."
        )
        info.setStyleSheet("QLabel { background-color: #FFF3CD; padding: 6px; border-radius: 4px; }")
        layout.addWidget(info)

        select_group = QGroupBox("Frame Selection")
        select_layout = QHBoxLayout(select_group)
        select_layout.addWidget(QLabel("Index:"))
        self.overlay_index_spin = QSpinBox()
        self.overlay_index_spin.setRange(0, 0)
        self.overlay_index_spin.valueChanged.connect(self._overlay_on_index_changed)
        select_layout.addWidget(self.overlay_index_spin)

        select_layout.addWidget(QLabel("File:"))
        self.overlay_file_combo = QComboBox()
        self.overlay_file_combo.currentIndexChanged.connect(self._overlay_on_file_changed)
        select_layout.addWidget(self.overlay_file_combo, stretch=1)

        btn_reload = QPushButton("Reload Frames")
        btn_reload.clicked.connect(self._overlay_reload)
        select_layout.addWidget(btn_reload)
        layout.addWidget(select_group)

        control_group = QGroupBox("Display Controls")
        control_layout = QHBoxLayout(control_group)
        control_layout.addWidget(QLabel("Stretch:"))
        self.overlay_scale_combo = QComboBox()
        self.overlay_scale_combo.addItems([
            "Auto Stretch (Siril)",
            "Asinh Stretch",
            "Midtone (MTF)",
            "Histogram Eq",
            "Log Stretch",
            "Sqrt Stretch",
            "Linear (1-99%)",
            "ZScale (IRAF)",
        ])
        self.overlay_scale_combo.currentIndexChanged.connect(self._overlay_on_stretch_changed)
        control_layout.addWidget(self.overlay_scale_combo)

        control_layout.addWidget(QLabel("Intensity:"))
        self.overlay_stretch_slider = QSlider(Qt.Horizontal)
        self.overlay_stretch_slider.setMinimum(1)
        self.overlay_stretch_slider.setMaximum(100)
        self.overlay_stretch_slider.setValue(25)
        self.overlay_stretch_slider.setFixedWidth(120)
        self.overlay_stretch_slider.sliderReleased.connect(self._overlay_redisplay)
        self.overlay_stretch_slider.valueChanged.connect(self._overlay_update_stretch_label)
        control_layout.addWidget(self.overlay_stretch_slider)

        self.overlay_stretch_value = QLabel("25")
        self.overlay_stretch_value.setFixedWidth(30)
        control_layout.addWidget(self.overlay_stretch_value)

        control_layout.addWidget(QLabel("Black:"))
        self.overlay_black_slider = QSlider(Qt.Horizontal)
        self.overlay_black_slider.setMinimum(0)
        self.overlay_black_slider.setMaximum(100)
        self.overlay_black_slider.setValue(0)
        self.overlay_black_slider.setFixedWidth(80)
        self.overlay_black_slider.sliderReleased.connect(self._overlay_redisplay)
        self.overlay_black_slider.valueChanged.connect(self._overlay_update_black_label)
        control_layout.addWidget(self.overlay_black_slider)

        self.overlay_black_value = QLabel("0")
        self.overlay_black_value.setFixedWidth(25)
        control_layout.addWidget(self.overlay_black_value)

        btn_reset_zoom = QPushButton("Reset Zoom")
        btn_reset_zoom.clicked.connect(self._overlay_reset_zoom)
        control_layout.addWidget(btn_reset_zoom)

        btn_reset_stretch = QPushButton("Reset Stretch")
        btn_reset_stretch.clicked.connect(self._overlay_reset_stretch)
        control_layout.addWidget(btn_reset_stretch)

        btn_2d_plot = QPushButton("2D Plot")
        btn_2d_plot.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; }")
        btn_2d_plot.clicked.connect(self._overlay_open_stretch_plot)
        control_layout.addWidget(btn_2d_plot)

        self.overlay_show_aapki = QCheckBox("AAPKI TSV")
        self.overlay_show_aapki.setChecked(True)
        self.overlay_show_aapki.stateChanged.connect(self._overlay_redisplay)
        control_layout.addWidget(self.overlay_show_aapki)

        self.overlay_show_iraf = QCheckBox("IRAF daofind")
        self.overlay_show_iraf.setChecked(True)
        self.overlay_show_iraf.stateChanged.connect(self._overlay_redisplay)
        control_layout.addWidget(self.overlay_show_iraf)

        control_layout.addStretch()
        layout.addWidget(control_group)

        self.overlay_fig = Figure(figsize=(10, 8))
        self.overlay_canvas = FigureCanvas(self.overlay_fig)
        self.overlay_canvas.setMinimumHeight(450)
        self.overlay_ax = self.overlay_fig.add_subplot(111)
        self.overlay_fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.95)
        self.overlay_canvas.setFocusPolicy(Qt.StrongFocus)
        self.overlay_canvas.mpl_connect("scroll_event", self._overlay_on_scroll)
        self.overlay_canvas.mpl_connect("button_press_event", self._overlay_on_button_press)
        self.overlay_canvas.mpl_connect("button_release_event", self._overlay_on_button_release)
        self.overlay_canvas.mpl_connect("motion_notify_event", self._overlay_on_motion)
        layout.addWidget(self.overlay_canvas, stretch=1)

        self.overlay_status = QLabel("No frame loaded.")
        self.overlay_status.setStyleSheet("QLabel { color: #555; padding: 4px; }")
        layout.addWidget(self.overlay_status)

        sc_prev = QShortcut(QKeySequence(Qt.Key_BracketLeft), widget)
        sc_prev.setContext(Qt.WidgetWithChildrenShortcut)
        sc_prev.activated.connect(lambda: self._overlay_navigate(-1))

        sc_next = QShortcut(QKeySequence(Qt.Key_BracketRight), widget)
        sc_next.setContext(Qt.WidgetWithChildrenShortcut)
        sc_next.activated.connect(lambda: self._overlay_navigate(1))

        sc_filter = QShortcut(QKeySequence("."), widget)
        sc_filter.setContext(Qt.WidgetWithChildrenShortcut)
        sc_filter.activated.connect(self._overlay_cycle_filter)

        self._overlay_reload()
        return widget

    def _build_iraf_param_defaults(self) -> dict:
        p = self.iraf_params
        return {
            "scale": p.scale,
            "emission": p.emission,
            "datamax": p.datamax,
            "noise": p.noise,
            "readnoise": p.readnoise,
            "epadu": p.epadu,
            "exposure": p.exposure,
            "itime": p.itime,
            "seeing_g": p.seeing_g,
            "seeing_r": p.seeing_r,
            "seeing_i": p.seeing_i,
            "seeing_default": p.seeing_default,
            "sigma_g": p.sigma_g,
            "sigma_r": p.sigma_r,
            "sigma_i": p.sigma_i,
            "sigma_default": p.sigma_default,
            "threshold_g": p.threshold_g,
            "threshold_r": p.threshold_r,
            "threshold_i": p.threshold_i,
            "threshold_default": p.threshold_default,
            "nsigma": p.nsigma,
            "ratio": p.ratio,
            "theta": p.theta,
            "sharplo_g": p.sharplo_g,
            "sharplo_r": p.sharplo_r,
            "sharplo_i": p.sharplo_i,
            "sharplo_default": p.sharplo_default,
            "sharphi": p.sharphi,
            "roundlo": p.roundlo,
            "roundhi": p.roundhi,
            "datamin_g": p.datamin_g,
            "datamin_r": p.datamin_r,
            "datamin_i": p.datamin_i,
            "datamin_default": p.datamin_default,
            "calgorithm": p.calgorithm,
            "cbox_mult": p.cbox_mult,
            "cthreshold": p.cthreshold,
            "minsnratio": p.minsnratio,
            "cmaxiter": p.cmaxiter,
            "maxshift": p.maxshift,
            "clean": p.clean,
            "rclean": p.rclean,
            "rclip": p.rclip,
            "kclean": p.kclean,
            "salgorithm": p.salgorithm,
            "annulus_mult": p.annulus_mult,
            "dannulus_mult": p.dannulus_mult,
            "skyvalue": p.skyvalue,
            "smaxiter": p.smaxiter,
            "sloclip": p.sloclip,
            "shiclip": p.shiclip,
            "snreject": p.snreject,
            "sloreject": p.sloreject,
            "shireject": p.shireject,
            "khist": p.khist,
            "binsize": p.binsize,
            "smooth": p.smooth,
            "rgrow": p.rgrow,
            "aperture_mult": p.aperture_mult,
            "zmag": p.zmag,
            "mkapert": p.mkapert,
            "pix_scale": p.pix_scale,
            "sigma_ref": p.sigma_ref,
        }

    def _load_iraf_toml_config(self) -> tuple[dict, dict]:
        toml_path = Path("parameters.toml")
        if not toml_path.exists():
            return {}, {}
        try:
            data = toml_path.read_text(encoding="utf-8")
            cfg = tomllib.loads(data)
        except Exception:
            return {}, {}

        tools = cfg.get("tools", {}) if isinstance(cfg, dict) else {}
        iraf_cfg = tools.get("iraf", {}) if isinstance(tools, dict) else {}
        if not iraf_cfg:
            iraf_cfg = cfg.get("iraf", {})
        raw_filters = iraf_cfg.get("filters", {})
        raw_aliases = iraf_cfg.get("filter_aliases", {})

        key_map = {
            "seeing_arcsec": "seeing",
            "seeing": "seeing",
            "sigma": "sigma",
            "threshold": "threshold",
            "sharplo": "sharplo",
            "datamin": "datamin",
            "nsigma": "nsigma",
            "ratio": "ratio",
            "theta": "theta",
            "sharphi": "sharphi",
            "roundlo": "roundlo",
            "roundhi": "roundhi",
            "calgorithm": "calgorithm",
            "cbox_mult": "cbox_mult",
            "cthreshold": "cthreshold",
            "minsnratio": "minsnratio",
            "cmaxiter": "cmaxiter",
            "maxshift": "maxshift",
            "clean": "clean",
            "rclean": "rclean",
            "rclip": "rclip",
            "kclean": "kclean",
            "salgorithm": "salgorithm",
            "annulus_mult": "annulus_mult",
            "dannulus_mult": "dannulus_mult",
            "skyvalue": "skyvalue",
            "smaxiter": "smaxiter",
            "sloclip": "sloclip",
            "shiclip": "shiclip",
            "snreject": "snreject",
            "sloreject": "sloreject",
            "shireject": "shireject",
            "khist": "khist",
            "binsize": "binsize",
            "smooth": "smooth",
            "rgrow": "rgrow",
            "aperture_mult": "aperture_mult",
            "zmag": "zmag",
            "mkapert": "mkapert",
            "scale": "scale",
            "emission": "emission",
            "datamax": "datamax",
            "noise": "noise",
            "readnoise": "readnoise",
            "epadu": "epadu",
            "exposure": "exposure",
            "itime": "itime",
            "sigma_ref": "sigma_ref",
        }

        filter_params = {}
        if isinstance(raw_filters, dict):
            for fkey, fvals in raw_filters.items():
                if not isinstance(fvals, dict):
                    continue
                key = str(fkey).strip().lower()
                params = {}
                for pkey, pval in fvals.items():
                    pname = str(pkey).strip().lower().replace("-", "_").replace(" ", "_")
                    pname = key_map.get(pname, pname)
                    params[pname] = pval
                filter_params[key] = params

        filter_aliases = {}
        if isinstance(raw_aliases, dict):
            for akey, aval in raw_aliases.items():
                if aval is None:
                    continue
                filter_aliases[str(akey).strip().lower()] = str(aval).strip().lower()

        return filter_params, filter_aliases

    def _overlay_image_dir(self) -> Path:
        return Path(self.cmp_image_edit.text())

    def _overlay_reload(self):
        img_dir = self._overlay_image_dir()
        if not img_dir.exists():
            self.overlay_file_list = []
            self.overlay_keys = []
            self.overlay_image_map = {}
            self.overlay_key_to_index = {}
            self.overlay_file_combo.clear()
            self.overlay_index_spin.setRange(0, 0)
            self.overlay_image_data = None
            self._overlay_render_empty(f"Image dir not found: {img_dir}")
            self._cmp_log(f"Overlay: image dir not found: {img_dir}")
            return

        files = sorted([p.name for p in img_dir.glob("*.fit*")])
        self.overlay_file_list = list(files)
        self.overlay_keys = []
        self.overlay_image_map = {}
        self.overlay_key_to_index = {}
        self.overlay_filter_cache = {}
        self._overlay_normalized_cache = None
        self.overlay_last_image_dir = img_dir

        for idx, fname in enumerate(self.overlay_file_list):
            key = _normalize_frame_key(Path(fname).stem)
            self.overlay_keys.append(key)
            if key not in self.overlay_image_map:
                self.overlay_image_map[key] = img_dir / fname
            if key not in self.overlay_key_to_index:
                self.overlay_key_to_index[key] = idx

        self.overlay_file_combo.blockSignals(True)
        self.overlay_file_combo.clear()
        self.overlay_file_combo.addItems(self.overlay_file_list)
        self.overlay_file_combo.blockSignals(False)

        if self.overlay_file_list:
            self.overlay_index_spin.setRange(0, max(0, len(self.overlay_file_list) - 1))
            idx = min(self.overlay_current_index, len(self.overlay_file_list) - 1)
            self.overlay_index_spin.setValue(idx)
        else:
            self.overlay_index_spin.setRange(0, 0)
            self.overlay_image_data = None
            self._overlay_render_empty("No FITS files found.")

        self._overlay_refresh_maps()
        self._cmp_log(
            f"Overlay: frames={len(self.overlay_file_list)} | image_dir={img_dir}"
        )

    def _overlay_refresh_maps(self):
        aapki_dir = Path(self.cmp_aapki_edit.text())
        iraf_dir = Path(self.cmp_iraf_edit.text())

        self.overlay_aapki_map = {}
        self.overlay_iraf_map = {}

        if aapki_dir.exists():
            for p in aapki_dir.rglob("*_photometry.tsv"):
                key = _normalize_frame_key(p.stem)
                self.overlay_aapki_map.setdefault(key, p)

        if iraf_dir.exists():
            for p in iraf_dir.rglob("*.coo"):
                key = _normalize_frame_key(p.stem)
                self.overlay_iraf_map.setdefault(key, p)
            if not self.overlay_iraf_map:
                for p in iraf_dir.rglob("*.txt"):
                    key = _normalize_frame_key(p.stem)
                    self.overlay_iraf_map.setdefault(key, p)

        self._cmp_log(
            f"Overlay: aapki={len(self.overlay_aapki_map)} | iraf={len(self.overlay_iraf_map)}"
        )

    def _overlay_on_file_changed(self, index):
        if index < 0 or index >= len(self.overlay_file_list):
            return
        self.overlay_current_index = index
        self.overlay_index_spin.blockSignals(True)
        self.overlay_index_spin.setValue(index)
        self.overlay_index_spin.blockSignals(False)
        self._overlay_load_current()

    def _overlay_on_index_changed(self, index):
        if index < 0 or index >= len(self.overlay_file_list):
            return
        self.overlay_current_index = index
        self.overlay_file_combo.blockSignals(True)
        self.overlay_file_combo.setCurrentIndex(index)
        self.overlay_file_combo.blockSignals(False)
        self._overlay_load_current()

    def _overlay_load_current(self):
        if not self.overlay_file_list:
            return
        img_dir = self._overlay_image_dir()
        fname = self.overlay_file_list[self.overlay_current_index]
        fpath = img_dir / fname
        self.overlay_xlim_original = None
        self.overlay_ylim_original = None
        if not fpath.exists():
            self._overlay_render_empty(f"Missing file: {fpath}")
            self._cmp_log(f"Overlay: missing file {fpath}")
            return
        try:
            with fits.open(fpath) as hdul:
                data = hdul[0].data
                header = hdul[0].header
        except Exception as e:
            self._overlay_render_empty(f"Failed to load: {fname}")
            self._cmp_log(f"Overlay: failed to load {fname}: {e}")
            return

        if data is None:
            self._overlay_render_empty(f"No image data: {fname}")
            self._cmp_log(f"Overlay: no data in {fname}")
            return

        if data.ndim > 2:
            data = data[0]

        self.overlay_image_data = data.astype(np.float32)
        self.overlay_header = header
        self._overlay_normalized_cache = None
        self._overlay_imshow_obj = None
        self._overlay_reset_stretch_plot_values()

        self._overlay_display()
        self.overlay_canvas.setFocus()

    def _overlay_display(self):
        if self.overlay_image_data is None:
            return

        normalized = self._overlay_normalize_image()
        if normalized is None:
            self._overlay_render_empty("No valid image data.")
            return

        stretched = self._overlay_apply_stretch(normalized)

        xlim_current = self.overlay_ax.get_xlim() if self.overlay_xlim_original else None
        ylim_current = self.overlay_ax.get_ylim() if self.overlay_ylim_original else None

        self.overlay_ax.clear()
        self._overlay_imshow_obj = self.overlay_ax.imshow(
            stretched,
            cmap="gray",
            origin="lower",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        fname = self.overlay_file_list[self.overlay_current_index]
        key = _normalize_frame_key(Path(fname).stem)

        aapki_xy = self._overlay_get_aapki_xy(key)
        iraf_xy = self._overlay_get_iraf_xy(key)

        aapki_n = 0
        iraf_n = 0

        if self.overlay_show_aapki.isChecked() and aapki_xy.size:
            aapki_n = aapki_xy.shape[0]
            self.overlay_ax.scatter(
                aapki_xy[:, 0],
                aapki_xy[:, 1],
                s=18,
                facecolors="none",
                edgecolors="#00C853",
                linewidths=0.8,
                label="AAPKI TSV",
            )

        if self.overlay_show_iraf.isChecked() and iraf_xy.size:
            iraf_n = iraf_xy.shape[0]
            self.overlay_ax.scatter(
                iraf_xy[:, 0],
                iraf_xy[:, 1],
                s=20,
                marker="x",
                color="#FF5252",
                linewidths=0.8,
                label="IRAF daofind",
            )

        if self.overlay_show_aapki.isChecked() or self.overlay_show_iraf.isChecked():
            self.overlay_ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

        self.overlay_ax.set_xlabel("X (pixels)")
        self.overlay_ax.set_ylabel("Y (pixels)")
        stretch_name = self.overlay_scale_combo.currentText()
        self.overlay_ax.set_title(f"{fname} | {stretch_name}")

        if self.overlay_xlim_original is None:
            self.overlay_xlim_original = self.overlay_ax.get_xlim()
            self.overlay_ylim_original = self.overlay_ax.get_ylim()
        elif xlim_current is not None and ylim_current is not None:
            self.overlay_ax.set_xlim(xlim_current)
            self.overlay_ax.set_ylim(ylim_current)

        self.overlay_canvas.draw_idle()

        filt = self._overlay_get_filter(fname)
        self.overlay_status.setText(
            f"Frame: {fname} | Filter: {filt} | AAPKI: {aapki_n} | IRAF: {iraf_n}"
        )

        self._cmp_log(
            f"Overlay: {fname} | filter={filt} | aapki={aapki_n} | iraf={iraf_n}"
        )

    def _overlay_render_empty(self, message: str):
        self.overlay_ax.clear()
        self.overlay_ax.text(0.5, 0.5, message, ha="center", va="center")
        self.overlay_canvas.draw_idle()
        self._overlay_imshow_obj = None
        self.overlay_status.setText(message)
        self.overlay_xlim_original = None
        self.overlay_ylim_original = None

    def _overlay_get_aapki_xy(self, key: str) -> np.ndarray:
        path = self.overlay_aapki_map.get(key)
        if path is None:
            return np.zeros((0, 2), dtype=float)
        try:
            df = _read_aapki_tsv(path)
        except Exception:
            return np.zeros((0, 2), dtype=float)
        x_col = _pick_first(df.columns, ["xcenter", "x", "x_init"])
        y_col = _pick_first(df.columns, ["ycenter", "y", "y_init"])
        if x_col is None or y_col is None:
            return np.zeros((0, 2), dtype=float)
        xy = df[[x_col, y_col]].to_numpy(float)
        xy = xy[np.isfinite(xy).all(axis=1)]
        return xy

    def _overlay_shift_for_frame(self, key: str) -> tuple[float, float]:
        for row in self.frame_rows:
            if row.get("frame") == key:
                sx = row.get("best_shift_x", np.nan)
                sy = row.get("best_shift_y", np.nan)
                if np.isfinite(sx):
                    sx = sx - BASE_IRAF_SHIFT
                else:
                    sx = 0.0
                if np.isfinite(sy):
                    sy = sy - BASE_IRAF_SHIFT
                else:
                    sy = 0.0
                return float(sx), float(sy)
        return 0.0, 0.0

    def _filter_for_frame_key(self, frame_key: str) -> str | None:
        path = self.overlay_image_map.get(frame_key)
        if path is None:
            return None
        fname = path.name
        return self._overlay_get_filter(fname)

    def _overlay_get_iraf_xy(self, key: str) -> np.ndarray:
        path = self.overlay_iraf_map.get(key)
        if path is None:
            return np.zeros((0, 2), dtype=float)
        try:
            if path.suffix.lower() == ".coo":
                df = _read_iraf_coo(path)
            else:
                df = _read_iraf_txt(path)
        except Exception:
            return np.zeros((0, 2), dtype=float)
        if "x" not in df.columns or "y" not in df.columns:
            return np.zeros((0, 2), dtype=float)
        xy = df[["x", "y"]].to_numpy(float)
        xy = xy[np.isfinite(xy).all(axis=1)]
        shift_x, shift_y = self._overlay_shift_for_frame(key)
        if shift_x or shift_y:
            xy[:, 0] += shift_x
            xy[:, 1] += shift_y
        return xy

    def _overlay_get_filter(self, fname: str) -> str:
        if fname in self.overlay_filter_cache:
            return self.overlay_filter_cache[fname]
        img_dir = self._overlay_image_dir()
        fpath = img_dir / fname
        filt = ""
        try:
            header = fits.getheader(fpath)
            filt = str(header.get("FILTER", "")).strip().lower()
        except Exception:
            filt = ""
        if not filt:
            filt = "unknown"
        self.overlay_filter_cache[fname] = filt
        return filt

    def _overlay_navigate(self, direction: int):
        if not self.overlay_file_list:
            return
        new_index = (self.overlay_current_index + direction) % len(self.overlay_file_list)
        self.overlay_index_spin.setValue(new_index)

    def _overlay_cycle_filter(self):
        if not self.overlay_file_list:
            return
        current = self.overlay_file_list[self.overlay_current_index]
        current_filter = self._overlay_get_filter(current)
        filters = {}
        for idx, fname in enumerate(self.overlay_file_list):
            filt = self._overlay_get_filter(fname)
            filters.setdefault(filt, []).append(idx)
        if len(filters) <= 1:
            self._cmp_log("Overlay: only one filter available.")
            return
        filter_names = sorted(filters.keys())
        if current_filter in filter_names:
            cur_idx = filter_names.index(current_filter)
        else:
            cur_idx = -1
        next_filter = filter_names[(cur_idx + 1) % len(filter_names)]
        next_idx = filters[next_filter][0]
        self.overlay_index_spin.setValue(next_idx)

    def _overlay_on_stretch_changed(self, index):
        self._overlay_normalized_cache = None
        self._overlay_reset_stretch_plot_values()
        self._overlay_display()

    def _overlay_update_stretch_label(self, value):
        self.overlay_stretch_value.setText(str(value))

    def _overlay_update_black_label(self, value):
        self.overlay_black_value.setText(str(value))

    def _overlay_reset_stretch(self):
        self.overlay_scale_combo.setCurrentIndex(0)
        self.overlay_stretch_slider.setValue(25)
        self.overlay_black_slider.setValue(0)
        self._overlay_normalized_cache = None
        self._overlay_reset_stretch_plot_values()
        self._overlay_display()

    def _overlay_reset_zoom(self):
        if self.overlay_xlim_original is not None and self.overlay_ylim_original is not None:
            self.overlay_ax.set_xlim(self.overlay_xlim_original)
            self.overlay_ax.set_ylim(self.overlay_ylim_original)
            self.overlay_canvas.draw_idle()

    def _overlay_redisplay(self):
        self._overlay_display()

    def _overlay_open_stretch_plot(self):
        """Open stretch plot window showing histogram with draggable min/max markers"""
        if self.overlay_image_data is None:
            QMessageBox.warning(self, "Warning", "Load an image first")
            return

        if self.overlay_stretch_plot_dialog is not None and self.overlay_stretch_plot_dialog.isVisible():
            self.overlay_stretch_plot_dialog.raise_()
            self.overlay_stretch_plot_dialog.activateWindow()
            self._overlay_update_stretch_plot()
            return

        self.overlay_stretch_plot_dialog = QDialog(self)
        self.overlay_stretch_plot_dialog.setWindowTitle("2D Plot - Stretch Control")
        self.overlay_stretch_plot_dialog.resize(500, 250)

        layout = QVBoxLayout(self.overlay_stretch_plot_dialog)

        self.overlay_stretch_plot_info_label = QLabel("Drag min/max markers to adjust stretch")
        self.overlay_stretch_plot_info_label.setStyleSheet(
            "QLabel { padding: 5px; background-color: #E3F2FD; border-radius: 3px; }"
        )
        layout.addWidget(self.overlay_stretch_plot_info_label)

        self.overlay_stretch_plot_fig = Figure(figsize=(6, 2.5))
        self.overlay_stretch_plot_canvas = FigureCanvas(self.overlay_stretch_plot_fig)
        self.overlay_stretch_plot_ax = self.overlay_stretch_plot_fig.add_subplot(111)
        self.overlay_stretch_plot_fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9)

        self.overlay_stretch_plot_canvas.mpl_connect('button_press_event', self._overlay_on_stretch_plot_press)
        self.overlay_stretch_plot_canvas.mpl_connect('motion_notify_event', self._overlay_on_stretch_plot_motion)
        self.overlay_stretch_plot_canvas.mpl_connect('button_release_event', self._overlay_on_stretch_plot_release)

        layout.addWidget(self.overlay_stretch_plot_canvas)

        hint_label = QLabel("Click and drag < > markers to adjust min/max | Changes apply in real-time")
        hint_label.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        layout.addWidget(hint_label)

        self.overlay_stretch_plot_dialog.show()
        self._overlay_update_stretch_plot()

    def _overlay_update_stretch_plot(self):
        """Update the stretch plot histogram and markers"""
        if self.overlay_stretch_plot_ax is None or self.overlay_image_data is None:
            return

        ax = self.overlay_stretch_plot_ax
        ax.clear()

        data = self.overlay_image_data.copy()
        finite_mask = np.isfinite(data)
        if not finite_mask.any():
            return

        flat = data[finite_mask].flatten()
        p_low, p_high = np.percentile(flat, [1, 99])
        display_data = flat[(flat >= p_low) & (flat <= p_high)]
        if len(display_data) == 0:
            display_data = flat

        self._overlay_stretch_data_range = (float(p_low), float(p_high))

        if self._overlay_stretch_vmin is None or self._overlay_stretch_vmax is None:
            stretch_idx = self.overlay_scale_combo.currentIndex()
            if stretch_idx == 6:
                vmin = np.percentile(flat, 1)
                vmax = np.percentile(flat, 99)
            elif stretch_idx == 7:
                vmin, vmax = self._overlay_calculate_zscale()
            else:
                _, median_val, std_val = sigma_clipped_stats(flat, sigma=3.0, maxiters=5)
                vmin = max(np.min(flat), median_val - 2.8 * std_val)
                vmax = min(np.max(flat), np.percentile(flat, 99.9))

            if vmax <= vmin:
                vmin = np.min(flat)
                vmax = np.max(flat)

            self._overlay_stretch_vmin = float(vmin)
            self._overlay_stretch_vmax = float(vmax)

        ax.hist(display_data, bins=128, color='#3a6ea5', edgecolor='none', alpha=0.7)
        ax.set_xlim(p_low, p_high)

        vmin = self._overlay_stretch_vmin
        vmax = self._overlay_stretch_vmax

        vmin_display = max(p_low, min(p_high, vmin))
        vmax_display = max(p_low, min(p_high, vmax))

        self._overlay_stretch_marker_min_line = ax.axvline(
            vmin_display, color='#FF5722', linewidth=2, linestyle='-', label=f"Min: {vmin:.1f}"
        )
        self._overlay_stretch_marker_max_line = ax.axvline(
            vmax_display, color='#4CAF50', linewidth=2, linestyle='-', label=f"Max: {vmax:.1f}"
        )

        y_max = ax.get_ylim()[1]
        ax.text(vmin_display, y_max * 0.95, '<', color='#FF5722', fontsize=14,
                ha='center', va='top', fontweight='bold')
        ax.text(vmax_display, y_max * 0.95, '>', color='#4CAF50', fontsize=14,
                ha='center', va='top', fontweight='bold')

        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Count')
        ax.set_title('Image Histogram')
        ax.legend(loc='upper right', fontsize=8)

        if self.overlay_stretch_plot_info_label:
            stretch_name = self.overlay_scale_combo.currentText()
            self.overlay_stretch_plot_info_label.setText(
                f"Stretch: {stretch_name} | Min: {vmin:.2f} | Max: {vmax:.2f}"
            )

        self.overlay_stretch_plot_canvas.draw_idle()

    def _overlay_on_stretch_plot_press(self, event):
        """Handle mouse press on stretch plot"""
        if event.inaxes != self.overlay_stretch_plot_ax or event.xdata is None:
            return
        if self._overlay_stretch_vmin is None or self._overlay_stretch_vmax is None:
            return

        x = event.xdata
        dist_to_min = abs(x - self._overlay_stretch_vmin)
        dist_to_max = abs(x - self._overlay_stretch_vmax)
        self._overlay_stretch_drag_target = "min" if dist_to_min < dist_to_max else "max"
        self._overlay_stretch_dragging = True

    def _overlay_on_stretch_plot_motion(self, event):
        """Handle mouse motion on stretch plot (dragging)"""
        if not self._overlay_stretch_dragging or event.xdata is None:
            return

        x = event.xdata
        if self._overlay_stretch_drag_target == "min":
            new_val = min(x, self._overlay_stretch_vmax - 1)
            self._overlay_stretch_vmin = new_val
        else:
            new_val = max(x, self._overlay_stretch_vmin + 1)
            self._overlay_stretch_vmax = new_val

        self._overlay_update_stretch_plot()
        self._overlay_apply_custom_stretch()

    def _overlay_on_stretch_plot_release(self, event):
        """Handle mouse release on stretch plot"""
        self._overlay_stretch_dragging = False
        self._overlay_stretch_drag_target = None

    def _overlay_apply_custom_stretch(self):
        """Apply custom vmin/vmax stretch to the overlay image"""
        if self.overlay_image_data is None:
            return
        if self._overlay_stretch_vmin is None or self._overlay_stretch_vmax is None:
            return

        vmin = self._overlay_stretch_vmin
        vmax = self._overlay_stretch_vmax
        if vmax <= vmin:
            vmax = vmin + 1

        data = self.overlay_image_data.copy()
        normalized = (data - vmin) / (vmax - vmin + 1e-10)
        normalized = np.clip(normalized, 0, 1)
        stretched = self._overlay_apply_stretch(normalized)

        if self._overlay_imshow_obj is not None:
            self._overlay_imshow_obj.set_data(stretched)
            self.overlay_canvas.draw_idle()

    def _overlay_reset_stretch_plot_values(self):
        """Reset stretch plot values when changing image or stretch mode"""
        self._overlay_stretch_vmin = None
        self._overlay_stretch_vmax = None
        if self.overlay_stretch_plot_dialog and self.overlay_stretch_plot_dialog.isVisible():
            self._overlay_update_stretch_plot()

    def _overlay_normalize_image(self):
        if self.overlay_image_data is None:
            return None

        stretch_idx = self.overlay_scale_combo.currentIndex()
        cache_key = (id(self.overlay_image_data), stretch_idx)
        if self._overlay_normalized_cache is not None:
            if self._overlay_normalized_cache[0] == cache_key:
                return self._overlay_normalized_cache[1].copy()

        data = self.overlay_image_data
        finite = np.isfinite(data)
        if not finite.any():
            return np.zeros_like(data)

        if stretch_idx == 6:  # Linear (1-99%)
            vmin = np.percentile(data[finite], 1)
            vmax = np.percentile(data[finite], 99)
        elif stretch_idx == 7:  # ZScale (IRAF)
            vmin, vmax = self._overlay_calculate_zscale()
        else:
            _, median_val, std_val = sigma_clipped_stats(data[finite], sigma=3.0, maxiters=5)
            vmin = max(np.min(data[finite]), median_val - 2.8 * std_val)
            vmax = min(np.max(data[finite]), np.percentile(data[finite], 99.9))

        if vmax <= vmin:
            vmin = float(np.min(data[finite]))
            vmax = float(np.max(data[finite]))

        normalized = (data - vmin) / (vmax - vmin + 1e-10)
        normalized = np.clip(normalized, 0, 1)

        self._overlay_normalized_cache = (cache_key, normalized)
        return normalized.copy()

    def _overlay_calculate_zscale(self):
        finite = np.isfinite(self.overlay_image_data)
        if not finite.any():
            return 0.0, 1.0
        data = self.overlay_image_data[finite]
        _, median_val, std_val = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
        vmin = float(median_val - 2.8 * std_val)
        vmax_percentile = np.percentile(data, 99.5)
        vmax_sigma = median_val + 6.0 * std_val
        vmax = float(min(vmax_percentile, vmax_sigma))
        if vmax <= vmin:
            vmin = float(np.min(data))
            vmax = float(np.max(data))
        return vmin, vmax

    def _overlay_apply_stretch(self, data):
        stretch_idx = self.overlay_scale_combo.currentIndex()
        intensity = self.overlay_stretch_slider.value() / 100.0
        black_point = self.overlay_black_slider.value() / 100.0

        data = np.clip((data - black_point) / (1.0 - black_point + 1e-10), 0, 1)

        if stretch_idx == 0:
            return self._overlay_stretch_auto_siril(data, intensity)
        if stretch_idx == 1:
            return self._overlay_stretch_asinh(data, intensity)
        if stretch_idx == 2:
            return self._overlay_stretch_mtf(data, intensity)
        if stretch_idx == 3:
            return self._overlay_stretch_histogram_eq(data)
        if stretch_idx == 4:
            return self._overlay_stretch_log(data, intensity)
        if stretch_idx == 5:
            return self._overlay_stretch_sqrt(data, intensity)
        return data

    def _overlay_stretch_auto_siril(self, data, intensity):
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return data
        median_val = np.median(finite)
        mad = np.median(np.abs(finite - median_val))
        sigma = mad * 1.4826
        shadows = max(0.0, median_val - 2.8 * sigma)
        highlights = 1.0
        stretched = (data - shadows) / (highlights - shadows + 1e-10)
        stretched = np.clip(stretched, 0, 1)
        midtone = 0.15 + (1.0 - intensity) * 0.35
        return self._overlay_mtf_function(stretched, midtone)

    def _overlay_stretch_asinh(self, data, intensity):
        beta = 1.0 + intensity * 15.0
        stretched = np.arcsinh(data * beta) / np.arcsinh(beta)
        return np.clip(stretched, 0, 1)

    def _overlay_stretch_mtf(self, data, intensity):
        midtone = 0.05 + (1.0 - intensity) * 0.45
        return self._overlay_mtf_function(data, midtone)

    def _overlay_mtf_function(self, data, midtone):
        m = np.clip(midtone, 0.001, 0.999)
        result = np.zeros_like(data)
        mask = data > 0
        result[mask] = (m - 1) * data[mask] / ((2 * m - 1) * data[mask] - m)
        result[data == 0] = 0
        result[data == 1] = 1
        return np.clip(result, 0, 1)

    def _overlay_stretch_histogram_eq(self, data):
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return data
        hist, bin_edges = np.histogram(finite.flatten(), bins=65536, range=(0, 1))
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        stretched = np.interp(data, bin_centers, cdf)
        return np.clip(stretched, 0, 1)

    def _overlay_stretch_log(self, data, intensity):
        a = 100 + intensity * 900
        stretched = np.log(1 + a * data) / np.log(1 + a)
        return np.clip(stretched, 0, 1)

    def _overlay_stretch_sqrt(self, data, intensity):
        power = 0.2 + (1.0 - intensity) * 0.8
        stretched = np.power(data, power)
        return np.clip(stretched, 0, 1)

    def _overlay_on_scroll(self, event):
        if event.inaxes != self.overlay_ax:
            return
        zoom_factor = 1.2 if event.button == "up" else 0.8
        xlim = self.overlay_ax.get_xlim()
        ylim = self.overlay_ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return
        x_range = (xlim[1] - xlim[0]) * zoom_factor
        y_range = (ylim[1] - ylim[0]) * zoom_factor
        new_xlim = [
            xdata - x_range * (xdata - xlim[0]) / (xlim[1] - xlim[0]),
            xdata + x_range * (xlim[1] - xdata) / (xlim[1] - xlim[0]),
        ]
        new_ylim = [
            ydata - y_range * (ydata - ylim[0]) / (ylim[1] - ylim[0]),
            ydata + y_range * (ylim[1] - ydata) / (ylim[1] - ylim[0]),
        ]
        self.overlay_ax.set_xlim(new_xlim)
        self.overlay_ax.set_ylim(new_ylim)
        self.overlay_canvas.draw_idle()

    def _overlay_on_button_press(self, event):
        if event.button == 3:
            self.overlay_panning = True
            self.overlay_pan_start = (event.xdata, event.ydata)

    def _overlay_on_button_release(self, event):
        if event.button == 3:
            self.overlay_panning = False
            self.overlay_pan_start = None

    def _overlay_on_motion(self, event):
        if not self.overlay_panning or self.overlay_pan_start is None:
            return
        if event.inaxes != self.overlay_ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        dx = self.overlay_pan_start[0] - event.xdata
        dy = self.overlay_pan_start[1] - event.ydata
        xlim = self.overlay_ax.get_xlim()
        ylim = self.overlay_ax.get_ylim()
        self.overlay_ax.set_xlim([xlim[0] + dx, xlim[1] + dx])
        self.overlay_ax.set_ylim([ylim[0] + dy, ylim[1] + dy])
        self.overlay_canvas.draw_idle()

    # ========================================================================
    # Run Photometry
    # ========================================================================
    def run_photometry(self):
        self._apply_params()

        data_dir = Path(self.data_edit.text())
        output_dir = Path(self.out_edit.text())

        if not data_dir.exists():
            QMessageBox.warning(self, "Error", f"Data directory not found:\n{data_dir}")
            return

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()

        self._auto_save_params()
        filter_params, filter_aliases = self._load_iraf_toml_config()
        param_defaults = self._build_iraf_param_defaults()

        self.worker = IRAFPhotometryWorker(
            data_dir=data_dir,
            output_dir=output_dir,
            file_pattern=self.pattern_edit.text(),
            params=self.iraf_params,
            auto_sigma=self.auto_sigma_check.isChecked(),
            skip_existing=self.skip_existing_check.isChecked(),
            filter_params=filter_params,
            filter_aliases=filter_aliases,
            param_defaults=param_defaults,
        )

        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._on_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)

        self.worker.start()

    def stop_photometry(self):
        if self.worker:
            self.worker.stop()

    def _on_progress(self, current, total, message):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"{current}/{total}: {message}")

    def _on_log(self, msg):
        self.log_text.append(msg)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_finished(self, result):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        n = len(result.get("results", []))
        self.progress_label.setText(f"Completed: {n} images processed")
        QMessageBox.information(self, "Complete",
            f"IRAF photometry completed.\n{n} images processed.\nOutput: {result.get('output_dir', '')}")

    def _on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", msg)

    def _cmp_log(self, msg: str):
        if not hasattr(self, "cmp_log_text") or self.cmp_log_text is None:
            return
        self.cmp_log_text.append(msg)
        scrollbar = self.cmp_log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ========================================================================
    # Comparison
    # ========================================================================
    def run_comparison(self):
        aapki_dir = Path(self.cmp_aapki_edit.text())
        iraf_dir = Path(self.cmp_iraf_edit.text())
        tol = self.cmp_tol.value()

        if hasattr(self, "cmp_log_text") and self.cmp_log_text is not None:
            self.cmp_log_text.clear()
        self._cmp_log("Starting comparison")
        self._cmp_log(f"AAPKI dir: {aapki_dir}")
        self._cmp_log(f"IRAF dir: {iraf_dir}")
        self._cmp_log(f"Tolerance: {tol:.2f} px")

        if not aapki_dir.exists():
            self._cmp_log(f"ERROR: AAPKI dir not found: {aapki_dir}")
            QMessageBox.warning(self, "Error", f"AAPKI dir not found: {aapki_dir}")
            return
        if not iraf_dir.exists():
            self._cmp_log(f"ERROR: IRAF dir not found: {iraf_dir}")
            QMessageBox.warning(self, "Error", f"IRAF dir not found: {iraf_dir}")
            return

        self._overlay_refresh_maps()

        # Collect files
        aapki_map = {}
        for p in aapki_dir.rglob("*_photometry.tsv"):
            key = _normalize_frame_key(p.stem)
            aapki_map.setdefault(key, p)

        iraf_map = {}
        for p in iraf_dir.rglob("*.txt"):
            key = _normalize_frame_key(p.stem)
            iraf_map.setdefault(key, p)

        self._cmp_log(f"AAPKI files: {len(aapki_map)}")
        self._cmp_log(f"IRAF files: {len(iraf_map)}")

        frames = sorted(set(aapki_map) & set(iraf_map))
        if not frames:
            self._cmp_log("ERROR: No matching frames found.")
            QMessageBox.warning(self, "Error", "No matching frames found.")
            return
        self._cmp_log(f"Matched frames: {len(frames)}")

        self.frame_rows = []
        self.frame_matches = {}
        all_matches = []

        def _match_with_iraf(iraf_df, aapki_df, x_col, y_col, mag_col):
            axy = aapki_df[[x_col, y_col]].to_numpy(float)
            ixy = iraf_df[["x", "y"]].to_numpy(float)

            if axy.size == 0 or ixy.size == 0:
                return pd.DataFrame()

            tree = cKDTree(axy)
            dist, idx = tree.query(ixy, distance_upper_bound=tol)
            mask = np.isfinite(dist) & (dist <= tol)

            if not np.any(mask):
                return pd.DataFrame()

            match = pd.DataFrame({
                "iraf_x": iraf_df.loc[mask, "x"].to_numpy(),
                "iraf_y": iraf_df.loc[mask, "y"].to_numpy(),
                "iraf_mag": iraf_df.loc[mask, "mag"].to_numpy(),
                "aapki_x": aapki_df.loc[idx[mask], x_col].to_numpy(),
                "aapki_y": aapki_df.loc[idx[mask], y_col].to_numpy(),
                "aapki_mag": aapki_df.loc[idx[mask], mag_col].to_numpy(),
                "dist_px": dist[mask],
            })
            match["dx"] = match["aapki_x"] - match["iraf_x"]
            match["dy"] = match["aapki_y"] - match["iraf_y"]
            match["dmag"] = match["aapki_mag"] - match["iraf_mag"]
            return match

        for frame in frames:
            aapki_df = _read_aapki_tsv(aapki_map[frame])
            iraf_df = _read_iraf_txt(iraf_map[frame])

            mag_col = _pick_first(aapki_df.columns, ["mag_inst", "mag", "mag_raw"])
            x_col = _pick_first(aapki_df.columns, ["xcenter", "x", "x_init"])
            y_col = _pick_first(aapki_df.columns, ["ycenter", "y", "y_init"])

            n_aapki_total = len(aapki_df)
            n_iraf_total = len(iraf_df)

            if not all([mag_col, x_col, y_col]):
                self._cmp_log(
                    f"{frame}: missing columns (mag={mag_col}, x={x_col}, y={y_col})"
                )
                continue

            match = _match_with_iraf(iraf_df, aapki_df, x_col, y_col, mag_col)
            if not match.empty:
                dx_med = float(np.nanmedian(match["dx"]))
                dy_med = float(np.nanmedian(match["dy"]))
            else:
                dx_med = np.nan
                dy_med = np.nan

            shift_x = _auto_axis_shift(dx_med)
            shift_y = _auto_axis_shift(dy_med)
            if (shift_x != 0.0) or (shift_y != 0.0):
                iraf_adj = iraf_df.copy()
                iraf_adj["x"] = iraf_adj["x"] + shift_x
                iraf_adj["y"] = iraf_adj["y"] + shift_y
                match = _match_with_iraf(iraf_adj, aapki_df, x_col, y_col, mag_col)

            if match.empty:
                self._cmp_log(f"{frame}: matched 0 (no pairs within {tol:.2f}px)")
                self.frame_rows.append({
                    "frame": frame,
                    "n": 0,
                    "dmag_med": np.nan,
                    "dmag_std": np.nan,
                    "dx_med": np.nan,
                    "dy_med": np.nan,
                    "dist_med": np.nan,
                    "dist_p95": np.nan,
                    "frac_within_tol": 0.0,
                    "best_shift_x": np.nan,
                    "best_shift_y": np.nan,
                    "n_iraf_total": n_iraf_total,
                    "n_aapki_total": n_aapki_total,
                })
                self.frame_matches[frame] = pd.DataFrame()
                continue

            dmag_med = float(np.nanmedian(match["dmag"]))
            dmag_std = float(np.nanstd(match["dmag"]))
            dx_med = float(np.nanmedian(match["dx"]))
            dy_med = float(np.nanmedian(match["dy"]))

            dist_vals = match["dist_px"].to_numpy(float)
            dist_med = float(np.nanmedian(dist_vals)) if dist_vals.size else np.nan
            dist_p95 = float(np.nanpercentile(dist_vals, 95)) if dist_vals.size else np.nan
            frac_within = float(np.mean(dist_vals <= tol)) if dist_vals.size else 0.0

            best_shift_x = BASE_IRAF_SHIFT + shift_x
            best_shift_y = BASE_IRAF_SHIFT + shift_y

            self._cmp_log(
                f"{frame}: matched {len(match)} | dmag={dmag_med:.4f}±{dmag_std:.4f} "
                f"dx={dx_med:.3f} dy={dy_med:.3f} "
                f"dist_med={dist_med:.3f} p95={dist_p95:.3f} "
                f"shift=({best_shift_x:.1f},{best_shift_y:.1f}) "
                f"n_iraf={n_iraf_total} n_aapki={n_aapki_total}"
            )

            self.frame_rows.append({
                "frame": frame,
                "n": len(match),
                "dmag_med": dmag_med,
                "dmag_std": dmag_std,
                "dx_med": dx_med,
                "dy_med": dy_med,
                "dist_med": dist_med,
                "dist_p95": dist_p95,
                "frac_within_tol": frac_within,
                "best_shift_x": best_shift_x,
                "best_shift_y": best_shift_y,
                "n_iraf_total": n_iraf_total,
                "n_aapki_total": n_aapki_total,
            })
            self.frame_matches[frame] = match
            all_matches.append(match.assign(frame=frame))

        self.matched_all = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()

        # Update table
        self.cmp_table.setRowCount(len(self.frame_rows))
        for i, row in enumerate(self.frame_rows):
            items = [
                row["frame"],
                str(row["n"]),
                f"{row['dmag_med']:.4f}" if np.isfinite(row["dmag_med"]) else "nan",
                f"{row['dmag_std']:.4f}" if np.isfinite(row["dmag_std"]) else "nan",
                f"{row['dx_med']:.3f}" if np.isfinite(row["dx_med"]) else "nan",
                f"{row['dy_med']:.3f}" if np.isfinite(row["dy_med"]) else "nan",
                f"{row['dist_med']:.3f}" if np.isfinite(row["dist_med"]) else "nan",
                f"{row['dist_p95']:.3f}" if np.isfinite(row["dist_p95"]) else "nan",
                f"{row['frac_within_tol']:.3f}",
                f"{row['best_shift_x']:.1f}" if np.isfinite(row["best_shift_x"]) else "nan",
                f"{row['best_shift_y']:.1f}" if np.isfinite(row["best_shift_y"]) else "nan",
                str(int(row["n_iraf_total"])),
                str(int(row["n_aapki_total"])),
            ]
            for col, text in enumerate(items):
                self.cmp_table.setItem(i, col, QTableWidgetItem(text))
        self.cmp_table.resizeColumnsToContents()

        # Update summary
        if not self.matched_all.empty:
            total = len(self.matched_all)
            med = np.nanmedian(self.matched_all["dmag"])
            std = np.nanstd(self.matched_all["dmag"])
            self.cmp_summary.setText(
                f"Total: {total} matched stars | dmag median: {med:.4f} | dmag std: {std:.4f}"
            )
            self._cmp_log(
                f"Total matched: {total} | dmag median: {med:.4f} | dmag std: {std:.4f}"
            )

            frame_filters = {}
            for row in self.frame_rows:
                frame_key = row.get("frame")
                if frame_key is None:
                    continue
                filt = self._filter_for_frame_key(frame_key)
                if filt:
                    frame_filters[frame_key] = filt

            if frame_filters:
                df = self.matched_all.copy()
                df["filter"] = df["frame"].map(frame_filters)
                self._cmp_log("Filter summary:")
                for filt, grp in df.groupby("filter"):
                    dmag_med = float(np.nanmedian(grp["dmag"]))
                    dmag_std = float(np.nanstd(grp["dmag"]))
                    dist_med = float(np.nanmedian(grp["dist_px"])) if "dist_px" in grp else np.nan
                    dist_p95 = float(np.nanpercentile(grp["dist_px"], 95)) if "dist_px" in grp else np.nan
                    frac_within = float(np.mean(grp["dist_px"] <= tol)) if "dist_px" in grp else np.nan
                    self._cmp_log(
                        f"  {filt}: n={len(grp)} | dmag={dmag_med:.4f}±{dmag_std:.4f} "
                        f"dist_med={dist_med:.3f} p95={dist_p95:.3f} frac<=tol={frac_within:.3f}"
                    )
        else:
            self.cmp_summary.setText("No matches found.")
            self._cmp_log("No matches found.")

        if self.cmp_table.rowCount() > 0:
            self.cmp_table.selectRow(0)

    def _plot_comparison(self):
        items = self.cmp_table.selectedItems()
        if not items:
            return

        frame = self.cmp_table.item(items[0].row(), 0).text()
        match = self.frame_matches.get(frame, pd.DataFrame())

        self._overlay_set_frame_key(frame)

        self.cmp_fig.clear()

        if match.empty:
            ax = self.cmp_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"{frame}: No matches", ha="center", va="center")
            self.cmp_canvas.draw_idle()
            return

        ax1 = self.cmp_fig.add_subplot(221)
        ax2 = self.cmp_fig.add_subplot(222)
        ax3 = self.cmp_fig.add_subplot(223)
        ax4 = self.cmp_fig.add_subplot(224)

        # dmag vs mag
        ax1.scatter(match["aapki_mag"], match["dmag"], s=10, alpha=0.6)
        ax1.axhline(0, color="red", ls="--")
        med = np.nanmedian(match["dmag"])
        ax1.axhline(med, color="blue", ls=":")
        ax1.set_xlabel("AAPKI mag")
        ax1.set_ylabel("dmag (AAPKI - IRAF)")
        ax1.set_title(f"dmag vs mag (med={med:.4f})")

        # dmag hist
        ax2.hist(match["dmag"].dropna(), bins=30, alpha=0.7, edgecolor="black")
        ax2.axvline(0, color="red", ls="--")
        ax2.axvline(med, color="blue", ls=":")
        ax2.set_xlabel("dmag")
        ax2.set_title(f"std={np.nanstd(match['dmag']):.4f}")

        # dx vs dy
        ax3.scatter(match["dx"], match["dy"], s=10, alpha=0.6)
        ax3.axhline(0, color="gray", lw=0.5)
        ax3.axvline(0, color="gray", lw=0.5)
        ax3.set_xlabel("dx [px]")
        ax3.set_ylabel("dy [px]")
        ax3.set_title("Position offset")
        ax3.set_aspect("equal")

        # 1:1
        ax4.scatter(match["iraf_mag"], match["aapki_mag"], s=10, alpha=0.6)
        lims = [min(match["iraf_mag"].min(), match["aapki_mag"].min()),
                max(match["iraf_mag"].max(), match["aapki_mag"].max())]
        ax4.plot(lims, lims, "r--")
        ax4.set_xlabel("IRAF mag")
        ax4.set_ylabel("AAPKI mag")
        ax4.set_title("1:1 comparison")

        self.cmp_fig.suptitle(f"{frame} (N={len(match)})")
        self.cmp_fig.tight_layout()
        self.cmp_canvas.draw_idle()

    def _overlay_set_frame_key(self, frame_key: str):
        if not hasattr(self, "overlay_index_spin") or not frame_key:
            return
        img_dir = self._overlay_image_dir()
        if self.overlay_last_image_dir is None or self.overlay_last_image_dir != img_dir:
            self._overlay_reload()
        if not self.overlay_key_to_index:
            return
        idx = self.overlay_key_to_index.get(frame_key)
        if idx is None:
            self._overlay_render_empty(f"No matching image for frame: {frame_key}")
            self._cmp_log(f"Overlay: no image for frame {frame_key}")
            return
        if idx != self.overlay_current_index:
            self.overlay_index_spin.setValue(idx)
        else:
            self._overlay_load_current()

    def export_comparison(self):
        if self.matched_all is None or self.matched_all.empty:
            self._cmp_log("Export skipped: no comparison data available.")
            QMessageBox.information(self, "Export", "No data to export.")
            return

        out_dir = Path(self.cmp_aapki_edit.text()) / "iraf_comparison"
        out_dir.mkdir(parents=True, exist_ok=True)

        self.matched_all.to_csv(out_dir / "iraf_compare_all.csv", index=False)
        pd.DataFrame(self.frame_rows).to_csv(out_dir / "iraf_compare_summary.csv", index=False)

        self._cmp_log(f"Exported CSVs to: {out_dir}")
        QMessageBox.information(self, "Export", f"Saved to:\n{out_dir}")

    # ========================================================================
    # Utilities
    # ========================================================================
    def _browse_dir(self, line_edit):
        path = QFileDialog.getExistingDirectory(self, "Select Directory", line_edit.text())
        if path:
            line_edit.setText(path)
