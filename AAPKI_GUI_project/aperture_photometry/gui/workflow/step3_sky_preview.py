"""
Step 3: Sky Preview & QC Window
Interactive image viewer with click-to-measure stats
Based on AAPKI Cell 6 with imexamine-like keyboard shortcuts
"""

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QMessageBox, QTextEdit, QComboBox, QDialog,
    QFormLayout, QLineEdit, QDialogButtonBox, QSplitter, QApplication,
    QSlider, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from pathlib import Path
import json
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.visualization import ZScaleInterval, ImageNormalize
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats
from photutils.detection import DAOStarFinder
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from .step_window_base import StepWindowBase
from ...utils.step_paths import step2_cropped_dir, step9_dir, crop_is_active


class SkyPreviewWindow(StepWindowBase):
    """
    Step 3: Sky Preview & QC
    Click on stars to measure FWHM, SNR, magnitude
    """

    def __init__(self, params, file_manager, project_state, main_window):
        """Initialize sky preview window"""
        self.file_manager = file_manager

        # Image data
        self.current_filename = None
        self.image_data = None
        self.header = None
        self.current_file_index = 0

        # Matplotlib components
        self.figure = None
        self.canvas = None
        self.ax = None

        # Measurement apertures and overlays
        self.aperture_patches = []
        self.cursor_crosshair_v = None
        self.cursor_crosshair_h = None
        self.cursor_aperture_patch = None
        self.cursor_x = None
        self.cursor_y = None

        # Photometry parameters from parameters.toml (hud5x section)
        hud5 = getattr(params.P, "_hud5", {})
        self.aperture_scale = float(hud5.get("5x.aperture_scale", "") or 1.0)
        self.annulus_in_scale = float(hud5.get("5x.annulus_in_scale", "") or 4.0)
        self.annulus_out_scale = float(hud5.get("5x.annulus_out_scale", "") or 2.0)  # width
        self.min_r_ap_px = float(hud5.get("5x.min_r_ap_px", "") or 12.0)
        self.min_r_in_px = float(hud5.get("5x.min_r_in_px", "") or 24.0)
        self.min_r_out_px = float(hud5.get("5x.min_r_out_px", "") or 36.0)
        self.sigma_clip_value = float(hud5.get("5x.sigma_clip", "") or 3.0)
        # FWHM seed in arcsec (converted to px using pixel_scale)
        self.fwhm_seed_arcsec = float(getattr(params.P, "fwhm_guess_arcsec", None) or 2.5)

        # File list
        self.file_list = []
        self.use_cropped = False
        self.cropped_dir = None

        # Persistent analysis windows
        self.histogram_dialog = None
        self.radial_profile_dialog = None
        self.last_radial_fwhm_px = None
        self.last_radial_center = None
        self.hist_fig = None
        self.hist_canvas = None
        self.hist_ax = None
        self.hist_stats_label = None
        self.prof_fig = None
        self.prof_canvas = None
        self.prof_ax = None
        self.last_measurement = None
        self._pending_xlim = None
        self._pending_ylim = None
        self._file_filter_map = {}
        self._file_frame_key_map = {}
        self._frame_key_map = {}
        self._frame_keys_by_filter = {}
        self._filter_order = []

        # Stretch plot window (2D Plot)
        self.stretch_plot_dialog = None
        self.stretch_plot_canvas = None
        self.stretch_plot_ax = None
        self.stretch_plot_fig = None
        self.stretch_plot_info_label = None
        self._stretch_vmin = None
        self._stretch_vmax = None
        self._stretch_data_range = None
        self._stretch_dragging = False
        self._stretch_drag_target = None  # "min" or "max"
        self._stretch_marker_min_line = None
        self._stretch_marker_max_line = None

        # Initialize base class
        super().__init__(
            step_index=2,
            step_name="Sky Preview & QC",
            params=params,
            project_state=project_state,
            main_window=main_window
        )

        # Setup step-specific UI
        self.setup_step_ui()

        # Restore state AFTER UI is created
        self.restore_state()

    def setup_step_ui(self):
        """Setup step-specific UI components"""

        # === Info Label ===
        info_label = QLabel(
            "Keyboard: [m] measure at cursor | [h] histogram | [g] radial profile | [.] next filter | [[/]] prev/next frame\n"
            "Mouse: Wheel to zoom | Right-click drag to pan"
        )
        info_label.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        self.content_layout.addWidget(info_label)

        # === File Selector ===
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Image:"))

        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self.on_file_changed)
        file_layout.addWidget(self.file_combo)

        btn_load = QPushButton("Load Image")
        btn_load.clicked.connect(self.load_selected_image)
        file_layout.addWidget(btn_load)

        btn_params = QPushButton("⚙ Photometry Parameters")
        btn_params.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 5px 10px; }")
        btn_params.clicked.connect(self.open_parameters_dialog)
        file_layout.addWidget(btn_params)

        file_layout.addStretch()
        self.content_layout.addLayout(file_layout)

        # === Image Viewer ===
        viewer_group = QGroupBox("Image Viewer")
        viewer_layout = QVBoxLayout(viewer_group)

        # Control bar - Row 1: Stretch method
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Stretch:"))

        self.scale_combo = QComboBox()
        self.scale_combo.addItems([
            'Auto Stretch (Siril)',
            'Asinh Stretch',
            'Midtone (MTF)',
            'Histogram Eq',
            'Log Stretch',
            'Sqrt Stretch',
            'Linear (1-99%)',
            'ZScale (IRAF)'
        ])
        self.scale_combo.currentIndexChanged.connect(self.on_stretch_changed)
        control_layout.addWidget(self.scale_combo)

        # Stretch parameter slider
        control_layout.addWidget(QLabel("Intensity:"))
        self.stretch_slider = QSlider(Qt.Horizontal)
        self.stretch_slider.setMinimum(1)
        self.stretch_slider.setMaximum(100)
        self.stretch_slider.setValue(25)
        self.stretch_slider.setFixedWidth(120)
        # Update only when slider is released (prevents lag)
        self.stretch_slider.sliderReleased.connect(self.redisplay_image)
        self.stretch_slider.valueChanged.connect(self.update_stretch_label)
        control_layout.addWidget(self.stretch_slider)

        self.stretch_value_label = QLabel("25")
        self.stretch_value_label.setFixedWidth(30)
        control_layout.addWidget(self.stretch_value_label)

        # Black point slider
        control_layout.addWidget(QLabel("Black:"))
        self.black_slider = QSlider(Qt.Horizontal)
        self.black_slider.setMinimum(0)
        self.black_slider.setMaximum(100)
        self.black_slider.setValue(0)
        self.black_slider.setFixedWidth(80)
        # Update only when slider is released (prevents lag)
        self.black_slider.sliderReleased.connect(self.redisplay_image)
        self.black_slider.valueChanged.connect(self.update_black_label)
        control_layout.addWidget(self.black_slider)

        self.black_value_label = QLabel("0")
        self.black_value_label.setFixedWidth(25)
        control_layout.addWidget(self.black_value_label)

        btn_reset_zoom = QPushButton("Reset Zoom")
        btn_reset_zoom.clicked.connect(self.reset_zoom)
        control_layout.addWidget(btn_reset_zoom)

        btn_reset_stretch = QPushButton("Reset Stretch")
        btn_reset_stretch.clicked.connect(self.reset_stretch)
        control_layout.addWidget(btn_reset_stretch)

        btn_2d_plot = QPushButton("2D Plot")
        btn_2d_plot.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; }")
        btn_2d_plot.clicked.connect(self.open_stretch_plot)
        control_layout.addWidget(btn_2d_plot)

        control_layout.addStretch()
        viewer_layout.addLayout(control_layout)

        # Matplotlib canvas
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self._imshow_obj = None  # Cache imshow object for fast updates

        # Enable keyboard focus for canvas
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()

        # Connect mouse and keyboard events
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)

        viewer_layout.addWidget(self.canvas)

        # === Create splitter for viewer and stats side-by-side ===
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(viewer_group)

        # === Stats Display (right side) ===
        stats_group = QGroupBox("Measurement Statistics")
        stats_layout = QVBoxLayout(stats_group)

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 10pt; }")
        self.stats_text.setPlainText("Press 'm' at cursor to measure\n'h' for histogram | 'g' for radial profile")

        self.stats_text.setMinimumWidth(360)
        stats_layout.addWidget(self.stats_text)
        main_splitter.addWidget(stats_group)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 2)

        self.content_layout.addWidget(main_splitter)

        # Zoom/pan state
        self.xlim_original = None
        self.ylim_original = None
        self.panning = False
        self.pan_start = None

        # Populate file list
        self.populate_file_list()

    def populate_file_list(self):
        """Populate file combo box with cropped files"""
        # Check for cropped files
        crop_active = crop_is_active(self.params.P.result_dir)
        cropped_dir = step2_cropped_dir(self.params.P.result_dir)
        legacy_cropped = self.params.P.result_dir / "cropped"

        if crop_active and cropped_dir.exists() and list(cropped_dir.glob("*.fit*")):
            # Use cropped files
            files = sorted([f.name for f in cropped_dir.glob("*.fit*")])
            self.use_cropped = True
            self.cropped_dir = cropped_dir
        elif crop_active and legacy_cropped.exists() and list(legacy_cropped.glob("*.fit*")):
            files = sorted([f.name for f in legacy_cropped.glob("*.fit*")])
            self.use_cropped = True
            self.cropped_dir = legacy_cropped
        else:
            # Use original files
            if not self.file_manager.filenames:
                try:
                    self.file_manager.scan_files()
                except:
                    pass
            files = self.file_manager.filenames
            self.use_cropped = False
            self.cropped_dir = None

        self.file_list = list(files)
        self.file_combo.clear()
        self.file_combo.addItems(files)
        self._file_filter_map = {}
        self._file_frame_key_map = {}
        self._frame_key_map = {}
        self._frame_keys_by_filter = {}
        self._filter_order = []
        self._load_filter_map_from_index()

    def load_selected_image(self, keep_view=False):
        """Load selected image"""
        filename = self.file_combo.currentText()
        if not filename:
            return

        try:
            if keep_view and self.ax is not None:
                self._pending_xlim = self.ax.get_xlim()
                self._pending_ylim = self.ax.get_ylim()
            else:
                self._pending_xlim = None
                self._pending_ylim = None

            # Determine path
            if self.use_cropped and self.cropped_dir is not None:
                file_path = self.cropped_dir / filename
            else:
                file_path = self.params.P.data_dir / filename

            # Load FITS
            hdul = fits.open(file_path)
            self.image_data = hdul[0].data.astype(float).copy()
            self.header = hdul[0].header.copy()
            hdul.close()

            self.current_filename = filename
            self.current_file_index = self.file_combo.currentIndex()
            self._file_filter_map.setdefault(filename, self._extract_filter_from_header(self.header))
            self._file_frame_key_map.setdefault(
                filename,
                self._extract_frame_key(filename, self._file_filter_map.get(filename, ""))
            )

            # Clear caches for new image
            self._normalized_cache = None
            self._imshow_obj = None
            self.xlim_original = None
            self.ylim_original = None
            self.reset_stretch_plot_values()  # Reset stretch plot values for new image

            # Display (full redraw for new image)
            self.display_image(full_redraw=True)

            # Set focus to canvas for keyboard shortcuts
            self.canvas.setFocus()

            if not (keep_view and self.last_measurement is not None):
                self.stats_text.setPlainText(
                    f"Loaded: {filename}\n"
                    f"Size: {self.image_data.shape}\n"
                    f"Press 'm' at cursor to measure | 'h' for histogram | 'g' for radial profile"
                )

            if keep_view:
                self._restore_measurement_overlay()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")

    def display_image(self, full_redraw=False):
        """Display image with selected stretch"""
        if self.image_data is None:
            return

        # Normalize and apply stretch
        normalized = self.normalize_image()
        if normalized is None:
            return

        stretched = self.apply_stretch(normalized)

        # Fast update: use set_data if imshow object exists
        if self._imshow_obj is not None and not full_redraw:
            # Just update the data (much faster)
            self._imshow_obj.set_data(stretched)

            # Update title
            stretch_name = self.scale_combo.currentText()
            self.ax.set_title(f"{self.current_filename} | {stretch_name}")

            # Use blit for even faster rendering
            self.canvas.draw_idle()
            return

        # Full redraw (first time or forced)
        xlim_current = self._pending_xlim if self._pending_xlim is not None else (
            self.ax.get_xlim() if self.xlim_original else None
        )
        ylim_current = self._pending_ylim if self._pending_ylim is not None else (
            self.ax.get_ylim() if self.ylim_original else None
        )

        self.ax.clear()
        self._imshow_obj = None

        # Display stretched image (already in 0-1 range)
        self._imshow_obj = self.ax.imshow(
            stretched,
            cmap='gray',
            origin='lower',
            vmin=0,
            vmax=1,
            interpolation='nearest'
        )

        self.ax.set_xlabel("X (pixels)")
        self.ax.set_ylabel("Y (pixels)")

        # Show stretch method in title
        stretch_name = self.scale_combo.currentText()
        self.ax.set_title(f"{self.current_filename} | {stretch_name}")

        # Store or restore zoom limits
        if self.xlim_original is None:
            self.xlim_original = self.ax.get_xlim()
            self.ylim_original = self.ax.get_ylim()
        elif xlim_current is not None:
            self.ax.set_xlim(xlim_current)
            self.ax.set_ylim(ylim_current)
        elif self._pending_xlim is not None:
            self.ax.set_xlim(self._pending_xlim)
            self.ax.set_ylim(self._pending_ylim)

        self._pending_xlim = None
        self._pending_ylim = None

        self.canvas.draw()

    def on_stretch_changed(self, index):
        """Handle stretch method change"""
        self._normalized_cache = None  # Clear cache when method changes
        self.reset_stretch_plot_values()  # Reset stretch plot values
        self.display_image()  # Just update data, not full redraw

    def reset_stretch(self):
        """Reset stretch parameters to defaults"""
        self.stretch_slider.setValue(25)
        self.black_slider.setValue(0)
        self.scale_combo.setCurrentIndex(0)
        self._normalized_cache = None
        self.reset_stretch_plot_values()  # Reset stretch plot values
        self.display_image()

    def update_stretch_label(self, value):
        """Update intensity label without redisplaying"""
        self.stretch_value_label.setText(str(value))

    def update_black_label(self, value):
        """Update black point label without redisplaying"""
        self.black_value_label.setText(str(value))

    def apply_stretch(self, data):
        """
        Apply selected stretch to normalized data [0,1]
        Returns stretched data in [0,1] range
        """
        stretch_idx = self.scale_combo.currentIndex()
        intensity = self.stretch_slider.value() / 100.0  # 0.01 to 1.0
        black_point = self.black_slider.value() / 100.0  # 0 to 1.0

        # Apply black point
        data = np.clip((data - black_point) / (1.0 - black_point + 1e-10), 0, 1)

        if stretch_idx == 0:  # Auto Stretch (Siril-style)
            return self.stretch_auto_siril(data, intensity)
        elif stretch_idx == 1:  # Asinh Stretch
            return self.stretch_asinh(data, intensity)
        elif stretch_idx == 2:  # Midtone (MTF)
            return self.stretch_mtf(data, intensity)
        elif stretch_idx == 3:  # Histogram Equalization
            return self.stretch_histogram_eq(data)
        elif stretch_idx == 4:  # Log Stretch
            return self.stretch_log(data, intensity)
        elif stretch_idx == 5:  # Sqrt Stretch
            return self.stretch_sqrt(data, intensity)
        elif stretch_idx == 6:  # Linear (1-99%)
            return data  # Already normalized
        elif stretch_idx == 7:  # ZScale (IRAF)
            return data  # Already normalized
        else:
            return data

    def stretch_auto_siril(self, data, intensity):
        """
        Siril-style auto stretch using asinh with automatic parameters
        Based on Siril's autostretch algorithm
        """
        # Compute background statistics
        finite = data[np.isfinite(data)]
        if len(finite) == 0:
            return data

        median_val = np.median(finite)
        mad = np.median(np.abs(finite - median_val))  # Median Absolute Deviation
        sigma = mad * 1.4826  # Convert MAD to sigma

        # Siril parameters
        shadow_clipping = -2.8  # Standard shadow clipping
        target_background = 0.25  # Target background level

        # Calculate shadows (black point)
        shadows = median_val + shadow_clipping * sigma
        shadows = max(0, shadows)

        # Calculate highlight protection
        highlights = 1.0

        # Normalize
        stretched = (data - shadows) / (highlights - shadows + 1e-10)
        stretched = np.clip(stretched, 0, 1)

        # Apply MTF for midtone adjustment based on intensity
        midtone = 0.15 + (1.0 - intensity) * 0.35  # Intensity controls midtone
        stretched = self.mtf_function(stretched, midtone)

        return stretched

    def stretch_asinh(self, data, intensity):
        """
        Arcsinh stretch (PixInsight-style)
        Preserves color ratios while compressing dynamic range
        """
        # Intensity controls the stretch factor (higher = more stretch)
        beta = 1.0 + intensity * 15.0  # Range: 1 to 16

        # Apply asinh stretch
        stretched = np.arcsinh(data * beta) / np.arcsinh(beta)

        return np.clip(stretched, 0, 1)

    def stretch_mtf(self, data, intensity):
        """
        Midtone Transfer Function (PixInsight-style)
        """
        # Intensity controls midtone level
        # Lower intensity = brighter midtones (more visible faint details)
        midtone = 0.05 + (1.0 - intensity) * 0.45  # Range: 0.05 to 0.5

        return self.mtf_function(data, midtone)

    def mtf_function(self, data, midtone):
        """
        MTF (Midtone Transfer Function) formula from PixInsight
        m = midtone parameter (0 to 1, typically 0.5 for no change)
        """
        # Avoid division issues
        m = np.clip(midtone, 0.001, 0.999)

        # PixInsight MTF formula
        result = np.zeros_like(data)
        mask = data > 0
        result[mask] = (m - 1) * data[mask] / ((2 * m - 1) * data[mask] - m)
        result[data == 0] = 0
        result[data == 1] = 1

        return np.clip(result, 0, 1)

    def stretch_histogram_eq(self, data):
        """
        Histogram Equalization
        """
        # Flatten and remove invalid values
        finite = data[np.isfinite(data)]
        if len(finite) == 0:
            return data

        # Compute histogram
        hist, bin_edges = np.histogram(finite.flatten(), bins=65536, range=(0, 1))
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]  # Normalize

        # Interpolate
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        stretched = np.interp(data, bin_centers, cdf)

        return np.clip(stretched, 0, 1)

    def stretch_log(self, data, intensity):
        """
        Logarithmic stretch
        """
        # Intensity controls the log base effect
        a = 100 + intensity * 900  # Range: 100 to 1000

        stretched = np.log(1 + a * data) / np.log(1 + a)

        return np.clip(stretched, 0, 1)

    def stretch_sqrt(self, data, intensity):
        """
        Square root / Power stretch
        """
        # Intensity controls power (lower = more stretch)
        power = 0.2 + (1.0 - intensity) * 0.8  # Range: 0.2 to 1.0

        stretched = np.power(data, power)

        return np.clip(stretched, 0, 1)

    def normalize_image(self):
        """
        Normalize image data to [0, 1] range based on selected method
        Uses caching to avoid recalculating for same image/method
        """
        if self.image_data is None:
            return None

        # Check cache
        stretch_idx = self.scale_combo.currentIndex()
        cache_key = (id(self.image_data), stretch_idx)
        if hasattr(self, '_normalized_cache') and self._normalized_cache is not None:
            if self._normalized_cache[0] == cache_key:
                return self._normalized_cache[1].copy()

        finite = np.isfinite(self.image_data)
        if not finite.any():
            return np.zeros_like(self.image_data)

        data = self.image_data.copy()

        if stretch_idx == 6:  # Linear (1-99%)
            vmin = np.percentile(data[finite], 1)
            vmax = np.percentile(data[finite], 99)
        elif stretch_idx == 7:  # ZScale (IRAF)
            vmin, vmax = self.calculate_zscale()
        else:  # Auto methods - use robust background estimation
            mean_val, median_val, std_val = sigma_clipped_stats(data[finite], sigma=3.0, maxiters=5)
            vmin = max(np.min(data[finite]), median_val - 2.8 * std_val)
            vmax = min(np.max(data[finite]), np.percentile(data[finite], 99.9))

        # Normalize to [0, 1]
        if vmax <= vmin:
            vmin = np.min(data[finite])
            vmax = np.max(data[finite])

        normalized = (data - vmin) / (vmax - vmin + 1e-10)
        normalized = np.clip(normalized, 0, 1)

        # Cache the result
        self._normalized_cache = (cache_key, normalized)

        return normalized.copy()

    def calculate_zscale(self):
        """
        Calculate ZScale (IRAF algorithm) vmin/vmax
        """
        finite = np.isfinite(self.image_data)
        if not finite.any():
            return 0, 1

        data = self.image_data[finite]
        mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=3.0, maxiters=5)

        vmin = float(median_val - 2.8 * std_val)
        vmax_percentile = np.percentile(data, 99.5)
        vmax_sigma = median_val + 6.0 * std_val
        vmax = float(min(vmax_percentile, vmax_sigma))

        if vmax <= vmin:
            vmin = float(np.min(data))
            vmax = float(np.max(data))

        return vmin, vmax

    def redisplay_image(self):
        """Redisplay with new scaling"""
        self.display_image()

    def on_file_changed(self, index):
        """Handle file selection change"""
        # Auto-load when selection changes
        pass

    def on_button_press(self, event):
        """Handle mouse button press"""
        if event.button == 3:  # Right click - start pan
            self.panning = True
            self.pan_start = (event.xdata, event.ydata)

    def measure_star(self, x, y):
        """Measure star properties at (x, y)"""
        if self.image_data is None:
            return

        try:
            # Clear previous apertures
            for patch in self.aperture_patches:
                try:
                    patch.remove()
                except:
                    pass
            self.aperture_patches = []

            # Get local region for FWHM estimation
            size = 25
            y0 = max(0, y - size)
            y1 = min(self.image_data.shape[0], y + size)
            x0 = max(0, x - size)
            x1 = min(self.image_data.shape[1], x + size)

            cutout = self.image_data[y0:y1, x0:x1]
            cutout_min = float(np.min(cutout)) if cutout.size else float("nan")
            cutout_max = float(np.max(cutout)) if cutout.size else float("nan")

            # Estimate local background
            mean, median, std = sigma_clipped_stats(cutout, sigma=3.0)

            # Convert FWHM seed from arcsec to pixels
            pixscale = self.params.P.pixel_scale_arcsec
            fwhm_seed_px = self.fwhm_seed_arcsec / pixscale if np.isfinite(pixscale) and pixscale > 0 else 6.0

            # Estimate FWHM using radial profile method (more accurate per-star)
            fwhm = fwhm_seed_px
            fwhm_method = "seed"
            try:
                # Find centroid using DAOStarFinder
                import warnings
                from photutils.utils.exceptions import NoDetectionsWarning

                xc, yc = float(x), float(y)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", NoDetectionsWarning)
                    finder = DAOStarFinder(threshold=median + 3*std, fwhm=fwhm_seed_px)
                    sources = finder(cutout - median)

                if sources is not None and len(sources) > 0:
                    # Find closest source to click
                    dx = sources['xcentroid'] - (x - x0)
                    dy = sources['ycentroid'] - (y - y0)
                    dist = np.sqrt(dx**2 + dy**2)
                    idx = np.argmin(dist)
                    if dist[idx] < 10:
                        xc = float(x0 + sources['xcentroid'][idx])
                        yc = float(y0 + sources['ycentroid'][idx])

                # Calculate FWHM from radial profile
                rmax = 30
                ry0 = max(0, int(yc - rmax))
                ry1 = min(self.image_data.shape[0], int(yc + rmax))
                rx0 = max(0, int(xc - rmax))
                rx1 = min(self.image_data.shape[1], int(xc + rmax))

                region = self.image_data[ry0:ry1, rx0:rx1]
                yy, xx = np.mgrid[ry0:ry1, rx0:rx1]
                rr = np.sqrt((xx - xc)**2 + (yy - yc)**2)

                # Background subtraction
                _, reg_median, _ = sigma_clipped_stats(region, sigma=3.0)
                region_sub = region - reg_median

                # Radial bins
                dr = 0.5
                edges = np.arange(0, rmax, dr)
                centers = 0.5 * (edges[:-1] + edges[1:])
                profile = np.full_like(centers, np.nan)

                for i in range(len(centers)):
                    mask = (rr >= edges[i]) & (rr < edges[i+1])
                    if np.any(mask):
                        vals = region_sub[mask]
                        vals = vals[np.isfinite(vals)]
                        if vals.size > 0:
                            profile[i] = np.mean(vals)

                # Estimate FWHM from profile
                peak = np.nanmax(profile) if np.isfinite(profile).any() else 0
                if peak > 0:
                    half = 0.5 * peak
                    idx_half = np.where((profile[:-1] >= half) & (profile[1:] < half))[0]
                    if len(idx_half) > 0:
                        i = idx_half[0]
                        x1_p, y1_p = centers[i], profile[i]
                        x2_p, y2_p = centers[i+1], profile[i+1]
                        if y1_p != y2_p:
                            r_half = x1_p + (half - y1_p) * (x2_p - x1_p) / (y2_p - y1_p)
                            fwhm = 2.0 * r_half
                            fwhm_method = "radial"
            except:
                pass

            # Define aperture and annulus using FWHM seed × scale (fixed size, not per-star)
            # Measured FWHM is for display only, aperture size uses seed value
            r_ap = max(self.min_r_ap_px, fwhm_seed_px * self.aperture_scale)
            r_in = max(self.min_r_in_px, fwhm_seed_px * self.annulus_in_scale)
            r_out = max(self.min_r_out_px, r_in + fwhm_seed_px * self.annulus_out_scale)
            self.last_measurement = {
                "x": float(x),
                "y": float(y),
                "r_ap": float(r_ap),
                "r_in": float(r_in),
                "r_out": float(r_out),
            }

            aperture = CircularAperture((x, y), r=r_ap)
            annulus = CircularAnnulus((x, y), r_in=r_in, r_out=r_out)

            # Calculate stats
            ap_stats = ApertureStats(self.image_data, aperture)
            an_stats = ApertureStats(self.image_data, annulus, sigma_clip=None)

            # Sky background - handle both scalar and array returns
            sky_median = float(an_stats.median[0]) if hasattr(an_stats.median, '__len__') else float(an_stats.median)
            sky_std = float(an_stats.std[0]) if hasattr(an_stats.std, '__len__') else float(an_stats.std)

            # Source flux - handle both scalar and array returns
            flux_total = float(ap_stats.sum[0]) if hasattr(ap_stats.sum, '__len__') else float(ap_stats.sum)
            flux_sky = sky_median * float(ap_stats.sum_aper_area.value)
            flux_source = flux_total - flux_sky

            # SNR calculation
            gain = self.params.P.gain_e_per_adu
            rdnoise = self.params.P.rdnoise_e

            flux_e = flux_source * gain
            sky_e = flux_sky * gain
            n_pix = ap_stats.sum_aper_area.value

            snr = flux_e / np.sqrt(flux_e + sky_e + n_pix * rdnoise**2)

            # Magnitude
            exptime = self.header.get('EXPTIME', 1.0)
            zp = self.params.P.zp_initial
            mag = -2.5 * np.log10(flux_source / exptime) + zp if flux_source > 0 else float("nan")
            mag_err = 1.0857 / snr if snr > 0 else float("nan")  # 2.5/ln(10)
            mag_str = f"{mag:.2f}" if np.isfinite(mag) else "NaN"
            mag_err_str = f"{mag_err:.3f}" if np.isfinite(mag_err) else "NaN"

            # FWHM in arcsec
            pixscale = self.params.P.pixel_scale_arcsec
            fwhm_arcsec = fwhm * pixscale if np.isfinite(pixscale) else np.nan

            # Display results with FWHM in arcsec first
            stats_text = f"""
Measurement at ({x}, {y})
─────────────────────────
File: {self.current_filename}
EXPTIME: {exptime:.1f} s
Local Pixel Min/Max: {cutout_min:.2f} / {cutout_max:.2f} ADU

Sky Background:
  Median: {sky_median:.2f} ADU
  StdDev: {sky_std:.2f} ADU

Aperture (r={r_ap:.1f} px):
  Total: {flux_total:.0f} ADU
  Sky:   {flux_sky:.0f} ADU
  Source: {flux_source:.0f} ADU

FWHM: {fwhm_arcsec:.2f}" ({fwhm:.2f} px; {fwhm_method})
SNR:  {snr:.1f}
Mag:  {mag_str} ± {mag_err_str}
"""
            self.stats_text.setPlainText(stats_text)

            # Draw apertures
            ap_circle = Circle((x, y), r_ap, fill=False, edgecolor='red', linewidth=2)
            an_circle_in = Circle((x, y), r_in, fill=False, edgecolor='cyan', linewidth=1, linestyle='--')
            an_circle_out = Circle((x, y), r_out, fill=False, edgecolor='cyan', linewidth=1, linestyle='--')

            self.ax.add_patch(ap_circle)
            self.ax.add_patch(an_circle_in)
            self.ax.add_patch(an_circle_out)

            self.aperture_patches = [ap_circle, an_circle_in, an_circle_out]

            self.canvas.draw()

        except Exception as e:
            self.stats_text.setPlainText(f"Measurement failed:\n{str(e)}")

    def _restore_measurement_overlay(self):
        if self.last_measurement is None or self.image_data is None:
            return
        x = int(self.last_measurement["x"])
        y = int(self.last_measurement["y"])
        if x < 0 or y < 0 or y >= self.image_data.shape[0] or x >= self.image_data.shape[1]:
            return
        self.measure_star(x, y)

    def on_button_release(self, event):
        """Handle mouse button release"""
        if event.button == 3:
            self.panning = False
            self.pan_start = None

    def on_motion(self, event):
        """Handle mouse motion for panning"""
        # Handle panning
        if self.panning and self.pan_start is not None and event.inaxes == self.ax:
            dx = self.pan_start[0] - event.xdata
            dy = self.pan_start[1] - event.ydata

            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            self.ax.set_xlim([xlim[0] + dx, xlim[1] + dx])
            self.ax.set_ylim([ylim[0] + dy, ylim[1] + dy])

            self.canvas.draw()
            return

        # Update cursor position for keyboard shortcuts
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            self.cursor_x = event.xdata
            self.cursor_y = event.ydata
        else:
            self.cursor_x = None
            self.cursor_y = None

    def on_scroll(self, event):
        """Handle mouse wheel zoom"""
        if event.inaxes != self.ax:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        zoom_factor = 1.2 if event.button == 'up' else 0.8

        xdata, ydata = event.xdata, event.ydata

        x_range = (xlim[1] - xlim[0]) * zoom_factor
        y_range = (ylim[1] - ylim[0]) * zoom_factor

        new_xlim = [xdata - x_range * (xdata - xlim[0]) / (xlim[1] - xlim[0]),
                    xdata + x_range * (xlim[1] - xdata) / (xlim[1] - xlim[0])]
        new_ylim = [ydata - y_range * (ydata - ylim[0]) / (ylim[1] - ylim[0]),
                    ydata + y_range * (ylim[1] - ydata) / (ylim[1] - ylim[0])]

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw()

    def reset_zoom(self):
        """Reset zoom to original view"""
        if self.xlim_original is not None:
            self.ax.set_xlim(self.xlim_original)
            self.ax.set_ylim(self.ylim_original)
            self.canvas.draw()

    # ========== Keyboard Event Handling ==========

    def keyPressEvent(self, event):
        """Handle Qt keyboard events (overridden from QWidget)"""
        key = event.key()
        text = event.text().lower()

        if text == 'm' and self.cursor_x is not None and self.cursor_y is not None:
            self.measure_star_at_cursor()
        elif text == 'h' and self.cursor_x is not None and self.cursor_y is not None:
            self.show_histogram()
        elif text == 'g' and self.cursor_x is not None and self.cursor_y is not None:
            self.show_radial_profile()
        elif key == Qt.Key_Period or text == '.':  # . key
            self.cycle_filter()
        elif key == Qt.Key_BracketLeft:  # '['
            self.navigate_frame(-1)
        elif key == Qt.Key_BracketRight:  # ']'
            self.navigate_frame(1)
        else:
            # Call parent implementation for other keys
            super().keyPressEvent(event)

    def on_key_press(self, event):
        """Handle matplotlib keyboard shortcuts (backup method)"""
        if event.key == 'm' and self.cursor_x is not None and self.cursor_y is not None:
            # Measure at cursor position
            self.measure_star_at_cursor()
        elif event.key == 'h' and self.cursor_x is not None and self.cursor_y is not None:
            # Show histogram
            self.show_histogram()
        elif event.key == 'g' and self.cursor_x is not None and self.cursor_y is not None:
            # Show radial profile
            self.show_radial_profile()
        elif event.key == '.':
            # Cycle to next filter
            self.cycle_filter()
        elif event.key == '[':
            # Previous frame
            self.navigate_frame(-1)
        elif event.key == ']':
            # Next frame
            self.navigate_frame(1)

    def measure_star_at_cursor(self):
        """Measure star properties at current cursor position (m key)"""
        if self.image_data is None or self.cursor_x is None or self.cursor_y is None:
            return

        x = int(self.cursor_x)
        y = int(self.cursor_y)

        # Use the existing measure_star method
        self.measure_star(x, y)

    def navigate_frame(self, direction):
        """Navigate to previous/next frame within the SAME filter"""
        if not self.file_list:
            return

        # Ensure filter map is built
        self._build_frame_key_map()

        # Get current filter from header if possible
        current_filter = self._file_filter_map.get(self.current_filename)
        if current_filter is None and self.header is not None:
            current_filter = self._extract_filter_from_header(self.header)
            if current_filter:
                self._file_filter_map[self.current_filename] = current_filter

        if not current_filter:
            current_filter = ""

        # Build list of file indices for current filter
        filter_indices = []
        for idx, fname in enumerate(self.file_list):
            fkey = self._file_filter_map.get(fname, "")
            if fkey == current_filter:
                filter_indices.append(idx)

        # If filter_indices is empty or has only 1 item, try filename-based filter
        if len(filter_indices) <= 1 and current_filter:
            filter_indices = []
            filter_suffix = f"_{current_filter.lower()}"
            for idx, fname in enumerate(self.file_list):
                base = fname.lower()
                for ext in ('.fits', '.fit', '.fts'):
                    if base.endswith(ext):
                        base = base[:-len(ext)]
                        break
                if base.endswith(filter_suffix) or f"_{current_filter.lower()}." in fname.lower():
                    filter_indices.append(idx)

        if not filter_indices or len(filter_indices) <= 1:
            # Fallback: cycle through all files
            self.current_file_index = (self.current_file_index + direction) % len(self.file_list)
        else:
            # Find current position within filter's files
            try:
                pos = filter_indices.index(self.current_file_index)
            except ValueError:
                pos = 0
            # Move within filter only
            pos = (pos + direction) % len(filter_indices)
            self.current_file_index = filter_indices[pos]

        self.file_combo.setCurrentIndex(self.current_file_index)
        self.load_selected_image(keep_view=True)

    def cycle_filter(self):
        """Cycle to next filter, keeping the same frame position (index within filter)"""
        if not self.file_list or self.image_data is None:
            return
        self._build_frame_key_map()
        current_filter = self._file_filter_map.get(self.current_filename, "")
        filters = [f for f in self._filter_order if f]
        if len(filters) <= 1:
            self.stats_text.append("\n[Filter Cycle] Only one filter found")
            return

        # Find next filter
        try:
            current_filter_idx = filters.index(current_filter)
            next_filter = filters[(current_filter_idx + 1) % len(filters)]
        except ValueError:
            next_filter = filters[0]

        # Build file indices for current and next filter
        current_filter_indices = []
        next_filter_indices = []
        for idx, fname in enumerate(self.file_list):
            fkey = self._file_filter_map.get(fname, "")
            if fkey == current_filter:
                current_filter_indices.append(idx)
            if fkey == next_filter:
                next_filter_indices.append(idx)

        if not next_filter_indices:
            return

        # Find current position within current filter's files
        try:
            pos_in_filter = current_filter_indices.index(self.current_file_index)
        except ValueError:
            pos_in_filter = 0

        # Go to same position in next filter (or last if shorter)
        target_pos = min(pos_in_filter, len(next_filter_indices) - 1)
        self.current_file_index = next_filter_indices[target_pos]

        self.file_combo.setCurrentIndex(self.current_file_index)
        self.load_selected_image(keep_view=True)
        self.stats_text.append(f"\n[Filter Cycle] Switched to {next_filter}")

    @staticmethod
    def _normalize_filter_key(value: str | None) -> str:
        if value is None:
            return ""
        return str(value).strip().upper()

    def _extract_filter_from_header(self, header) -> str:
        if header is None:
            return ""
        for key in ("FILTER", "FILTER1", "FILTER2", "FILTNAM"):
            val = header.get(key)
            if val:
                return self._normalize_filter_key(val)
        return ""

    @staticmethod
    def _infer_filter_from_filename(fname: str) -> str:
        base = Path(fname).name
        for ext in (".fits", ".fit", ".fts", ".fz", ".gz"):
            if base.lower().endswith(ext):
                base = base[: -len(ext)]
        parts = [p for p in base.replace(".", "_").replace("-", "_").split("_") if p]
        for token in reversed(parts):
            cand = token.lower()
            if 1 <= len(cand) <= 3 and cand.isalpha():
                return cand.upper()
        return ""

    def _extract_frame_key(self, fname: str, filter_key: str) -> str:
        name = Path(fname).name
        base = name
        for ext in (".fits", ".fit", ".fts", ".fz", ".gz"):
            if base.lower().endswith(ext):
                base = base[: -len(ext)]
        base_lower = base.lower()
        filt = str(filter_key or "").lower() or self._infer_filter_from_filename(fname).lower()
        if filt:
            for sep in ("-", "_", "."):
                suffix = f"{sep}{filt}"
                if base_lower.endswith(suffix):
                    return base[: -len(suffix)]
        return base

    def _load_filter_map_from_index(self):
        idx_path = step9_dir(self.params.P.result_dir) / "photometry_index.csv"
        if not idx_path.exists():
            idx_path = self.params.P.result_dir / "photometry_index.csv"
        if not idx_path.exists():
            return
        try:
            df = pd.read_csv(idx_path)
        except Exception:
            return
        if "file" not in df.columns or "filter" not in df.columns:
            return
        for fname, filt in zip(df["file"].astype(str), df["filter"].astype(str)):
            fkey = self._normalize_filter_key(filt)
            if not fkey:
                fkey = self._infer_filter_from_filename(fname)
            self._file_filter_map[fname] = fkey
            if fname.lower().startswith("crop_"):
                self._file_filter_map[fname[5:]] = fkey
            if fname.lower().startswith("cropped_"):
                self._file_filter_map[fname[8:]] = fkey

    def _build_frame_key_map(self):
        if self._frame_key_map:
            return
        for idx, fname in enumerate(self.file_list):
            fkey = self._file_filter_map.get(fname)
            if not fkey:
                try:
                    if self.use_cropped and self.cropped_dir is not None:
                        fpath = self.cropped_dir / fname
                    else:
                        fpath = self.params.P.data_dir / fname
                    hdr = fits.getheader(fpath)
                    fkey = self._extract_filter_from_header(hdr)
                except Exception:
                    fkey = ""
                if not fkey:
                    fkey = self._infer_filter_from_filename(fname)
                self._file_filter_map[fname] = fkey
            frame_key = self._extract_frame_key(fname, fkey)
            self._file_frame_key_map[fname] = frame_key
            self._frame_key_map[(frame_key, fkey)] = idx
            self._frame_keys_by_filter.setdefault(fkey, []).append(frame_key)
        self._filter_order = [f for f in sorted(self._frame_keys_by_filter.keys()) if f]

    def show_histogram(self):
        """Show histogram of cursor region (h key) - persistent window"""
        if self.image_data is None or self.cursor_x is None or self.cursor_y is None:
            return

        x = int(self.cursor_x)
        y = int(self.cursor_y)

        # Get region around cursor
        size = 50
        y0 = max(0, y - size)
        y1 = min(self.image_data.shape[0], y + size)
        x0 = max(0, x - size)
        x1 = min(self.image_data.shape[1], x + size)

        region = self.image_data[y0:y1, x0:x1]
        region_finite = region[np.isfinite(region)]

        if region_finite.size == 0:
            return

        # Create or update histogram window
        if self.histogram_dialog is None or not self.histogram_dialog.isVisible():
            # Create new dialog
            self.histogram_dialog = QDialog(self)
            self.histogram_dialog.setWindowTitle("Histogram (Press 'h' to update)")
            self.histogram_dialog.resize(700, 500)
            self.histogram_dialog.setWindowFlags(Qt.Window)  # Make it resizable

            layout = QVBoxLayout(self.histogram_dialog)

            # Create matplotlib figure
            self.hist_fig = Figure(figsize=(7, 5))
            self.hist_canvas = FigureCanvas(self.hist_fig)
            self.hist_ax = self.hist_fig.add_subplot(111)

            layout.addWidget(NavigationToolbar(self.hist_canvas, self.histogram_dialog))
            layout.addWidget(self.hist_canvas)

            # Stats text
            self.hist_stats_label = QLabel()
            self.hist_stats_label.setStyleSheet("QLabel { font-family: monospace; padding: 10px; background-color: #f0f0f0; }")
            layout.addWidget(self.hist_stats_label)

            self.histogram_dialog.show()
            self._move_dialog_to_side(self.histogram_dialog)

        # Update plot
        self.hist_ax.clear()
        mean, median, std = sigma_clipped_stats(region_finite, sigma=3.0)

        self.hist_ax.hist(region_finite.flatten(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        self.hist_ax.axvline(median, color='red', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
        self.hist_ax.axvline(mean, color='orange', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')

        self.hist_ax.set_xlabel('Pixel Value (ADU)')
        self.hist_ax.set_ylabel('Count')
        self.hist_ax.set_title(f'Histogram at ({x}, {y}) | {x0}:{x1}, {y0}:{y1} ({region_finite.size} pixels)')
        self.hist_ax.legend()
        self.hist_ax.grid(True, alpha=0.3)

        self.hist_fig.tight_layout()
        self.hist_canvas.draw()

        # Update stats
        self.hist_stats_label.setText(
            f"Mean: {mean:.2f} ADU  |  Median: {median:.2f} ADU  |  StdDev: {std:.2f} ADU  |  "
            f"Min: {np.min(region_finite):.2f} ADU  |  Max: {np.max(region_finite):.2f} ADU"
        )

    def show_radial_profile(self):
        """Show radial profile plot (g key)"""
        if self.image_data is None or self.cursor_x is None or self.cursor_y is None:
            return

        x = int(self.cursor_x)
        y = int(self.cursor_y)

        try:
            # Recenter on star
            size = 25
            y0 = max(0, y - size)
            y1 = min(self.image_data.shape[0], y + size)
            x0 = max(0, x - size)
            x1 = min(self.image_data.shape[1], x + size)

            cutout = self.image_data[y0:y1, x0:x1]
            mean, median, std = sigma_clipped_stats(cutout, sigma=3.0)

            # Convert FWHM seed from arcsec to pixels
            pixscale = self.params.P.pixel_scale_arcsec
            fwhm_seed_px = self.fwhm_seed_arcsec / pixscale if np.isfinite(pixscale) and pixscale > 0 else 6.0

            # Try to find center
            xc, yc = float(x), float(y)
            try:
                import warnings
                from photutils.utils.exceptions import NoDetectionsWarning

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", NoDetectionsWarning)
                    finder = DAOStarFinder(threshold=median + 3*std, fwhm=fwhm_seed_px)
                    sources = finder(cutout - median)

                if sources is not None and len(sources) > 0:
                    dx = sources['xcentroid'] - (x - x0)
                    dy = sources['ycentroid'] - (y - y0)
                    dist = np.sqrt(dx**2 + dy**2)
                    idx = np.argmin(dist)
                    if dist[idx] < 10:
                        xc = float(x0 + sources['xcentroid'][idx])
                        yc = float(y0 + sources['ycentroid'][idx])
            except:
                pass

            # Calculate radial profile
            rmax = 50
            y0 = max(0, int(yc - rmax))
            y1 = min(self.image_data.shape[0], int(yc + rmax))
            x0 = max(0, int(xc - rmax))
            x1 = min(self.image_data.shape[1], int(xc + rmax))

            self.last_radial_center = (float(xc), float(yc))

            region = self.image_data[y0:y1, x0:x1]
            yy, xx = np.mgrid[y0:y1, x0:x1]
            rr = np.sqrt((xx - xc)**2 + (yy - yc)**2)

            # Background subtraction
            mean, median, std = sigma_clipped_stats(region, sigma=3.0)
            region_sub = region - median

            # Radial bins
            dr = 0.5
            edges = np.arange(0, rmax, dr)
            centers = 0.5 * (edges[:-1] + edges[1:])
            profile = np.full_like(centers, np.nan)

            for i in range(len(centers)):
                mask = (rr >= edges[i]) & (rr < edges[i+1])
                if np.any(mask):
                    vals = region_sub[mask]
                    vals = vals[np.isfinite(vals)]
                    if vals.size > 0:
                        profile[i] = np.mean(vals)

            # Create or reuse radial profile window
            if self.radial_profile_dialog is None or not self.radial_profile_dialog.isVisible():
                self.radial_profile_dialog = QDialog(self)
                self.radial_profile_dialog.setWindowTitle("Radial Profile (Press 'g' to update)")
                self.radial_profile_dialog.resize(700, 500)
                self.radial_profile_dialog.setWindowFlags(Qt.Window)  # Make it resizable

                layout = QVBoxLayout(self.radial_profile_dialog)

                self.prof_fig = Figure(figsize=(7, 5))
                self.prof_canvas = FigureCanvas(self.prof_fig)
                self.prof_ax = self.prof_fig.add_subplot(111)

                layout.addWidget(NavigationToolbar(self.prof_canvas, self.radial_profile_dialog))
                layout.addWidget(self.prof_canvas)

                close_btn = QPushButton("Close")
                close_btn.clicked.connect(self.radial_profile_dialog.close)
                layout.addWidget(close_btn)

                self.radial_profile_dialog.show()
                self._move_dialog_to_side(self.radial_profile_dialog)

            # Plot profile
            self.prof_ax.clear()
            self.prof_ax.plot(centers, profile, 'o-', color='steelblue', markersize=4, linewidth=1.5)
            self.prof_ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            self.prof_ax.set_xlabel('Radius (pixels)')
            self.prof_ax.set_ylabel('Pixel Value - Background (ADU)')
            self.prof_ax.set_title(f'Radial Profile (centered at {int(xc)}, {int(yc)})')
            self.prof_ax.grid(True, alpha=0.3)

            # Estimate FWHM from profile
            self.last_radial_fwhm_px = None
            fwhm_note = "FWHM: N/A"
            pixscale = self.params.P.pixel_scale_arcsec
            peak = np.nanmax(profile) if np.isfinite(profile).any() else 0
            if peak > 0:
                half = 0.5 * peak
                idx = np.where((profile[:-1] >= half) & (profile[1:] < half))[0]
                if len(idx) > 0:
                    i = idx[0]
                    x1, y1 = centers[i], profile[i]
                    x2, y2 = centers[i+1], profile[i+1]
                    if y1 != y2:
                        r_half = x1 + (half - y1) * (x2 - x1) / (y2 - y1)
                        fwhm_est = 2.0 * r_half
                        self.last_radial_fwhm_px = float(fwhm_est)
                        fwhm_arcsec = fwhm_est * pixscale if np.isfinite(pixscale) else np.nan
                        if np.isfinite(fwhm_arcsec):
                            fwhm_note = f'FWHM: {fwhm_arcsec:.2f}" ({fwhm_est:.2f} px; radial)'
                        else:
                            fwhm_note = f"FWHM: {fwhm_est:.2f} px (radial)"
                        self.prof_ax.axvline(r_half, color='orange', linestyle='--', linewidth=2, label=f'FWHM ~{fwhm_arcsec:.2f}" ({fwhm_est:.2f} px)')
                        self.prof_ax.axhline(half, color='green', linestyle=':', linewidth=1, alpha=0.7)
                        self.prof_ax.legend()

            self.prof_fig.tight_layout()
            self.prof_ax.text(0.02, 0.95, fwhm_note, transform=self.prof_ax.transAxes,
                              ha='left', va='top', fontsize=10,
                              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            self.prof_canvas.draw()

        except Exception as e:
            self.stats_text.append(f"\n[Radial Profile] Error: {str(e)}")

    def _move_dialog_to_side(self, dialog):
        """Place tool dialog to the right of the main window."""
        if dialog is None:
            return
        main_geo = self.frameGeometry()
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        x = min(main_geo.topRight().x() + 10, avail.right() - dialog.width())
        y = min(main_geo.top(), avail.bottom() - dialog.height())
        dialog.move(x, y)

    # ========== Parameter Dialog ==========

    def open_parameters_dialog(self):
        """Open photometry parameters adjustment dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Photometry Parameters")
        dialog.resize(400, 350)

        layout = QVBoxLayout(dialog)

        # Info label
        info = QLabel("Adjust aperture photometry parameters.\nChanges apply immediately to measurements.")
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; margin-bottom: 10px; }")
        layout.addWidget(info)

        # Form layout
        form_layout = QFormLayout()

        # Create input fields - scale based parameters
        fwhm_seed_edit = QLineEdit(str(self.fwhm_seed_arcsec))
        ap_scale_edit = QLineEdit(str(self.aperture_scale))
        ann_in_scale_edit = QLineEdit(str(self.annulus_in_scale))
        ann_out_scale_edit = QLineEdit(str(self.annulus_out_scale))
        min_r_ap_edit = QLineEdit(str(self.min_r_ap_px))
        min_r_in_edit = QLineEdit(str(self.min_r_in_px))
        min_r_out_edit = QLineEdit(str(self.min_r_out_px))
        sigma_clip_edit = QLineEdit(str(self.sigma_clip_value))

        form_layout.addRow("FWHM Seed (arcsec):", fwhm_seed_edit)
        form_layout.addRow("Aperture Scale (× FWHM):", ap_scale_edit)
        form_layout.addRow("Annulus Inner Scale (× FWHM):", ann_in_scale_edit)
        form_layout.addRow("Annulus Outer Width (× FWHM):", ann_out_scale_edit)
        form_layout.addRow("Min Aperture Radius (px):", min_r_ap_edit)
        form_layout.addRow("Min Annulus Inner (px):", min_r_in_edit)
        form_layout.addRow("Min Annulus Outer (px):", min_r_out_edit)
        form_layout.addRow("Sigma Clipping (σ):", sigma_clip_edit)

        layout.addLayout(form_layout)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)

        def save_parameters():
            try:
                self.fwhm_seed_arcsec = float(fwhm_seed_edit.text())
                self.aperture_scale = float(ap_scale_edit.text())
                self.annulus_in_scale = float(ann_in_scale_edit.text())
                self.annulus_out_scale = float(ann_out_scale_edit.text())
                self.min_r_ap_px = float(min_r_ap_edit.text())
                self.min_r_in_px = float(min_r_in_edit.text())
                self.min_r_out_px = float(min_r_out_edit.text())
                self.sigma_clip_value = float(sigma_clip_edit.text())

                # Save to project state for persistence
                self.save_state()

                QMessageBox.information(dialog, "Success", "Parameters updated and saved!")
                dialog.accept()
            except ValueError as e:
                QMessageBox.critical(dialog, "Error", f"Invalid parameter value:\n{str(e)}")

        button_box.accepted.connect(save_parameters)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.exec_()

    def open_stretch_plot(self):
        """Open stretch plot window showing histogram with draggable min/max markers"""
        if self.image_data is None:
            QMessageBox.warning(self, "Warning", "Load an image first")
            return

        # If dialog exists and is visible, just raise it
        if self.stretch_plot_dialog is not None and self.stretch_plot_dialog.isVisible():
            self.stretch_plot_dialog.raise_()
            self.stretch_plot_dialog.activateWindow()
            self._update_stretch_plot()
            return

        # Create dialog
        self.stretch_plot_dialog = QDialog(self)
        self.stretch_plot_dialog.setWindowTitle("2D Plot - Stretch Control")
        self.stretch_plot_dialog.resize(500, 250)

        layout = QVBoxLayout(self.stretch_plot_dialog)

        # Info label
        self.stretch_plot_info_label = QLabel("Drag min/max markers to adjust stretch")
        self.stretch_plot_info_label.setStyleSheet("QLabel { padding: 5px; background-color: #E3F2FD; border-radius: 3px; }")
        layout.addWidget(self.stretch_plot_info_label)

        # Matplotlib figure for histogram
        self.stretch_plot_fig = Figure(figsize=(6, 2.5))
        self.stretch_plot_canvas = FigureCanvas(self.stretch_plot_fig)
        self.stretch_plot_ax = self.stretch_plot_fig.add_subplot(111)
        self.stretch_plot_fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9)

        # Connect mouse events
        self.stretch_plot_canvas.mpl_connect('button_press_event', self._on_stretch_plot_press)
        self.stretch_plot_canvas.mpl_connect('motion_notify_event', self._on_stretch_plot_motion)
        self.stretch_plot_canvas.mpl_connect('button_release_event', self._on_stretch_plot_release)

        layout.addWidget(self.stretch_plot_canvas)

        # Hint label
        hint_label = QLabel("Click and drag < > markers to adjust min/max | Changes apply in real-time")
        hint_label.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        layout.addWidget(hint_label)

        self.stretch_plot_dialog.show()
        self._update_stretch_plot()

    def _update_stretch_plot(self):
        """Update the stretch plot histogram and markers"""
        if self.stretch_plot_ax is None or self.image_data is None:
            return

        ax = self.stretch_plot_ax
        ax.clear()

        # Get image data statistics
        data = self.image_data.copy()
        finite_mask = np.isfinite(data)
        if not finite_mask.any():
            return

        flat = data[finite_mask].flatten()

        # Use percentile range for histogram display
        p_low, p_high = np.percentile(flat, [1, 99])
        display_data = flat[(flat >= p_low) & (flat <= p_high)]

        if len(display_data) == 0:
            display_data = flat

        # Store data range for marker calculations
        self._stretch_data_range = (float(p_low), float(p_high))

        # Calculate current vmin/vmax from normalize_image logic
        if self._stretch_vmin is None or self._stretch_vmax is None:
            # Get initial values based on current stretch
            stretch_idx = self.scale_combo.currentIndex()
            if stretch_idx == 6:  # Linear 1-99%
                vmin = np.percentile(flat, 1)
                vmax = np.percentile(flat, 99)
            elif stretch_idx == 7:  # ZScale
                vmin, vmax = self.calculate_zscale()
            else:
                _, median_val, std_val = sigma_clipped_stats(flat, sigma=3.0, maxiters=5)
                vmin = max(np.min(flat), median_val - 2.8 * std_val)
                vmax = min(np.max(flat), np.percentile(flat, 99.9))

            if vmax <= vmin:
                vmin = np.min(flat)
                vmax = np.max(flat)

            self._stretch_vmin = float(vmin)
            self._stretch_vmax = float(vmax)

        # Plot histogram
        hist, bins, _ = ax.hist(display_data, bins=128, color='#3a6ea5', edgecolor='none', alpha=0.7)
        ax.set_xlim(p_low, p_high)

        # Draw min/max markers
        vmin = self._stretch_vmin
        vmax = self._stretch_vmax

        # Clamp markers to display range for visualization
        vmin_display = max(p_low, min(p_high, vmin))
        vmax_display = max(p_low, min(p_high, vmax))

        self._stretch_marker_min_line = ax.axvline(vmin_display, color='#FF5722', linewidth=2, linestyle='-', label=f'Min: {vmin:.1f}')
        self._stretch_marker_max_line = ax.axvline(vmax_display, color='#4CAF50', linewidth=2, linestyle='-', label=f'Max: {vmax:.1f}')

        # Add marker text
        y_max = ax.get_ylim()[1]
        ax.text(vmin_display, y_max * 0.95, '◀', color='#FF5722', fontsize=14, ha='center', va='top', fontweight='bold')
        ax.text(vmax_display, y_max * 0.95, '▶', color='#4CAF50', fontsize=14, ha='center', va='top', fontweight='bold')

        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Count')
        ax.set_title('Image Histogram')
        ax.legend(loc='upper right', fontsize=8)

        # Update info label
        if self.stretch_plot_info_label:
            stretch_name = self.scale_combo.currentText()
            self.stretch_plot_info_label.setText(
                f"Stretch: {stretch_name} | Min: {vmin:.2f} | Max: {vmax:.2f}"
            )

        self.stretch_plot_canvas.draw_idle()

    def _on_stretch_plot_press(self, event):
        """Handle mouse press on stretch plot"""
        if event.inaxes != self.stretch_plot_ax or event.xdata is None:
            return

        if self._stretch_vmin is None or self._stretch_vmax is None:
            return

        x = event.xdata
        dist_to_min = abs(x - self._stretch_vmin)
        dist_to_max = abs(x - self._stretch_vmax)

        # Select closer marker
        if dist_to_min < dist_to_max:
            self._stretch_drag_target = "min"
        else:
            self._stretch_drag_target = "max"

        self._stretch_dragging = True

    def _on_stretch_plot_motion(self, event):
        """Handle mouse motion on stretch plot (dragging)"""
        if not self._stretch_dragging or event.xdata is None:
            return

        x = event.xdata

        if self._stretch_drag_target == "min":
            # Don't let min exceed max
            new_val = min(x, self._stretch_vmax - 1)
            self._stretch_vmin = new_val
        else:
            # Don't let max go below min
            new_val = max(x, self._stretch_vmin + 1)
            self._stretch_vmax = new_val

        # Update plot and image in real-time
        self._update_stretch_plot()
        self._apply_custom_stretch()

    def _on_stretch_plot_release(self, event):
        """Handle mouse release on stretch plot"""
        self._stretch_dragging = False
        self._stretch_drag_target = None

    def _apply_custom_stretch(self):
        """Apply custom vmin/vmax stretch to the image"""
        if self.image_data is None:
            return

        if self._stretch_vmin is None or self._stretch_vmax is None:
            return

        vmin = self._stretch_vmin
        vmax = self._stretch_vmax

        # Normalize with custom vmin/vmax
        data = self.image_data.copy()
        if vmax <= vmin:
            vmax = vmin + 1

        normalized = (data - vmin) / (vmax - vmin + 1e-10)
        normalized = np.clip(normalized, 0, 1)

        # Apply current stretch function
        stretched = self.apply_stretch(normalized)

        if self._imshow_obj is not None:
            self._imshow_obj.set_data(stretched)
            self.canvas.draw_idle()

    def reset_stretch_plot_values(self):
        """Reset stretch plot values when changing image or stretch mode"""
        self._stretch_vmin = None
        self._stretch_vmax = None
        if self.stretch_plot_dialog and self.stretch_plot_dialog.isVisible():
            self._update_stretch_plot()

    def validate_step(self) -> bool:
        """Validate if step can be completed"""
        # Step 3 is optional QC, always allow completion
        return True

    def save_state(self):
        """Save step state including photometry parameters"""
        state_data = {
            "current_file": self.current_filename,
            "use_cropped": self.use_cropped,
            # Photometry parameters (scale-based)
            "fwhm_seed_arcsec": self.fwhm_seed_arcsec,
            "aperture_scale": self.aperture_scale,
            "annulus_in_scale": self.annulus_in_scale,
            "annulus_out_scale": self.annulus_out_scale,
            "min_r_ap_px": self.min_r_ap_px,
            "min_r_in_px": self.min_r_in_px,
            "min_r_out_px": self.min_r_out_px,
            "sigma_clip_value": self.sigma_clip_value,
        }
        self.project_state.store_step_data("sky_preview", state_data)

    def restore_state(self):
        """Restore step state including photometry parameters"""
        state_data = self.project_state.get_step_data("sky_preview")
        if state_data:
            # Restore photometry parameters
            if "fwhm_seed_arcsec" in state_data:
                self.fwhm_seed_arcsec = float(state_data["fwhm_seed_arcsec"])
            if "aperture_scale" in state_data:
                self.aperture_scale = float(state_data["aperture_scale"])
            if "annulus_in_scale" in state_data:
                self.annulus_in_scale = float(state_data["annulus_in_scale"])
            if "annulus_out_scale" in state_data:
                self.annulus_out_scale = float(state_data["annulus_out_scale"])
            if "min_r_ap_px" in state_data:
                self.min_r_ap_px = float(state_data["min_r_ap_px"])
            if "min_r_in_px" in state_data:
                self.min_r_in_px = float(state_data["min_r_in_px"])
            if "min_r_out_px" in state_data:
                self.min_r_out_px = float(state_data["min_r_out_px"])
            if "sigma_clip_value" in state_data:
                self.sigma_clip_value = float(state_data["sigma_clip_value"])
