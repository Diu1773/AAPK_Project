"""
Enhanced FITS Image Viewer with zoom, pan, and scaling options
"""

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel
from PyQt5.QtCore import Qt


class FITSImageViewer(QWidget):
    """
    Enhanced FITS image viewer with:
    - Mouse wheel zoom
    - Right-click pan
    - Multiple scaling options (ZScale, Linear, Log, etc)
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.image_data = None
        self.display_data = None

        # Scaling options
        self.scale_mode = 'zscale'  # zscale, linear, log, sqrt, asinh

        # Zoom/Pan state
        self.xlim_original = None
        self.ylim_original = None
        self.panning = False
        self.pan_start = None

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Setup viewer UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Control bar
        control_layout = QHBoxLayout()

        # Scale mode selector
        control_layout.addWidget(QLabel("Scale:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(['ZScale (Auto)', 'Linear (1-99%)', 'Log', 'Sqrt', 'Asinh'])
        self.scale_combo.currentIndexChanged.connect(self.on_scale_changed)
        control_layout.addWidget(self.scale_combo)

        # Reset zoom button
        btn_reset = QPushButton("Reset Zoom")
        btn_reset.clicked.connect(self.reset_zoom)
        control_layout.addWidget(btn_reset)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        # Matplotlib canvas
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Connect mouse events
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

        layout.addWidget(self.canvas)

    def set_image(self, image_data, title="FITS Image"):
        """Set image data to display"""
        self.image_data = image_data.copy()
        self.display_image(title=title)

    def display_image(self, title="FITS Image"):
        """Display image with current scaling"""
        if self.image_data is None:
            return

        self.ax.clear()

        # Apply scaling
        vmin, vmax = self.calculate_scale()

        # Display image
        self.ax.imshow(
            self.image_data,
            cmap='gray',
            origin='lower',
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest'
        )

        self.ax.set_xlabel("X (pixels)")
        self.ax.set_ylabel("Y (pixels)")
        self.ax.set_title(title)

        # Store original limits for reset
        if self.xlim_original is None:
            self.xlim_original = self.ax.get_xlim()
            self.ylim_original = self.ax.get_ylim()

        self.canvas.draw()

    def calculate_scale(self):
        """Calculate vmin/vmax based on selected scaling mode"""
        if self.image_data is None:
            return 0, 1

        scale_mode = self.scale_combo.currentIndex()

        if scale_mode == 0:  # ZScale (Auto)
            return self.zscale()
        elif scale_mode == 1:  # Linear (1-99%)
            vmin = np.percentile(self.image_data, 1)
            vmax = np.percentile(self.image_data, 99)
            return vmin, vmax
        elif scale_mode == 2:  # Log
            data_positive = self.image_data - self.image_data.min() + 1
            vmin = np.percentile(np.log10(data_positive), 1)
            vmax = np.percentile(np.log10(data_positive), 99)
            return vmin, vmax
        elif scale_mode == 3:  # Sqrt
            data_positive = self.image_data - self.image_data.min()
            vmin = np.percentile(np.sqrt(data_positive), 1)
            vmax = np.percentile(np.sqrt(data_positive), 99)
            return vmin, vmax
        elif scale_mode == 4:  # Asinh
            vmin = np.percentile(np.arcsinh(self.image_data), 1)
            vmax = np.percentile(np.arcsinh(self.image_data), 99)
            return vmin, vmax

        return self.image_data.min(), self.image_data.max()

    def zscale(self):
        """
        Simple ZScale implementation (approximation)
        Returns vmin, vmax for optimal contrast
        """
        # Use central region for statistics
        h, w = self.image_data.shape
        cy, cx = h // 2, w // 2
        size = min(h, w) // 4

        y0 = max(0, cy - size)
        y1 = min(h, cy + size)
        x0 = max(0, cx - size)
        x1 = min(w, cx + size)

        central = self.image_data[y0:y1, x0:x1]

        # Use median and std for robust estimation
        median = np.median(central)
        std = np.std(central)

        vmin = median - 2 * std
        vmax = median + 6 * std

        # Clip to data range
        vmin = max(vmin, self.image_data.min())
        vmax = min(vmax, self.image_data.max())

        return vmin, vmax

    def on_scale_changed(self, index):
        """Handle scale mode change"""
        self.display_image()

    def on_scroll(self, event):
        """Handle mouse wheel zoom"""
        if event.inaxes != self.ax:
            return

        # Get current limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Zoom factor
        zoom_factor = 1.2 if event.button == 'up' else 0.8

        # Get cursor position
        xdata, ydata = event.xdata, event.ydata

        # Calculate new limits centered on cursor
        x_range = (xlim[1] - xlim[0]) * zoom_factor
        y_range = (ylim[1] - ylim[0]) * zoom_factor

        new_xlim = [xdata - x_range * (xdata - xlim[0]) / (xlim[1] - xlim[0]),
                    xdata + x_range * (xlim[1] - xdata) / (xlim[1] - xlim[0])]
        new_ylim = [ydata - y_range * (ydata - ylim[0]) / (ylim[1] - ylim[0]),
                    ydata + y_range * (ylim[1] - ydata) / (ylim[1] - ylim[0])]

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw()

    def on_button_press(self, event):
        """Handle mouse button press"""
        if event.button == 3:  # Right click
            self.panning = True
            self.pan_start = (event.xdata, event.ydata)

    def on_button_release(self, event):
        """Handle mouse button release"""
        if event.button == 3:
            self.panning = False
            self.pan_start = None

    def on_motion(self, event):
        """Handle mouse motion for panning"""
        if not self.panning or self.pan_start is None:
            return
        if event.inaxes != self.ax:
            return

        # Calculate pan offset
        dx = self.pan_start[0] - event.xdata
        dy = self.pan_start[1] - event.ydata

        # Get current limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Apply pan
        self.ax.set_xlim([xlim[0] + dx, xlim[1] + dx])
        self.ax.set_ylim([ylim[0] + dy, ylim[1] + dy])

        self.canvas.draw()

    def reset_zoom(self):
        """Reset zoom to original view"""
        if self.xlim_original is not None:
            self.ax.set_xlim(self.xlim_original)
            self.ax.set_ylim(self.ylim_original)
            self.canvas.draw()

    def get_figure(self):
        """Get matplotlib figure"""
        return self.figure

    def get_ax(self):
        """Get matplotlib axes"""
        return self.ax

    def get_canvas(self):
        """Get matplotlib canvas"""
        return self.canvas
