"""
Step 2: Image Crop Window
Allows user to select crop region on reference image and apply crop to all files
"""

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QMessageBox, QProgressDialog
)
from PyQt5.QtCore import Qt
from pathlib import Path
import json
import shutil
import numpy as np
from astropy.io import fits
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .step_window_base import StepWindowBase
from ...utils.step_paths import step2_dir, step2_cropped_dir, crop_rect_path


class CropSelectorWindow(StepWindowBase):
    """Step 2: Image Crop Selection"""

    def __init__(self, params, file_manager, project_state, main_window):
        """
        Initialize crop selector window

        Args:
            params: Parameters object
            file_manager: FileManager instance
            project_state: ProjectState object
            main_window: Main window reference
        """
        self.file_manager = file_manager

        # Crop rectangle coordinates
        self.crop_x0 = None
        self.crop_y0 = None
        self.crop_x1 = None
        self.crop_y1 = None

        # Matplotlib components
        self.figure = None
        self.canvas = None
        self.ax = None
        self.rect_patch = None  # Rectangle patch for visualization
        self.original_image_data = None  # Original uncropped image
        self.displayed_image_data = None  # Currently displayed (may be cropped)
        self.ref_filename = None
        self.is_cropped = False  # Track if images have been cropped
        self.crop_skipped = False

        # Mouse tracking for rectangle drawing
        self.drawing = False
        self.start_x = None
        self.start_y = None

        # Initialize base class
        super().__init__(
            step_index=1,
            step_name="Image Crop",
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
            "1. Load reference image\n"
            "2. Draw rectangle by clicking and dragging\n"
            "3. Click 'Apply Crop' to save cropped images to 'result/cropped/' folder"
        )
        info_label.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        self.content_layout.addWidget(info_label)

        # === Image Viewer ===
        viewer_group = QGroupBox("Reference Image Viewer")
        viewer_layout = QVBoxLayout(viewer_group)

        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)

        viewer_layout.addWidget(self.canvas)

        # Controls below image
        image_controls_layout = QHBoxLayout()

        self.btn_crop = QPushButton("Apply Crop")
        self.btn_crop.setEnabled(False)
        self.btn_crop.setMinimumHeight(35)
        self.btn_crop.setStyleSheet("""
            QPushButton {
                background-color: #FF5722;
                color: white;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #E64A19;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        self.btn_crop.clicked.connect(self.apply_crop)
        image_controls_layout.addWidget(self.btn_crop)

        self.btn_reset = QPushButton("Reset to Original")
        self.btn_reset.setMinimumHeight(35)
        self.btn_reset.clicked.connect(self.reset_to_original)
        image_controls_layout.addWidget(self.btn_reset)

        self.btn_skip = QPushButton("Skip Crop")
        self.btn_skip.setMinimumHeight(35)
        self.btn_skip.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #546E7A;
            }
        """)
        self.btn_skip.clicked.connect(self.skip_crop)
        image_controls_layout.addWidget(self.btn_skip)

        image_controls_layout.addStretch()

        viewer_layout.addLayout(image_controls_layout)
        self.content_layout.addWidget(viewer_group)

        # === Controls ===
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)

        # Load Image button
        btn_load = QPushButton("Load Reference Image")
        btn_load.clicked.connect(self.load_reference_image)
        controls_layout.addWidget(btn_load)

        controls_layout.addStretch()

        self.content_layout.addWidget(controls_group)

        # === Crop Info ===
        crop_info_group = QGroupBox("Status")
        crop_info_layout = QVBoxLayout(crop_info_group)

        self.crop_info_label = QLabel("No image loaded")
        self.crop_info_label.setStyleSheet("QLabel { font-weight: bold; color: blue; }")
        crop_info_layout.addWidget(self.crop_info_label)

        self.content_layout.addWidget(crop_info_group)

    def load_reference_image(self):
        """Load reference image from file manager"""
        try:
            # Check if files are available from Step 1
            if not self.file_manager.filenames:
                # Try to scan files
                try:
                    self.file_manager.scan_files()
                except:
                    QMessageBox.warning(
                        self, "No Files",
                        "Please complete Step 1 first to scan files."
                    )
                    return

            # Use reference from Step 1 if available, otherwise use first file
            if self.file_manager.ref_filename and self.file_manager.ref_filename in self.file_manager.filenames:
                # Use reference selected in Step 1
                self.ref_filename = self.file_manager.ref_filename
            elif self.ref_filename and self.ref_filename in self.file_manager.filenames:
                # Use previously selected reference from this step
                pass
            else:
                # Fallback to first file
                self.ref_filename = self.file_manager.filenames[0]

            file_path = self.params.P.data_dir / self.ref_filename

            # Load FITS data from original
            hdul = fits.open(file_path)
            self.original_image_data = hdul[0].data.astype(float).copy()
            hdul.close()

            self.displayed_image_data = self.original_image_data.copy()
            self.is_cropped = False

            # Display image
            self.display_image()

            # Update status
            self.crop_info_label.setText(f"Loaded: {self.ref_filename}\nReady to draw crop region")

            self.update_navigation_buttons()

        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to load reference image:\n{str(e)}"
            )

    def display_image(self):
        """Display image with proper scaling"""
        if self.displayed_image_data is None:
            return

        self.ax.clear()

        # Calculate percentile-based scaling for better visualization
        vmin = np.percentile(self.displayed_image_data, 1)
        vmax = np.percentile(self.displayed_image_data, 99)

        # Display image
        self.ax.imshow(
            self.displayed_image_data,
            cmap='gray',
            origin='lower',
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest'
        )

        self.ax.set_xlabel("X (pixels)")
        self.ax.set_ylabel("Y (pixels)")

        status = "CROPPED" if self.is_cropped else "ORIGINAL"
        self.ax.set_title(f"Reference: {self.ref_filename} [{status}]")

        # Draw existing rectangle if coordinates are set
        if self.crop_x0 is not None and not self.is_cropped:
            width = self.crop_x1 - self.crop_x0
            height = self.crop_y1 - self.crop_y0
            rect = Rectangle((self.crop_x0, self.crop_y0), width, height,
                           linewidth=2, edgecolor='red', facecolor='none')
            self.ax.add_patch(rect)

        self.canvas.draw()

    def on_mouse_press(self, event):
        """Handle mouse press event"""
        if self.is_cropped or self.original_image_data is None:
            return
        if event.inaxes != self.ax:
            return

        self.drawing = True
        self.start_x = int(event.xdata)
        self.start_y = int(event.ydata)

        # Clear any existing rectangle patch
        if self.rect_patch is not None:
            self.rect_patch.remove()
            self.rect_patch = None

    def on_mouse_move(self, event):
        """Handle mouse move event - optimized with patch update"""
        if not self.drawing or self.is_cropped:
            return
        if event.inaxes != self.ax:
            return

        # Get current position
        current_x = int(event.xdata)
        current_y = int(event.ydata)

        # Update rectangle coordinates
        x0 = min(self.start_x, current_x)
        x1 = max(self.start_x, current_x)
        y0 = min(self.start_y, current_y)
        y1 = max(self.start_y, current_y)

        # Remove old rectangle patch if exists
        if self.rect_patch is not None:
            self.rect_patch.remove()

        # Create and add new rectangle patch
        width = x1 - x0
        height = y1 - y0
        self.rect_patch = Rectangle((x0, y0), width, height,
                                    linewidth=2, edgecolor='red', facecolor='none',
                                    animated=False)
        self.ax.add_patch(self.rect_patch)

        # Fast redraw (only canvas, not full figure)
        self.canvas.draw_idle()

    def on_mouse_release(self, event):
        """Handle mouse release event"""
        if not self.drawing:
            return

        self.drawing = False

        if event.inaxes != self.ax:
            return

        # Get final coordinates
        end_x = int(event.xdata)
        end_y = int(event.ydata)

        # Ensure x0 < x1 and y0 < y1
        self.crop_x0 = min(self.start_x, end_x)
        self.crop_x1 = max(self.start_x, end_x)
        self.crop_y0 = min(self.start_y, end_y)
        self.crop_y1 = max(self.start_y, end_y)

        # Validate minimum size
        if (self.crop_x1 - self.crop_x0) < 50 or (self.crop_y1 - self.crop_y0) < 50:
            QMessageBox.warning(self, "Too Small", "Crop region must be at least 50×50 pixels")
            self.crop_x0 = None
            self.crop_y0 = None
            self.crop_x1 = None
            self.crop_y1 = None
            self.display_image()
            return

        # Update info label
        width = self.crop_x1 - self.crop_x0
        height = self.crop_y1 - self.crop_y0
        self.crop_info_label.setText(
            f"Crop Region Selected: ({self.crop_x0}, {self.crop_y0}) → ({self.crop_x1}, {self.crop_y1})\n"
            f"Size: {width} × {height} pixels\n"
            f"Click 'Apply Crop' to crop all images"
        )

        # Enable crop button
        self.btn_crop.setEnabled(True)

        # Redraw final rectangle
        self.display_image()

    def apply_crop(self):
        """Apply crop to all FITS files"""
        if not self.validate_crop_region():
            return

        # Confirm with user
        reply = QMessageBox.question(
            self, "Apply Crop",
            f"This will create cropped files in 'result/step2_crop/cropped' folder.\n"
            f"Original files will not be modified.\n\n"
            f"Crop {len(self.file_manager.filenames)} images\n"
            f"Region: ({self.crop_x0}, {self.crop_y0}) → ({self.crop_x1}, {self.crop_y1})\n\n"
            f"Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return
        self.crop_skipped = False

        # Create progress dialog
        progress = QProgressDialog("Cropping images...", "Cancel", 0, len(self.file_manager.filenames), self)
        progress.setWindowTitle("Processing")
        progress.setWindowModality(Qt.WindowModal)

        try:
            # Create cropped directory only
            cropped_dir = step2_cropped_dir(self.params.P.result_dir)
            cropped_dir.mkdir(parents=True, exist_ok=True)

            # Crop all files
            for i, filename in enumerate(self.file_manager.filenames):
                if progress.wasCanceled():
                    break

                progress.setValue(i)
                progress.setLabelText(f"Cropping {filename}...")

                original_path = self.params.P.data_dir / filename
                cropped_path = cropped_dir / filename

                # Read FITS data and header from original
                hdul = fits.open(original_path)
                original_data = hdul[0].data.copy()
                header = hdul[0].header.copy()
                hdul.close()

                # Crop data
                cropped_data = original_data[self.crop_y0:self.crop_y1, self.crop_x0:self.crop_x1]

                # Update header with crop info
                header['CROPPED'] = (True, 'Image has been cropped')
                header['CROP_X0'] = (self.crop_x0, 'Crop region start X')
                header['CROP_Y0'] = (self.crop_y0, 'Crop region start Y')
                header['CROP_X1'] = (self.crop_x1, 'Crop region end X')
                header['CROP_Y1'] = (self.crop_y1, 'Crop region end Y')

                # Save cropped file
                hdu = fits.PrimaryHDU(data=cropped_data, header=header)
                hdu.writeto(cropped_path, overwrite=True)

            progress.setValue(len(self.file_manager.filenames))

            # Update displayed image to show cropped result
            self.displayed_image_data = self.original_image_data[self.crop_y0:self.crop_y1, self.crop_x0:self.crop_x1]
            self.is_cropped = True

            # Redisplay cropped image (without rectangle)
            self.display_image()

            # Save crop info
            self.save_crop_info()

            # Update status
            width = self.crop_x1 - self.crop_x0
            height = self.crop_y1 - self.crop_y0
            self.crop_info_label.setText(
                f"Crop Applied!\n"
                f"Cropped files saved to: result/step2_crop/cropped/\n"
                f"Size: {width} × {height} pixels"
            )

            # Disable crop button
            self.btn_crop.setEnabled(False)

            # Update navigation buttons to enable Mark Complete
            self.update_navigation_buttons()

            QMessageBox.information(
                self, "Crop Complete",
                f"Successfully cropped {len(self.file_manager.filenames)} images!\n\n"
                f"Cropped files: result/step2_crop/cropped/\n"
                f"Original files: unchanged"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Crop Error",
                f"Failed to crop images:\n{str(e)}"
            )

    def reset_to_original(self):
        """Reset selection and reload original image"""
        if self.is_cropped:
            # Just reload original
            try:
                file_path = self.params.P.data_dir / self.ref_filename
                hdul = fits.open(file_path)
                self.original_image_data = hdul[0].data.astype(float).copy()
                hdul.close()

                self.displayed_image_data = self.original_image_data.copy()
                self.is_cropped = False

                # Clear crop coordinates
                self.crop_x0 = None
                self.crop_y0 = None
                self.crop_x1 = None
                self.crop_y1 = None

                # Redisplay
                self.display_image()

                self.crop_info_label.setText(f"Loaded: {self.ref_filename}\nReady to draw crop region")
                self.btn_crop.setEnabled(False)
                self.update_navigation_buttons()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to reload:\n{str(e)}")
            return

        # Just reset selection if not cropped
        # Reset crop coordinates
        self.crop_x0 = None
        self.crop_y0 = None
        self.crop_x1 = None
        self.crop_y1 = None

        # Disable crop button
        self.btn_crop.setEnabled(False)

        # Reset display to original
        if self.original_image_data is not None:
            self.displayed_image_data = self.original_image_data.copy()
            self.display_image()

        self.crop_info_label.setText(f"Loaded: {self.ref_filename}\nReady to draw crop region")
        self.update_navigation_buttons()

    def skip_crop(self):
        """Mark crop step as skipped (use original images)."""
        reply = QMessageBox.question(
            self, "Skip Crop",
            "Skip cropping and use original images?\n"
            "Any existing cropped outputs will be ignored.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self.crop_skipped = True
        self.is_cropped = False
        self.crop_x0 = None
        self.crop_y0 = None
        self.crop_x1 = None
        self.crop_y1 = None
        if self.rect_patch is not None:
            self.rect_patch.remove()
            self.rect_patch = None

        if self.original_image_data is not None:
            self.displayed_image_data = self.original_image_data.copy()
            self.display_image()

        self.btn_crop.setEnabled(False)
        self.save_skip_info()
        self.crop_info_label.setText("Crop skipped. Original images will be used.")
        self.update_navigation_buttons()

    def validate_crop_region(self) -> bool:
        """Validate if crop region is valid"""
        if self.crop_x0 is None or self.crop_y0 is None or self.crop_x1 is None or self.crop_y1 is None:
            QMessageBox.warning(
                self, "No Crop Region",
                "Please draw a crop region first by clicking and dragging on the image."
            )
            return False
        return True

    def validate_step(self) -> bool:
        """Validate if step can be completed"""
        # Step can be completed if crop has been applied
        return self.is_cropped or self.crop_skipped

    def save_crop_info(self):
        """Save crop info to JSON"""
        crop_data = {
            "x0": int(self.crop_x0),
            "y0": int(self.crop_y0),
            "x1": int(self.crop_x1),
            "y1": int(self.crop_y1),
            "ref_file": self.ref_filename or "",
            "skipped": False
        }

        # Save to result directory
        output_dir = step2_dir(self.params.P.result_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rect_path = output_dir / "crop_rect.json"
        with open(rect_path, 'w') as f:
            json.dump(crop_data, f, indent=2)

        # Also save to project state
        self.project_state.store_step_data("crop", crop_data)

        print(f"Crop region saved: {rect_path}")

    def save_skip_info(self):
        """Persist skip state and disable crop marker."""
        crop_data = {
            "skipped": True,
            "ref_file": self.ref_filename or ""
        }
        self.project_state.store_step_data("crop", crop_data)
        rect_path = crop_rect_path(self.params.P.result_dir)
        try:
            rect_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def save_state(self):
        """Save step state to project"""
        if self.is_cropped:
            self.save_crop_info()
        elif self.crop_skipped:
            self.save_skip_info()

    def restore_state(self):
        """Restore step state from project"""
        # First, check if there's a reference from Step 1
        step1_data = self.project_state.get_step_data("file_selection")
        if step1_data and step1_data.get("reference_frame"):
            # Use reference from Step 1
            if not self.ref_filename:  # Only if not already set
                self.ref_filename = step1_data.get("reference_frame")

        # Try to restore crop state
        state_data = self.project_state.get_step_data("crop")

        if state_data:
            if state_data.get("skipped"):
                self.crop_skipped = True
                self.is_cropped = False
                if state_data.get("ref_file"):
                    self.ref_filename = state_data.get("ref_file")
                try:
                    self.load_reference_image()
                except Exception:
                    pass
                self.crop_info_label.setText("Crop skipped. Original images will be used.")
                self.update_navigation_buttons()
                return

            self.crop_x0 = state_data.get("x0")
            self.crop_y0 = state_data.get("y0")
            self.crop_x1 = state_data.get("x1")
            self.crop_y1 = state_data.get("y1")

            # Use crop's ref_file if available
            if state_data.get("ref_file"):
                self.ref_filename = state_data.get("ref_file")

            self.is_cropped = True  # Assume cropped if we have saved state

            # Try to load image
            try:
                self.load_reference_image()

                # Update info label
                if self.crop_x0 is not None:
                    width = self.crop_x1 - self.crop_x0
                    height = self.crop_y1 - self.crop_y0
                    self.crop_info_label.setText(
                        f"Crop Previously Applied\n"
                        f"Cropped files in: result/step2_crop/cropped/\n"
                        f"Size: {width} × {height} pixels"
                    )
            except:
                pass
