"""
Step 10: PSF Photometry (placeholder).
"""

from __future__ import annotations

from PyQt5.QtWidgets import QVBoxLayout, QLabel

from .step_window_base import StepWindowBase


class PsfPhotometryWindow(StepWindowBase):
    """Step 10: PSF Photometry placeholder window."""

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        super().__init__(
            step_index=9,
            step_name="PSF Photometry",
            params=params,
            project_state=project_state,
            main_window=main_window,
        )
        self.setup_step_ui()
        self.restore_state()

    def setup_step_ui(self):
        info = QLabel(
            "PSF photometry stage placeholder.\n"
            "Frame-by-frame PSF modeling will be implemented here."
        )
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        self.content_layout.addWidget(info)

    def validate_step(self) -> bool:
        return False

    def save_state(self):
        self.project_state.store_step_data("psf_photometry", {})

    def restore_state(self):
        return
