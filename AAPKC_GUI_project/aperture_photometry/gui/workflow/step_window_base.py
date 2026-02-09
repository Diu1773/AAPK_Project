"""
Base class for step windows
Each step gets its own window with Previous/Next navigation
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


class StepWindowBase(QMainWindow):
    """Base class for workflow step windows with navigation buttons"""

    # Signals
    step_completed = pyqtSignal(int)  # Emit step index when completed
    go_to_step = pyqtSignal(int)  # Request to go to another step

    def __init__(self, step_index: int, step_name: str, params, project_state, main_window, parent=None):
        """
        Initialize step window

        Args:
            step_index: Index of this step (0-based)
            step_name: Display name of this step
            params: Parameters object
            project_state: ProjectState object
            main_window: Reference to main window
            parent: Parent widget
        """
        super().__init__(parent)

        self.step_index = step_index
        self.step_name = step_name
        self.params = params
        self.project_state = project_state
        self.main_window = main_window

        self.completed = False

        # Connect to main window
        self.step_completed.connect(main_window.on_step_completed)

        # Setup base UI
        self.setWindowTitle(f"Step {step_index + 1}: {step_name}")
        self.setMinimumSize(900, 700)

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Title
        self.title_label = QLabel(f"Step {step_index + 1}: {step_name}")
        self.title_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.title_label)

        # Content area (to be filled by subclasses)
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.main_layout.addWidget(self.content_widget)

        # Navigation buttons
        self.setup_navigation_buttons()

        # NOTE: restore_state() should be called by subclass after UI setup

    def setup_navigation_buttons(self):
        """Setup Previous/Next/Complete buttons at bottom"""
        nav_layout = QHBoxLayout()

        # Previous button (GREEN - always enabled if not first step)
        self.btn_previous = QPushButton("← Previous Step")
        self.btn_previous.setMinimumHeight(40)
        self.btn_previous.setFont(QFont("Arial", 10, QFont.Bold))
        self.btn_previous.clicked.connect(self.go_previous)

        if self.step_index > 0:
            # Green style for Previous
            self.btn_previous.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: 2px solid #45a049;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            self.btn_previous.setEnabled(True)
        else:
            # Disabled for Step 1
            self.btn_previous.setEnabled(False)

        nav_layout.addWidget(self.btn_previous)

        nav_layout.addStretch()

        # Complete button (initially disabled)
        self.btn_complete = QPushButton("Mark as Complete")
        self.btn_complete.setMinimumHeight(40)
        self.btn_complete.setFont(QFont("Arial", 10, QFont.Bold))
        self.btn_complete.setEnabled(False)
        self.btn_complete.clicked.connect(self.mark_complete)
        nav_layout.addWidget(self.btn_complete)

        # Next button (RED when incomplete, GREEN when complete)
        self.btn_next = QPushButton("Next Step →")
        self.btn_next.setMinimumHeight(40)
        self.btn_next.setFont(QFont("Arial", 10, QFont.Bold))
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self.go_next)

        # Initial red style (not complete)
        self.btn_next.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: 2px solid #da190b;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
                border: 2px solid #AAAAAA;
            }
        """)

        nav_layout.addWidget(self.btn_next)

        self.main_layout.addLayout(nav_layout)

        # Update button states
        self.update_navigation_buttons()

    def update_navigation_buttons(self):
        """Update navigation button states"""
        # Complete button: enable if validation passes
        can_complete = self.validate_step()
        self.btn_complete.setEnabled(can_complete)

        # Next button: enable if step is completed
        is_completed = self.project_state.is_step_completed(self.step_index)
        self.btn_next.setEnabled(is_completed)

        # Update complete button style
        if is_completed:
            # Complete button - green when done
            self.btn_complete.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: 2px solid #45a049;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            self.btn_complete.setText("✓ Completed")

            # Next button - GREEN when complete
            self.btn_next.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: 2px solid #45a049;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
        else:
            # Complete button - default style
            self.btn_complete.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: 2px solid #1976D2;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
                QPushButton:disabled {
                    background-color: #CCCCCC;
                    color: #666666;
                }
            """)
            self.btn_complete.setText("Mark as Complete")

            # Next button - RED when not complete
            self.btn_next.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    border: 2px solid #da190b;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:disabled {
                    background-color: #CCCCCC;
                    color: #666666;
                    border: 2px solid #AAAAAA;
                }
            """)

    def validate_step(self) -> bool:
        """
        Validate if step can be marked as complete
        Override in subclasses

        Returns:
            True if step can be completed
        """
        return False

    def mark_complete(self):
        """Mark this step as complete"""
        if not self.validate_step():
            QMessageBox.warning(
                self, "Cannot Complete",
                "Please complete all required tasks before marking this step as complete."
            )
            return

        # Save step data
        self.save_state()

        # Save parameters to TOML file
        if hasattr(self.params, 'save_toml'):
            self.params.save_toml()

        # Mark as completed
        self.project_state.mark_step_completed(self.step_index)
        self.completed = True

        # Emit signal
        self.step_completed.emit(self.step_index)

        # Update buttons
        self.update_navigation_buttons()

        QMessageBox.information(
            self, "Step Complete",
            f"Step {self.step_index + 1} marked as complete!\n\n"
            f"Click 'Next Step' to proceed or close this window to return to main menu."
        )

    def go_previous(self):
        """Go to previous step"""
        if self.step_index > 0:
            self.close()
            self.main_window.open_step(self.step_index - 1)

    def go_next(self):
        """Go to next step"""
        if not self.project_state.is_step_completed(self.step_index):
            QMessageBox.warning(
                self, "Step Not Complete",
                "Please complete this step before proceeding to the next."
            )
            return

        self.close()
        self.main_window.open_step(self.step_index + 1)

    def save_state(self):
        """
        Save step-specific state
        Override in subclasses to save data
        """
        pass

    def restore_state(self):
        """
        Restore step-specific state
        Override in subclasses to restore data
        """
        pass

    def closeEvent(self, event):
        """Handle window close"""
        # Auto-save state when closing
        if self.validate_step():
            self.save_state()

        # Always save parameters to TOML file when closing
        if hasattr(self.params, 'save_toml'):
            self.params.save_toml()

        event.accept()
