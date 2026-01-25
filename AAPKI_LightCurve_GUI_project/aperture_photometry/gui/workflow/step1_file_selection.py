"""
Step 1: File Selection and Header Reading
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTableWidget, QTableWidgetItem, QGroupBox,
    QLineEdit, QFileDialog, QMessageBox, QHeaderView
)
from PyQt5.QtCore import Qt
from pathlib import Path


class FileSelectionDialog(QDialog):
    """
    File selection dialog for FITS files
    Shows file list and header summary
    """

    def __init__(self, params, file_manager, parent=None):
        """
        Initialize file selection dialog

        Args:
            params: Parameters object
            file_manager: FileManager instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.params = params
        self.file_manager = file_manager

        self.setup_ui()
        self.load_files()

    def setup_ui(self):
        """Setup user interface"""
        self.setWindowTitle("Step 1: File Selection")
        self.setMinimumSize(900, 600)

        layout = QVBoxLayout(self)

        # === Directory Selection ===
        dir_group = QGroupBox("Data Directory")
        dir_layout = QHBoxLayout(dir_group)

        self.dir_edit = QLineEdit(str(self.params.P.data_dir))
        self.dir_edit.setReadOnly(True)
        dir_layout.addWidget(QLabel("Directory:"))
        dir_layout.addWidget(self.dir_edit)

        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_directory)
        dir_layout.addWidget(btn_browse)

        layout.addWidget(dir_group)

        # === File Prefix Filter ===
        filter_group = QGroupBox("File Filter")
        filter_layout = QHBoxLayout(filter_group)

        filter_layout.addWidget(QLabel("Filename Prefix:"))
        self.prefix_edit = QLineEdit(self.params.P.filename_prefix)
        self.prefix_edit.setMaximumWidth(150)
        filter_layout.addWidget(self.prefix_edit)

        btn_rescan = QPushButton("Rescan Files")
        btn_rescan.clicked.connect(self.rescan_files)
        filter_layout.addWidget(btn_rescan)

        filter_layout.addStretch()

        self.file_count_label = QLabel("Files: 0")
        filter_layout.addWidget(self.file_count_label)

        layout.addWidget(filter_group)

        # === Header Table ===
        table_group = QGroupBox("FITS Headers")
        table_layout = QVBoxLayout(table_group)

        self.header_table = QTableWidget()
        self.header_table.setColumnCount(6)
        self.header_table.setHorizontalHeaderLabels([
            "Filename", "DATE-OBS", "FILTER", "EXPTIME", "AIRMASS", "IMAGETYP"
        ])
        self.header_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.header_table.horizontalHeader().setStretchLastSection(True)
        self.header_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.header_table.setEditTriggers(QTableWidget.NoEditTriggers)

        table_layout.addWidget(self.header_table)

        layout.addWidget(table_group)

        # === Reference Frame Selection ===
        ref_group = QGroupBox("Reference Frame")
        ref_layout = QHBoxLayout(ref_group)

        ref_layout.addWidget(QLabel("Selected Reference:"))
        self.ref_label = QLabel("(not selected)")
        self.ref_label.setStyleSheet("QLabel { font-weight: bold; color: blue; }")
        ref_layout.addWidget(self.ref_label)

        ref_layout.addStretch()

        btn_auto_ref = QPushButton("Auto-Select Reference")
        btn_auto_ref.clicked.connect(self.auto_select_reference)
        ref_layout.addWidget(btn_auto_ref)

        btn_use_selected = QPushButton("Use Selected Row as Reference")
        btn_use_selected.clicked.connect(self.use_selected_as_reference)
        ref_layout.addWidget(btn_use_selected)

        layout.addWidget(ref_group)

        # === Action Buttons ===
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(btn_cancel)

        self.btn_accept = QPushButton("Accept and Continue")
        self.btn_accept.clicked.connect(self.accept_files)
        self.btn_accept.setDefault(True)
        self.btn_accept.setEnabled(False)
        button_layout.addWidget(self.btn_accept)

        layout.addLayout(button_layout)

    def browse_directory(self):
        """Browse for data directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Data Directory",
            str(self.params.P.data_dir)
        )
        if dir_path:
            self.params.P.data_dir = Path(dir_path)
            self.dir_edit.setText(dir_path)
            self.rescan_files()

    def rescan_files(self):
        """Rescan files with current prefix"""
        # Update prefix in params
        self.params.P.filename_prefix = self.prefix_edit.text()

        # Scan files
        try:
            self.load_files()
        except Exception as e:
            QMessageBox.warning(
                self,
                "Scan Error",
                f"Failed to scan files:\n{str(e)}"
            )

    def load_files(self):
        """Load files and headers"""
        try:
            # Scan for files
            filenames = self.file_manager.scan_files()

            # Read headers
            df_headers = self.file_manager.read_headers()

            self.file_count_label.setText(f"Files: {len(filenames)}")

            # Populate table
            self._populate_header_table(df_headers)

            # Auto-select reference
            self.auto_select_reference()

            # Enable accept button
            self.btn_accept.setEnabled(len(filenames) > 0)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load files:\n{str(e)}"
            )

    def _populate_header_table(self, df_headers):
        self.header_table.setRowCount(len(df_headers))
        for i, row in df_headers.iterrows():
            self.header_table.setItem(i, 0, QTableWidgetItem(str(row["Filename"])))
            self.header_table.setItem(i, 1, QTableWidgetItem(str(row["DATE-OBS"])))
            self.header_table.setItem(i, 2, QTableWidgetItem(str(row["FILTER"])))
            self.header_table.setItem(i, 3, QTableWidgetItem(str(row["EXPTIME"])))
            self.header_table.setItem(i, 4, QTableWidgetItem(str(row["AIRMASS"])))
            self.header_table.setItem(i, 5, QTableWidgetItem(str(row["IMAGETYP"])))

    def auto_select_reference(self):
        """Automatically select reference frame"""
        try:
            ref_filename = self.file_manager.select_reference_frame()
            self.ref_label.setText(ref_filename)

            # Highlight reference row in table
            for i in range(self.header_table.rowCount()):
                item = self.header_table.item(i, 0)
                if item and item.text() == ref_filename:
                    self.header_table.selectRow(i)
                    break

        except Exception as e:
            QMessageBox.warning(
                self,
                "Reference Selection Error",
                f"Failed to select reference frame:\n{str(e)}"
            )

    def use_selected_as_reference(self):
        """Use currently selected row as reference frame"""
        current_row = self.header_table.currentRow()
        if current_row >= 0:
            item = self.header_table.item(current_row, 0)
            if item:
                ref_filename = item.text()
                self.file_manager.ref_filename = ref_filename
                self.ref_label.setText(ref_filename)
        else:
            QMessageBox.information(
                self,
                "No Selection",
                "Please select a row in the table first."
            )

    def accept_files(self):
        """Accept file selection and close dialog"""
        if not self.file_manager.filenames:
            QMessageBox.warning(
                self,
                "No Files",
                "No files have been loaded."
            )
            return

        if not self.file_manager.ref_filename:
            QMessageBox.warning(
                self,
                "No Reference",
                "No reference frame has been selected."
            )
            return

        # Success
        self.accept()
