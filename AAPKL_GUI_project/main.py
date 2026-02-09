#!/usr/bin/env python3
"""
AAPKI Light Curve Toolkit - KNUEMAO
Main entry point for GUI application
"""

import sys
import os
import warnings
from pathlib import Path

# Suppress astropy FITSFixedWarning (MJD-OBS auto-fix spam)
warnings.filterwarnings("ignore", message=".*datfix.*MJD-OBS.*", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="astropy")
try:
    from astropy.wcs import FITSFixedWarning
    warnings.filterwarnings("ignore", category=FITSFixedWarning)
except ImportError:
    pass

# Suppress matplotlib tight_layout warnings
warnings.filterwarnings("ignore", message=".*tight_layout.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*figure layout.*", category=UserWarning)

# Add aperture_photometry to Python path
sys.path.insert(0, str(Path(__file__).parent))

from PyQt5.QtWidgets import QApplication
from aperture_photometry.gui.main_window_workflow import MainWindowWorkflow


def main():
    """Main application entry point"""
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("AAPKI Light Curve Toolkit")
    app.setOrganizationName("KNUEMAO")

    # Set working directory to script location
    os.chdir(Path(__file__).parent)

    # Create and show main window
    try:
        window = MainWindowWorkflow()
        window.show()
    except Exception as e:
        print(f"Failed to initialize application: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run event loop
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
