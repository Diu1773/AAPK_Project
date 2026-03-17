#!/usr/bin/env python3
"""
AAPK Light Curve Toolkit - KNUEMAO
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
from PyQt5.QtGui import QFont, QFontDatabase
from aperture_photometry.gui.main_window_workflow import MainWindowWorkflow


def _configure_app_fonts(app: QApplication) -> None:
    """Configure a Korean-capable font for both Qt and matplotlib."""
    candidate_font_files = [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "malgun.ttf",
        Path("/mnt/c/Windows/Fonts/malgun.ttf"),
    ]
    candidate_families = [
        "Malgun Gothic",
        "맑은 고딕",
        "NanumGothic",
        "Nanum Gothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "AppleGothic",
        "Arial Unicode MS",
    ]

    resolved_family = None
    try:
        import matplotlib
        from matplotlib import font_manager

        for font_path in candidate_font_files:
            if not font_path.exists():
                continue
            try:
                font_manager.fontManager.addfont(str(font_path))
                resolved_family = font_manager.FontProperties(fname=str(font_path)).get_name()
                break
            except Exception:
                continue

        if resolved_family is None:
            available = {f.name.lower(): f.name for f in font_manager.fontManager.ttflist}
            for family in candidate_families:
                if family.lower() in available:
                    resolved_family = available[family.lower()]
                    break

        if resolved_family:
            matplotlib.rcParams["font.family"] = [resolved_family, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
            matplotlib.rcParams["mathtext.default"] = "regular"
            matplotlib.rcParams["axes.formatter.use_mathtext"] = False
    except Exception:
        pass

    try:
        db = QFontDatabase()
        qt_families = {family.lower(): family for family in db.families()}
        qt_family = None

        if resolved_family and resolved_family.lower() in qt_families:
            qt_family = qt_families[resolved_family.lower()]
        else:
            for family in candidate_families:
                if family.lower() in qt_families:
                    qt_family = qt_families[family.lower()]
                    break

        if qt_family:
            app.setFont(QFont(qt_family, 9))
    except Exception:
        pass


def main():
    """Main application entry point"""
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("AAPKI Light Curve Toolkit")
    app.setOrganizationName("KNUEMAO")
    _configure_app_fonts(app)

    # Set working directory to script location
    os.chdir(Path(__file__).parent)

    # Create and show main window
    try:
        window = MainWindowWorkflow()
        window.show()
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Failed to initialize application: {e}")
        print(tb)
        from PyQt5.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Startup Error")
        msg.setText(str(e))
        msg.setDetailedText(tb)
        msg.exec_()
        return 1

    # Run event loop
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
