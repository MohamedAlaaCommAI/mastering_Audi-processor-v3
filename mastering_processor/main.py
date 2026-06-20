"""Entry point for the Mastering Processor."""
from __future__ import annotations

import sys
import logging

from PyQt5.QtWidgets import QApplication

from mastering_processor.gui.main_window import MasteringGUI
from mastering_processor.gui.style import STYLE_SHEET


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE_SHEET)
    app.setApplicationName("Mastering Processor")

    window = MasteringGUI(samplerate=44100, blocksize=1024)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
